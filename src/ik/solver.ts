/**
 * The chunk solve (`dream_ik.solver.ChunkIK`): tool-pose targets to joint commands, one arm at a time.
 *
 * Per row (`solve`, the "tick" solve): reach the target tool pose (position 50 : angle 10 on the SE(3) log error), a
 * stiff push back inside the margined limits, a light pull toward the rest pose and a pull toward the row before,
 * solved by Levenberg-Marquardt from the row before; the solution is clipped to the margined limits, then the command
 * steps toward it under the per-joint speed caps, scaled by the most-limited joint so its direction is kept.
 *
 * The trajectory solve (`solveTraj`, "traj") solves every row in one problem, warm started from the tick solve: the
 * pose, limit and rest costs of each row, joint velocity, acceleration and jerk costs along the chunk, and the arm's
 * last three commands fixed as its history. Its answer is clipped and speed capped from the last command as the tick
 * solve's is. Obstacles, regions and the two-arm cost are not implemented: no built-in arm here has them.
 *
 * pyroki solves in float32 with analytic Jacobians; this is float64 with central differences, so the two agree to
 * a fraction of a millimetre on the tool path, not bit for bit.
 */

import { detail, DreamConfigError } from "../errors.js";
import { apply3, cross3, dot3, mul3, rotvec, transpose3, type Mat3, type Vec3 } from "../linalg.js";
import type { Arm } from "./arms.js";

/** A tool pose: position (m) and row-major rotation, in the arm's base frame. */
export interface Pose {
  position: Vec3;
  rotation: Mat3;
}

/** The trajectory solve's weights (`ServeSpec.traj_*`): residual weights on row differences. */
export interface TrajWeights {
  velocity: number;
  acceleration: number;
  jerk: number;
  /** Times the three above where the chunk meets the last three commands. */
  history: number;
}

export const TRAJ_WEIGHTS: Readonly<TrajWeights> = { velocity: 5, acceleration: 50, jerk: 200, history: 5 };
/** The traj solve's orientation weight on an arm with fewer than six free joints, as a share of its own. */
export const TRAJ_FEW_JOINTS_ORI_SCALE = 0.5;

/** What a chunk solve answers (`solver.IKResult`), one arm. */
export interface IkResult {
  /** `[rows][dof]` commanded joints, spec order. */
  q: number[][];
  /** `[rows][dof]` the solver's answer before the speed cap. */
  qSolution: number[][];
  /** Per row, metres between the commanded tool position and its target. */
  posMiss: number[];
  /** Per row, radians between the commanded tool rotation and its target. */
  angMiss: number[];
  /** Per row, whether the speed cap shortened the step. */
  atCap: boolean[];
}

const sumSq = (r: readonly number[]) => r.reduce((s, v) => s + v * v, 0);

/** `log(target^-1 * pose)` in SE(3): `[translation part, rotation vector]`, as jaxlie's `SE3.log`. */
function poseError(target: Pose, pose: Pose): number[] {
  const rt = transpose3(target.rotation);
  const r = mul3(rt, pose.rotation);
  const t = apply3(rt, [pose.position[0] - target.position[0], pose.position[1] - target.position[1], pose.position[2] - target.position[2]]);
  const omega = rotvec(r);
  const theta2 = dot3(omega, omega);
  const theta = Math.sqrt(theta2);
  // V^-1 = I - [w]x / 2 + (1 / theta^2) (1 - theta sin(theta) / (2 (1 - cos(theta)))) [w]x^2
  const c = theta < 1e-6 ? 1 / 12 : (1 - (theta * Math.sin(theta)) / (2 * (1 - Math.cos(theta)))) / theta2;
  const wxt = cross3(omega, t);
  const wxwxt = cross3(omega, wxt);
  return [t[0] - 0.5 * wxt[0] + c * wxwxt[0], t[1] - 0.5 * wxt[1] + c * wxwxt[1], t[2] - 0.5 * wxt[2] + c * wxwxt[2], ...omega];
}

/** One row's residuals: pose, the limits both ways, rest, and (the tick solve) the pull toward `qPrev`. */
function residual(arm: Arm, q: number[], target: Pose, qPrev: readonly number[] | null, lo: readonly number[], hi: readonly number[], ori: number): number[] {
  const w = arm.data.weights;
  const e = poseError(target, arm.tool(q));
  const out = e.map((v, i) => v * (i < 3 ? w.pos! : ori));
  const n = arm.dof;
  for (let j = 0; j < n; j++) out.push(w.limit! * Math.max(0, q[j]! - hi[j]!));
  for (let j = 0; j < n; j++) out.push(w.limit! * Math.max(0, lo[j]! - q[j]!));
  for (let j = 0; j < n; j++) out.push((q[j]! - arm.data.restPose[j]!) * arm.data.restWeight[j]!);
  if (qPrev) for (let j = 0; j < n; j++) out.push((q[j]! - qPrev[j]!) * w.last_pose!);
  return out;
}

/** Central-difference Jacobian of a residual function, `[residuals][dof]`. */
function numericJacobian(f: (q: number[]) => number[], q: readonly number[], size: number): number[][] {
  const h = 1e-6;
  const jac = Array.from({ length: size }, () => new Array<number>(q.length).fill(0));
  for (let j = 0; j < q.length; j++) {
    const up = [...q];
    const down = [...q];
    up[j]! += h;
    down[j]! -= h;
    const ru = f(up);
    const rd = f(down);
    for (let i = 0; i < size; i++) jac[i]![j] = (ru[i]! - rd[i]!) / (2 * h);
  }
  return jac;
}

/**
 * The normal equations of a least-squares problem at a point: `g = J^T r` and `N = J^T J`, stored dense (`band` the
 * size) or as a symmetric band (`band` diagonals below the main one, row-major with stride `band + 1`).
 */
interface Normal {
  cost: number;
  gradient: Float64Array;
  matrix: Float64Array;
  band: number;
}

const at = (n: Normal, i: number, j: number): number => (j > i ? at(n, j, i) : i - j > n.band ? 0 : n.matrix[i * (n.band + 1) + i - j]!);

/** Solves `(S N S + lambda I) x = b` by Cholesky on the band; null when it is not positive definite. */
function solveDamped(n: Normal, scaler: Float64Array, lambda: number, b: Float64Array): Float64Array | null {
  const size = b.length;
  const band = n.band;
  const stride = band + 1;
  const l = new Float64Array(size * stride);
  for (let i = 0; i < size; i++) {
    for (let j = Math.max(0, i - band); j <= i; j++) {
      let s = at(n, i, j) * scaler[i]! * scaler[j]! + (i === j ? lambda : 0);
      for (let k = Math.max(0, i - band, j - band); k < j; k++) s -= l[i * stride + i - k]! * l[j * stride + j - k]!;
      if (i === j) {
        if (!(s > 0) || !Number.isFinite(s)) return null;
        l[i * stride] = Math.sqrt(s);
      } else l[i * stride + i - j] = s / l[j * stride]!;
    }
  }
  const x = new Float64Array(size);
  for (let i = 0; i < size; i++) {
    let s = b[i]!;
    for (let j = Math.max(0, i - band); j < i; j++) s -= l[i * stride + i - j]! * x[j]!;
    x[i] = s / l[i * stride]!;
  }
  for (let i = size - 1; i >= 0; i--) {
    let s = x[i]!;
    for (let j = i + 1; j <= Math.min(size - 1, i + band); j++) s -= l[j * stride + j - i]! * x[j]!;
    x[i] = s / l[i * stride]!;
  }
  return x;
}

/** `v^T N v`. */
function quadratic(n: Normal, v: Float64Array): number {
  let total = 0;
  for (let i = 0; i < v.length; i++) {
    total += at(n, i, i) * v[i]! * v[i]!;
    for (let j = Math.max(0, i - n.band); j < i; j++) total += 2 * at(n, i, j) * v[i]! * v[j]!;
  }
  return total;
}

/** jaxls's Levenberg-Marquardt settings as dream_ik runs it (`TrustRegionConfig(lambda_initial=1.0)`, defaults else). */
const LM = { lambdaInitial: 1.0, lambdaFactor: 2.0, lambdaMin: 1e-5, lambdaMax: 1e6, stepQualityMin: 1e-3, maxIterations: 100, costTolerance: 1e-5, gradientTolerance: 1e-4, gradientStartStep: 10, parameterTolerance: 1e-6 };

/**
 * Levenberg-Marquardt as jaxls solves it: the Jacobian's columns scaled once at the start (`1 / (1 + norm) + 1`), a
 * step accepted when its actual reduction is at least 1e-3 of the linear model's, the damping doubled (growing
 * geometrically) on a refusal and halved on an acceptance, and the stop at a relative cost change under 1e-5, a
 * gradient under 1e-4 (from the tenth try), a step under 1e-6 of the point, or 100 tries in all.
 *
 * @param evaluate The cost at a point, and with `linearize` its normal equations.
 */
function levenbergMarquardt(x0: Float64Array, evaluate: (x: Float64Array, linearize: boolean) => Normal): Float64Array {
  let x = x0;
  let state = evaluate(x, true);
  const size = x.length;
  const scaler = new Float64Array(size);
  for (let i = 0; i < size; i++) scaler[i] = 1 / (1 + Math.sqrt(at(state, i, i))) + 1;
  let lambda = LM.lambdaInitial;
  let iterations = 0;
  let lastAccepted = true;
  let terminated = false;
  while (iterations < LM.maxIterations && !terminated && !(!lastAccepted && lambda >= LM.lambdaMax) && Number.isFinite(state.cost)) {
    const atb = new Float64Array(size);
    for (let i = 0; i < size; i++) atb[i] = -scaler[i]! * state.gradient[i]!;
    let tryLambda = lambda / LM.lambdaFactor;
    let growth = LM.lambdaFactor;
    let accepted = false;
    let proposal: Float64Array | null = null;
    while (!accepted && !terminated && tryLambda < LM.lambdaMax && iterations < LM.maxIterations) {
      tryLambda = Math.min(tryLambda * growth, LM.lambdaMax);
      growth *= 2;
      iterations++;
      const delta = solveDamped(state, scaler, tryLambda, atb);
      if (delta === null) continue;
      const dx = delta.map((v, i) => v * scaler[i]!);
      proposal = x.map((v, i) => v + dx[i]!);
      const cost = evaluate(proposal, false).cost;
      const predicted = 2 * delta.reduce((s, v, i) => s + v * atb[i]!, 0) - quadratic(state, dx);
      accepted = !Number.isNaN(cost) && (state.cost - cost) / predicted >= LM.stepQualityMin;
      const gradient = iterations >= LM.gradientStartStep && Math.max(...atb.map(Math.abs)) < LM.gradientTolerance;
      const costConverged = accepted && Math.abs(cost - state.cost) / state.cost < LM.costTolerance;
      const parameters = accepted && Math.hypot(...dx) / (Math.hypot(...x) + LM.parameterTolerance) < LM.parameterTolerance;
      terminated = costConverged || gradient || parameters;
    }
    lambda = Math.max(LM.lambdaMin, accepted ? tryLambda / LM.lambdaFactor : tryLambda);
    lastAccepted = accepted;
    if (accepted) {
      x = proposal!;
      state = evaluate(x, true);
    }
  }
  return x;
}

/** The normal equations of a residual function at `q` (dense), by central differences. */
function linearize(f: (q: number[]) => number[], q: Float64Array, withDerivatives: boolean): Normal {
  const point = Array.from(q);
  const r = f(point);
  const n = point.length;
  const normal: Normal = { cost: sumSq(r), gradient: new Float64Array(n), matrix: new Float64Array(n * n), band: n - 1 };
  if (!withDerivatives) return normal;
  const jac = numericJacobian(f, point, r.length);
  for (let a = 0; a < n; a++) {
    normal.gradient[a] = jac.reduce((s, row, i) => s + row[a]! * r[i]!, 0);
    for (let b = 0; b <= a; b++) normal.matrix[a * n + a - b] = jac.reduce((s, row) => s + row[a]! * row[b]!, 0);
  }
  return normal;
}

/** One row by Levenberg-Marquardt from `qPrev` (the row before's solution). */
function solveRow(arm: Arm, target: Pose, qPrev: readonly number[], lo: readonly number[], hi: readonly number[]): number[] {
  const f = (q: number[]) => residual(arm, q, target, qPrev, lo, hi, arm.data.weights.ori!);
  return Array.from(levenbergMarquardt(Float64Array.from(qPrev), (q, d) => linearize(f, q, d)));
}

const clip = (q: readonly number[], lo: readonly number[], hi: readonly number[]) => q.map((v, j) => Math.min(hi[j]!, Math.max(lo[j]!, v)));

/**
 * Per joint, the largest step per row: `min(maxSpeed, speedCap) / hz`.
 *
 * @throws DreamConfigError `speedCap` is not positive (Infinity is), or `hz` is not a positive, finite number.
 */
export function maxStep(arm: Arm, speedCap: number, hz: number): number[] {
  if (!(typeof speedCap === "number" && speedCap > 0)) throw new DreamConfigError(`speedCap must be a positive number of rad/s (Infinity for none), got ${speedCap}`, detail("range", "speedCap"));
  if (!(typeof hz === "number" && hz > 0 && Number.isFinite(hz))) throw new DreamConfigError(`hz must be a positive, finite number of rows per second, got ${hz}`, detail("range", "hz"));
  return arm.data.maxSpeed.map((v) => Math.min(v, speedCap) / hz);
}

/** The command toward `target`, scaled by the most-limited joint so its direction is kept; and whether it was capped. */
export function capStep(prev: readonly number[], target: readonly number[], step: readonly number[]): { q: number[]; capped: boolean } {
  const delta = target.map((v, j) => v - prev[j]!);
  const scale = Math.min(1, ...delta.map((d, j) => step[j]! / Math.max(Math.abs(d), 1e-12)));
  return { q: prev.map((v, j) => v + scale * delta[j]!), capped: scale < 1 };
}

function result(arm: Arm, targets: readonly Pose[], q: number[][], qSolution: number[][], atCap: boolean[]): IkResult {
  const posMiss: number[] = [];
  const angMiss: number[] = [];
  q.forEach((row, k) => {
    const reached = arm.tool(row);
    const target = targets[k]!;
    posMiss.push(Math.hypot(...reached.position.map((v, i) => v - target.position[i]!)));
    angMiss.push(Math.hypot(...rotvec(mul3(reached.rotation, transpose3(target.rotation)))));
  });
  return { q, qSolution, posMiss, angMiss, atCap };
}

/**
 * The tick solve of one arm's chunk (`ChunkIK.solve`): row by row from `q0`, each warm started from and pulled toward
 * the row before's solution, under the speed caps.
 *
 * @param q0 `[dof]` the current joints, spec order.
 * @param speedCap The caller's own per-joint cap, rad/s; Infinity for none (the arm's caps still hold).
 * @param hz Rows per second.
 */
export function solve(arm: Arm, targets: readonly Pose[], q0: readonly number[], speedCap: number, hz: number): IkResult {
  // A joint already past its margin may stay there, never go further.
  const lo = arm.lo.map((v, j) => Math.min(v, q0[j]!));
  const hi = arm.hi.map((v, j) => Math.max(v, q0[j]!));
  const step = maxStep(arm, speedCap, hz);
  let qCmd = [...q0];
  let qPrev = [...q0];
  const q: number[][] = [];
  const qSolution: number[][] = [];
  const atCap: boolean[] = [];
  for (const target of targets) {
    const qSol = clip(solveRow(arm, target, qPrev, lo, hi), lo, hi);
    const { q: next, capped } = capStep(qCmd, qSol, step);
    qCmd = next;
    qPrev = qSol;
    q.push(next);
    qSolution.push(qSol);
    atCap.push(capped);
  }
  return result(arm, targets, q, qSolution, atCap);
}

/**
 * The trajectory solve of one arm's chunk (`ChunkIK.solve_traj`): every row at once, continuing the last commands.
 *
 * @param q0 `[dof]` the last command, spec order.
 * @param qBefore The commands before it, newest first: one (`[dof]`) or two rows (`[2][dof]`; the second sets the
 *   jerk into the chunk). Pass `[q0, q0]` for an arm that was still.
 */
export function solveTraj(
  arm: Arm,
  targets: readonly Pose[],
  q0: readonly number[],
  qBefore: readonly (readonly number[])[],
  speedCap: number,
  hz: number,
  weights: Readonly<TrajWeights> = TRAJ_WEIGHTS,
): IkResult {
  const n = arm.dof;
  const third = qBefore.length >= 2;
  const before = qBefore[0] ?? q0;
  const before2 = third ? qBefore[1]! : before;
  const warm = solve(arm, targets, q0, Infinity, hz);
  const lo = arm.lo.map((v, j) => Math.min(v, q0[j]!));
  const hi = arm.hi.map((v, j) => Math.max(v, q0[j]!));
  const free = arm.lo.filter((v, j) => arm.hi[j]! > v).length;
  const ori = free >= 6 ? arm.data.weights.ori! : arm.data.weights.ori! * TRAJ_FEW_JOINTS_ORI_SCALE;
  const local = (row: number[], t: number) => residual(arm, row, targets[t]!, null, lo, hi, ori);
  const band = 3 * n;
  const stride = band + 1;
  const rows = targets.length;
  const evaluate = (flat: Float64Array, derivatives: boolean): Normal => {
    const path = Array.from({ length: rows }, (_, t) => Array.from(flat.subarray(t * n, (t + 1) * n)));
    const normal = new Float64Array(rows * n * stride);
    const gradient = new Float64Array(rows * n);
    let cost = 0;
    path.forEach((row, t) => {
      const r = local(row, t);
      cost += sumSq(r);
      if (!derivatives) return;
      const jac = numericJacobian((q) => local(q, t), row, r.length);
      for (let j = 0; j < n; j++) {
        const index = t * n + j;
        gradient[index]! += jac.reduce((s, line, i) => s + line[j]! * r[i]!, 0);
        for (let k = 0; k <= j; k++) normal[index * stride + j - k]! += jac.reduce((s, line) => s + line[j]! * line[k]!, 0);
      }
    });
    // Negative rows are the fixed command history: -1 the last, -2 the one before, -3 the one before that.
    const value = (t: number, j: number) => (t >= 0 ? path[t]![j]! : t === -1 ? q0[j]! : t === -2 ? before[j]! : before2[j]!);
    const temporal = (ticks: number[], coefficients: number[], weight: number) => {
      if (!weight || ticks.some((t) => t >= rows)) return;
      for (let j = 0; j < n; j++) {
        const r = weight * ticks.reduce((s, t, i) => s + coefficients[i]! * value(t, j), 0);
        cost += r * r;
        if (!derivatives) continue;
        for (let a = 0; a < ticks.length; a++) {
          if (ticks[a]! < 0) continue;
          const ia = ticks[a]! * n + j;
          const ca = coefficients[a]! * weight;
          gradient[ia]! += ca * r;
          for (let b = 0; b < ticks.length; b++) {
            if (ticks[b]! < 0) continue;
            const ib = ticks[b]! * n + j;
            if (ia >= ib) normal[ia * stride + ia - ib]! += ca * coefficients[b]! * weight;
          }
        }
      }
    };
    const { velocity, acceleration, jerk, history } = weights;
    for (let t = 1; t < rows; t++) temporal([t, t - 1], [1, -1], velocity);
    for (let t = 2; t < rows; t++) temporal([t, t - 1, t - 2], [1, -2, 1], acceleration);
    for (let t = 3; t < rows; t++) temporal([t, t - 1, t - 2, t - 3], [1, -3, 3, -1], jerk);
    temporal([0, -1], [1, -1], history * velocity);
    temporal([0, -1, -2], [1, -2, 1], history * acceleration);
    temporal([1, 0, -1], [1, -2, 1], history * acceleration);
    temporal([0, -1, -2, -3], [1, -3, 3, -1], third ? history * jerk : 0);
    temporal([1, 0, -1, -2], [1, -3, 3, -1], history * jerk);
    temporal([2, 1, 0, -1], [1, -3, 3, -1], history * jerk);
    return { cost, gradient, matrix: normal, band };
  };
  const solved = levenbergMarquardt(Float64Array.from(warm.qSolution.flat()), evaluate);
  const path = Array.from({ length: rows }, (_, t) => Array.from(solved.subarray(t * n, (t + 1) * n)));
  const qSolution = path.map((row) => clip(row, lo, hi));
  const step = maxStep(arm, speedCap, hz);
  let prev = [...q0];
  const q: number[][] = [];
  const atCap: boolean[] = [];
  for (const row of qSolution) {
    const { q: next, capped } = capStep(prev, row, step);
    prev = next;
    q.push(next);
    atCap.push(capped);
  }
  return result(arm, targets, q, qSolution, atCap);
}
