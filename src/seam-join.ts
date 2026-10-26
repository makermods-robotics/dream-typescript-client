/**
 * The seam join (`dream.seam_join`): how each chunk takes over from the motion the arm is on.
 *
 * The join replaces the rows of the new plan from the clock row through {@link RAMP_ROWS} after it. Those rows start
 * exactly where the old motion goes next, carry its velocity, blend it into the new plan's velocity, and land exactly
 * on the plan's row at the end of the window: per step `k` of `N`, `u_k = (1 - w_k) v_old + w_k (plan[k+1] - plan[k])`
 * with `w_k = (k + 1) / (N + 1)`, integrated from the old motion's next position, the miss removed along the quintic
 * smoothstep. The blend runs in joint space, or on the end effector's pose when the caller gives the arm's
 * {@link Kinematics}, solved back to joints with damped least squares; joint space is the fallback when that fails.
 */

import { invert, mul3, rotationMatrix, rotvec, solveLinear, transpose3, type Mat3, type Vec3 } from "./linalg.js";

/** Rows the join spans after the clock row. */
export const RAMP_ROWS = 16;
/** Damping of the least-squares IK step, `dq = J^T (J J^T + lambda^2 I)^-1 e`. */
export const IK_DAMPING = 1e-3;
export const IK_MAX_ITERATIONS = 8;
export const IK_TOLERANCE_M = 1e-5;
export const IK_TOLERANCE_RAD = 1e-4;
export const FALLBACK_RESIDUAL_M = 1e-3;
export const FALLBACK_RESIDUAL_RAD = 0.01;
/** Largest move of any joint from a row's seed to its solution before the seam falls back to joint space. */
export const FALLBACK_SEED_JUMP_RAD = 0.25;
/** Times, in rows relative to the seam, of the old-motion rows the fit takes. */
export const OLD_PATH_TIMES = [-1, 0, 1, 2] as const;

export type Rows = number[][];

/** One arm's forward kinematics at the end effector the seam is joined on (`seam_join.Kinematics`). */
export interface Kinematics {
  /** The end effector's pose for the joints (radians): position in metres and a row-major rotation, base frame. */
  pose(joints: readonly number[]): { position: Vec3; rotation: Mat3 };
  /** `[6][joints]`: rows 0..2 the linear velocity of the end-effector point, rows 3..5 the angular velocity. */
  jacobian(joints: readonly number[]): number[][];
}

/** `10s^3 - 15s^4 + 6s^5`: 0 at 0, 1 at 1, zero velocity and acceleration at both ends. */
export const smoothstep5 = (s: number): number => s * s * s * (10 + s * (-15 + 6 * s));

/**
 * The old motion's position and per-row velocity at time 0, from a least-squares fit through its rows: a quadratic
 * through three or more rows, a line through two, a constant (at rest) through one.
 */
export function fitMotion(rows: Rows, times: readonly number[]): { now: number[]; velocity: number[] } {
  const columns = rows[0]!.length;
  if (rows.length === 1) return { now: [...rows[0]!], velocity: new Array<number>(columns).fill(0) };
  const order = Math.min(3, rows.length);
  const basis = times.map((t) => Array.from({ length: order }, (_, p) => t ** p));
  const normal = Array.from({ length: order }, (_, i) => Array.from({ length: order }, (_, j) => basis.reduce((s, b) => s + b[i]! * b[j]!, 0)));
  const inverse = invert(normal);
  const coefficient = (p: number, c: number) => basis.reduce((s, b, k) => s + inverse[p]!.reduce((t, v, q) => t + v * b[q]!, 0) * rows[k]![c]!, 0);
  return {
    now: Array.from({ length: columns }, (_, c) => coefficient(0, c)),
    velocity: Array.from({ length: columns }, (_, c) => coefficient(1, c)),
  };
}

/** Step `k` of `N` is `(1 - w_k) velocity + w_k planSteps[k]`, `w_k = (k + 1) / (N + 1)`. */
export function blendedSteps(velocity: readonly number[], planSteps: Rows): Rows {
  const count = planSteps.length;
  return planSteps.map((step, k) => {
    const w = (k + 1) / (count + 1);
    return step.map((v, c) => (1 - w) * velocity[c]! + w * v);
  });
}

/**
 * Joins the old motion into `plan` over its rows: row 0 is `now`, the last row is `plan`'s exactly.
 *
 * @param plan The plan's window rows, `[N + 1][columns]`, `N` at least 1.
 */
export function velocityBlend(now: readonly number[], velocity: readonly number[], plan: Rows, landing = true): Rows {
  const steps = blendedSteps(velocity, plan.slice(1).map((row, k) => row.map((v, c) => v - plan[k]![c]!)));
  const path: Rows = [[...now]];
  for (const step of steps) path.push(path.at(-1)!.map((v, c) => v + step[c]!));
  if (!landing) return path;
  const count = plan.length - 1;
  const miss = path[count]!.map((v, c) => v - plan[count]![c]!);
  return path.map((row, k) => row.map((v, c) => v - smoothstep5(k / count) * miss[c]!));
}

const pose6 = (position: Vec3, rotation: Mat3, reference: Mat3): number[] => [...position, ...rotvec(mul3(rotation, transpose3(reference)))];

/**
 * The velocity blend done on the end effector's pose and solved back to joints, or null when the solve fails
 * (`seam_join.eef_window`).
 */
export function eefWindow(oldPath: Rows, times: readonly number[], plan: Rows, jointBlend: Rows, kinematics: Kinematics): Rows | null {
  const count = plan.length - 1;
  const oldPoses = oldPath.map((row) => kinematics.pose(row));
  let nearest = 0;
  times.forEach((t, i) => {
    if (Math.abs(t) < Math.abs(times[nearest]!)) nearest = i;
  });
  const reference = oldPoses[nearest]!.rotation;
  const { now, velocity } = fitMotion(oldPoses.map((p) => pose6(p.position, p.rotation, reference)), times);
  const planPoses = plan.map((row) => kinematics.pose(row));
  const planSteps = planPoses.slice(1).map((p1, k) => {
    const p0 = planPoses[k]!;
    return [...p1.position.map((v, i) => v - p0.position[i]!), ...rotvec(mul3(p1.rotation, transpose3(p0.rotation)))];
  });
  const steps = blendedSteps(velocity, planSteps);
  const positions: Vec3[] = [[now[0]!, now[1]!, now[2]!]];
  const rotations: Mat3[] = [mul3(rotationMatrix([now[3]!, now[4]!, now[5]!]), reference)];
  for (const step of steps) {
    const p = positions.at(-1)!;
    positions.push([p[0] + step[0]!, p[1] + step[1]!, p[2] + step[2]!]);
    rotations.push(mul3(rotationMatrix([step[3]!, step[4]!, step[5]!]), rotations.at(-1)!));
  }
  const last = planPoses[count]!;
  const missP = positions[count]!.map((v, i) => v - last.position[i]!);
  const missR = rotvec(mul3(rotations[count]!, transpose3(last.rotation)));
  const out = jointBlend.map((row) => [...row]);
  out[count] = [...plan[count]!];
  for (let k = 1; k < count; k++) {
    const w = smoothstep5(k / count);
    const targetP = positions[k]!.map((v, i) => v - w * missP[i]!) as Vec3;
    const targetR = mul3(rotationMatrix([-w * missR[0], -w * missR[1], -w * missR[2]]), rotations[k]!);
    const seed = out[k - 1]!.map((v, j) => v + (jointBlend[k]![j]! - jointBlend[k - 1]![j]!));
    const solved = solve(kinematics, seed, targetP, targetR);
    if (solved === null || Math.max(...solved.map((v, j) => Math.abs(v - seed[j]!))) > FALLBACK_SEED_JUMP_RAD) return null;
    out[k] = solved;
  }
  return out;
}

/**
 * Joint angles that put the end effector at the target pose, by damped least squares from `seed`; null when the final
 * error exceeds {@link FALLBACK_RESIDUAL_M} or {@link FALLBACK_RESIDUAL_RAD}. Joint limits are not applied.
 */
export function solve(kinematics: Kinematics, seed: readonly number[], targetP: Readonly<Vec3>, targetR: Readonly<Mat3>): number[] | null {
  let joints = [...seed];
  let errorP = Infinity;
  let errorR = Infinity;
  const damping = IK_DAMPING * IK_DAMPING;
  for (let iteration = 0; iteration <= IK_MAX_ITERATIONS; iteration++) {
    const { position, rotation } = kinematics.pose(joints);
    const error = [...targetP.map((v, i) => v - position[i]!), ...rotvec(mul3(targetR, transpose3(rotation)))];
    errorP = Math.hypot(error[0]!, error[1]!, error[2]!);
    errorR = Math.hypot(error[3]!, error[4]!, error[5]!);
    if (errorP < IK_TOLERANCE_M && errorR < IK_TOLERANCE_RAD) return joints;
    if (iteration === IK_MAX_ITERATIONS) break;
    const jacobian = kinematics.jacobian(joints);
    const jjt = jacobian.map((a, i) => jacobian.map((b, j) => a.reduce((s, v, k) => s + v * b[k]!, 0) + (i === j ? damping : 0)));
    const y = solveLinear(jjt, error);
    if (y === null) return null;
    joints = joints.map((q, k) => q + jacobian.reduce((s, row, i) => s + row[k]! * y[i]!, 0));
  }
  return errorP > FALLBACK_RESIDUAL_M || errorR > FALLBACK_RESIDUAL_RAD ? null : joints;
}
