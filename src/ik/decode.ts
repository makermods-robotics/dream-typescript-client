/**
 * The EEF head decoded to joints (`dream_ik.decode`): Savitzky-Golay over the hand path, the chunk solve from the
 * joints the chunk continues at, and the seam blend that eases in from them.
 */

import { detail, DreamActionError, DreamConfigError } from "../errors.js";
import { firstFailing, firstHole, isNumbers } from "../finite.js";
import { mul3, rot6ToMat, rotationMatrix, rotvec, transpose3, type Vec3 } from "../linalg.js";
import { savgolFilter } from "../smoothing.js";
import type { Arm } from "./arms.js";
import { maxStep, solve, solveTraj, TRAJ_WEIGHTS, type IkResult, type Pose, type TrajWeights } from "./solver.js";

/** How an EEF head is solved (`dream_ik.specs.ServeSpec`). */
export interface ServeSpec {
  /** Savitzky-Golay window over the hand targets, rows, odd; 0 turns it off. */
  smoothWindow: number;
  smoothPolyorder: number;
  /** Rows a tick solve eases in over from the last command; 0 still clips and caps. */
  seamBlendTicks: number;
  solve: "tick" | "traj";
  traj: TrajWeights;
}

/** dream_ik's `SERVE`: what dream's own decode runs. */
export const SERVE: Readonly<ServeSpec> = { smoothWindow: 15, smoothPolyorder: 3, seamBlendTicks: 12, solve: "tick", traj: TRAJ_WEIGHTS };

/** Most rows one chunk solve takes (Dream-1 answers 48): the solve's cost grows with them. */
export const MAX_SOLVE_ROWS = 512;

/**
 * Tool poses from `[rows][3]` positions and `[rows][6]` rotation_6d (`decode.head_poses`).
 *
 * @param keys What the two are called in errors, e.g. the reply's `action.right_eef_xyz` and `action.right_eef_rot6`.
 * @throws DreamActionError They are not 1 to {@link MAX_SOLVE_ROWS} matching rows of 3 and 6 finite numbers, or a
 *   rotation_6d has a near-zero or near-parallel column.
 */
export function headPoses(xyz: readonly (readonly number[])[], rot6: readonly (readonly number[])[], keys: readonly [string, string] = ["xyz", "rot6"]): Pose[] {
  if (!Array.isArray(xyz) || !Array.isArray(rot6) || xyz.length < 1 || xyz.length > MAX_SOLVE_ROWS || rot6.length !== xyz.length) {
    throw new DreamActionError(
      `${keys[0]} and ${keys[1]} must be matching rows, 1 to ${MAX_SOLVE_ROWS}, got ${Array.isArray(xyz) ? xyz.length : typeof xyz} and ${Array.isArray(rot6) ? rot6.length : typeof rot6}`,
      detail(Array.isArray(xyz) && Array.isArray(rot6) && rot6.length === xyz.length ? "range" : "shape", keys[0]),
    );
  }
  // Array.from, not map(): map() skips a hole, which would come back as a pose of `undefined`.
  return Array.from(xyz, (p, k) => {
    const r = rot6[k]!;
    for (const [key, row, width] of [[keys[0], p, 3], [keys[1], r, 6]] as const) {
      if (!(Array.isArray(row) || ArrayBuffer.isView(row)) || (row as ArrayLike<unknown>).length !== width || firstHole(row as ArrayLike<unknown>) >= 0) {
        throw new DreamActionError(`${key} row ${k} is not ${width} numbers`, detail("shape", key, k));
      }
      if (!isNumbers(row, true)) {
        throw new DreamActionError(`${key} row ${k} holds a non-finite number`, detail("non_finite", key, k));
      }
    }
    const rotation = rot6ToMat(r);
    if (rotation === null) {
      throw new DreamActionError(
        `${keys[1]} row ${k} names no rotation: a column of near-zero norm, or two near-parallel columns (${JSON.stringify(r)})`,
        detail("degenerate_rotation", keys[1], k),
      );
    }
    return { position: [p[0]!, p[1]!, p[2]!], rotation };
  });
}

/**
 * Savitzky-Golay over the rows of poses: positions as they are, rotations as rotation vectors relative to the first
 * row. The window shrinks to the chunk (odd); 0 or too short is off (`decode.smooth_poses`).
 */
export function smoothPoses(poses: readonly Pose[], window: number = SERVE.smoothWindow, polyorder: number = SERVE.smoothPolyorder): Pose[] {
  const n = poses.length;
  const w = Math.min(window | 1, n % 2 ? n : n - 1);
  if (window <= 0 || w <= polyorder) return [...poses];
  const r0 = poses[0]!.rotation;
  const r0t = transpose3(r0);
  const positions = savgolFilter(poses.map((p) => [...p.position]), w, polyorder);
  const vectors = savgolFilter(poses.map((p) => [...rotvec(mul3(r0t, p.rotation))]), w, polyorder);
  return poses.map((_, k) => ({ position: positions[k] as Vec3, rotation: mul3(r0, rotationMatrix(vectors[k] as Vec3)) }));
}

/**
 * The poses with `rows` more: the last `k` rows' mean step continued, the position in a line and the rotation about a
 * fixed axis of the hand's own frame (`decode.extend_poses`).
 */
export function extendPoses(poses: readonly Pose[], rows: number, k = 4): Pose[] {
  if (rows <= 0) return [...poses];
  const span = Math.min(k, poses.length - 1);
  const last = poses.at(-1)!;
  // One pose has no motion to continue: it is held (Python divides by the zero span and answers NaN).
  if (span < 1) return [...poses, ...Array.from({ length: rows }, () => last)];
  const back = poses[poses.length - 1 - span]!;
  const dp = last.position.map((v, i) => (v - back.position[i]!) / span);
  const w = rotvec(mul3(transpose3(back.rotation), last.rotation)).map((v) => v / span) as Vec3;
  const out = [...poses];
  for (let s = 1; s <= rows; s++) {
    out.push({
      position: [last.position[0] + s * dp[0]!, last.position[1] + s * dp[1]!, last.position[2] + s * dp[2]!],
      rotation: mul3(last.rotation, rotationMatrix([s * w[0], s * w[1], s * w[2]])),
    });
  }
  return out;
}

/**
 * Eases a chunk in from the last command (`decode.seam_blend`): the plan shifted by its start offset, the shift fading
 * to zero over `ticks` rows, clipped to the margined limits and each row kept within `step` of the one before.
 */
export function seamBlend(q: readonly (readonly number[])[], qFrom: readonly number[], ticks: number, lo: readonly number[], hi: readonly number[], step: readonly number[]): number[][] {
  const shift = qFrom.map((v, j) => v - q[0]![j]!);
  let prev = [...qFrom];
  return q.map((row, t) => {
    const w = Math.min(1, Math.max(0, 1 - (t + 1) / (ticks + 1)));
    const eased = row.map((v, j) => Math.min(hi[j]!, Math.max(lo[j]!, v + w * shift[j]!)));
    prev = eased.map((v, j) => Math.min(prev[j]! + step[j]!, Math.max(prev[j]! - step[j]!, v)));
    return prev;
  });
}

/** Options of {@link solveChunk}. */
export interface SolveChunkOptions {
  /** The caller's own per-joint cap, rad/s; Infinity (the default) for none: the arm's caps still hold. */
  speedCap?: number;
  /** Rows per second the chunk is played at: the speed caps are per row. */
  hz: number;
  serve?: Partial<ServeSpec>;
  /** The trajectory solve's history: the commands before `qFrom`, newest first (two set the jerk into the chunk); `[qFrom]` when left out. */
  qBefore?: readonly (readonly number[])[];
}

/**
 * One arm's chunk, the three steps of the contract (`decode.solve_chunk`): Savitzky-Golay over the hand path, the
 * solve from `qFrom` under the caps, the seam blend from `qFrom` (tick only).
 *
 * @param xyz `[rows][3]` the head's tool positions, metres, in the arm's base frame.
 * @param rot6 `[rows][6]` its rotations, rotation_6d.
 * @param qFrom `[dof]` the joints the chunk continues from, spec order.
 * @returns The joint commands (`q`, `[rows][dof]`) and the solver's misses, all finite.
 * @throws DreamConfigError An option is out of range: `hz` not positive and finite, `speedCap` not positive (Infinity
 *   is), `qFrom` or `qBefore` (one or two rows) not the arm's joints as finite numbers, an unknown solve, `fences`.
 * @throws DreamActionError The targets are not 1 to {@link MAX_SOLVE_ROWS} matching rows of finite numbers, a rotation
 *   is degenerate, or the solve answered a non-finite number.
 */
export function solveChunk(arm: Arm, xyz: readonly (readonly number[])[], rot6: readonly (readonly number[])[], qFrom: readonly number[], options: SolveChunkOptions): IkResult {
  if ((options.serve as { fences?: unknown } | undefined)?.fences) {
    throw new DreamConfigError("solveChunk: ServeSpec.fences (the hard keep-out and two-arm holds) is not implemented here", detail("unsupported", "serve.fences"));
  }
  const serve = { ...SERVE, ...options.serve };
  if (serve.solve !== "tick" && serve.solve !== "traj") throw new DreamConfigError(`solveChunk: serve.solve must be 'tick' or 'traj', got ${JSON.stringify(serve.solve)}`, detail("range", "serve.solve"));
  const speedCap = options.speedCap ?? Infinity;
  maxStep(arm, speedCap, options.hz); // checks both
  const joints = (row: unknown, key: string, index: number | null): readonly number[] => {
    const values = Array.isArray(row) || ArrayBuffer.isView(row) ? (row as ArrayLike<unknown>) : null;
    if (values === null || values.length !== arm.dof) {
      throw new DreamConfigError(`solveChunk: ${key}${index === null ? "" : ` row ${index}`} must be ${arm.name}'s ${arm.dof} joints, got ${values?.length ?? typeof row}`, detail("shape", key, index));
    }
    if (!isNumbers(values, true)) {
      throw new DreamConfigError(`solveChunk: ${key}${index === null ? "" : ` row ${index}`} holds a non-finite joint`, detail("non_finite", key, index));
    }
    return row as readonly number[];
  };
  joints(qFrom, "qFrom", null);
  const qBefore = options.qBefore ?? [qFrom];
  if (!Array.isArray(qBefore) || qBefore.length < 1 || qBefore.length > 2) {
    throw new DreamConfigError(`solveChunk: qBefore must be one or two rows of joints, newest first, got ${Array.isArray(qBefore) ? qBefore.length : typeof qBefore}`, detail("shape", "qBefore"));
  }
  // Indices, not forEach(): it skips a hole, which the solve would then read as `undefined` joints.
  for (let i = 0; i < qBefore.length; i++) joints(qBefore[i], "qBefore", i);
  const targets = smoothPoses(headPoses(xyz, rot6), serve.smoothWindow, serve.smoothPolyorder);
  // As dream_ik: no history means the command before was qFrom itself (one row, so no jerk into the chunk).
  let res = serve.solve === "traj" ? solveTraj(arm, targets, qFrom, qBefore, speedCap, options.hz, serve.traj) : solve(arm, targets, qFrom, speedCap, options.hz);
  if (serve.solve === "tick") res = { ...res, q: seamBlend(res.q, qFrom, serve.seamBlendTicks, arm.lo, arm.hi, maxStep(arm, speedCap, options.hz)) };
  for (const [key, rows] of [["q", res.q], ["posMiss", res.posMiss], ["angMiss", res.angMiss]] as const) {
    const row = firstFailing(rows as readonly unknown[], (r) => (typeof r === "number" ? Number.isFinite(r) : isNumbers(r, true)));
    if (row >= 0) throw new DreamActionError(`solveChunk: the solve answered a non-finite ${key} at row ${row}; nothing is returned`, detail("non_finite", key, row));
  }
  return res;
}

