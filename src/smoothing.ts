/**
 * Seam smoothing (`dream.smoothing`): shapes each returned chunk so it takes over from the motion the arm is on.
 *
 * Always on, with no setting. Per arm: the anchor is the row of the previous chunk the loop runs next (the one after
 * the joints it last commanded, when the caller passes `commanded`, else the row nearest the measured joints); the
 * clock row is the round trip in rows (or every row the previous plan had left, for an arm that ran out and holds); a
 * Savitzky-Golay filter at twice the row rate removes per-row jitter from the clock row on, with five earlier rows as
 * context; then the seam join (`seam-join.ts`) replaces {@link RAMP_ROWS} rows from the clock row with the velocity
 * blend into the new plan. Rows before the clock row continue the previous plan. The first chunk of an episode, and
 * an arm off its plan, are filtered only.
 */

import { DreamActionError, DreamConfigError, detail } from "./errors.js";
import { every, firstFailing, firstHole, isNumbers } from "./finite.js";
import { invert, roundHalfEven } from "./linalg.js";
import { eefWindow, fitMotion, OLD_PATH_TIMES, RAMP_ROWS, velocityBlend, type Kinematics, type Rows } from "./seam-join.js";

/** Each chunk key with the observation key holding the measured pose of the arm it drives, on the YAM wire. */
export const ARM_POSE_KEYS: Readonly<Record<string, string>> = { q_r_chunk: "state.arm_r.joint_pos", q_l_chunk: "state.arm_l.joint_pos" };
/** The same for a body's own parts (client IK, an embodiment config): `state.<side>_arm`. */
export const SIDE_POSE_KEYS: Readonly<Record<string, string>> = { q_r_chunk: "state.right_arm", q_l_chunk: "state.left_arm" };
export const SAVGOL_WINDOW = 11;
export const SAVGOL_POLYORDER = 3;
/** The filter runs on the chunk resampled at this many samples per row (linear in between). */
export const UPSAMPLE = 2;
/** Rows of the new chunk before the clock row the filter takes as left context. */
export const FILTER_CONTEXT_ROWS = 5;
/** Largest L2 joint distance (rad) from the previous plan for the arm to count as on it. */
export const MAX_OFFSET_RAD = 0.5;

/** Rows a fence returns for one arm's joined chunk: the rows the arm may run (client IK's reach area). */
export type Fence = (key: string, rows: Rows) => Rows;
/**
 * The last step on one arm's chunk before it is returned and kept as the plan the next seam continues: the rows kept
 * within the arm's limits and speed caps, from `from` (the joints last commanded, else the measured ones). `first` is
 * true for the arm's first chunk of the episode; `observation`, when given, is the one the chunk answers.
 */
export type Projection = (key: string, rows: Rows, from: readonly number[] | null, first: boolean, observation?: Readonly<Record<string, unknown>> | null) => Rows;

const designs = new Map<string, { basis: number[][]; fit: number[][] }>();

function design(window: number, polyorder: number): { basis: number[][]; fit: number[][] } {
  const key = `${window}:${polyorder}`;
  let found = designs.get(key);
  if (found) return found;
  const half = Math.floor(window / 2);
  const basis = Array.from({ length: window }, (_, i) => Array.from({ length: polyorder + 1 }, (_, p) => (i - half) ** p));
  const n = polyorder + 1;
  // pinv(V) = (V^T V)^-1 V^T for the full-rank Vandermonde basis (numpy's SVD agrees to rounding).
  const inverse = invert(Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => basis.reduce((s, r) => s + r[i]! * r[j]!, 0))));
  const fit = Array.from({ length: n }, (_, i) => basis.map((_, k) => inverse[i]!.reduce((s, v, j) => s + v * basis[k]![j]!, 0)));
  designs.set(key, (found = { basis, fit }));
  return found;
}


/**
 * Savitzky-Golay filter along the rows, each column on its own (`smoothing.savgol_filter`, scipy's `interp` mode at
 * the edges). Returns `rows` itself when there are fewer rows than the window.
 */
export function savgolFilter(rows: Rows, window: number = SAVGOL_WINDOW, polyorder: number = SAVGOL_POLYORDER): Rows {
  const count = rows.length;
  if (count < window) return rows;
  const columns = rows[0]!.length;
  const half = Math.floor(window / 2);
  const { basis, fit } = design(window, polyorder);
  const centre = fit[0]!;
  const out = rows.map((row) => [...row]);
  for (let i = half; i < count - half; i++) {
    for (let c = 0; c < columns; c++) {
      let acc = 0;
      for (let k = 0; k < window; k++) acc += rows[i - half + k]![c]! * centre[k]!;
      out[i]![c] = acc;
    }
  }
  const edge = (start: number, basisRows: number[][], target: number) => {
    for (let c = 0; c < columns; c++) {
      const coefficients = fit.map((f) => f.reduce((s, w, k) => s + w * rows[start + k]![c]!, 0));
      basisRows.forEach((b, r) => {
        out[target + r]![c] = b.reduce((s, v, p) => s + v * coefficients[p]!, 0);
      });
    }
  };
  edge(0, basis.slice(0, half), 0);
  edge(count - window, basis.slice(half + 1), count - half);
  return out;
}

/** The chunk with every column filtered from row `clock` on, at {@link UPSAMPLE} samples per row; earlier rows are context. */
export function filterFrom(chunk: Rows, clock: number): Rows {
  const start = Math.max(0, clock - FILTER_CONTEXT_ROWS);
  const segment = chunk.slice(start);
  const count = segment.length;
  const columns = chunk[0]!.length;
  const upsampled: Rows = [];
  for (let i = 0; i <= (count - 1) * UPSAMPLE; i++) {
    const t = i / UPSAMPLE;
    const lo = Math.floor(t);
    const hi = Math.min(lo + 1, count - 1);
    const x = t - lo;
    upsampled.push(Array.from({ length: columns }, (_, c) => (x === 0 ? segment[lo]![c]! : segment[lo]![c]! + x * (segment[hi]![c]! - segment[lo]![c]!))));
  }
  const filtered = savgolFilter(upsampled, SAVGOL_WINDOW, SAVGOL_POLYORDER);
  if (filtered === upsampled) return chunk;
  const out = chunk.map((row) => [...row]);
  for (let row = clock; row < chunk.length; row++) out[row] = [...filtered[(row - start) * UPSAMPLE]!];
  return out;
}

const distance = (a: readonly number[], b: readonly number[]) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]!) ** 2, 0));

/** Index of the row nearest `pose` (L2), or null when even that row is farther than `maxOffsetRad`. */
export function nearestRow(chunk: Rows, pose: readonly number[], maxOffsetRad: number): number | null {
  const distances = chunk.map((row) => distance(row, pose));
  let index = 0;
  distances.forEach((d, i) => {
    if (d < distances[index]!) index = i;
  });
  // A plan that ends at rest repeats its last row: the arm holding there is at its end.
  if (distances.slice(index).every((d) => d === distances[index])) index = distances.length - 1;
  return distances[index]! <= maxOffsetRad ? index : null;
}

/** Row of the previous chunk the loop runs next at the observation, or null for an arm off that plan. */
export function anchorRow(previous: Rows, pose: readonly number[], commanded: readonly number[] | null): number | null {
  if (commanded === null) return nearestRow(previous, pose, MAX_OFFSET_RAD);
  const last = commanded.slice(0, previous[0]!.length);
  if (distance(pose, last) > MAX_OFFSET_RAD) return null;
  const row = nearestRow(previous, last, MAX_OFFSET_RAD);
  return row === null ? null : Math.min(row + 1, previous.length - 1);
}

/** The clock row and the old-motion rows the join continues (`smoothing.old_motion`). */
export function oldMotion(previous: Rows, anchor: number, rowsInFlight: number): { clock: number; rows: Rows; times: number[] } {
  const last = previous.length - 1;
  const remaining = last - anchor;
  const clock = roundHalfEven(rowsInFlight);
  if (clock >= remaining) return { clock: remaining, rows: [[...previous[last]!]], times: [0] };
  const rows: Rows = [];
  const times: number[] = [];
  for (const t of OLD_PATH_TIMES) {
    const i = anchor + clock + t;
    if (i >= 0 && i <= last) {
      rows.push([...previous[i]!]);
      times.push(t);
    }
  }
  return { clock, rows, times };
}

/** Where a new chunk takes over one arm, and the two commands before it (`SeamSmoother.takeover_history`). */
export interface Takeover {
  clock: number;
  joints: number[];
  /** The previous chunk's rows one and two before `joints` (its first row where there is none), newest first. */
  before: [number[], number[]];
}

/** Joins each chunk onto the motion the arm is on; keeps the previous chunk per arm (`smoothing.SeamSmoother`). */
export class SeamSmoother {
  private previous = new Map<string, Rows>();
  private readonly kinematics: Readonly<Record<string, Kinematics>>;
  private readonly poseKeys: Readonly<Record<string, string>>;

  /** @throws DreamConfigError `actionHz` is not a positive, finite number. */
  constructor(
    readonly actionHz: number,
    kinematics?: Readonly<Record<string, Kinematics>> | null,
    poseKeys: Readonly<Record<string, string>> = ARM_POSE_KEYS,
    private readonly fence: Fence | null = null,
    private readonly project: Projection | null = null,
  ) {
    if (!(typeof actionHz === "number" && actionHz > 0 && Number.isFinite(actionHz))) {
      throw new DreamConfigError(`actionHz must be a positive, finite number of rows per second, got ${String(actionHz)}`, detail("range", "actionHz"));
    }
    this.kinematics = kinematics ?? {};
    this.poseKeys = poseKeys;
  }

  /** Forgets the previous chunks: a fresh episode's first chunk is filtered only. */
  reset(): void {
    this.previous.clear();
  }

  /** Whether a chunk of `key` was returned this episode. */
  has(key: string): boolean {
    return this.previous.has(key);
  }

  /**
   * `action` with each arm's chunk filtered and joined onto the motion that arm is on; every other key untouched.
   *
   * Every arm is prepared and checked before any is kept: when one throws, no arm's plan changes.
   *
   * @param rttS The seconds the loop executed between the observation and the reply (the round trip, by default):
   *   with the row rate, the rows it ran meanwhile.
   * @param commanded Per chunk key, the joints the loop last commanded that arm before the observation.
   * @throws DreamActionError A chunk is not rows of finite numbers, or the join made one that is not.
   */
  smooth(
    observation: Readonly<Record<string, unknown>>,
    action: Record<string, unknown>,
    rttS: number,
    commanded?: Readonly<Record<string, readonly number[]>> | null,
  ): Record<string, unknown> {
    const joined = this.prepare(observation, action, rttS, commanded);
    joined.commit();
    return joined.result;
  }

  /**
   * As {@link smooth}, but no arm's plan is kept until `commit()`: a caller that may still drop the result (a call
   * cancelled meanwhile) leaves the plans the arms are on as they were. `first` lists the chunk keys whose chunk is
   * the first of the episode.
   */
  prepare(
    observation: Readonly<Record<string, unknown>>,
    action: Record<string, unknown>,
    rttS: number,
    commanded?: Readonly<Record<string, readonly number[]>> | null,
  ): { result: Record<string, unknown>; first: string[]; commit: () => void } {
    const result: Record<string, unknown> = { ...action };
    const plans = new Map<string, Rows>();
    for (const [key, poseKey] of Object.entries(this.poseKeys)) {
      if (!(key in action)) continue;
      const chunk = numericRows(action[key], key);
      if (chunk === null || !chunk.length) continue;
      const pose = flatPose(observation[poseKey]);
      let smoothed = this.join(key, chunk, pose, commanded?.[key] ?? null, rttS);
      if (this.fence) smoothed = this.fence(key, smoothed);
      // What is returned is what the next seam continues from.
      if (this.project) smoothed = this.project(key, smoothed, commanded?.[key] ?? pose, !this.previous.has(key), observation);
      // Finite in, but the filter's sums can still overflow (rows alternating near ±1.7e308): never a command.
      const bad = firstFailing(smoothed, (row) => isNumbers(row, true));
      if (bad >= 0) throw new DreamActionError(`action key '${key}' row ${bad} is not finite after the seam join: the reply's numbers overflow when filtered`, detail("non_finite", key, bad));
      plans.set(key, smoothed);
      result[key] = smoothed.map((row) => [...row]);
    }
    const first = [...plans.keys()].filter((key) => !this.previous.has(key));
    return { result, first, commit: () => plans.forEach((plan, key) => this.previous.set(key, plan)) };
  }

  /** The shape of the plan kept for `key`, or null when none is. */
  planShape(key: string): { rows: number; width: number } | null {
    const plan = this.previous.get(key);
    return plan ? { rows: plan.length, width: plan[0]!.length } : null;
  }

  /**
   * Makes `rows` the plan the next seam of `key` continues: the rows the loop actually installed (`takeoverAt`).
   *
   * @throws DreamActionError `rows` is not the kept plan's shape, or holds a non-finite number; the plan is unchanged.
   */
  adopt(key: string, rows: Rows): void {
    const shape = this.planShape(key);
    if (!shape) return;
    // Indices, not some()/every(): those skip a hole, which would be kept as `undefined`.
    if (rows.length !== shape.rows || !every(rows, (row) => Array.isArray(row) && row.length === shape.width && firstHole(row) < 0)) {
      throw new DreamActionError(`the plan adopted for '${key}' is not ${shape.rows} rows of ${shape.width}, as the one kept`, detail("shape", key));
    }
    const bad = firstFailing(rows, (row) => isNumbers(row, true));
    if (bad >= 0) throw new DreamActionError(`the plan adopted for '${key}' row ${bad} is not finite`, detail("non_finite", key, bad));
    this.previous.set(key, rows.map((row) => [...row]));
  }

  /** Per chunk key, how many joints the arm's measured pose in the observation holds, or null when it holds none. */
  armJoints(observation: unknown): Record<string, number | null> {
    const out: Record<string, number | null> = {};
    for (const [key, poseKey] of Object.entries(this.poseKeys)) {
      const value = typeof observation === "object" && observation !== null ? (observation as Record<string, unknown>)[poseKey] : undefined;
      const pose = flatPose(value);
      const flat = pose !== null && isNumbers(value);
      out[key] = flat ? pose.length : null;
    }
    return out;
  }

  /** Where a new chunk of `[count][width]` takes over one arm, or null when it will not be joined. */
  takeoverHistory(key: string, count: number, width: number, pose: readonly number[], rttS: number, commanded: readonly number[] | null): Takeover | null {
    const previous = this.previous.get(key);
    if (!previous || previous.length !== count || previous[0]!.length !== width) return null;
    const joints = previous.map((row) => row.slice(0, pose.length));
    const anchor = anchorRow(joints, pose, commanded);
    if (anchor === null) return null;
    const { clock } = oldMotion(joints, anchor, rttS * this.actionHz);
    const at = anchor + clock;
    return { clock, joints: [...joints[at]!], before: [[...joints[Math.max(at - 1, 0)]!], [...joints[Math.max(at - 2, 0)]!]] };
  }

  private join(key: string, chunk: Rows, pose: number[] | null, commanded: readonly number[] | null, rttS: number): Rows {
    if (pose === null) return filterFrom(chunk, 0);
    const jointCount = Math.min(pose.length, chunk[0]!.length - 1);
    let previous = this.previous.get(key) ?? null;
    if (previous && (previous.length !== chunk.length || previous[0]!.length !== chunk[0]!.length)) previous = null;
    const head = (rows: Rows) => rows.map((row) => row.slice(0, jointCount));
    const anchor = previous ? anchorRow(head(previous), pose.slice(0, jointCount), commanded) : null;
    if (!previous || anchor === null) return filterFrom(chunk, 0);
    const { clock, rows, times } = oldMotion(head(previous), anchor, rttS * this.actionHz);
    const plan = filterFrom(chunk, clock);
    const count = Math.min(RAMP_ROWS, chunk.length - 1 - clock);
    if (count < 1) return plan;
    const window = head(plan.slice(clock, clock + count + 1));
    const { now, velocity } = fitMotion(rows, times);
    let joints = velocityBlend(now, velocity, window);
    const kinematics = this.kinematics[key];
    if (kinematics) joints = eefWindow(rows, times, window, joints, kinematics) ?? joints;
    const out = plan.map((row) => [...row]);
    joints.forEach((row, k) => row.forEach((v, c) => (out[clock + k]![c] = v)));
    // Rows before the clock row continue the previous plan, the motion the arm is on.
    for (let k = 0; k < clock; k++) {
      const past = previous[Math.min(anchor + k, previous.length - 1)]!;
      for (let c = 0; c < jointCount; c++) out[k]![c] = past[c]!;
    }
    return out;
  }
}

/** A chunk as rows of numbers; null for a shape that cannot be joined (not two-dimensional). */
function numericRows(value: unknown, key: string): Rows | null {
  if (!Array.isArray(value)) {
    // A null head is kept as it is, and so is a bare number, as Python leaves a chunk that is not two-dimensional.
    if (ArrayBuffer.isView(value) || typeof value === "number" || value === null) return null;
    throw new DreamActionError(`action key '${key}' is not rows of numbers: got ${typeof value}`);
  }
  if (!value.length) return null;
  // Not two-dimensional (a flat list, or rows of rows): left as it is, as Python leaves it.
  if (!Array.isArray(value[0]) && !ArrayBuffer.isView(value[0])) return null;
  if (Array.isArray(value[0]) && value[0].some((item) => Array.isArray(item) || ArrayBuffer.isView(item))) return null;
  // Rows of no value: Python's filter fails on them (numpy's ValueError); refused here.
  if ((value[0] as ArrayLike<unknown>).length === 0) throw new DreamActionError(`action key '${key}' is rows of no values`, detail("shape", key, 0));
  // Holes first: every(), forEach() and map() skip them, so a sparse chunk would pass and be joined as `undefined`.
  const hole = firstHole(value);
  if (hole >= 0) throw new DreamActionError(`action key '${key}' is not rows of numbers: row ${hole} is a hole (a sparse array's missing element)`, detail("shape", key, hole));
  const rows = value.map((row) => (ArrayBuffer.isView(row) ? Array.from(row as unknown as ArrayLike<number>) : row));
  if (!rows.every(Array.isArray)) {
    if (rows.every((row) => typeof row === "number")) return null;
    throw new DreamActionError(`action key '${key}' is not rows of numbers: some rows are not lists`);
  }
  const width = (rows[0] as unknown[]).length;
  (rows as unknown[][]).forEach((row, i) => {
    if (row.length !== width) throw new DreamActionError(`action key '${key}' is not rows of numbers: the rows have different lengths`, detail("shape", key, i));
    if (!isNumbers(row)) throw new DreamActionError(`action key '${key}' is not rows of numbers: row ${i} holds a non-number or a hole`, detail("shape", key, i));
    if (!isNumbers(row, true)) throw new DreamActionError(`action key '${key}' row ${i} holds a non-finite number (NaN or an infinity)`, detail("non_finite", key, i));
  });
  return rows as Rows;
}

/** The measured joint pose as a flat array, or null when the value is not numbers (a hole is not a number). */
export function flatPose(value: unknown): number[] | null {
  if (value === undefined || value === null) return null;
  if (typeof value === "number") return [value];
  if (ArrayBuffer.isView(value)) return Array.from(value as unknown as ArrayLike<number>, Number);
  if (!Array.isArray(value)) return null;
  // Walked by index: flat() drops holes, which would shorten the pose and shift every joint after one.
  const flat: number[] = [];
  const walk = (items: readonly unknown[]): boolean => {
    for (let i = 0; i < items.length; i++) {
      if (!(i in items)) return false;
      const v = items[i];
      if (Array.isArray(v)) {
        if (!walk(v)) return false;
      } else if (typeof v === "number") flat.push(v);
      else return false;
    }
    return true;
  };
  return walk(value) ? flat : null;
}
