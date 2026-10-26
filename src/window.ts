/**
 * The frame window (`dream._window`): which camera ticks the client keeps, and the last few frames of each camera.
 *
 * A caller hands the client every camera tick, at whatever rate its cameras run, while a checkpoint reads each
 * camera's window at the rate it was trained on, `obsFps`. So the window keeps one tick every `1 / obsFps` seconds,
 * deciding on the caller's timestamps, never on a count of calls:
 *
 * - The first tick of an episode is kept, and the next one is due a period after it.
 * - A later tick is kept when it is at or past the due time less a slack: half the median interval between the last
 *   {@link SLACK_CALLS} distinct timestamps, at most half a period. A tick repeating the last timestamp is not counted.
 * - After a kept tick the next one is due a period later, anchored on the schedule so jitter never drifts it; a tick
 *   kept half a period or more after its due time (the caller paused) restarts the schedule from itself.
 *
 * Every tick, kept or not, replaces the latest observation, so what is sent is always the newest state.
 */

import { DreamObservationError } from "./errors.js";

/** Frames per window key once the window is full, unless `obsWindow` says otherwise. */
export const OBS_WINDOW = 9;
/** Frames kept per second, unless `obsFps` says otherwise. */
export const OBS_FPS = 5.0;
/** Ticks whose intervals size the slack. */
export const SLACK_CALLS = 8;

/** Python's `statistics.median`. */
export function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

/** One episode's frames, a ring per frame key, and its latest observation (`_window.Window`). */
export class ObservationWindow {
  private rings = new Map<string, string[]>();
  private latest: Record<string, unknown> | null = null;
  private recent: number[] = [];
  private due: number | null = null;

  /**
   * @param size Frames per ring once it is full; at least 1.
   * @param periodS Seconds between kept ticks; positive.
   */
  constructor(
    readonly size: number = OBS_WINDOW,
    readonly periodS: number = 1 / OBS_FPS,
  ) {}

  /**
   * Takes one camera tick: keeps its frames when they are due, and its observation in any case. Nothing changes when
   * a check fails.
   *
   * @returns Whether the frames were kept.
   * @throws DreamObservationError The tick is not valid, its frame keys are not the episode's, or its timestamp is
   *   earlier than the last tick's.
   */
  add(frames: Readonly<Record<string, string>>, observation: Readonly<Record<string, unknown>>, timestampS: number): boolean {
    checkTick(frames, observation, timestampS);
    const keys = Object.keys(frames);
    if (this.rings.size) {
      const missing = [...this.rings.keys()].filter((key) => !(key in frames)).sort();
      const extra = keys.filter((key) => !this.rings.has(key)).sort();
      if (missing.length || extra.length) {
        throw new DreamObservationError(
          `frame keys changed within the episode: missing ${pyList(missing)}, extra ${pyList(extra)}; send the same ` +
            "frame keys on every setObservation(), or call resetEpisode() first",
        );
      }
    }
    const last = this.recent.at(-1);
    if (last !== undefined && timestampS < last) {
      throw new DreamObservationError(
        `setObservation() timestamp ${timestampS} is earlier than the previous call's, ${last}; timestamps never go ` +
          "back within an episode",
      );
    }
    if (last === undefined || timestampS > last) {
      this.recent.push(timestampS);
      if (this.recent.length > SLACK_CALLS) this.recent.shift();
    }
    this.latest = snapshot(observation);
    const gaps = this.recent.slice(1).map((t, i) => t - this.recent[i]!);
    const slack = gaps.length ? Math.min(median(gaps) / 2, this.periodS / 2) : 0;
    const due = this.due;
    if (due !== null && timestampS < due - slack) return false;
    const anchor = due !== null && timestampS < due + this.periodS / 2 ? due : timestampS;
    this.due = anchor + this.periodS;
    for (const key of keys) {
      let ring = this.rings.get(key);
      if (!ring) this.rings.set(key, (ring = []));
      ring.push(frames[key]!);
      if (ring.length > this.size) ring.shift();
    }
    return true;
  }

  /**
   * The observation to send: the latest tick's (a deep copy of its numbers, taken when it was added), plus each frame
   * key's window, oldest frame first. New arrays, so a later tick changes nothing in it.
   *
   * @throws DreamObservationError No tick since the episode started.
   */
  buildObservation(): Record<string, unknown> {
    if (this.latest === null) {
      throw new DreamObservationError("no observation since the episode started: call setObservation(...) before getAction()");
    }
    const out: Record<string, unknown> = { ...this.latest };
    for (const [key, ring] of this.rings) out[key] = [...ring];
    return out;
  }

  /** Forgets the episode: the frames, the latest observation and the schedule; the next tick is a first one. */
  clear(): void {
    this.rings = new Map();
    this.latest = null;
    this.recent = [];
    this.due = null;
  }
}

function checkTick(frames: unknown, observation: unknown, timestampS: unknown): void {
  if (!isPlainRecord(frames)) {
    throw new DreamObservationError(`setObservation() frames must be a mapping of observation key to one frame, got ${typeName(frames)}`);
  }
  const keys = Object.keys(frames);
  if (!keys.length) throw new DreamObservationError("setObservation() frames is empty: pass one frame per window key");
  for (const key of keys) {
    const frame = frames[key];
    if (!key) {
      throw new DreamObservationError("frame key '' must be a non-empty string, the observation key the checkpoint reads the window under");
    }
    if (frame instanceof ArrayBuffer || ArrayBuffer.isView(frame) || (typeof Blob !== "undefined" && frame instanceof Blob)) {
      throw new DreamObservationError(`frame '${key}' is bytes; send each frame as a base64 string, not bytes`);
    }
    if (typeof frame !== "string" || !frame) {
      const got = typeof frame === "string" ? "an empty string" : typeName(frame);
      throw new DreamObservationError(
        `frame '${key}' must be one frame, a non-empty base64 JPEG string (the client keeps the window), got ${got}`,
      );
    }
  }
  if (!isPlainRecord(observation)) {
    throw new DreamObservationError(`setObservation() observation must be a mapping, got ${typeName(observation)}`);
  }
  for (const key of keys) {
    if (key in observation) {
      throw new DreamObservationError(
        `observation key '${key}' is also a frame key; leave it out of the observation, the client sends the window it ` +
          "keeps under that key",
      );
    }
  }
  if (typeof timestampS !== "number" || !Number.isFinite(timestampS)) {
    throw new DreamObservationError(
      `setObservation() timestamp must be a finite number of seconds on a monotonic clock, got ${String(timestampS)}`,
    );
  }
}

/**
 * A deep copy of an observation's numbers: arrays and typed arrays copied, plain objects walked, everything else
 * (strings, numbers) as is. A sensor buffer the caller reuses can then not change what was sent, or what the reply is
 * joined onto, after the fact.
 */
export function snapshot<T>(value: T): T {
  if (Array.isArray(value)) return value.map(snapshot) as T;
  if (ArrayBuffer.isView(value)) {
    if (value instanceof DataView) return new DataView(value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength)) as T;
    // The typed array's own slice() may be a view (Node's Buffer overrides it): the base one always copies.
    return copyTyped.call(value as unknown as Uint8Array) as T;
  }
  if (isPlainRecord(value) && (Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null)) {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, snapshot(item)])) as T;
  }
  return value;
}

const copyTyped = (Object.getPrototypeOf(Uint8Array.prototype) as Uint8Array).slice;

export function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value) && !ArrayBuffer.isView(value);
}

export function typeName(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  if (typeof value === "object") return (value as object).constructor?.name ?? "object";
  return typeof value;
}

/** A list of strings as Python prints it, for messages that match upstream's. */
export function pyList(items: readonly string[]): string {
  return `[${items.map((item) => `'${item}'`).join(", ")}]`;
}
