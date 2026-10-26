/**
 * Embodiments and the strict checks (`dream.client._compile_checks`, `_check`, `_check_per_key`, and the parser of
 * `shared.datasets.embodiments` without `parts`).
 *
 * The `embodiment` config key takes either form:
 *
 * - a shape spec, `{group: {key: {shape?, dtype?}}}`, flattened to `"<group>.<key>"` (the YAM station's wire:
 *   `state.arm_r.joint_pos`, `image_window.top`, and the `action` group's chunk keys);
 * - the embodiment config a body was registered with: `state` and `action` parts to `[start, end)` spans and `video`
 *   roles to dataset video keys. The observation then carries `state.<part>` (as wide as the span) and `video.<role>`
 *   windows, and the reply's `per_key` carries `action.<part>` rows.
 *
 * A config with `parts` (a body whose fine-tune trains other slots than its own parts) is refused: this package does
 * not convert parts yet.
 */

import { DreamActionError, DreamConfigError, DreamObservationError } from "./errors.js";
import { every, firstHole } from "./finite.js";
import { pyList, typeName } from "./window.js";

/** The keys every embodiment config names. */
export const CONFIG_KEYS = ["state", "action", "video"] as const;
/** The keys an embodiment config may add. */
export const OPTIONAL_KEYS = ["head_camera", "parts", "actions", "urdf"] as const;
/** Keys older configs carried, ignored when reading one. */
export const RETIRED_KEYS = ["state_space", "action_space"] as const;
/** The camera roles of the wrists, matched by name. */
export const WRIST_ROLES = ["wrist_l", "wrist_r"] as const;
/** The body the Dream-1 base models were trained on. */
export const YAM_BODY = "yam";
/** The six EEF parts of a body that records its end effector's pose, with their widths. */
export const EEF_PARTS: Readonly<Record<string, number>> = {
  right_eef_xyz: 3,
  right_eef_rot6: 6,
  right_grip: 1,
  left_eef_xyz: 3,
  left_eef_rot6: 6,
  left_grip: 1,
};
/** Most values `state` or `action` may hold. */
export const MAX_DIMS = 128;
const NAME = /^[a-z][a-z0-9_]*$/;
const NAME_RULE = "lowercase letters, digits and underscores, starting with a letter";

/** One entry of a shape spec: the key must be present; `shape` and `dtype` are checked when given. */
export interface ShapeEntry {
  shape?: number[];
  dtype?: string;
}
/** A shape spec: `{group: {key: entry}}`. */
export type ShapeSpec = Record<string, Record<string, ShapeEntry>>;

/** An embodiment config, as `add_embodiment()` takes it. */
export interface EmbodimentConfig {
  state: Record<string, [number, number]>;
  action: Record<string, [number, number]>;
  video: Record<string, string>;
  head_camera?: string;
  [key: string]: unknown;
}

/** A parsed embodiment config. */
export interface Body {
  state: Record<string, [number, number]>;
  action: Record<string, [number, number]>;
  video: Record<string, string>;
  headCamera: string | null;
}

/**
 * Whether an `embodiment` block is an embodiment config rather than a shape spec (`client._is_embodiment_config`):
 * it names `state`, `action` and `video`, and holds something that is not an object one or two levels down.
 */
export function isEmbodimentConfig(block: unknown): block is EmbodimentConfig {
  if (!isObject(block) || !CONFIG_KEYS.every((key) => key in block)) return false;
  for (const group of Object.values(block)) {
    if (!isObject(group) || !Object.values(group).every(isObject)) return true;
  }
  return false;
}

/**
 * Parses an embodiment config (`shared.datasets.embodiments.parse`, without `parts`).
 *
 * @throws DreamConfigError The config is not valid, or it carries `parts`, which this package does not convert.
 */
export function parseEmbodiment(config: unknown): Body {
  const fail = (why: string): never => {
    throw new DreamConfigError(
      "config key 'embodiment' names state, action and video with values that are not shape entries, so it is read as " +
        `an embodiment config, and that config is not valid: ${why}`,
    );
  };
  if (!isObject(config)) return fail("The embodiment config must be a JSON object with state, action and video.");
  for (const key of Object.keys(config)) {
    if (![...CONFIG_KEYS, ...OPTIONAL_KEYS, ...RETIRED_KEYS].includes(key as never)) {
      fail(`Unknown key '${key}' in the embodiment config; it takes ${[...CONFIG_KEYS, ...OPTIONAL_KEYS].join(", ")}.`);
    }
  }
  for (const key of CONFIG_KEYS) if (!(key in config)) fail(`The embodiment config has no '${key}'.`);
  if ("parts" in config || "actions" in config || "urdf" in config) {
    throw new DreamConfigError(
      "config key 'embodiment' is an embodiment config with parts (a body whose fine-tune trains other slots than its " +
        "own parts); @makermods/dreamlabs does not convert parts yet: use the Python client for this model",
    );
  }
  const state = spans("state", config.state, fail);
  const action = spans("action", config.action, fail);
  const video = config.video;
  if (!isObject(video) || !Object.keys(video).length) fail("video must be a non-empty object of camera role to the dataset's video key.");
  for (const [role, key] of Object.entries(video as Record<string, unknown>)) {
    if (!NAME.test(role)) fail(`video: role '${role}' is not a name dream takes; use ${NAME_RULE}.`);
    if (typeof key !== "string" || !key) fail(`video: role '${role}' must name the dataset's video key, a non-empty string.`);
    if (key === "." || key === ".." || /[/\\\0]/.test(key as string)) {
      fail(`video: role '${role}' names '${key as string}'; a video key is one name such as observation.images.overhead, not a path.`);
    }
  }
  const head = config.head_camera;
  if (head !== undefined && head !== null) {
    if (typeof head !== "string" || !head) fail("head_camera must name one of the video roles, a non-empty string.");
    if (!((head as string) in (video as object))) fail(`head_camera: '${head as string}' is not one of the video roles (${Object.keys(video as object).join(", ")}).`);
  }
  for (const [what, parts] of [["state", state], ["action", action]] as const) {
    const width = Math.max(...Object.values(parts).map(([, end]) => end));
    if (width > MAX_DIMS) fail(`${what} is ${width} values wide; the most dream trains is ${MAX_DIMS}.`);
  }
  return { state, action, video: { ...(video as Record<string, string>) }, headCamera: (head as string | undefined) ?? null };
}

function spans(what: string, raw: unknown, fail: (why: string) => never): Record<string, [number, number]> {
  if (!isObject(raw) || !Object.keys(raw).length) fail(`${what} must be a non-empty object of part name to [start, end].`);
  const parsed: Record<string, [number, number]> = {};
  for (const [part, span] of Object.entries(raw as Record<string, unknown>)) {
    if (!NAME.test(part)) fail(`${what}: part '${part}' is not a name dream takes; use ${NAME_RULE}.`);
    if (!(Array.isArray(span) && span.length === 2 && every(span, Number.isInteger))) fail(`${what}: part '${part}' must be [start, end], two whole numbers.`);
    const [start, end] = span as [number, number];
    if (start < 0 || end <= start) fail(`${what}: part '${part}' covers [${start}, ${end}); it needs 0 <= start < end.`);
    parsed[part] = [start, end];
  }
  let expected = 0;
  for (const [part, [start, end]] of Object.entries(parsed).sort((a, b) => a[1][0] - b[1][0] || a[1][1] - b[1][1])) {
    if (start > expected) fail(`${what}: nothing covers [${expected}, ${start}); part '${part}' starts at ${start}, expected ${expected}.`);
    if (start < expected) fail(`${what}: part '${part}' covers [${start}, ${end}), overlapping another part.`);
    expected = end;
  }
  return parsed;
}

// ---------- the strict checks ----------

/** What strict mode compares each observation and reply against, compiled once (`client._Checks`). */
export interface Checks {
  /** Dotted observation key to its entry (`{}` only requires the key). */
  observation: Record<string, ShapeEntry>;
  /** The observation group whose keys must each be a frame window: `image_window` for a spec, `video` for a config. */
  frameWindowGroup: string;
  /** Bare action key (`q_r_chunk`) to its entry; empty for an embodiment config. */
  action: Record<string, ShapeEntry>;
  /** `action.<part>` to its width, rows of which the reply's `per_key` must carry; empty for a spec. */
  perKeyWidths: Record<string, number>;
}

/** The dtypes a spec entry may name (numpy's names and the common aliases), each to its typed array. */
const DTYPES: Readonly<Record<string, string>> = {
  float32: "Float32Array", f4: "Float32Array", single: "Float32Array",
  float64: "Float64Array", f8: "Float64Array", float: "Float64Array", double: "Float64Array",
  int8: "Int8Array", i1: "Int8Array", uint8: "Uint8Array", u1: "Uint8Array",
  int16: "Int16Array", i2: "Int16Array", uint16: "Uint16Array", u2: "Uint16Array",
  int32: "Int32Array", i4: "Int32Array", uint32: "Uint32Array", u4: "Uint32Array",
  int64: "BigInt64Array", i8: "BigInt64Array", int: "BigInt64Array", uint64: "BigUint64Array", u8: "BigUint64Array",
};

/**
 * The checks of an `embodiment` block, either form; null for an empty spec, which checks nothing.
 *
 * @throws DreamConfigError The block is a malformed spec or an embodiment config that is not valid.
 */
export function compileChecks(embodiment: unknown): Checks | null {
  if (isEmbodimentConfig(embodiment)) {
    const body = parseEmbodiment(embodiment);
    const observation: Record<string, ShapeEntry> = {};
    for (const [part, [start, end]] of Object.entries(body.state)) observation[`state.${part}`] = { shape: [end - start] };
    for (const role of Object.keys(body.video)) observation[`video.${role}`] = {};
    const perKeyWidths: Record<string, number> = {};
    for (const [part, [start, end]] of Object.entries(body.action)) perKeyWidths[`action.${part}`] = end - start;
    return { observation, frameWindowGroup: "video", action: {}, perKeyWidths };
  }
  if (!isObject(embodiment)) throw new DreamConfigError(`config key 'embodiment' must be a dict of groups, got ${typeName(embodiment)}`);
  for (const [group, entries] of Object.entries(embodiment)) {
    if (!isObject(entries)) throw new DreamConfigError(`embodiment group '${group}' must be a dict of keys, got ${typeName(entries)}`);
    for (const [key, entry] of Object.entries(entries)) {
      const dotted = `${group}.${key}`;
      if (!isObject(entry)) throw new DreamConfigError(`embodiment entry '${dotted}' must be a dict with optional 'shape' and 'dtype', got ${typeName(entry)}`);
      if ("shape" in entry && !(Array.isArray(entry.shape) && every(entry.shape, Number.isInteger))) {
        throw new DreamConfigError(`embodiment entry '${dotted}' has shape ${JSON.stringify(entry.shape)}; expected a list of ints`);
      }
      if ("dtype" in entry && !(typeof entry.dtype === "string" && entry.dtype in DTYPES)) {
        throw new DreamConfigError(`embodiment entry '${dotted}' has dtype ${JSON.stringify(entry.dtype)}, which this client does not know`);
      }
    }
  }
  if (!Object.keys(embodiment).length) return null;
  const observation: Record<string, ShapeEntry> = {};
  for (const [group, entries] of Object.entries(embodiment as ShapeSpec)) {
    if (group === "action") continue;
    for (const [key, entry] of Object.entries(entries)) observation[`${group}.${key}`] = entry;
  }
  return { observation, frameWindowGroup: "image_window", action: { ...((embodiment as ShapeSpec).action ?? {}) }, perKeyWidths: {} };
}

/** numpy's `np.shape` of a value: lists, typed arrays and scalars; null for a ragged list, or one with a hole. */
export function shapeOf(value: unknown): number[] | null {
  if (ArrayBuffer.isView(value)) return [(value as unknown as ArrayLike<number>).length];
  if (!Array.isArray(value)) return [];
  if (!value.length) return [0];
  // A hole (a sparse array's missing element) has no shape: map() and every() would skip it, and JSON sends null.
  if (firstHole(value) >= 0) return null;
  const inner = value.map(shapeOf);
  if (inner.some((s) => s === null)) return null;
  const first = JSON.stringify(inner[0]);
  // numpy refuses a ragged list where it is nested; a list of scalars and lists is ragged too.
  if (!inner.every((s) => JSON.stringify(s) === first)) return null;
  return [value.length, ...inner[0]!];
}

/** ` (a hole at …)` when an array in `value` has one, naming the first one's index path; else "". */
function holeAt(value: unknown, path: number[] = []): string {
  if (!Array.isArray(value)) return "";
  const hole = firstHole(value);
  if (hole >= 0) return ` (a hole, a sparse array's missing element, at index ${[...path, hole].join(", ")})`;
  for (let i = 0; i < value.length; i++) {
    const found = holeAt(value[i], [...path, i]);
    if (found) return found;
  }
  return "";
}

const tuple = (shape: readonly number[]) => (shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(", ")})`);

/**
 * Checks keys and shapes (and the dtype of typed arrays) of a payload against a flattened spec (`client._check`).
 * Keys without a group (`prompt`) are metadata and not checked.
 */
export function check(
  payload: unknown,
  spec: Readonly<Record<string, ShapeEntry>>,
  kind: "observation" | "action",
  options: { allowExtra?: boolean; frameWindowGroup?: string } = {},
): void {
  const Err = kind === "observation" ? DreamObservationError : DreamActionError;
  if (!isObject(payload)) throw new Err(`${kind} must be a dict, got ${typeName(payload)}`);
  const missing = Object.keys(spec).filter((key) => !(key in payload)).sort();
  if (missing.length) throw new Err(`${kind} is missing keys declared by the configured embodiment: ${pyList(missing)}`);
  if (!options.allowExtra) {
    const unexpected = Object.keys(payload).filter((key) => !(key in spec) && key.includes(".")).sort();
    if (unexpected.length) throw new Err(`${kind} has keys not declared by the configured embodiment: ${pyList(unexpected)}`);
  }
  const group = options.frameWindowGroup ?? "image_window";
  for (const [key, entry] of Object.entries(spec)) {
    const value = payload[key];
    if (entry.shape) {
      const shape = shapeOf(value);
      if (shape === null) throw new Err(`${kind} key '${key}' is not an array of one shape${holeAt(value)}`);
      if (JSON.stringify(shape) !== JSON.stringify(entry.shape)) {
        throw new Err(`${kind} key '${key}' has shape ${tuple(shape)}, but the configured embodiment declares ${tuple(entry.shape)}`);
      }
    }
    if (entry.dtype && ArrayBuffer.isView(value) && value.constructor.name !== DTYPES[entry.dtype]) {
      throw new Err(`${kind} key '${key}' is a ${value.constructor.name}, but the configured embodiment declares ${entry.dtype}`);
    }
    if (key.startsWith(`${group}.`) && !(Array.isArray(value) && value.length > 0 && every(value, (item) => typeof item === "string"))) {
      throw new Err(`${kind} key '${key}' must be a non-empty list of base64 JPEG strings (a frame window, oldest first), got ${typeName(value)}`);
    }
  }
}

/**
 * Checks the reply's `per_key` against an embodiment config's action parts (`client._check_per_key`): rows of each
 * part's width (a width-1 part may come as `[H]`), every part with the same number of rows, at least one.
 */
export function checkPerKey(action: Readonly<Record<string, unknown>>, widths: Readonly<Record<string, number>>): void {
  const keys = Object.keys(widths);
  if (!keys.length) return;
  const perKey = action.per_key;
  if (!isObject(perKey)) {
    throw new DreamActionError(`action has no 'per_key' dict, where the serve returns the parts the configured embodiment declares: ${pyList([...keys].sort())}`);
  }
  const missing = keys.filter((key) => !(key in perKey)).sort();
  if (missing.length) throw new DreamActionError(`action 'per_key' lacks parts the configured embodiment declares: ${pyList(missing)}`);
  let horizon: number | null = null;
  for (const [key, width] of Object.entries(widths)) {
    const shape = shapeOf(perKey[key]);
    if (shape === null) throw new DreamActionError(`action 'per_key' entry '${key}' is not rows of one width`);
    const rows = shape.length === 1 && width === 1 ? [shape[0]!, 1] : shape;
    if (rows.length !== 2 || rows[1] !== width || rows[0] === 0) {
      throw new DreamActionError(
        `action 'per_key' entry '${key}' has shape ${tuple(shape)}, but the configured embodiment declares (H, ${width}): one row of the part's ${width} values per step`,
      );
    }
    if (horizon !== null && rows[0] !== horizon) throw new DreamActionError(`action 'per_key' entry '${key}' has ${rows[0]} rows where the other parts have ${horizon}`);
    horizon = rows[0]!;
  }
}

export function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value) && !ArrayBuffer.isView(value);
}
