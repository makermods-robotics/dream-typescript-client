/**
 * What a fine-tune was trained on, held against what a run sends it (`dream.trained_inputs`): each camera's aspect,
 * and the prompt.
 *
 * A fine-tune learned from the frames and tasks of its dataset. The model resizes every frame to one input size, so a
 * camera sent at another aspect than its training videos (a square crop, a letterbox) shows the model shapes it never
 * saw; a prompt in other words than the dataset's tasks is one it was never trained to follow. A frame's size alone is
 * not a line: at the training aspect, a larger or smaller frame becomes the same input.
 *
 * DreamClient.policy() reads, once per checkpoint, what one of your fine-tunes was trained on
 * ({@link fetchTrainedInputs}, `GET <data>/v1/fine_tuning/models/<id>/training_data`); each episode's first
 * getAction() holds its frames and prompt to it ({@link trainedNotes}), warns each line once a policy (a
 * {@link DreamTrainingWarning} that never refuses), and hands the episode's lines to every round trip
 * (`RoundTrip.trained`). Only a fine-tune's id ({@link FINE_TUNE_PREFIX}) is looked up.
 */

import { FOLD_MULTI, FOLD_RUNS } from "./casefold.data.js";
import { DreamServerError } from "./errors.js";
import { callService, type ServiceOptions } from "./models.js";

/** What every fine-tune's id starts with (`fine_tuned_model`, `dream-1-ft:<uid8>-<suffix>`): only these are looked up. */
export const FINE_TUNE_PREFIX = "dream-1-ft:";
/** How far a frame's aspect (width over height) may be from a training camera's, as a share of it. */
export const ASPECT_TOLERANCE = 0.05;
/** How alike a training prompt must be to the one sent (difflib's ratio) to be named as its closest. */
export const CLOSE_PROMPT = 0.6;

/** What a fine-tune was trained on: each camera role's frame sizes (`[width, height]`), and its dataset's tasks. */
export interface TrainedInputs {
  cameras: Record<string, number[][]>;
  prompts: string[];
}

/** The data service's path for a checkpoint's training inputs; the id quoted as Python's `quote(id, safe='')`. */
export function trainingDataPath(checkpoint: string): string {
  return `/v1/fine_tuning/models/${pyQuote(checkpoint)}/training_data`;
}

/**
 * What one of your fine-tunes was trained on (`trained_inputs.fetch`). One request, not retried.
 *
 * @returns null when the checkpoint is not one of your fine-tunes (`not_found`), or the service predates the fields.
 * @throws DreamError The data service could not be reached, or refused for another reason.
 */
export async function fetchTrainedInputs(checkpoint: string, options: ServiceOptions): Promise<TrainedInputs | null> {
  let answer: unknown;
  try {
    answer = await callService("GET", trainingDataPath(checkpoint), options);
  } catch (error) {
    if (error instanceof DreamServerError && error.code === "not_found") return null;
    throw error;
  }
  if (typeof answer !== "object" || answer === null || Array.isArray(answer) || !("cameras" in answer) || !("prompts" in answer)) return null;
  const { cameras, prompts } = answer as Record<string, unknown>;
  return checkedTrainedInputs({ cameras, prompts });
}

/**
 * Training inputs as this client compares them: per role, only the sizes that are a finite positive width and height;
 * only the prompts that are strings. What the service answered otherwise is left out, not compared: a malformed record
 * must never stop a policy (Python would fail getAction() on it).
 */
export function checkedTrainedInputs(found: { cameras: unknown; prompts: unknown }): TrainedInputs {
  const cameras: Record<string, number[][]> = {};
  if (typeof found.cameras === "object" && found.cameras !== null && !Array.isArray(found.cameras)) {
    for (const [role, sizes] of Object.entries(found.cameras)) {
      if (!Array.isArray(sizes)) continue;
      const good = sizes.filter(
        (size): size is number[] =>
          Array.isArray(size) && size.length === 2 && size.every((v) => typeof v === "number" && Number.isFinite(v) && v > 0),
      );
      if (good.length) cameras[role] = good.map(([w, h]) => [w!, h!]);
    }
  }
  const prompts = Array.isArray(found.prompts) ? found.prompts.filter((p): p is string => typeof p === "string") : [];
  return { cameras, prompts };
}

/**
 * The lines for one call's frames and prompt against what the checkpoint was trained on; none when they match
 * (`trained_inputs.notes`).
 *
 * @param frames Per frame key (`video.<role>`), the base64 JPEGs the call sends; the newest one's size is read.
 * @param prompt The call's `prompt`, or undefined when it sends none.
 * @returns One line per camera whose aspect is off every size its role was trained at, then one for a prompt that is
 *   none of the training tasks (case, spacing and end punctuation aside).
 */
export function trainedNotes(given: TrainedInputs, checkpoint: string, frames: Readonly<Record<string, readonly string[]>>, prompt: unknown): string[] {
  const out: string[] = [];
  const trained = checkedTrainedInputs(given);
  const cameras = trained.cameras;
  for (const [key, window] of Object.entries(frames)) {
    const role = key.startsWith("video.") ? key.slice("video.".length) : key;
    const sizes = Object.hasOwn(cameras, role) ? cameras[role] : undefined;
    const size = window.length ? jpegSize(window.at(-1)!) : null;
    if (!key.startsWith("video.") || !Array.isArray(sizes) || !sizes.length || size === null || !size[1]) continue;
    const [width, height] = size;
    if (sizes.every(([w, h]) => Math.abs(width / height / (w! / h!) - 1) > ASPECT_TOLERANCE)) {
      const shown = sizes.map(([w, h]) => `${w} x ${h} (aspect ${fixed2(w! / h!)})`).join(" or ");
      out.push(
        `${key} frames are ${width} x ${height} (aspect ${fixed2(width / height)}), but ${checkpoint} was trained on ` +
          `${shown}: the model resizes each frame to one input size, so it sees shapes it never learned; send ` +
          "this camera at its training aspect, cropped from the full frame, not padded or letterboxed",
      );
    }
  }
  const tasks = new Map<string, string>();
  for (const task of trained.prompts) tasks.set(plain(task), task);
  if (typeof prompt === "string" && tasks.size && !tasks.has(plain(prompt))) {
    const close = closestMatch(plain(prompt), [...tasks.keys()], CLOSE_PROMPT);
    const some = tasks.values().next().value!;
    const hint = close !== null ? `the closest is ${pyRepr(tasks.get(close)!)}` : `${tasks.size} tasks, such as ${pyRepr(some)}`;
    out.push(
      `the prompt ${pyRepr(prompt)} is none of the tasks ${checkpoint} was trained on (${hint}): a fine-tune follows ` +
        "the words of its training tasks best",
    );
  }
  return out;
}

/** A base64 JPEG's width and height from its start of frame, without decoding it; null when it has none. */
export function jpegSize(frame: string): [number, number] | null {
  const data = b64decode(frame);
  if (data === null) return null;
  const byte = (i: number) => data[i]!;
  let i = 2;
  while (i + 9 < data.length) {
    if (byte(i) !== 0xff) return null;
    const marker = byte(i + 1);
    const length = (byte(i + 2) << 8) | byte(i + 3);
    // A start of frame: height, then width.
    if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
      return [(byte(i + 7) << 8) | byte(i + 8), (byte(i + 5) << 8) | byte(i + 6)];
    }
    i += 2 + length;
  }
  return null;
}

const B64 = new Int8Array(128).fill(-1);
for (const [i, c] of [..."ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"].entries()) B64[c.charCodeAt(0)] = i;

/**
 * Python's `base64.b64decode(text, validate=False)` (`binascii.a2b_base64`, not strict): characters outside the
 * alphabet are skipped, a complete pad sequence ends the input, and a quad left incomplete is an error. Null for an
 * error, and for a non-ASCII character (Python's ValueError).
 */
export function b64decode(text: string): Uint8Array | null {
  const out = new Uint8Array(Math.floor((text.length * 3) / 4) + 3);
  let length = 0;
  let quad = 0;
  let left = 0;
  let pads = 0;
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (code > 0x7f) return null;
    if (code === 0x3d) {
      if (quad >= 2 && quad + ++pads >= 4) {
        quad = 0;
        break;
      }
      continue;
    }
    const value = B64[code]!;
    if (value < 0) continue;
    pads = 0;
    if (quad === 0) [quad, left] = [1, value];
    else if (quad === 1) [quad, out[length++], left] = [2, (left << 2) | (value >> 4), value & 0x0f];
    else if (quad === 2) [quad, out[length++], left] = [3, ((left << 4) | (value >> 2)) & 0xff, value & 0x03];
    else [quad, out[length++], left] = [0, ((left << 6) | value) & 0xff, 0];
  }
  return quad === 0 ? out.subarray(0, length) : null;
}

/** Python's `str.isspace()` characters, which `\s` in a Python str pattern and `str.strip()` take. */
const SPACE = "\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const SPACES = new RegExp(`[${SPACE}]+`, "g");
const EDGE_SPACES = new RegExp(`^[${SPACE}]+|[${SPACE}]+$`, "g");

/** A prompt as compared (`trained_inputs._plain`): lower case, single spaces, no punctuation at its ends. */
export function plain(text: string): string {
  const stripped = text.replace(SPACES, " ").replace(EDGE_SPACES, "").replace(/^[.!?,;:]+|[.!?,;:]+$/g, "").replace(EDGE_SPACES, "");
  return casefold(stripped);
}

let folds: Map<number, string> | null = null;

/** Python's `str.casefold()`: full Unicode case folding, from Python's own table (`src/casefold.data.ts`). */
export function casefold(text: string): string {
  if (!folds) {
    folds = new Map(FOLD_MULTI);
    for (let r = 0; r < FOLD_RUNS.length; r += 4) {
      const [start, count, step, delta] = [FOLD_RUNS[r]!, FOLD_RUNS[r + 1]!, FOLD_RUNS[r + 2]!, FOLD_RUNS[r + 3]!];
      for (let k = 0; k < count; k++) folds.set(start + k * step, String.fromCodePoint(start + k * step + delta));
    }
  }
  let out = "";
  for (const c of text) out += folds.get(c.codePointAt(0)!) ?? c;
  return out;
}

/** Python's `<` on str: by code point, not by UTF-16 unit. */
function codePointGreater(a: string, b: string): boolean {
  const x = Array.from(a, (c) => c.codePointAt(0)!);
  const y = Array.from(b, (c) => c.codePointAt(0)!);
  for (let i = 0; i < Math.min(x.length, y.length); i++) if (x[i] !== y[i]) return x[i]! > y[i]!;
  return x.length > y.length;
}

/**
 * `difflib.get_close_matches(word, possibilities, n=1, cutoff)`: the possibility most like `word` by
 * `SequenceMatcher.ratio()`, at least `cutoff`, ties to the greater string; null when none is.
 */
export function closestMatch(word: string, possibilities: readonly string[], cutoff: number): string | null {
  let best: { score: number; text: string } | null = null;
  for (const text of possibilities) {
    const score = sequenceRatio(text, word);
    if (score < cutoff) continue;
    if (!best || score > best.score || (score === best.score && codePointGreater(text, best.text))) best = { score, text };
  }
  return best?.text ?? null;
}

/**
 * `difflib.SequenceMatcher(None, a, b).ratio()`: twice the matched code points over both lengths, the matches found as
 * Python finds them (the longest block first, then each side of it), with its popular-element heuristic on `b`.
 */
export function sequenceRatio(aText: string, bText: string): number {
  const a = Array.from(aText);
  const b = Array.from(bText);
  if (!a.length && !b.length) return 1;
  const b2j = new Map<string, number[]>();
  b.forEach((c, j) => {
    let list = b2j.get(c);
    if (!list) b2j.set(c, (list = []));
    list.push(j);
  });
  if (b.length >= 200) {
    const most = Math.floor(b.length / 100) + 1;
    for (const [c, list] of [...b2j]) if (list.length > most) b2j.delete(c);
  }
  const longest = (alo: number, ahi: number, blo: number, bhi: number): [number, number, number] => {
    let [besti, bestj, bestsize] = [alo, blo, 0];
    let j2len = new Map<number, number>();
    for (let i = alo; i < ahi; i++) {
      const next = new Map<number, number>();
      for (const j of b2j.get(a[i]!) ?? []) {
        if (j < blo) continue;
        if (j >= bhi) break;
        const k = (j2len.get(j - 1) ?? 0) + 1;
        next.set(j, k);
        if (k > bestsize) [besti, bestj, bestsize] = [i - k + 1, j - k + 1, k];
      }
      j2len = next;
    }
    while (besti > alo && bestj > blo && a[besti - 1] === b[bestj - 1]) [besti, bestj, bestsize] = [besti - 1, bestj - 1, bestsize + 1];
    while (besti + bestsize < ahi && bestj + bestsize < bhi && a[besti + bestsize] === b[bestj + bestsize]) bestsize++;
    return [besti, bestj, bestsize];
  };
  let matched = 0;
  const queue: [number, number, number, number][] = [[0, a.length, 0, b.length]];
  while (queue.length) {
    const [alo, ahi, blo, bhi] = queue.pop()!;
    const [i, j, k] = longest(alo, ahi, blo, bhi);
    if (!k) continue;
    matched += k;
    if (alo < i && blo < j) queue.push([alo, i, blo, j]);
    if (i + k < ahi && j + k < bhi) queue.push([i + k, ahi, j + k, bhi]);
  }
  return (2 * matched) / (a.length + b.length);
}

/** Python's `f"{x:.2f}"` for a positive number: correctly rounded, an exact tie to the even digit. */
export function fixed2(x: number): string {
  // A tie at two decimals is a multiple of 1/8 with an odd numerator: the only ones a double holds exactly.
  if (Number.isInteger(x * 8) && (x * 8) % 2 !== 0 && !Number.isInteger(x * 100)) {
    const low = Math.floor(x * 100);
    return ((low % 2 === 0 ? low : low + 1) / 100).toFixed(2);
  }
  return x.toFixed(2);
}

/** What Python's `str.isprintable()` refuses: control, format, surrogate, private, unassigned and separator characters. */
const NOT_PRINTABLE = /[\p{Cc}\p{Cf}\p{Cs}\p{Co}\p{Cn}\p{Zl}\p{Zp}\p{Zs}]/u;

/** Python's `repr()` of a str. */
export function pyRepr(text: string): string {
  const quote = text.includes("'") && !text.includes('"') ? '"' : "'";
  let out = quote;
  for (const c of text) {
    const code = c.codePointAt(0)!;
    if (c === quote || c === "\\") out += `\\${c}`;
    else if (c === "\n") out += "\\n";
    else if (c === "\r") out += "\\r";
    else if (c === "\t") out += "\\t";
    else if (c !== " " && NOT_PRINTABLE.test(c)) {
      const hex = code.toString(16);
      out += code <= 0xff ? `\\x${hex.padStart(2, "0")}` : code <= 0xffff ? `\\u${hex.padStart(4, "0")}` : `\\U${hex.padStart(8, "0")}`;
    } else out += c;
  }
  return out + quote;
}

/** Python's `urllib.parse.quote(text, safe='')`. */
export function pyQuote(text: string): string {
  return encodeURIComponent(text).replace(/[!'()*]/g, (c) => `%${c.charCodeAt(0).toString(16).toUpperCase()}`);
}
