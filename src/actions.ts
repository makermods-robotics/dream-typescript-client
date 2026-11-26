/**
 * What an action dict may hold: the one definition the client and the proxy both apply, whatever `strict` says.
 *
 * Python (`dream.client`, `dream.smoothing`) returns the model's action dict with every key it does not filter
 * untouched, and filters each arm's chunk (`q_r_chunk`, `q_l_chunk`) through `np.asarray(..., float64)`, so a chunk
 * that is not rectangular, or holds a leaf numpy cannot read as a number, is a `DreamActionError`. This client holds
 * the same, and more where a value is a joint target:
 *
 * - Actuator heads: `q_r_chunk`, `q_l_chunk`, every array-valued part of `per_key`, and the heads the embodiment
 *   declares as numeric (a shape spec's action entries with a `shape`, an embodiment config's `per_key` parts). Each
 *   is null as a whole (kept, and left unfiltered), a finite number (a bare head, returned as it came, as Python does
 *   for a chunk that is not two-dimensional), or rectangular arrays whose every leaf is a finite number: no null,
 *   string, boolean or object leaf, no hole.
 * - Every number anywhere in the action, extensions included, is finite.
 * - Everything else is metadata and passes as Python passes it: labels and other extensions of any JSON type, nulls
 *   included, and `per_key` itself as a container (its non-array parts are labels).
 */

import { DreamActionError, detail } from "./errors.js";

/** The heads every action dict may carry: each arm's chunk. */
export const CHUNK_HEADS = ["q_r_chunk", "q_l_chunk"] as const;

/** Reads a leaf as a number: the number, or undefined when it is not one. Plain values by default. */
export type NumberReader = (value: unknown) => number | undefined;

const plainNumber: NumberReader = (value) => (typeof value === "number" ? value : undefined);

/** The heads a configured embodiment declares as numeric; empty when it declares none or cannot be read. */
export interface DeclaredHeads {
  /** Action keys of a shape spec whose entry has a `shape`. */
  action: readonly string[];
  /** `per_key` parts of an embodiment config's `action` (`action.<part>`). */
  perKey: readonly string[];
}

export const NO_DECLARED_HEADS: DeclaredHeads = { action: [], perKey: [] };

type Problem = { key: string; row: number | null; kind: "non_finite" | "null" | "hole" | "shape" | "ragged" };

const isArrayLike = (value: unknown): value is ArrayLike<unknown> => Array.isArray(value) || (ArrayBuffer.isView(value) && !(value instanceof DataView));

/** The first problem of an actuator head: null as a whole passes; else a finite number, or rectangular numeric arrays. */
function headProblem(value: unknown, key: string, read: NumberReader): Problem | null {
  if (value === null || value === undefined) return null;
  const number = read(value);
  if (number !== undefined) return Number.isFinite(number) ? null : { key, row: null, kind: "non_finite" };
  if (!isArrayLike(value)) return { key, row: null, kind: "shape" };
  // Walked level by level: every array of a level has the same length, and a level is all arrays or all numbers.
  let level: ArrayLike<unknown>[] = [value];
  // Per array of the level, the row (index at the top level) it belongs to.
  let rows: number[] = [0];
  for (let depth = 0; ; depth++) {
    const width = level[0]!.length;
    const next: ArrayLike<unknown>[] = [];
    const nextRows: number[] = [];
    let arrays = 0;
    let leaves = 0;
    for (let a = 0; a < level.length; a++) {
      const items = level[a]!;
      if (items.length !== width) return { key, row: depth === 0 ? null : rows[a]!, kind: "ragged" };
      const sparse = Array.isArray(items);
      for (let i = 0; i < items.length; i++) {
        const row = depth === 0 ? i : rows[a]!;
        if (sparse && !(i in items)) return { key, row, kind: "hole" };
        const item = items[i];
        if (item === null) return { key, row, kind: "null" };
        if (isArrayLike(item)) {
          arrays++;
          next.push(item);
          nextRows.push(row);
          continue;
        }
        const n = read(item);
        if (n === undefined) return { key, row, kind: "shape" };
        if (!Number.isFinite(n)) return { key, row, kind: "non_finite" };
        leaves++;
      }
    }
    if (arrays && leaves) return { key, row: null, kind: "ragged" };
    if (!arrays) return null;
    level = next;
    rows = nextRows;
  }
}

/** The first non-finite number anywhere in `value`, without recursion. */
function nonFinite(value: unknown, read: NumberReader): string | null {
  const stack: [unknown, string][] = [[value, ""]];
  while (stack.length) {
    const [node, path] = stack.pop()!;
    const n = read(node);
    if (n !== undefined) {
      if (!Number.isFinite(n)) return path;
      continue;
    }
    if (isArrayLike(node)) {
      for (let i = 0; i < node.length; i++) stack.push([node[i], path]);
    } else if (typeof node === "object" && node !== null) {
      for (const key of Object.keys(node)) stack.push([(node as Record<string, unknown>)[key], path ? `${path}.${key}` : key]);
    }
  }
  return null;
}

/** The actuator heads present in `action`, with their dotted keys. */
export function actuatorHeads(action: Readonly<Record<string, unknown>>, declared: DeclaredHeads = NO_DECLARED_HEADS): [string, unknown][] {
  const heads: [string, unknown][] = [];
  for (const key of new Set<string>([...CHUNK_HEADS, ...declared.action])) if (Object.hasOwn(action, key)) heads.push([key, action[key]]);
  const perKey = action.per_key;
  if (typeof perKey === "object" && perKey !== null && !Array.isArray(perKey)) {
    const parts = perKey as Record<string, unknown>;
    for (const part of Object.keys(parts)) {
      if (isArrayLike(parts[part]) || declared.perKey.includes(part)) heads.push([`per_key.${part}`, parts[part]]);
    }
  }
  return heads;
}

/**
 * Checks an action dict by the rules above.
 *
 * @param read How a leaf reads as a number (the proxy's numbers keep their source text).
 * @throws DreamActionError A rule is broken; the message and `detail` name the key and row.
 */
export function checkAction(action: unknown, declared: DeclaredHeads = NO_DECLARED_HEADS, read: NumberReader = plainNumber, what = "the model's reply"): void {
  if (typeof action !== "object" || action === null || Array.isArray(action)) return;
  const dict = action as Record<string, unknown>;
  for (const [key, value] of actuatorHeads(dict, declared)) {
    const problem = headProblem(value, key, read);
    if (problem === null) continue;
    const where = `key '${problem.key}'${problem.row === null ? "" : ` row ${problem.row}`}`;
    const kind = problem.kind === "null" || problem.kind === "non_finite" ? "non_finite" : "shape";
    const why = {
      null: "holds null where a number belongs",
      non_finite: "holds a non-finite number (NaN or an infinity; JSON's 1e400 reads as Infinity)",
      hole: "has a hole (a missing element of a sparse array)",
      shape: "holds something other than a number where a joint value belongs",
      ragged: "is not rectangular rows of numbers",
    }[problem.kind];
    throw new DreamActionError(`${what} ${why} at ${where}, so nothing is returned`, detail(kind, problem.key, problem.row));
  }
  const path = nonFinite(dict, read);
  if (path !== null) {
    throw new DreamActionError(`${what} holds a non-finite number at key '${path}' (JSON's 1e400 reads as Infinity), so nothing is returned`, detail("non_finite", path || null));
  }
}

/** A chunk head as rows of numbers (`[H][width]`), or null when it is not two-dimensional rows (null, a bare head). */
export function chunkRows(head: unknown): number[][] | null {
  if (!Array.isArray(head) || !head.length || !head.every((row) => Array.isArray(row) && row.every((v) => typeof v === "number"))) return null;
  return head as number[][];
}
