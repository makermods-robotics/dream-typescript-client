/** Finite-number checks of what the model answered: JSON has no NaN, but `1e400` parses as Infinity. */

import { DreamActionError, detail } from "./errors.js";

/**
 * Index of the first hole in `items` (a sparse array's missing element: `[1, , 3]`, or `delete a[1]`), or -1. Typed
 * arrays have none. `every`, `some`, `forEach`, `map`, `filter` and `flat` skip holes, so a check written with them
 * passes an array with a hole in it: every check here walks indices instead.
 */
export function firstHole(items: ArrayLike<unknown>): number {
  if (!Array.isArray(items)) return -1;
  for (let i = 0; i < items.length; i++) if (!(i in items)) return i;
  return -1;
}

/** Whether `test` holds for every element of `items`, a hole failing it (where `Array.prototype.every` skips holes). */
export function every<T>(items: ArrayLike<T>, test: (item: T, index: number) => boolean): boolean {
  const sparse = Array.isArray(items);
  for (let i = 0; i < items.length; i++) if ((sparse && !(i in items)) || !test(items[i]!, i)) return false;
  return true;
}

/** Index of the first element `test` fails, a hole failing it, or -1. */
export function firstFailing<T>(items: ArrayLike<T>, test: (item: T, index: number) => boolean): number {
  const sparse = Array.isArray(items);
  for (let i = 0; i < items.length; i++) if ((sparse && !(i in items)) || !test(items[i]!, i)) return i;
  return -1;
}

/** Whether `value` is an array or a typed array whose every element, holes failing, is a number (finite when `finite`). */
export function isNumbers(value: unknown, finite = false): value is ArrayLike<number> {
  if (!(Array.isArray(value) || (ArrayBuffer.isView(value) && !(value instanceof DataView)))) return false;
  return every(value as ArrayLike<unknown>, (v) => typeof v === "number" && (!finite || Number.isFinite(v)));
}

/**
 * Where the first non-finite number in a value is: its dotted key and its row (first index under the key). A hole in
 * an array (no element at all) is found too, with `hole` set, and a null inside an array, with `nulled` set. A null
 * value of an object key (`per_key`'s `action.right_arm: null`) is not a number's place and passes.
 */
export function firstNonFinite(value: unknown, key = "", row: number | null = null): { key: string; row: number | null; hole?: true; nulled?: true } | null {
  if (typeof value === "number") return Number.isFinite(value) ? null : { key, row };
  if (Array.isArray(value) || (ArrayBuffer.isView(value) && !(value instanceof DataView))) {
    const items = value as ArrayLike<unknown>;
    const sparse = Array.isArray(items);
    for (let i = 0; i < items.length; i++) {
      if (sparse && !(i in items)) return { key, row: row ?? i, hole: true };
      // Inside an array a null stands where a number belongs (a joint target): a proxy that re-serialised an
      // infinity writes it as null.
      if (items[i] === null) return { key, row: row ?? i, nulled: true };
      const found = firstNonFinite(items[i], key, row ?? i);
      if (found) return found;
    }
    return null;
  }
  if (typeof value === "object" && value !== null) {
    for (const [name, item] of Object.entries(value)) {
      const found = firstNonFinite(item, key ? `${key}.${name}` : name, null);
      if (found) return found;
    }
  }
  return null;
}

/** The actuator heads every action dict may carry: each arm's chunk. `per_key`'s parts are actuator heads too. */
export const ACTUATOR_HEADS = ["q_r_chunk", "q_l_chunk"] as const;

/**
 * Where a head's value first holds something other than a finite number in arrays: a NaN or an infinity, a null, a
 * hole, or anything else (an object, a string, a boolean) where a joint value belongs. The head as a whole may be null.
 */
function firstBadLeaf(value: unknown, key: string): { key: string; row: number | null; kind: "non_finite" | "null" | "hole" | "shape" } | null {
  if (value === null || value === undefined) return null;
  const visit = (node: unknown, row: number | null): { key: string; row: number | null; kind: "non_finite" | "null" | "hole" | "shape" } | null => {
    if (typeof node === "number") return Number.isFinite(node) ? null : { key, row, kind: "non_finite" };
    if (Array.isArray(node) || (ArrayBuffer.isView(node) && !(node instanceof DataView))) {
      const items = node as ArrayLike<unknown>;
      const sparse = Array.isArray(items);
      for (let i = 0; i < items.length; i++) {
        if (sparse && !(i in items)) return { key, row: row ?? i, kind: "hole" };
        if (items[i] === null) return { key, row: row ?? i, kind: "null" };
        const found = visit(items[i], row ?? i);
        if (found) return found;
      }
      return null;
    }
    return { key, row, kind: "shape" };
  };
  return visit(value, null);
}

/**
 * Checks every actuator head of a reply: `q_r_chunk`, `q_l_chunk`, every part of `per_key`, and the heads the
 * configured embodiment declares (`declared`). Each is null as a whole, or arrays whose every leaf is a finite number:
 * a joint target is never NaN, an infinity (JSON's `1e400`), null, a hole, or an object. Any other key is an extension,
 * passed through untouched, as in Python.
 *
 * @throws DreamActionError A head holds something else; the message and `detail` name the key and row.
 */
export function checkFiniteAction(action: unknown, declared: readonly string[] = [], what = "the model's reply"): void {
  if (typeof action !== "object" || action === null) return;
  const dict = action as Record<string, unknown>;
  const heads: [string, unknown][] = [...new Set([...ACTUATOR_HEADS, ...declared])].filter((key) => key in dict).map((key) => [key, dict[key]]);
  const perKey = dict.per_key;
  if (typeof perKey === "object" && perKey !== null && !Array.isArray(perKey)) {
    for (const [part, value] of Object.entries(perKey)) heads.push([`per_key.${part}`, value]);
  }
  for (const [key, value] of heads) {
    const found = firstBadLeaf(value, key);
    if (found === null) continue;
    const where = `key '${found.key}'${found.row === null ? "" : ` row ${found.row}`}`;
    if (found.kind === "null") throw new DreamActionError(`${what} holds null where a number belongs at ${where}, so nothing is returned`, detail("non_finite", found.key, found.row));
    if (found.kind === "hole") throw new DreamActionError(`${what} has a hole (a missing element of a sparse array) at ${where}, so nothing is returned`, detail("shape", found.key, found.row));
    if (found.kind === "shape") throw new DreamActionError(`${what} holds something other than a number where a joint value belongs at ${where}, so nothing is returned`, detail("shape", found.key, found.row));
    throw new DreamActionError(
      `${what} holds a non-finite number at ${where} (NaN or an infinity; JSON's 1e400 reads as Infinity), so nothing is returned`,
      detail("non_finite", found.key, found.row),
    );
  }
}
