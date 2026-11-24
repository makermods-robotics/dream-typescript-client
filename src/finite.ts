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
