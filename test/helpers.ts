import { readFileSync } from "node:fs";
import { expect } from "vitest";

/** A golden fixture written by scripts/gen_fixtures.py. */
export function fixture<T = any>(name: string): T {
  return JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8")) as T;
}

/** Deep equality with numbers compared to `tolerance` (absolute); null matches null only. */
export function expectClose(actual: unknown, expected: unknown, tolerance = 1e-9, path = "$"): void {
  if (typeof expected === "number") {
    expect(typeof actual, `${path} is a number`).toBe("number");
    expect(Math.abs((actual as number) - expected), `${path}: ${actual} vs ${expected}`).toBeLessThanOrEqual(tolerance);
    return;
  }
  if (Array.isArray(expected)) {
    const list = ArrayBuffer.isView(actual) ? Array.from(actual as unknown as ArrayLike<number>) : actual;
    expect(Array.isArray(list), `${path} is an array`).toBe(true);
    expect((list as unknown[]).length, `${path} length`).toBe(expected.length);
    expected.forEach((item, i) => expectClose((list as unknown[])[i], item, tolerance, `${path}[${i}]`));
    return;
  }
  if (expected !== null && typeof expected === "object") {
    expect(actual !== null && typeof actual === "object", `${path} is an object`).toBe(true);
    expect(Object.keys(actual as object).sort(), `${path} keys`).toEqual(Object.keys(expected).sort());
    for (const [key, value] of Object.entries(expected)) expectClose((actual as Record<string, unknown>)[key], value, tolerance, `${path}.${key}`);
    return;
  }
  expect(actual, path).toEqual(expected);
}

/** The largest absolute difference between two nested number arrays of one shape. */
export function maxDiff(a: unknown, b: unknown): number {
  if (typeof a === "number" && typeof b === "number") return Math.abs(a - b);
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return Infinity;
    return a.reduce((m: number, v, i) => Math.max(m, maxDiff(v, b[i])), 0);
  }
  return a === b ? 0 : Infinity;
}
