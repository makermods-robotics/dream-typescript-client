/**
 * Edge replies against Python's own outcomes (scripts/gen_edge_replies.py, edge_replies.json): every case through the
 * direct client (default and raw processing) and through createDreamProxy() with the client behind it, strict on and
 * off. Each outcome must be Python's, or the TypeScript outcome of the deliberate difference the case names (the
 * fixture's `deliberate`, each with its reason). No other difference passes.
 */

import { describe, expect, it } from "vitest";
import { PolicyClient } from "../src/index.js";
import { createDreamProxy } from "../src/proxy.js";
import { fixture } from "./helpers.js";

interface Outcome {
  action?: unknown;
  error?: string;
  session_id_length: number | null;
}

interface Case {
  name: string;
  reply: string;
  spec: unknown;
  observation: Record<string, unknown>;
  deliberate: Partial<Record<Path, string>>;
  python: { strict: Outcome; loose: Outcome };
}

type Path = "direct" | "raw" | "proxy";

const edge = fixture<{ deliberate: Record<string, string>; ts_outcome: Record<string, string>; cases: Case[] }>("edge_replies.json");
const KEY = "dk_SYNTHETIC_edge_key_0001";

/** A fixture value as JavaScript holds it: the spelled-out numbers decoded. */
function decode(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(decode);
  if (typeof value === "object" && value !== null) {
    const record = value as Record<string, unknown>;
    if ("$float" in record) return { nan: NaN, inf: Infinity, "-inf": -Infinity, "-0": -0 }[record.$float as string];
    if ("$int" in record) return Number(record.$int);
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(record)) Object.defineProperty(out, key, { value: decode(record[key]), enumerable: true, writable: true, configurable: true });
    return out;
  }
  return value;
}

/** Deep equality with numbers to 1e-9 (Python's filter is float64 as ours), -0 told from 0 where Python says -0. */
function same(got: unknown, want: unknown, path: string): void {
  if (typeof want === "number") {
    expect(typeof got, `${path} is a number`).toBe("number");
    if (Object.is(want, -0)) expect(Object.is(got, -0), `${path} is -0`).toBe(true);
    else expect(Math.abs((got as number) - want), `${path}: ${got} vs ${want}`).toBeLessThanOrEqual(1e-9);
    return;
  }
  if (Array.isArray(want)) {
    expect(Array.isArray(got), `${path} is an array`).toBe(true);
    expect((got as unknown[]).length, `${path} length`).toBe(want.length);
    want.forEach((item, i) => same((got as unknown[])[i], item, `${path}[${i}]`));
    return;
  }
  if (typeof want === "object" && want !== null) {
    expect(typeof got === "object" && got !== null, `${path} is an object`).toBe(true);
    expect(Object.keys(got as object).sort(), `${path} keys`).toEqual(Object.keys(want).sort());
    for (const key of Object.keys(want)) same((got as Record<string, unknown>)[key], (want as Record<string, unknown>)[key], `${path}.${key}`);
    return;
  }
  expect(got, path).toEqual(want);
}

async function run(entry: Case, path: Path, strict: boolean): Promise<{ action?: unknown; error?: string; sessionLength: number | null }> {
  let fetch = async () => new Response(entry.reply);
  if (path === "proxy") {
    const handler = createDreamProxy({ apiKey: KEY, allowedHosts: "any", fetch: async () => new Response(entry.reply) });
    fetch = async () => handler(new Request("http://proxy.test/v1/get_action", { method: "POST", body: "{}", headers: { "Dream-Client-Version": "x" } }));
  }
  const client = new PolicyClient({
    checkpoint: "c",
    baseUrl: path === "proxy" ? "http://proxy.test" : "https://gateway.test",
    apiKey: path === "proxy" ? null : KEY,
    proxyHeaders: path === "proxy",
    processing: path === "raw" ? "raw" : "default",
    embodiment: entry.spec ?? undefined,
    strict,
    fetch,
    onWarning: () => {},
  });
  try {
    const action = await client.getAction(entry.observation);
    return { action, sessionLength: client.sessionId?.length ?? null };
  } catch (error) {
    return { error: (error as Error).name, sessionLength: client.sessionId?.length ?? null };
  }
}

describe("edge replies, as Python handles them", () => {
  it("covers the cases the review names", () => {
    expect(edge.cases.length).toBeGreaterThanOrEqual(95);
    for (const reason of new Set(edge.cases.flatMap((c) => Object.values(c.deliberate)))) expect(edge.deliberate[reason!], reason).toBeDefined();
  });

  for (const entry of edge.cases) {
    it(entry.name, async () => {
      for (const path of ["direct", "raw", "proxy"] as const) {
        for (const strict of [true, false]) {
          const label = `${path} ${strict ? "strict" : "loose"}`;
          const python = entry.python[strict ? "strict" : "loose"];
          const got = await run(entry, path, strict);
          const reason = entry.deliberate[path];
          const expected = reason ? edge.ts_outcome[reason]! : (python.error ?? "ok");
          expect(got.error ?? "ok", `${label}${reason ? ` (deliberate: ${reason})` : ""}`).toBe(expected);
          if (got.error || expected !== "ok") continue;
          if (path === "raw") {
            // No filter: the action as the reply carries it.
            same(got.action, JSON.parse(entry.reply).actions, label);
          } else if (!reason) {
            same(got.action, decode(python.action), label);
            expect(got.sessionLength, `${label} session id`).toBe(python.session_id_length);
          }
        }
      }
    });
  }
});
