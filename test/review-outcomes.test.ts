/**
 * Outcome-level checks of the sixth review's findings that the tests added with the fixes do not already make, and of
 * replies from a fetch without body streams: what is returned, thrown, read or cancelled, never the wording of a
 * message beyond the kind of error.
 */

import { describe, expect, it } from "vitest";
import { PolicyClient } from "../src/index.js";
import type { PolicyIk } from "../src/internal.js";
import { callService, parseRaw } from "../src/internal.js";
import { answerFor } from "../src/proxy.js";

/** The documented ceiling on a direct reply (`MAX_REPLY_BYTES`), 16 MiB. */
const MAX_REPLY_BYTES = 16 * 1024 * 1024;
const OK = (actions: string) => `{"status":{"code":"DA-0000"},"session_id":"s","actions":${actions}}`;

describe("short command rows, strict off", () => {
  it("refuses rows shorter than the measured arm whatever strict says, default and raw", async () => {
    const pose = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    for (const processing of ["default", "raw"] as const) {
      const client = new PolicyClient({
        checkpoint: "c",
        baseUrl: "https://gateway.test",
        processing,
        strict: false,
        embodiment: { state: { "arm_r.joint_pos": { shape: [6] } } },
        fetch: async () => new Response(OK('{"q_r_chunk":[[0.1,0.2],[0.3,0.4]]}')),
      });
      const outcome = await client.getAction({ "state.arm_r.joint_pos": pose, prompt: "x" }).then(
        (action) => ({ returned: action }),
        (error: Error) => ({ error: error.name }),
      );
      expect(outcome, processing).toEqual({ error: "DreamActionError" });
    }
  });
});

describe("a reply past the ceiling", () => {
  /** A body without length that would never end: it says whether it was cancelled, and how much was read of it. */
  const endless = () => {
    const state = { read: 0, cancelled: false };
    const body = new ReadableStream<Uint8Array>({
      pull: (controller) => {
        state.read += 64 * 1024;
        if (state.read > 2 * MAX_REPLY_BYTES) controller.error(new Error("read past the ceiling"));
        else controller.enqueue(new Uint8Array(64 * 1024).fill(0x20));
      },
      cancel: () => void (state.cancelled = true),
    });
    return { state, body };
  };

  it("has its body stream cancelled, gateway and services alike", async () => {
    const gateway = endless();
    const client = new PolicyClient({ checkpoint: "c", baseUrl: "https://gateway.test", fetch: async () => new Response(gateway.body) });
    await expect(client.getAction({ prompt: "x" })).rejects.toMatchObject({ name: "DreamProtocolError" });
    expect(gateway.state.cancelled).toBe(true);
    const service = endless();
    await expect(callService("GET", "/v1/models", { url: "https://platform.test", fetch: async () => new Response(service.body) })).rejects.toMatchObject({ name: "DreamProtocolError" });
    expect(service.state.cancelled).toBe(true);
  });
});

describe("the proxy's parse of a wide answer", () => {
  it("stops at the value limit: well under the work the same answer takes unbounded", () => {
    // Counted as the calls a parse makes to RegExp.prototype.exec, Number() and parseFloat(), whatever it makes per
    // number: the bounded parse inside answerFor must stop near the million-value limit, half the two million values.
    const text = OK(`{"extra":[${"0,".repeat(2_000_000)}0]}`);
    const count = (run: () => unknown) => {
      const exec = RegExp.prototype.exec;
      const NumberBefore = globalThis.Number;
      const parseFloatBefore = globalThis.parseFloat;
      let calls = 0;
      RegExp.prototype.exec = function (this: RegExp, input: string) {
        calls++;
        return exec.call(this, input);
      };
      const counted = Object.assign(function (value?: unknown) {
        calls++;
        return NumberBefore(value);
      }, NumberBefore) as NumberConstructor;
      Object.setPrototypeOf(counted, NumberBefore);
      globalThis.Number = counted;
      globalThis.parseFloat = (input: string) => (calls++, parseFloatBefore(input));
      let result: unknown;
      try {
        result = run();
      } finally {
        RegExp.prototype.exec = exec;
        globalThis.Number = NumberBefore;
        globalThis.parseFloat = parseFloatBefore;
      }
      return { calls, result };
    };
    const unbounded = count(() => parseRaw(text));
    const bounded = count(() => answerFor("gateway", 200, new TextEncoder().encode(text), "dk_SYNTHETIC_outcome_key_01"));
    expect(bounded.result).toBeNull();
    expect(unbounded.calls).toBeGreaterThan(2_000_000);
    expect(bounded.calls).toBeLessThanOrEqual(0.6 * unbounded.calls);
  });
});

describe("a fetch without body streams (React Native, hand-written doubles)", () => {
  /** A Response-like reply with no body stream: only status and text(). */
  const bodiless = (text: string, status = 200) => ({ status, body: null, text: async () => text }) as unknown as Response;

  it("is read through text(): the gateway's and the platform's replies are parsed", async () => {
    const reply = '{"status":{"code":"DA-0001","reason":"new"},"session_id":"s","actions":{"q_r_chunk":[[0.5,0.25]]}}';
    const client = new PolicyClient({ checkpoint: "c", baseUrl: "https://gateway.test", processing: "raw", fetch: async () => bodiless(reply) });
    expect(await client.getAction({ prompt: "x" })).toEqual({ q_r_chunk: [[0.5, 0.25]] });
    expect(client.sessionId).toBe("s");
    const model = { id: "m", owner: "public", base_model: "m", embodiment: "yam", servable: true, created_at: null, cameras: ["top"] };
    expect(await callService("GET", "/v1/models", { url: "https://platform.test", fetch: async () => bodiless(JSON.stringify([model])) })).toEqual([model]);
  });

  it("refuses a reply past the ceiling as the streamed case does, and keeps an empty one empty", async () => {
    // Multi-byte text: the ceiling is in UTF-8 bytes, not characters.
    const wide = `{"status":{"code":"DA-0000"},"pad":"${"é".repeat(MAX_REPLY_BYTES / 2)}"}`;
    const client = new PolicyClient({ checkpoint: "c", baseUrl: "https://gateway.test", fetch: async () => bodiless(wide) });
    await expect(client.getAction({ prompt: "x" })).rejects.toMatchObject({ name: "DreamProtocolError", message: expect.stringMatching(/larger than/) });
    await expect(callService("GET", "/v1/models", { url: "https://platform.test", fetch: async () => bodiless(`[${" ".repeat(MAX_REPLY_BYTES)}]`) })).rejects.toMatchObject({ name: "DreamProtocolError", message: expect.stringMatching(/more than/) });
    // No content at all: an empty reply, as before (here, the gateway says it is not JSON).
    const empty = new PolicyClient({ checkpoint: "c", baseUrl: "https://gateway.test", fetch: async () => bodiless("") });
    await expect(empty.getAction({ prompt: "x" })).rejects.toMatchObject({ name: "DreamProtocolError", message: expect.stringMatching(/not JSON/) });
  });
});

describe("command rows on the IK path", () => {
  /** A client-side IK that answers `rows` for the right arm (six joints), and projects them unchanged. */
  const ik = (rows: number[][]): PolicyIk => ({
    joints: { q_r_chunk: 6, q_l_chunk: 6 },
    yamJoints: null,
    fence: null,
    head: "joint",
    observe: (observation) => ({ ...observation }),
    solve: (action) => ({ ...action, q_r_chunk: rows }),
    project: (_key, chunk) => chunk,
    sentChecks: (checks) => ({ ...checks }),
    replyWidths: (widths) => ({ ...widths }),
  });
  const reply = async () => new Response(OK('{"per_key":{"action.right_arm":[[0,0,0,0,0,0]]}}'));

  it("refuses solved rows shorter than the arm's joints and grip before anything is returned; full rows are returned", async () => {
    for (const rows of [[[0.1, 0.2]], [[0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0.5], [0, 0, 0, 0, 0, 0]]]) {
      const client = new PolicyClient({ checkpoint: "c", baseUrl: "https://gateway.test", ik: ik(rows), ikLeadMs: 0, fetch: reply });
      const outcome = await client.getAction({ prompt: "x" }).then(
        (action) => ({ returned: action }),
        (error: Error & { detail?: unknown }) => ({ error: error.name, detail: error.detail }),
      );
      expect(outcome, JSON.stringify(rows)).toMatchObject({ error: "DreamActionError", detail: { kind: "shape", key: "q_r_chunk" } });
    }
    const full = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]];
    const client = new PolicyClient({ checkpoint: "c", baseUrl: "https://gateway.test", ik: ik(full), ikLeadMs: 0, fetch: reply });
    expect((await client.getAction({ prompt: "x" })).q_r_chunk).toEqual(full);
  });
});
