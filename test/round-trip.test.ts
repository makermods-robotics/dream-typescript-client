/**
 * The 1.1 client's application hooks: onRoundTrip() (`DreamClient.on_round_trip`), waitForIk(), the envelope reasons,
 * and the training-input lines (end to end against Python in wire.test.ts).
 */

import { describe, expect, it } from "vitest";
import { DreamClient, DreamConfigError, DreamServerError, REASONS, type RoundTrip } from "../src/index.js";

const OK = { status: { code: "DA-0001", reason: "new" }, session_id: "ses_1", actions: { q_r_chunk: [[0.1, 0.2]] }, timing: { serve_infer_ms: 1, gateway_ms: 2 } };

function client(replies: unknown[]) {
  const fetch = async (url: string): Promise<Response> => Response.json(url.endsWith("/v1/end_session") ? { status: { code: "DA-0000" } } : replies.shift());
  return new DreamClient({ apiKey: "dk_test", baseUrl: "https://gateway.test", fetch, onWarning: () => {} });
}

describe("onRoundTrip", () => {
  it("hands every getAction() over, returned or refused, frames apart from the rest", async () => {
    const refusal = { status: { code: "DA-2000", reason: "pool_full", retryable: true, retry_after_s: 2 } };
    const dream = client([OK, refusal]);
    const trips: RoundTrip[] = [];
    const remove = dream.onRoundTrip((trip) => void trips.push(trip));
    await expect(dream.waitForIk(1)).rejects.toThrow(/no policy is open: call policy\(\{ checkpoint: \.\.\. \}\) before waitForIk\(\)/);
    await dream.policy({ checkpoint: "dreamlabs/dream-1-beta", strict: false });
    expect(await dream.waitForIk(60)).toBe(true);
    // No observation yet: nothing was sent, so nothing is handed over (as Python).
    await expect(dream.getAction()).rejects.toThrow(/no observation since the episode started/);
    expect(trips).toHaveLength(0);
    dream.setObservation({ "video.top": "Zg==" }, { "state.arm": [1, 2], prompt: "wave" }, { timestamp: 1 });
    const before = Date.now() / 1000;
    const action = await dream.getAction();
    expect(trips).toHaveLength(1);
    const [first] = trips;
    expect(first!.frames).toEqual({ "video.top": ["Zg=="] });
    expect(first!.observation).toEqual({ "state.arm": [1, 2], prompt: "wave" });
    expect(first!.action).toBe(action);
    expect(first!.error).toBeNull();
    expect(first!.timing?.serveInferMs).toBe(1);
    expect(first!.trained).toEqual([]);
    expect(first!.sentS).toBeGreaterThanOrEqual(before - 1);
    expect(Object.isFrozen(first)).toBe(true);
    await expect(dream.getAction()).rejects.toThrow(DreamServerError);
    expect(trips[1]!.action).toBeNull();
    expect(trips[1]!.error).toBeInstanceOf(DreamServerError);
    expect((trips[1]!.error as DreamServerError).reason).toBe("pool_full");
    remove();
    await expect(dream.getAction()).rejects.toThrow();
    expect(trips).toHaveLength(2);
    await dream.close();
  });

  it("throws what a callback throws out of getAction(), as Python", async () => {
    const dream = client([OK]);
    await dream.policy({ checkpoint: "dreamlabs/dream-1-beta", strict: false });
    dream.onRoundTrip(() => {
      throw new Error("view crashed");
    });
    dream.setObservation({ "video.top": "Zg==" }, { prompt: "wave" }, { timestamp: 1 });
    await expect(dream.getAction()).rejects.toThrow("view crashed");
    expect(() => dream.onRoundTrip(42 as never)).toThrow(DreamConfigError);
    await dream.close();
  });
});

describe("REASONS", () => {
  it("is shared.contracts.envelope 1.0.42's reasons per code", () => {
    expect(REASONS["DA-6000"]).toEqual(["internal", "store_write_failed", "cannot_scale", "service_unavailable", "connection_timed_out"]);
    expect(REASONS["DA-4000"]).toContain("client_outdated");
    expect(Object.values(REASONS).flat()).not.toContain("steering_unavailable");
  });
});
