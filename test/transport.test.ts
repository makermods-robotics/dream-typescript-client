import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  DreamClient,
  DreamConfigError,
  DreamError,
  DreamObservationError,
  DreamProtocolError,
  DreamSlowNetworkWarning,
  DreamTimeout,
  PolicyClient,
} from "../src/index.js";

let server: Server;
let url = "";
let handler: (body: any) => { status?: number; body: unknown; delayMs?: number; headers?: Record<string, string> } = () => ({ body: {} });
const seen: { path: string; headers: Record<string, unknown>; body: any }[] = [];

beforeAll(async () => {
  server = createServer((request, response) => {
    let text = "";
    request.on("data", (chunk) => (text += chunk));
    request.on("end", () => {
      const body = JSON.parse(text || "null");
      seen.push({ path: request.url!, headers: request.headers, body });
      const reply = handler(body);
      setTimeout(() => {
        response.writeHead(reply.status ?? 200, { "Content-Type": "application/json", ...reply.headers });
        response.end(typeof reply.body === "string" ? reply.body : JSON.stringify(reply.body));
      }, reply.delayMs ?? 0);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

afterAll(() => new Promise<void>((resolve) => server.close(() => resolve())));

const chunk = (value: number) => Array.from({ length: 48 }, (_, k) => [value + k * 0.001, 0, 0, 0, 0, 0, 0.5]);
const ok = (session = "ses_a", rtt?: number) => ({
  body: { status: { code: "DA-0000" }, session_id: session, actions: { q_r_chunk: chunk(0), q_l_chunk: chunk(0) }, timing: { serve_infer_ms: 100, gateway_ms: 120 } },
  delayMs: rtt,
});
const observation = { "state.arm_r.joint_pos": [0, 0, 0, 0, 0, 0], "state.arm_l.joint_pos": [0, 0, 0, 0, 0, 0], prompt: "x" };

describe("the transport", () => {
  it("aborts a slow reply as DreamTimeout, then works again", async () => {
    const client = new PolicyClient({ checkpoint: "c", baseUrl: url, timeoutMs: 400 });
    handler = () => ({ ...ok(), delayMs: 1500 });
    await expect(client.getAction(observation)).rejects.toThrow(DreamTimeout);
    expect(client.lastTiming?.responseBytes).toBe(0);
    handler = () => ok();
    await expect(client.getAction(observation)).resolves.toHaveProperty("q_r_chunk");
    await client.close();
  });

  it("reports an unreachable gateway as DreamProtocolError", async () => {
    const client = new PolicyClient({ checkpoint: "c", baseUrl: "http://127.0.0.1:9", timeoutMs: 2000 });
    await expect(client.getAction(observation)).rejects.toThrow(DreamProtocolError);
  });

  it("refuses a relative baseUrl where nothing can resolve it", () => {
    expect(() => new PolicyClient({ checkpoint: "c", baseUrl: "/api/dream" })).toThrow(DreamConfigError);
  });

  it("does not follow a redirect with the key", async () => {
    handler = () => ({ status: 307, body: "", headers: { Location: "http://example.invalid/steal" } });
    const client = new PolicyClient({ checkpoint: "c", baseUrl: url, apiKey: "dk_x" });
    await expect(client.getAction(observation)).rejects.toThrow(DreamProtocolError);
    expect(seen.at(-1)!.headers.authorization).toBe("Bearer dk_x");
  });

  it("names the key JSON cannot carry and sends nothing", async () => {
    const client = new PolicyClient({ checkpoint: "c", baseUrl: url });
    const before = seen.length;
    await expect(client.getAction({ ...observation, "state.bad": [NaN] })).rejects.toThrow(/observation key 'state.bad' cannot be sent as JSON/);
    await expect(client.getAction({ ...observation, blob: new ArrayBuffer(2) })).rejects.toThrow(DreamObservationError);
    expect(seen.length).toBe(before);
  });

  it("refuses a second getAction while one is in flight", async () => {
    const client = new PolicyClient({ checkpoint: "c", baseUrl: url });
    handler = () => ({ ...ok(), delayMs: 100 });
    const first = client.getAction(observation);
    await expect(client.getAction(observation)).rejects.toThrow(DreamError);
    await expect(client.resetEpisode()).rejects.toThrow(/in flight/);
    await first;
    await client.close();
    await expect(client.getAction(observation)).rejects.toThrow(/closed client/);
  });

  it("releases the session on reset and close, and starts a new one after", async () => {
    const client = new PolicyClient({ checkpoint: "ckpt", baseUrl: url });
    handler = (body) => (body.observation ? ok("ses_b") : { body: { status: { code: "DA-0000" } } });
    await client.getAction(observation);
    expect(client.sessionId).toBe("ses_b");
    await client.resetEpisode();
    expect(seen.at(-1)).toMatchObject({ path: "/v1/end_session", body: { checkpoint: "ckpt", session_id: "ses_b" } });
    expect(client.sessionId).toBeNull();
    await client.getAction(observation);
    expect(seen.at(-1)!.body.session_id).toBeNull();
    await client.close();
    expect(seen.at(-1)!.path).toBe("/v1/end_session");
  });

  it("warns once when the round trips stay slow, and never throws it", async () => {
    let clock = 0;
    const warnings: DreamSlowNetworkWarning[] = [];
    handler = () => {
      clock += 1500;
      return ok();
    };
    const client = new PolicyClient({ checkpoint: "c", baseUrl: url, now: () => clock, onWarning: (w) => warnings.push(w) });
    for (let i = 0; i < 7; i++) await client.getAction(observation);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]!.message).toMatch(/Network is slow: get_action round trips are about 1500 ms \(network 1380 ms\)/);
    expect(client.mode).toBe("sync");
    expect(client.refillDue(10, 38)).toBe(false);
    expect(client.refillDue(48, 0)).toBe(true);
  });
});

describe("DreamClient", () => {
  it("needs a policy first", async () => {
    const client = new DreamClient({ baseUrl: url });
    expect(() => client.setObservation({ "video.a": "Zg==" }, {})).toThrow(/no policy is open/);
    await expect(client.getAction()).rejects.toThrow(DreamConfigError);
    expect(() => client.mode).toThrow(DreamConfigError);
    expect(client.lastTiming).toBeNull();
  });

  it("refuses config keys it does not take, and checks types before anything is sent", async () => {
    const client = new DreamClient({ baseUrl: url, platformUrl: url });
    await expect(client.policy({ checkpoint: "c", host: "x" } as never)).rejects.toThrow(/'host'/);
    await expect(client.policy({ checkpoint: "c", obsWindow: 0, strict: false })).rejects.toThrow(/obsWindow/);
    await expect(client.policy({ checkpoint: "c", timeoutMs: 1.5, strict: false })).rejects.toThrow(/timeoutMs/);
    await expect(client.policy({ checkpoint: "c", robot: "robot" as never, strict: false })).rejects.toThrow(/robot/);
    await expect(client.policy({ checkpoint: "c", ik: { arm: "yam", head: "eef" }, strict: false })).rejects.toThrow(/does not solve/);
    await expect(client.policy({ checkpoint: "c", ik: { arm: "metal", head: "eef", region: {} } as never })).rejects.toThrow(/does not implement/);
  });
});
