/**
 * The client against a mock DreamAPI server that replays what the Python `dream.DreamClient` sent and was answered
 * (scripts/gen_fixtures.py, wire.json): every request the TypeScript client sends must be the one Python sent (method,
 * path, headers, JSON body), every answer it returns the one Python returned, every round trip it hands onRoundTrip()
 * the one Python handed on_round_trip(), and every line it warns about a fine-tune's training inputs Python's.
 */

import { createServer, type IncomingMessage, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { CLIENT_VERSION, DreamClient, DreamTrainingWarning, UPSTREAM, type PolicyConfig, type RoundTrip } from "../src/index.js";
import { CEILINGS } from "../scripts/fidelity.js";
import { expectClose, fixture, maxDiff } from "./helpers.js";

const wire = fixture("wire.json");
const upstreamVersion = UPSTREAM.dreamlabsClient.replace(/\./g, "\\.");

interface Expected {
  method: string;
  path: string;
  headers: Record<string, string>;
  body: unknown;
  reply: { status: number; body: string };
}

let server: Server;
let url = "";
let queue: Expected[] = [];
let models: unknown[] = [];
let modelRequests = 0;
let advance = 0;
const clock = { ms: 0 };
const failures: string[] = [];

function readBody(request: IncomingMessage): Promise<string> {
  return new Promise((resolve) => {
    let text = "";
    request.on("data", (chunk) => (text += chunk));
    request.on("end", () => resolve(text));
  });
}

beforeAll(async () => {
  server = createServer(async (request, response) => {
    const text = await readBody(request);
    const send = (status: number, body: string) => {
      response.writeHead(status, { "Content-Type": "application/json" });
      response.end(body);
    };
    if (request.url === "/v1/models") {
      modelRequests++;
      expect(request.headers.authorization).toBe("Bearer dk_test_key");
      return send(200, JSON.stringify(models));
    }
    const expected = queue.shift();
    if (!expected) {
      failures.push(`unexpected request ${request.method} ${request.url} ${text}`);
      return send(500, "{}");
    }
    try {
      expect(request.method).toBe(expected.method);
      expect(request.url).toBe(expected.path);
      if (expected.method === "GET") {
        // The data service's training_data: the platform call's headers, as `dream.datasets._call` sends them.
        expect(request.headers.authorization).toBe(expected.headers.Authorization);
        expect(request.headers.accept).toBe(expected.headers.Accept);
        expect(expected.headers["User-Agent"]).toBe(`dream-client/${UPSTREAM.dreamlabsClient}`);
        expect(request.headers["user-agent"]).toBe(`dream-client/${CLIENT_VERSION}`);
        expect(text).toBe("");
        expect(expected.body).toBeNull();
        send(expected.reply.status, expected.reply.body);
        return;
      }
      expect(request.headers["content-type"]).toBe(expected.headers["Content-Type"]);
      expect(request.headers.authorization).toBe(expected.headers.Authorization);
      // Python names its semver; this client names itself, deliberately not a semver (src/version.ts).
      expect(expected.headers["Dream-Client-Version"]).toBe(UPSTREAM.dreamlabsClient);
      expect(request.headers["dream-client-version"]).toBe(CLIENT_VERSION);
      expect(expected.headers["User-Agent"]).toMatch(new RegExp(`^dream-client/${upstreamVersion} [0-9a-f]{8}$`));
      expect(request.headers["user-agent"]).toBe(`dream-client/${CLIENT_VERSION} ${request.headers["user-agent"]!.slice(-8)}`);
      expect(request.headers["user-agent"]).toMatch(/ [0-9a-f]{8}$/);
      // The tool poses Python sends are dream_ik's float32 forward kinematics; ours are float64.
      expectClose(JSON.parse(text), expected.body, CEILINGS.wire.body, `${expected.path} body`);
    } catch (error) {
      failures.push(String(error));
    }
    if (expected.path === "/v1/get_action") clock.ms += advance;
    send(expected.reply.status, expected.reply.body);
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

afterAll(() => new Promise<void>((resolve) => server.close(() => resolve())));

const CONFIG_KEYS: Record<string, keyof PolicyConfig> = {
  checkpoint: "checkpoint",
  embodiment: "embodiment",
  strict: "strict",
  timeout_ms: "timeoutMs",
  obs_window: "obsWindow",
  obs_fps: "obsFps",
  execution_horizon: "executionHorizon",
  action_hz: "actionHz",
  sync_above_ms: "syncAboveMs",
  robot: "robot",
  ik: "ik",
};

const toConfig = (python: Record<string, unknown>): PolicyConfig =>
  Object.fromEntries(Object.entries(python).map(([key, value]) => [CONFIG_KEYS[key] ?? key, value])) as unknown as PolicyConfig;

async function outcome(run: () => unknown): Promise<{ value?: unknown; error?: any }> {
  try {
    return { value: await run() };
  } catch (error) {
    return { error };
  }
}

function expectError(step: any, error: any) {
  expect(error, `${step.op} should fail with ${step.error.type}: ${step.error.message}`).toBeDefined();
  expect(error.name, `${step.op}: ${error.message}`).toBe(step.error.type);
  for (const [python, ts] of [["code", "code"], ["reason", "reason"], ["retryable", "retryable"], ["retry_after_s", "retryAfterS"]] as const) {
    if (python in step.error) expect(error[ts], python).toEqual(step.error[python]);
  }
}

/** Joints our float64 solve answers differ from dream_ik's float32 one by less than this, radians (`npm run fidelity`). */
const IK_JOINT_TOLERANCE = CEILINGS.wire.ikJoints;

describe("DreamClient against the Python client's wire", () => {
  for (const scenario of wire.scenarios) {
    it(scenario.name, async () => {
      models = scenario.models;
      failures.length = 0;
      modelRequests = 0;
      const warnings: string[] = [];
      const trips: RoundTrip[] = [];
      const client = new DreamClient({
        apiKey: "dk_test_key",
        baseUrl: url,
        platformUrl: url,
        dataUrl: url,
        now: () => clock.ms,
        onWarning: (warning) => void (warning instanceof DreamTrainingWarning && warnings.push(warning.message)),
      });
      client.onRoundTrip((trip) => void trips.push(trip));
      let ik = false;
      for (const [index, step] of scenario.steps.entries()) {
        const label = `step ${index} (${step.op})`;
        warnings.length = 0;
        trips.length = 0;
        modelRequests = 0;
        queue = step.requests.filter((r: Expected) => r.path !== "/v1/models");
        const pythonListings = step.requests.length - queue.length;
        let result: { value?: unknown; error?: any };
        if (step.op === "policy") {
          ik = "ik" in step.config;
          const config = toConfig(step.config);
          if (config.ik) {
            // Python's seam takes the round trip only. And its joint head needs no embodiment id; ours requires the
            // arm's, here the one the model was trained on.
            config.ikLeadMs = 0;
            const record = scenario.models?.find((m: { id: string }) => m.id === config.checkpoint);
            if (config.ik.head === "joint" && record) config.ik = { ...config.ik, embodimentId: record.embodiment };
          }
          result = await outcome(() => client.policy(config));
        } else if (step.op === "set_observation") {
          result = await outcome(() => client.setObservation(step.frames, step.observation, { timestamp: step.timestamp }));
        } else if (step.op === "get_action") {
          clock.ms = step.sent_s * 1000;
          advance = step.rtt_s * 1000;
          result = await outcome(() => client.getAction({ commanded: step.commanded }));
        } else if (step.op === "reset_episode") {
          result = await outcome(() => client.resetEpisode());
        } else {
          result = await outcome(() => client.close());
        }
        expect(failures, label).toEqual([]);
        expect(queue.map((r) => r.path), `${label}: requests Python sent that we did not`).toEqual([]);
        expect(warnings, `${label}: training warnings`).toEqual(step.warnings);
        // One models listing per policy() where Python makes one or more: this client reuses one listing for the
        // embodiment config and the IK's record within a call, where Python lists the models for each.
        expect(modelRequests, `${label}: model listings`).toBe(Math.min(pythonListings, 1));
        expect(trips.length, `${label}: round trips`).toBe(step.round_trips.length);
        trips.forEach((trip, i) => {
          const want = step.round_trips[i];
          expect(trip.frames, `${label}: round trip frames`).toEqual(want.frames);
          expectClose(trip.observation, want.observation, 1e-12, `${label}: round trip observation`);
          expect(trip.action === null ? null : trip.action === result.value ? "returned" : "other", `${label}: round trip action`).toBe(want.action);
          expect(trip.timing && { serve_infer_ms: trip.timing.serveInferMs, gateway_ms: trip.timing.gatewayMs }, `${label}: round trip timing`).toEqual(want.timing);
          expect(trip.error?.name ?? null, `${label}: round trip error`).toBe(want.error);
          expect(trip.trained, `${label}: round trip lines`).toEqual(want.trained);
        });
        if (step.error) {
          expectError(step, result.error);
          continue;
        }
        expect(result.error, `${label}: ${result.error?.message}`).toBeUndefined();
        if (step.op === "get_action") {
          const { action, mode, session_id: sessionId, timing } = step.result;
          expect(client.mode).toBe(mode);
          expect(client.sessionId).toBe(sessionId);
          expect(client.lastTiming?.serveInferMs ?? null).toBe(timing.serve_infer_ms ?? null);
          expect(client.lastTiming?.gatewayMs ?? null).toBe(timing.gateway_ms ?? null);
          const got = result.value as Record<string, unknown>;
          if (!ik) expectClose(got, action, CEILINGS.wire.actions, label);
          else {
            expect(Object.keys(got).sort(), label).toEqual(Object.keys(action).sort());
            for (const key of Object.keys(action)) {
              if (key === "q_r_chunk" || key === "q_l_chunk") {
                const diff = maxDiff(got[key], action[key]);
                expect(diff, `${label} ${key}: largest joint difference to dream_ik`).toBeLessThan(IK_JOINT_TOLERANCE);
              } else expectClose(got[key], action[key], CEILINGS.wire.actions, `${label} ${key}`);
            }
          }
        }
        if (step.op === "reset_episode") expect(client.sessionId).toBe(step.result.session_id);
      }
      await client.close();
    });
  }
});
