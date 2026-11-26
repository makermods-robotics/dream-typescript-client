/**
 * The client against a mock DreamAPI server that replays what the Python `dream.DreamClient` sent and was answered
 * (scripts/gen_fixtures.py, wire.json): every request the TypeScript client sends must be the one Python sent (method,
 * path, headers, JSON body), every answer it returns the one Python returned, every round trip it hands onRoundTrip()
 * the one Python handed on_round_trip(), and every line it warns about a fine-tune's training inputs Python's.
 *
 * A step ends only when the server has received every request Python made in it (or a bound passes): the client's
 * session release gives up after two seconds and carries on, so on a stalled machine its request can reach the server
 * after the call returned. A request that reaches the server in another step than Python sent it fails, naming both.
 * The client's requests carry a token of this run: anything else reaching the port (another local process probing a
 * fresh listener) is answered 404 and noted as a stray, never matched against Python's requests.
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
  /** The step Python sent it in (set here). */
  step?: number;
}

/** The header and value the client's own requests carry in this run. */
const RUN_HEADER = "x-wire-run";
const RUN_TOKEN = crypto.randomUUID();
/** Requests from anything but the client: noted, answered 404, never matched. */
const strays: string[] = [];

/** How long a step waits for the requests Python made in it to reach the server, milliseconds. */
const STEP_DRAIN_MS = 15_000;

let server: Server;
let url = "";
/** The scenario's requests other than the models listing, in the order Python sent them, each with its step. */
let expectedAll: Expected[] = [];
/** How many of `expectedAll` the server has received. */
let received = 0;
/** The step the test is running. */
let currentStep = 0;
/** Wakes the step waiting for the server to have received `received >= target` requests. */
let drained: { target: number; resolve: () => void } | null = null;
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
    // A request its client gave up on still counts as received, with what arrived of it.
    request.on("close", () => resolve(text));
  });
}

beforeAll(async () => {
  server = createServer(async (request, response) => {
    const text = await readBody(request);
    if (request.headers[RUN_HEADER] !== RUN_TOKEN) {
      strays.push(`${request.method} ${request.url} (Host ${request.headers.host ?? "none"}, User-Agent ${request.headers["user-agent"] ?? "none"})`);
      response.writeHead(404).end();
      return;
    }
    const send = (status: number, body: string) => {
      response.writeHead(status, { "Content-Type": "application/json" });
      response.end(body);
    };
    if (request.url === "/v1/models") {
      modelRequests++;
      if (request.headers.authorization !== "Bearer dk_test_key") failures.push(`step ${currentStep}: models listing without the key`);
      return send(200, JSON.stringify(models));
    }
    const expected = expectedAll[received];
    if (!expected) {
      failures.push(`step ${currentStep}: unexpected request ${request.method} ${request.url} ${text}`);
      return send(500, "{}");
    }
    received++;
    if (drained && received >= drained.target) drained.resolve();
    if (expected.step !== currentStep || expected.method !== request.method || expected.path !== request.url) {
      const owner = expectedAll.findIndex((e, i) => i >= received - 1 && e.method === request.method && e.path === request.url);
      failures.push(
        `${request.method} ${request.url} reached the server during step ${currentStep}; Python sent ` +
          (owner >= 0 ? `it in step ${expectedAll[owner]!.step}` : "no such request") +
          `, and next expected ${expected.method} ${expected.path} from step ${expected.step}`,
      );
      return send(expected.reply.status, expected.reply.body);
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
      strays.length = 0;
      modelRequests = 0;
      expectedAll = scenario.steps.flatMap((step: { requests: Expected[] }, index: number) =>
        step.requests.filter((r) => r.path !== "/v1/models").map((r) => ({ ...r, step: index })),
      );
      received = 0;
      let throughStep = 0;
      const warnings: string[] = [];
      const trips: RoundTrip[] = [];
      const client = new DreamClient({
        apiKey: "dk_test_key",
        // Marks the client's own requests, as the server tells them from strangers'.
        fetch: (input, init) => globalThis.fetch(input, { ...init, headers: { ...(init.headers as Record<string, string>), [RUN_HEADER]: RUN_TOKEN } }),
        baseUrl: url,
        platformUrl: url,
        dataUrl: url,
        now: () => clock.ms,
        onWarning: (warning) => void (warning instanceof DreamTrainingWarning && warnings.push(warning.message)),
      });
      client.onRoundTrip((trip) => void trips.push(trip));
      let ik = false;
      for (const [index, step] of scenario.steps.entries()) {
        let label = `step ${index} (${step.op})`;
        warnings.length = 0;
        trips.length = 0;
        modelRequests = 0;
        currentStep = index;
        const stepRequests = step.requests.filter((r: Expected) => r.path !== "/v1/models");
        throughStep += stepRequests.length;
        const pythonListings = step.requests.length - stepRequests.length;
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
        // The step is over once the server has every request Python sent in it, a late one included.
        if (received < throughStep) {
          let timer: ReturnType<typeof setTimeout> | undefined;
          await Promise.race([
            new Promise<void>((resolve) => (drained = { target: throughStep, resolve })),
            new Promise<void>((resolve) => (timer = setTimeout(resolve, STEP_DRAIN_MS))),
          ]);
          clearTimeout(timer);
          drained = null;
        }
        // Strays never fail a step by themselves; when the step fails, its message names them.
        if (strays.length) label += ` [stray requests to the port so far: ${strays.join("; ")}]`;
        expect(failures, label).toEqual([]);
        expect(
          expectedAll.slice(received, throughStep).map((r) => `${r.method} ${r.path}`),
          `${label}: requests Python sent in this step that the server did not receive within ${STEP_DRAIN_MS} ms`,
        ).toEqual([]);
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
