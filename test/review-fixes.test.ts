/**
 * The client side of the proxy and of the training-input check: createClient()'s services, the key kept out of
 * messages and warnings, the headers a worker sends behind a proxy, malformed training records, URL checks and the
 * round-trip registrations.
 */

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { DreamCallbackWarning, DreamClient, DreamConfigError, DreamTrainingWarning, type RoundTrip } from "../src/index.js";
import { callService, checkServiceUrl } from "../src/internal.js";
import { createClient } from "../src/node/index.js";

const KEY = "dk_dummy_review_key";
const dir = mkdtempSync(join(tmpdir(), "dreamlabs-ts-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/** A credentials file of staging services, and an environment that names only it. */
function staging(): Record<string, string> {
  const file = join(dir, "credentials");
  writeFileSync(file, `DREAM_API_KEY=${KEY}\nDREAM_GATEWAY_URL=https://gateway.staging.test\nDREAM_API_URL=https://platform.staging.test\nDREAM_DATA_URL=https://data.staging.test\n`);
  return { DREAM_CREDENTIALS: file };
}

interface Call {
  url: string;
  headers: Record<string, string>;
}

function recording(answer: (url: string) => Response) {
  const calls: Call[] = [];
  const fetch = async (url: string, init: RequestInit) => {
    calls.push({ url, headers: { ...(init.headers as Record<string, string>) } });
    return answer(url);
  };
  return { calls, fetch };
}

describe("createClient()", () => {
  it("takes all three services and the key from the credentials, the data service included", async () => {
    const { calls, fetch } = recording((url) => (url.endsWith("/training_data") ? Response.json({ cameras: {}, prompts: [] }) : Response.json({ status: { code: "DA-0000" } })));
    const client = createClient({ fetch, onWarning: () => {} }, staging());
    expect([client.baseUrl, client.platformUrl, client.dataUrl]).toEqual(["https://gateway.staging.test", "https://platform.staging.test", "https://data.staging.test"]);
    await client.policy({ checkpoint: "dream-1-ft:0a1b2c3d-towels", strict: false });
    expect(calls.map((c) => c.url)).toEqual(["https://data.staging.test/v1/fine_tuning/models/dream-1-ft%3A0a1b2c3d-towels/training_data"]);
    expect(calls[0]!.headers.Authorization).toBe(`Bearer ${KEY}`);
    await client.close();
  });

  it("with proxyUrl takes nothing from the credentials: no key, and the proxy for every service", async () => {
    const { calls, fetch } = recording(() => Response.json({ cameras: {}, prompts: [] }));
    const client = createClient({ proxyUrl: "https://app.example.com/api/dream", fetch, onWarning: () => {} }, staging());
    expect([client.baseUrl, client.platformUrl, client.dataUrl]).toEqual(Array(3).fill("https://app.example.com/api/dream"));
    await client.policy({ checkpoint: "dream-1-ft:0a1b2c3d-towels", strict: false });
    expect(calls[0]!.url).toBe("https://app.example.com/api/dream/v1/fine_tuning/models/dream-1-ft%3A0a1b2c3d-towels/training_data");
    expect(calls[0]!.headers.Authorization).toBeUndefined();
    // Behind a proxy the client sends no User-Agent of its own: the proxy names it.
    expect(calls[0]!.headers["User-Agent"]).toBeUndefined();
    await client.close();
  });
});

describe("the key in messages", () => {
  it("is cut from a service's error and from the training warning, even when the service echoes it", async () => {
    const echo = () => Response.json({ error: { code: "unavailable", message: `bad header Authorization: Bearer ${KEY}` } }, { status: 503 });
    const warnings: string[] = [];
    const client = new DreamClient({ apiKey: KEY, baseUrl: "https://gateway.test", dataUrl: "https://data.test", fetch: recording(echo).fetch, onWarning: (w) => void warnings.push(w.message) });
    await client.policy({ checkpoint: "dream-1-ft:0a1b2c3d-towels", strict: false });
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain("Bearer <api key>");
    expect(warnings[0]).not.toContain(KEY);
    const error = await callService("GET", "/v1/models", { url: "https://platform.test", apiKey: KEY, fetch: recording(echo).fetch }).catch((e: Error) => e);
    expect(String(error)).not.toContain(KEY);
    const unreachable = await callService("GET", "/v1/models", { url: "https://platform.test", apiKey: KEY, fetch: () => Promise.reject(new TypeError(`refused ${KEY}`)) }).catch((e: Error) => e);
    expect(String(unreachable)).not.toContain(KEY);
    expect((unreachable as Error).cause).toBeUndefined();
    await client.close();
  });
});

describe("a worker", () => {
  it("sends no User-Agent, which it cannot set and a cross-origin preflight would ask for", async () => {
    const scope = globalThis as { WorkerGlobalScope?: unknown };
    scope.WorkerGlobalScope = function WorkerGlobalScope() {};
    try {
      const { calls, fetch } = recording(() => Response.json([]));
      await callService("GET", "/v1/models", { url: "https://proxy.test", fetch });
      expect(calls[0]!.headers["User-Agent"]).toBeUndefined();
    } finally {
      delete scope.WorkerGlobalScope;
    }
    const { calls, fetch } = recording(() => Response.json([]));
    await callService("GET", "/v1/models", { url: "https://platform.test", fetch, proxyHeaders: true });
    expect(calls[0]!.headers["User-Agent"]).toBeUndefined();
    expect(calls[0]!.headers["Dream-Client-Version"]).toBeDefined();
  });
});

describe("a malformed training record", () => {
  it("leaves the policy usable: the bad sizes are left out, the call is sent", async () => {
    const reply = { status: { code: "DA-0001", reason: "new" }, session_id: "s", actions: { q_r_chunk: [[0.1]] } };
    const { calls, fetch } = recording((url) =>
      url.endsWith("/training_data") ? Response.json({ cameras: { front: [null, [640, "x"]], top: "big" }, prompts: [null, "wave"] }) : Response.json(reply),
    );
    const warnings: unknown[] = [];
    const client = new DreamClient({ apiKey: KEY, baseUrl: "https://gateway.test", dataUrl: "https://data.test", fetch, onWarning: (w) => void warnings.push(w) });
    await client.policy({ checkpoint: "dream-1-ft:0a1b2c3d-towels", strict: false });
    client.setObservation({ "video.front": "/9j/wAARCAHgAoADAAAAAAAAAAAA/9lqcGVnLTk5" }, { prompt: "wave" }, { timestamp: 1 });
    expect(await client.getAction()).toEqual({ q_r_chunk: [[0.1]] });
    expect(calls.map((c) => c.url.split("/v1/")[1])).toEqual(["fine_tuning/models/dream-1-ft%3A0a1b2c3d-towels/training_data", "get_action"]);
    expect(warnings.filter((w) => w instanceof DreamTrainingWarning)).toEqual([]);
    await client.close();
  });
});

describe("service URLs", () => {
  it("refuse a query or fragment marker and a user name, and come back as parsed", () => {
    for (const url of ["https://gateway.test/?", "https://gateway.test#", "https://gateway.test/api?x", "https://u:p@gateway.test", "//evil.test/api"]) {
      expect(() => checkServiceUrl(url, "baseUrl", true), url).toThrow(DreamConfigError);
      expect(() => new DreamClient({ baseUrl: url }), url).toThrow(DreamConfigError);
    }
    expect(checkServiceUrl("HTTPS://Gateway.Test:443/api/dream/", "baseUrl", true)).toBe("https://gateway.test/api/dream");
    expect(checkServiceUrl("https://gateway.test/", "gatewayUrl", false)).toBe("https://gateway.test");
    expect(checkServiceUrl("/api/dream/", "baseUrl", true)).toBe("/api/dream");
  });
});

describe("onRoundTrip registrations", () => {
  it("are one per call: the same function added twice is two, and a removal handle removes only its own, once", async () => {
    const replies = () => Response.json({ status: { code: "DA-0001", reason: "new" }, session_id: "s", actions: {} });
    const client = new DreamClient({ apiKey: KEY, baseUrl: "https://gateway.test", fetch: recording(replies).fetch });
    const trips: RoundTrip[] = [];
    const fn = (trip: RoundTrip) => void trips.push(trip);
    const first = client.onRoundTrip(fn);
    client.onRoundTrip(fn);
    await client.policy({ checkpoint: "dreamlabs/dream-1-beta", strict: false });
    client.setObservation({ "video.top": "Zg==" }, {}, { timestamp: 1 });
    await client.getAction();
    expect(trips).toHaveLength(2);
    first();
    first();
    await client.getAction();
    expect(trips).toHaveLength(3);
    await client.close();
  });
});

describe("hold fixes, client side", () => {
  it("createClient(): an option given as undefined keeps the credentials' value; apiKey null asks for none", async () => {
    const env = staging();
    const client = createClient({ baseUrl: undefined, platformUrl: undefined, dataUrl: undefined, apiKey: undefined }, env);
    expect([client.baseUrl, client.platformUrl, client.dataUrl]).toEqual(["https://gateway.staging.test", "https://platform.staging.test", "https://data.staging.test"]);
    const { calls, fetch } = recording(() => Response.json([]));
    await createClient({ fetch, apiKey: undefined }, env).getModels();
    expect(calls[0]!.headers.Authorization).toBe(`Bearer ${KEY}`);
    const keyless = recording(() => Response.json([]));
    await createClient({ fetch: keyless.fetch, apiKey: null, platformUrl: "https://platform.staging.test" }, env).getModels();
    expect(keyless.calls[0]!.headers.Authorization).toBeUndefined();
  });

  it("keeps the key out of an error's fields and causes, a warning and a round trip", async () => {
    const { inspect } = await import("node:util");
    const echoCode = () => Response.json({ error: { code: `bad_key_${KEY}`, message: "no" } }, { status: 400 });
    const refused = await callService("GET", "/v1/models", { url: "https://platform.test", apiKey: KEY, fetch: recording(echoCode).fetch }).catch((e: unknown) => e);
    expect(inspect(refused, { depth: 10 })).not.toContain(KEY);
    const garbled = () => new Response(`{"oops": "${KEY}"`, { status: 200 });
    const bad = await callService("GET", "/v1/models", { url: "https://platform.test", apiKey: KEY, fetch: recording(garbled).fetch }).catch((e: unknown) => e);
    expect((bad as Error).cause).toBeUndefined();
    expect(inspect(bad, { depth: 10 })).not.toContain(KEY);
    // The gateway's own refusal echoing the key (escaped, too) in its envelope.
    const escapedKey = Array.from(KEY, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`).join("");
    const gateway = new DreamClient({
      apiKey: KEY,
      baseUrl: "https://gateway.test",
      dataUrl: "https://data.test",
      fetch: async (url) =>
        url.endsWith("/training_data")
          ? Response.json({ cameras: {}, prompts: [`say ${KEY}`] })
          : new Response(`{"status":{"code":"DA-2000","reason":"${escapedKey}","detail":"${KEY}"}}`),
      onWarning: (w) => void warnings.push(w.message),
    });
    const warnings: string[] = [];
    const trips: RoundTrip[] = [];
    gateway.onRoundTrip((trip) => void trips.push(trip));
    await gateway.policy({ checkpoint: "dream-1-ft:0a1b2c3d-towels", strict: false });
    gateway.setObservation({ "video.top": "Zg==" }, { prompt: "wave" }, { timestamp: 1 });
    const error = await gateway.getAction().catch((e: unknown) => e);
    expect(inspect(error, { depth: 10 })).not.toContain(KEY);
    expect(warnings.join("\n")).toContain("<api key>");
    expect(warnings.join("\n")).not.toContain(KEY);
    expect(inspect(trips, { depth: 10 })).not.toContain(KEY);
    expect(trips[0]!.trained.join("\n")).toContain("<api key>");
    await gateway.close();
  });

  it("checks a service URL as written, keeps a root mount, and never shows a password", () => {
    for (const url of ["https:gateway.test", "https:///gateway.test", "https://gateway.test:0", "https://gateway.test:65536", "https://gateway.test/a/../v1", "https://gateway.test/./x"]) {
      expect(() => checkServiceUrl(url, "baseUrl", true), url).toThrow(DreamConfigError);
    }
    expect(() => checkServiceUrl("https://gateway.test/path/..", "DREAM_GATEWAY_URL", false)).toThrow(DreamConfigError);
    expect(() => checkServiceUrl("https://gateway.test/v2", "DREAM_GATEWAY_URL", false)).toThrow(DreamConfigError);
    const leaked = (() => {
      try {
        checkServiceUrl("https://me:hunter2@gateway.test/?", "baseUrl", true);
      } catch (error) {
        return String(error);
      }
      return "";
    })();
    expect(leaked).toContain("<userinfo>@gateway.test");
    expect(leaked).not.toContain("hunter2");
    expect(checkServiceUrl("https://[::1]:8443/api/", "baseUrl", true)).toBe("https://[::1]:8443/api");
    const root = new DreamClient({ proxyUrl: "/" });
    expect([root.baseUrl, root.platformUrl, root.dataUrl]).toEqual(["", "", ""]);
  });

  it("reports a rejected async round-trip callback as a warning, never an unhandled rejection", async () => {
    const warnings: unknown[] = [];
    const replies = () => Response.json({ status: { code: "DA-0001", reason: "new" }, session_id: "s", actions: {} });
    const client = new DreamClient({ apiKey: KEY, baseUrl: "https://gateway.test", fetch: recording(replies).fetch, onWarning: (w) => void warnings.push(w) });
    client.onRoundTrip(async () => {
      throw new Error(`view lost ${KEY}`);
    });
    await client.policy({ checkpoint: "dreamlabs/dream-1-beta", strict: false });
    client.setObservation({ "video.top": "Zg==" }, {}, { timestamp: 1 });
    await client.getAction();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toBeInstanceOf(DreamCallbackWarning);
    expect((warnings[0] as Error).message).toBe("an onRoundTrip() callback's promise was rejected: view lost <api key>");
    await client.close();
  });
});
