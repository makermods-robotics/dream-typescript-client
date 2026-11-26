/**
 * The client side of the proxy and of the training-input check: createClient()'s services, the key kept out of
 * messages and warnings, the headers a worker sends behind a proxy, malformed training records, URL checks and the
 * round-trip registrations.
 */

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { DreamCallbackWarning, DreamClient, DreamConfigError, DreamTrainingWarning, PolicyClient, type RoundTrip } from "../src/index.js";
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
    expect(leaked).toContain("baseUrl must be");
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

describe("third review, client side", () => {
  it("never keeps a failed request's configuration as an error's cause", async () => {
    const { inspect } = await import("node:util");
    const failing = Object.assign(new TypeError(`fetch failed for ${KEY}`), {
      code: "ECONNREFUSED",
      config: { headers: { Authorization: `Bearer ${KEY}` } },
      cause: Object.assign(new Error("connect refused"), { request: { headers: { authorization: `Bearer ${KEY}` } }, errno: -111 }),
    });
    const client = new PolicyClient({ checkpoint: "c", baseUrl: "https://gateway.test", apiKey: KEY, fetch: () => Promise.reject(failing) });
    const error = await client.getAction({ prompt: "x" }).catch((e: unknown) => e);
    const shown = inspect(error, { depth: 20, showHidden: true });
    expect(shown).not.toContain(KEY);
    expect(shown).not.toContain("config");
    const cause = (error as Error).cause as Error & { code?: string };
    expect(cause.code).toBe("ECONNREFUSED");
    expect(cause.message).toBe("fetch failed for <api key>");
    expect(((cause.cause as Error & { errno?: number }).errno)).toBe(-111);
    const odd = new PolicyClient({ checkpoint: "c", baseUrl: "https://gateway.test", apiKey: KEY, fetch: () => Promise.reject(Object.create(null)) });
    await expect(odd.getAction({ prompt: "x" })).rejects.toThrow(/could not reach Dream Labs/);
  });

  it("refuses dot segments in any spelling, takes DNS and IDN hosts, and never prints the authority", () => {
    for (const url of ["https://proxy.test/application/%2e%2e/route", "https://proxy.test/a/.%2E/b", "https://proxy.test/%2E/b", "/api/%2e%2e/x"]) {
      expect(() => checkServiceUrl(url, "baseUrl", true), url).toThrow(/dot segment/);
    }
    expect(checkServiceUrl("https://proxy.example./api", "baseUrl", true)).toBe("https://proxy.example./api");
    expect(checkServiceUrl("https://bücher.example", "baseUrl", true)).toBe("https://xn--bcher-kva.example");
    for (const url of ["//me:hunter2@proxy.test/api", "https://me:p@ss@proxy.test/", "https://me:hunter2@proxy.test"]) {
      let message = "";
      try {
        checkServiceUrl(url, "baseUrl", true);
      } catch (error) {
        message = String(error);
      }
      expect(message, url).toMatch(/baseUrl must be/);
      expect(message, url).not.toMatch(/hunter2|p@ss|ss@|me:/);
    }
  });

  it("reports a callback rejecting with an unprintable value without throwing", async () => {
    const warnings: Error[] = [];
    const replies = () => Response.json({ status: { code: "DA-0001", reason: "new" }, session_id: "s", actions: {} });
    const client = new DreamClient({ apiKey: KEY, baseUrl: "https://gateway.test", fetch: recording(replies).fetch, onWarning: (w) => void warnings.push(w) });
    client.onRoundTrip(() => Promise.reject(Object.create(null)));
    client.onRoundTrip(() => Promise.reject({ toString: () => { throw new Error("no"); } }));
    await client.policy({ checkpoint: "dreamlabs/dream-1-beta", strict: false });
    client.setObservation({ "video.top": "Zg==" }, {}, { timestamp: 1 });
    await client.getAction();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(warnings.map((w) => w.message)).toEqual([
      "an onRoundTrip() callback's promise was rejected: (a value that cannot be shown)",
      "an onRoundTrip() callback's promise was rejected: (a value that cannot be shown)",
    ]);
    await client.close();
  });
});

describe("fourth review, client side", () => {
  const raw = (reply: string, embodiment?: unknown) =>
    new PolicyClient({ checkpoint: "c", baseUrl: "https://gateway.test", processing: "raw", embodiment, fetch: async () => new Response(reply) });

  it("returns no object, null or string where a joint value belongs, even raw and with a declared shape", async () => {
    const spec = { action: { q_r_chunk: { shape: [1, 1] } } };
    for (const [actions, kind] of [['{"q_r_chunk":[[{"x":0}]]}', "shape"], ['{"q_r_chunk":[[null]]}', "non_finite"], ['{"per_key":{"action.right_arm":[["0.1"]]}}', "shape"]] as const) {
      const error = await raw(`{"status":{"code":"DA-0000"},"session_id":"s","actions":${actions}}`, actions.startsWith('{"q_r') ? spec : undefined)
        .getAction({ prompt: "x" })
        .catch((e: { name: string; detail?: { kind: string } }) => e);
      expect(error, actions).toMatchObject({ name: "DreamActionError" });
      if (!actions.startsWith('{"q_r_chunk":[[{')) expect((error as { detail: { kind: string } }).detail.kind, actions).toBe(kind);
    }
    // Extensions pass through, nulls and strings included; a null head stays allowed.
    const fine = '{"status":{"code":"DA-0000"},"session_id":"s","actions":{"q_r_chunk":[[0.5]],"extra":[null],"head_label":"right","per_key":{"action.right_arm":null}}}';
    expect(await raw(fine, spec).getAction({ prompt: "x" })).toEqual({ q_r_chunk: [[0.5]], extra: [null], head_label: "right", per_key: { "action.right_arm": null } });
  });

  it("adopts a long session id exactly and releases it", async () => {
    const id = "s".repeat(1000);
    const seen: string[] = [];
    const client = new PolicyClient({
      checkpoint: "c",
      baseUrl: "https://gateway.test",
      processing: "raw",
      fetch: async (url, init) => {
        seen.push(`${url.split("/v1/")[1]} ${JSON.parse(String(init.body)).session_id}`);
        return new Response(JSON.stringify({ status: { code: "DA-0001", reason: "new" }, session_id: id, actions: {} }));
      },
    });
    await client.getAction({ prompt: "x" });
    expect(client.sessionId).toBe(id);
    await client.close();
    expect(seen).toEqual(["get_action null", `end_session ${id}`]);
  });

  it("checks https where the key is attached: relative URLs against the page, the internal transport and calls, literal true only", async () => {
    const scope = globalThis as { location?: unknown };
    scope.location = { href: "http://remote.example/app/" };
    try {
      expect(() => new DreamClient({ apiKey: KEY, platformUrl: "/api" })).toThrow(/plain http/);
      expect(() => new DreamClient({ proxyUrl: "/api/dream" })).not.toThrow();
    } finally {
      delete scope.location;
    }
    scope.location = { href: "https://app.example/" };
    try {
      expect(() => new DreamClient({ apiKey: KEY, platformUrl: "/api" })).not.toThrow();
    } finally {
      delete scope.location;
    }
    const { Transport } = await import("../src/internal.js");
    expect(() => new Transport({ baseUrl: "http://gateway.example", timeoutMs: 1000, apiKey: KEY })).toThrow(/plain http/);
    expect(() => new Transport({ baseUrl: "http://gateway.example", timeoutMs: 1000, apiKey: null })).not.toThrow();
    await expect(callService("GET", "/v1/models", { url: "http://platform.example", apiKey: KEY, fetch: async () => Response.json([]) })).rejects.toThrow(/plain http/);
    expect(() => new DreamClient({ apiKey: KEY, baseUrl: "http://gateway.example", allowInsecureHttp: "false" as never })).toThrow(/plain http/);
    expect(() => new DreamClient({ apiKey: KEY, baseUrl: "http://gateway.example", allowInsecureHttp: true })).not.toThrow();
    // This computer, by every loopback spelling.
    for (const url of ["http://127.0.0.1:9", "http://[::1]:9", "http://[::ffff:127.0.0.1]:9", "http://robot.localhost:9"]) {
      expect(() => new DreamClient({ apiKey: KEY, baseUrl: url }), url).not.toThrow();
    }
  });

  it("survives an error whose cause getter throws", async () => {
    const hostile = new TypeError(`fetch failed ${KEY}`);
    Object.defineProperty(hostile, "cause", {
      get() {
        throw new Error("getter exploded");
      },
    });
    const client = new PolicyClient({ checkpoint: "c", baseUrl: "https://gateway.test", apiKey: KEY, fetch: () => Promise.reject(hostile) });
    const error = await client.getAction({ prompt: "x" }).catch((e: Error) => e);
    expect(error.name).toBe("DreamProtocolError");
    expect(error.message).toContain("fetch failed <api key>");
    expect((error.cause as Error).message).toBe("fetch failed <api key>");
  });
});

describe("fifth review, client side", () => {
  it("sends the key only to the URL it checked: a relative URL resolved as fetch resolves it, refused with no page", async () => {
    const urls: string[] = [];
    const fetch = async (url: string) => (urls.push(url), Response.json({ status: { code: "DA-0001", reason: "new" }, session_id: "s", actions: {} }));
    const scope = globalThis as { document?: unknown; location?: unknown };
    scope.document = { baseURI: "http://remote.example/app/" };
    scope.location = { href: "http://remote.example/app/" };
    try {
      expect(() => new PolicyClient({ checkpoint: "c", baseUrl: "/api/dream", apiKey: KEY, fetch })).toThrow(/plain http/);
      const { Transport } = await import("../src/internal.js");
      // Checked again where the key is attached, against the page as it is when the request is made.
      const transport = new Transport({ baseUrl: "https://gateway.test", timeoutMs: 1000, apiKey: KEY, fetch });
      (transport as unknown as { baseUrl: string }).baseUrl = "/api/dream";
      await expect(transport.request({})).rejects.toThrow(/plain http/);
      expect(urls).toEqual([]);
    } finally {
      delete scope.document;
      delete scope.location;
    }
    scope.document = { baseURI: "https://app.example/base/page" };
    scope.location = { href: "https://app.example/" };
    try {
      await new PolicyClient({ checkpoint: "c", baseUrl: "/api/dream", apiKey: KEY, fetch }).getAction({ prompt: "x" });
      expect(urls).toEqual(["https://app.example/api/dream/v1/get_action"]);
    } finally {
      delete scope.document;
      delete scope.location;
    }
    // No page to resolve against: with a key it is refused before anything is sent.
    const { Transport } = await import("../src/internal.js");
    const transport = new Transport({ baseUrl: "/api/dream", timeoutMs: 1000, apiKey: KEY, fetch });
    await expect(transport.request({})).rejects.toThrow(/no page to resolve it against/);
    await expect(callService("GET", "/v1/models", { url: "/api", apiKey: KEY, fetch })).rejects.toThrow(/no page to resolve it against/);
    expect(urls).toHaveLength(1);
  });

  it("never lets a throwing message getter escape a platform call's last attempt", async () => {
    const hostile = {
      get message(): string {
        throw new Error("getter exploded");
      },
      toString() {
        throw new Error("no string either");
      },
    };
    const error = (await callService("GET", "/v1/models", { url: "https://platform.test", apiKey: KEY, fetch: () => Promise.reject(hostile) }).catch((e: unknown) => e)) as Error;
    expect(error.name).toBe("DreamProtocolError");
    expect(error.message).toContain("(a value that cannot be shown)");
  });

  it("keeps an action key named __proto__ as a key, and a null head through default processing", async () => {
    const reply = '{"status":{"code":"DA-0000"},"session_id":"s","actions":{"q_r_chunk":null,"__proto__":{"polluted":true},"per_key":{"__proto__":[[1,2]]}}}';
    const action = await new PolicyClient({ checkpoint: "c", baseUrl: "https://gateway.test", fetch: async () => new Response(reply) }).getAction({ prompt: "x" });
    expect(Object.getPrototypeOf(action)).toBe(Object.prototype);
    expect(Object.keys(action).sort()).toEqual(["__proto__", "per_key", "q_r_chunk"]);
    expect((action as Record<string, unknown>).polluted).toBeUndefined();
    expect(Object.getOwnPropertyDescriptor(action, "__proto__")!.value).toEqual({ polluted: true });
    expect(Object.keys(action.per_key!)).toEqual(["__proto__"]);
    expect(action.q_r_chunk).toBeNull();
  });

  it("refuses an integer a double cannot hold before delivery, raw and with strict off", async () => {
    for (const token of ["9007199254740993", "1000000000000000001"]) {
      const reply = `{"status":{"code":"DA-0000"},"session_id":"s","actions":{"head":[[${token}]]}}`;
      const client = new PolicyClient({ checkpoint: "c", baseUrl: "https://gateway.test", processing: "raw", strict: false, embodiment: { action: { head: { shape: [1, 1] } } }, fetch: async () => new Response(reply) });
      await expect(client.getAction({ prompt: "x" }), token).rejects.toThrow(/integer a JavaScript number cannot hold exactly/);
    }
    const exact = new PolicyClient({ checkpoint: "c", baseUrl: "https://gateway.test", processing: "raw", fetch: async () => new Response('{"status":{"code":"DA-0000"},"session_id":"s","actions":{"head":[[9007199254740992,1000000000000000128]]}}') });
    expect(await exact.getAction({ prompt: "x" })).toEqual({ head: [[9007199254740992, 1000000000000000128]] });
    // A declared [1,1] joint is checked whatever strict says: 1e400 is refused, raw included.
    const infinite = new PolicyClient({ checkpoint: "c", baseUrl: "https://gateway.test", processing: "raw", strict: false, embodiment: { action: { head: { shape: [1, 1] } } }, fetch: async () => new Response('{"status":{"code":"DA-0000"},"session_id":"s","actions":{"head":[[1e400]]}}') });
    await expect(infinite.getAction({ prompt: "x" })).rejects.toMatchObject({ name: "DreamActionError" });
  });
});

describe("sixth review, client side", () => {
  const ok = (actions: unknown) => async () => Response.json({ status: { code: "DA-0000" }, session_id: "s", actions });
  const six = [0, 0, 0, 0, 0, 0];

  it("never prints an inexact integer's token: a key of digits echoed as a number stays out of the message", async () => {
    const digits = "12345678901234567890123";
    const client = new PolicyClient({ checkpoint: "c", baseUrl: "https://gateway.test", apiKey: digits, processing: "raw", fetch: async () => new Response(`{"status":{"code":"DA-0000"},"session_id":"s","actions":{"extra":${digits}}}`) });
    const error = (await client.getAction({ prompt: "x" }).catch((e: Error) => e)) as Error;
    expect(error.name).toBe("DreamActionError");
    expect(error.message).not.toContain(digits);
  });

  it("sends the key for a relative URL only to the page's own origin, whatever its <base> says", async () => {
    const urls: string[] = [];
    const fetch = async (url: string) => (urls.push(url), Response.json({ status: { code: "DA-0001", reason: "new" }, session_id: "s", actions: {} }));
    const scope = globalThis as { document?: unknown; location?: unknown };
    scope.document = { baseURI: "https://elsewhere.example/" };
    scope.location = { href: "https://app.example/page" };
    try {
      const client = new PolicyClient({ checkpoint: "c", baseUrl: "/api/dream", apiKey: KEY, fetch });
      await expect(client.getAction({ prompt: "x" })).rejects.toThrow(/another origin than the page's/);
      await expect(callService("GET", "/v1/models", { url: "/api", apiKey: KEY, fetch })).rejects.toThrow(/another origin than the page's/);
      await expect(callService("GET", "/v1/models", { url: "//elsewhere.example/api", apiKey: KEY, fetch })).rejects.toThrow(/another origin than the page's/);
      expect(urls).toEqual([]);
      // Without a key, and with a whole URL, nothing changes.
      await callService("GET", "/v1/models", { url: "/api", fetch });
      await new PolicyClient({ checkpoint: "c", baseUrl: "https://elsewhere.example/api/dream", apiKey: KEY, fetch }).getAction({ prompt: "x" });
      expect(urls).toEqual(["https://elsewhere.example/api/v1/models", "https://elsewhere.example/api/dream/v1/get_action"]);
    } finally {
      delete scope.document;
      delete scope.location;
    }
  });

  it("refuses a chunk whose rows are narrower than the arm's joints and grip, on either arm, raw or joined", async () => {
    for (const processing of ["raw", "default"] as const) {
      for (const [key, pose] of [["q_r_chunk", "state.arm_r.joint_pos"], ["q_l_chunk", "state.arm_l.joint_pos"]] as const) {
        const embodiment = { state: { [pose.slice("state.".length)]: { shape: [6] } } };
        for (const rows of [[[0.1, 0.2]], [[0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]]) {
          const client = new PolicyClient({ checkpoint: "c", baseUrl: "https://gateway.test", processing, embodiment, fetch: ok({ [key]: rows }) });
          const error = (await client.getAction({ [pose]: six }).catch((e: unknown) => e)) as Error & { detail?: unknown };
          expect(error.name, `${processing} ${key} ${JSON.stringify(rows)}`).toBe("DreamActionError");
          expect(error.detail).toMatchObject({ kind: "shape", key });
        }
        // The arm's joints and its grip pass unchanged (two rows: too few for the filter to touch).
        const rows = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]];
        const client = new PolicyClient({ checkpoint: "c", baseUrl: "https://gateway.test", processing, embodiment, fetch: ok({ [key]: rows }) });
        expect(await client.getAction({ [pose]: six }), `${processing} ${key}`).toEqual({ [key]: rows });
      }
    }
    // Joined onto a previous plan: a narrow second chunk is refused too, and the plan kept is the first one.
    let reply: unknown = { q_r_chunk: Array.from({ length: 12 }, () => [0, 0, 0, 0, 0, 0, 0.5]) };
    const client = new PolicyClient({ checkpoint: "c", baseUrl: "https://gateway.test", fetch: async () => Response.json({ status: { code: "DA-0000" }, session_id: "s", actions: reply }) });
    await client.getAction({ "state.arm_r.joint_pos": six });
    reply = { q_r_chunk: Array.from({ length: 12 }, () => [0.1, 0.2]) };
    await expect(client.getAction({ "state.arm_r.joint_pos": six })).rejects.toMatchObject({ name: "DreamActionError", detail: { kind: "shape", key: "q_r_chunk", row: 0 } });
    // Without a measured pose the arm's width is not known: left as it was.
    reply = { q_r_chunk: [[0.1, 0.2]] };
    expect(await client.getAction({ prompt: "x" })).toEqual({ q_r_chunk: [[0.1, 0.2]] });
  });

  it("returns nothing when an onRoundTrip() callback closes the client or fires the caller's signal", async () => {
    const replies = () => Response.json({ status: { code: "DA-0001", reason: "new" }, session_id: "s", actions: { q_r_chunk: null } });
    const client = new DreamClient({ apiKey: KEY, baseUrl: "https://gateway.test", fetch: recording(replies).fetch });
    let act: () => void = () => undefined;
    client.onRoundTrip(() => act());
    await client.policy({ checkpoint: "dreamlabs/dream-1-beta", strict: false });
    client.setObservation({ "video.top": "Zg==" }, {}, { timestamp: 1 });
    const controller = new AbortController();
    act = () => controller.abort();
    await expect(client.getAction({ signal: controller.signal })).rejects.toMatchObject({ name: "DreamCancelled" });
    act = () => void client.close();
    await expect(client.getAction()).rejects.toMatchObject({ name: "DreamCancelled" });
    await client.close();
  });

  it("stops reading a reply at the byte limit, as it streams, gateway and services alike", async () => {
    // The documented default, 16 MiB.
    const MAX_REPLY_BYTES = 16 * 1024 * 1024;
    let pulled = 0;
    let calls = 0;
    // A fast reply with no Content-Length, three times the limit.
    const huge = () => {
      calls++;
      pulled = 0;
      return new Response(
        new ReadableStream<Uint8Array>({
          pull(controller) {
            if (pulled >= 3 * MAX_REPLY_BYTES) return controller.close();
            pulled += 1 << 20;
            controller.enqueue(new Uint8Array(1 << 20).fill(0x20));
          },
        }),
      );
    };
    const client = new PolicyClient({ checkpoint: "c", baseUrl: "https://gateway.test", apiKey: KEY, fetch: async () => huge() });
    await expect(client.getAction({ prompt: "x" })).rejects.toThrow(/larger than 16777216 bytes/);
    expect(pulled).toBeLessThanOrEqual(MAX_REPLY_BYTES + 3 * (1 << 20));
    calls = 0;
    await expect(callService("GET", "/v1/models", { url: "https://platform.test", apiKey: KEY, fetch: async () => huge() }, undefined, true)).rejects.toThrow(/more than 16777216 bytes/);
    expect(pulled).toBeLessThanOrEqual(MAX_REPLY_BYTES + 3 * (1 << 20));
    // Not retried: the same service would answer the same.
    expect(calls).toBe(1);
    expect((await import("../src/transport.js")).MAX_REPLY_BYTES).toBe(MAX_REPLY_BYTES);
  });
});
