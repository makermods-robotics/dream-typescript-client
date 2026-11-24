/**
 * createDreamProxy() against a mock Dream Labs: what it forwards and how, what it refuses, the CORS answers, the body
 * limit, that the key never comes back, and a DreamClient pointed at it with `proxyUrl`; nodeHandler() on node:http.
 */

import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import { fixture } from "./helpers.js";

const wire = fixture("wire.json");
import { CLIENT_VERSION, DreamClient, DreamConfigError, PolicyClient } from "../src/index.js";
import { nodeHandler } from "../src/node/index.js";
import { ANSWER_LIMITS, answerFor, createDreamProxy, readAll, type DreamProxyOptions, type ScanWork } from "../src/proxy.js";

const KEY = "dk_secret_proxy_key";
const PAGE = "https://app.example.com";

interface Seen {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: string;
}

/** A mock of the gateway, the platform and the data service, recording what reached it. */
function upstream(answer: (seen: Seen) => Response | Promise<Response> = () => Response.json({ status: { code: "DA-0000" } })) {
  const seen: Seen[] = [];
  const fetch = async (url: string, init: RequestInit): Promise<Response> => {
    const body = init.body === undefined ? "" : new TextDecoder().decode(init.body as Uint8Array);
    const entry: Seen = { url, method: String(init.method), headers: Object.fromEntries(Object.entries(init.headers as Record<string, string>).map(([k, v]) => [k.toLowerCase(), v])), body };
    // Every outbound call refuses to follow a redirect (the key never goes to another host), and carries a signal.
    expect(init.redirect).toBe("manual");
    expect(init.signal).toBeInstanceOf(AbortSignal);
    seen.push(entry);
    return answer(entry);
  };
  return { seen, fetch };
}

function proxy(options: Partial<DreamProxyOptions> = {}, mock = upstream()) {
  const handler = createDreamProxy({
    apiKey: KEY,
    allowedHosts: "any",
    allowedOrigins: [PAGE],
    gatewayUrl: "https://gateway.test",
    platformUrl: "https://platform.test",
    dataUrl: "https://data.test",
    basePath: "/api/dream",
    fetch: mock.fetch,
    ...options,
  });
  return { handler, ...mock };
}

/** A GET as the client sends it through a proxy: with Dream-Client-Version. */
const get = (url: string, headers: Record<string, string> = {}) => new Request(url, { headers: { "Dream-Client-Version": CLIENT_VERSION, ...headers } });

const post = (path: string, body: string, headers: Record<string, string> = {}) =>
  new Request(`https://app.example.com/api/dream${path}`, { method: "POST", body, headers: { "Content-Type": "application/json", ...headers } });

async function text(response: Response): Promise<string> {
  return response.text();
}

describe("createDreamProxy", () => {
  it("refuses malformed options, and needs a host policy", () => {
    const any = { allowedHosts: "any" } as const;
    expect(() => createDreamProxy({ apiKey: "", ...any })).toThrow(DreamConfigError);
    expect(() => createDreamProxy({ apiKey: KEY } as DreamProxyOptions)).toThrow(/needs allowedHosts/);
    expect(() => createDreamProxy({ apiKey: KEY, ...any, allowedOrigins: ["https://app.example.com/"] })).toThrow(/origins as a browser sends them/);
    expect(() => createDreamProxy({ apiKey: KEY, ...any, allowedOrigins: ["*"] })).toThrow(/origins as a browser sends them/);
    expect(() => createDreamProxy({ apiKey: KEY, ...any, gatewayUrl: "https://gateway.test/v2" })).toThrow(/with no path/);
    expect(() => createDreamProxy({ apiKey: KEY, ...any, maxBodyBytes: 0 })).toThrow(/maxBodyBytes/);
    expect(() => createDreamProxy({ apiKey: "dk_\u201ckey\u201d", ...any })).toThrow(DreamConfigError);
    expect(() => createDreamProxy({ apiKey: KEY, ...any, basePath: "api" })).toThrow(/basePath/);
    expect(() => createDreamProxy({ apiKey: KEY, ...any, clientVersion: "1.1 7" })).toThrow(/clientVersion/);
  });

  it("refuses upstream URLs whose path a query, a fragment or a user name could move", async () => {
    for (const gatewayUrl of ["https://gateway.test/?", "https://gateway.test/#", "https://gateway.test?x=1", "https://user:pw@gateway.test", "ftp://gateway.test"]) {
      expect(() => createDreamProxy({ apiKey: KEY, allowedHosts: "any", gatewayUrl }), gatewayUrl).toThrow(DreamConfigError);
    }
    // What is accepted is reduced to its origin, so /v1/... always lands in the path.
    const { handler, seen } = proxy({ gatewayUrl: "HTTPS://Gateway.test:443/" });
    await handler(post("/v1/get_action", "{}"));
    expect(seen[0]!.url).toBe("https://gateway.test/v1/get_action");
  });

  it("forwards get_action to the gateway with the key, this package's identity and the User-Agent built from the instance", async () => {
    const { handler, seen } = proxy();
    const body = JSON.stringify({ checkpoint: "dreamlabs/dream-1-beta", session_id: null, reset: false, observation: { prompt: "wave" } });
    const reply = await handler(
      post("/v1/get_action", body, {
        Origin: PAGE,
        "Dream-Client-Version": CLIENT_VERSION,
        "Dream-Client-Instance": "0a1b2c3d",
        Authorization: "Bearer someone-elses",
        Cookie: "session=1",
        "X-Other": "dropped",
      }),
    );
    expect(reply.status).toBe(200);
    expect(await reply.json()).toEqual({ status: { code: "DA-0000" } });
    expect(reply.headers.get("access-control-allow-origin")).toBe(PAGE);
    expect(reply.headers.get("vary")).toBe("Origin");
    expect(seen).toHaveLength(1);
    expect(seen[0]!.url).toBe("https://gateway.test/v1/get_action");
    expect(seen[0]!.method).toBe("POST");
    expect(seen[0]!.body).toBe(body);
    expect(seen[0]!.headers).toEqual({
      authorization: `Bearer ${KEY}`,
      accept: "application/json",
      "content-type": "application/json",
      "dream-client-version": CLIENT_VERSION,
      "user-agent": `dream-client/${CLIENT_VERSION} 0a1b2c3d`,
    });
  });

  it("never lets the caller choose the version sent upstream: a semver from a page would unlock parts checkpoints", async () => {
    const { handler, seen } = proxy();
    await handler(post("/v1/end_session", "{}", { "User-Agent": "dream-client/1.1.7 00c0ffee", "Dream-Client-Version": "1.1.7" }));
    expect(seen[0]!.url).toBe("https://gateway.test/v1/end_session");
    expect(seen[0]!.headers["dream-client-version"]).toBe(CLIENT_VERSION);
    // A non-browser client's instance is kept; its version is not.
    expect(seen[0]!.headers["user-agent"]).toBe(`dream-client/${CLIENT_VERSION} 00c0ffee`);
    await handler(post("/v1/get_action", "{}", { "Dream-Client-Version": "bad version", "Dream-Client-Instance": "XYZ", "User-Agent": "Mozilla/5.0" }));
    expect(seen[1]!.headers["dream-client-version"]).toBe(CLIENT_VERSION);
    expect(seen[1]!.headers["user-agent"]).toMatch(new RegExp(`^dream-client/${CLIENT_VERSION.replace(/[.+]/g, "\\$&")} [0-9a-f]{8}$`));
    await handler(new Request("https://app.example.com/api/dream/v1/models", { headers: { "Dream-Client-Version": "1.1.7" } }));
    expect(seen[2]!.headers["user-agent"]).toBe(`dream-client/${CLIENT_VERSION}`);
    // Only the server's own configuration may name another version.
    const configured = proxy({ clientVersion: "1.1.7" });
    await configured.handler(post("/v1/get_action", "{}", { "Dream-Client-Version": CLIENT_VERSION, "Dream-Client-Instance": "0a1b2c3d" }));
    expect(configured.seen[0]!.headers["dream-client-version"]).toBe("1.1.7");
    expect(configured.seen[0]!.headers["user-agent"]).toBe("dream-client/1.1.7 0a1b2c3d");
  });

  it("forwards the models list to the platform and a fine-tune's training data to the data service, as GETs", async () => {
    const { handler, seen } = proxy({}, upstream(() => Response.json([{ id: "m" }], { status: 200 })));
    const models = await handler(get("https://app.example.com/api/dream/v1/models"));
    expect(await models.json()).toEqual([{ id: "m" }]);
    expect(seen[0]).toEqual({
      url: "https://platform.test/v1/models",
      method: "GET",
      body: "",
      headers: { authorization: `Bearer ${KEY}`, accept: "application/json", "user-agent": `dream-client/${CLIENT_VERSION}` },
    });
    await handler(get("https://app.example.com/api/dream/v1/fine_tuning/models/dream-1-ft%3A0a1b2c3d-towels/training_data"));
    expect(seen[1]!.url).toBe("https://data.test/v1/fine_tuning/models/dream-1-ft%3A0a1b2c3d-towels/training_data");
    expect(seen[1]!.headers["user-agent"]).toBe(`dream-client/${CLIENT_VERSION}`);
  });

  it("refuses every other route and method, and a training-data path that is not a fine-tune's", async () => {
    const { handler, seen } = proxy();
    for (const path of ["/v1/start_session", "/v1/models/x", "/v2/get_action", "/v1/fine_tuning/models/dreamlabs%2Fdream-1-beta/training_data", "/v1/fine_tuning/models/dream-1-ft%3Aa%2F..%2Fb/training_data", "/v1/fine_tuning/jobs"]) {
      expect((await handler(get(`https://app.example.com/api/dream${path}`))).status, path).toBe(404);
    }
    expect((await handler(new Request("https://app.example.com/other/v1/models"))).status).toBe(404);
    const wrong = await handler(new Request("https://app.example.com/api/dream/v1/get_action"));
    expect(wrong.status).toBe(405);
    expect(wrong.headers.get("allow")).toBe("POST, OPTIONS");
    expect((await handler(post("/v1/models", "{}"))).status).toBe(405);
    expect(seen).toHaveLength(0);
  });

  it("answers the CORS preflight for an allowed origin only, and refuses a foreign origin's requests", async () => {
    const { handler, seen } = proxy({ allowedHeaders: ["X-App-Auth"] });
    const preflight = await handler(new Request("https://app.example.com/api/dream/v1/get_action", { method: "OPTIONS", headers: { Origin: PAGE, "Access-Control-Request-Method": "POST" } }));
    expect(preflight.status).toBe(204);
    expect(preflight.headers.get("access-control-allow-origin")).toBe(PAGE);
    expect(preflight.headers.get("access-control-allow-methods")).toBe("POST, OPTIONS");
    expect(preflight.headers.get("access-control-allow-headers")).toBe("Content-Type, Accept, Dream-Client-Version, Dream-Client-Instance, X-App-Auth");
    const foreign = await handler(new Request("https://app.example.com/api/dream/v1/get_action", { method: "OPTIONS", headers: { Origin: "https://evil.example" } }));
    expect(foreign.status).toBe(403);
    expect(foreign.headers.get("access-control-allow-origin")).toBeNull();
    expect((await handler(post("/v1/get_action", "{}", { Origin: "https://evil.example" }))).status).toBe(403);
    expect((await handler(post("/v1/get_action", "{}", { Origin: "null" }))).status).toBe(403);
    expect(seen).toHaveLength(0);
    // No origins allowed (the default): any request from a page is refused, and no wildcard is ever answered.
    const closed = proxy({ allowedOrigins: [] });
    expect((await closed.handler(post("/v1/get_action", "{}", { Origin: PAGE }))).status).toBe(403);
    const plain = await closed.handler(post("/v1/get_action", "{}"));
    expect(plain.status).toBe(200);
    expect(plain.headers.get("access-control-allow-origin")).toBeNull();
  });

  it("refuses a host it does not answer to (DNS rebinding)", async () => {
    const { handler, seen } = proxy({ allowedHosts: ["app.example.com"] }, upstream(() => Response.json([])));
    expect((await handler(get("https://rebound.example/api/dream/v1/models"))).status).toBe(421);
    expect((await handler(get("https://APP.example.com/api/dream/v1/models"))).status).toBe(200);
    expect(seen).toHaveLength(1);
  });

  it("refuses a body over the limit, declared or streamed", async () => {
    const { handler, seen } = proxy({ maxBodyBytes: 1000 });
    expect((await handler(post("/v1/get_action", "x".repeat(1001)))).status).toBe(413);
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (let i = 0; i < 5; i++) controller.enqueue(new Uint8Array(400));
        controller.close();
      },
    });
    const chunked = new Request("https://app.example.com/api/dream/v1/get_action", { method: "POST", body: stream, duplex: "half" } as RequestInit);
    expect((await handler(chunked)).status).toBe(413);
    expect((await handler(post("/v1/get_action", "x".repeat(1000)))).status).toBe(200);
    expect(seen).toHaveLength(1);
  });

  it("gives an allowed page's sign-in refusal its CORS headers, and the hook no body", async () => {
    let bodySeen: string | null = null;
    const { handler, seen } = proxy({
      authorize: async (request) => {
        bodySeen = await request.text();
        return request.headers.get("x-app-auth") === "ok" ? true : new Response('{"error":"sign in"}', { status: 401, headers: { "Content-Type": "application/json" } });
      },
    });
    const refused = await handler(post("/v1/get_action", "{}", { Origin: PAGE }));
    expect(refused.status).toBe(401);
    expect(refused.headers.get("access-control-allow-origin")).toBe(PAGE);
    expect(await refused.json()).toEqual({ error: "sign in" });
    const body = JSON.stringify({ observation: { prompt: "wave" } });
    expect((await handler(post("/v1/get_action", body, { Origin: PAGE, "X-App-Auth": "ok" }))).status).toBe(200);
    expect(bodySeen).toBe("");
    expect(seen[0]!.body).toBe(body);
  });

  it("runs your sign-in before anything is forwarded", async () => {
    const { handler, seen } = proxy({ authorize: (request) => (request.headers.get("x-app-auth") === "ok" ? true : request.headers.has("x-app-auth") ? new Response("go away", { status: 418 }) : false) });
    expect((await handler(post("/v1/get_action", "{}"))).status).toBe(401);
    expect((await handler(post("/v1/get_action", "{}", { "X-App-Auth": "no" }))).status).toBe(418);
    expect((await handler(post("/v1/get_action", "{}", { "X-App-Auth": "ok" }))).status).toBe(200);
    expect(seen).toHaveLength(1);
  });

  it("withholds an answer that carries the key", async () => {
    for (const echo of [`{"echo":"Bearer ${KEY}"}`, `<html>${KEY}</html>`]) {
      const { handler } = proxy({}, upstream(() => new Response(echo, { status: 400 })));
      const reply = await handler(post("/v1/get_action", "{}", { Origin: PAGE }));
      expect(reply.status).toBe(502);
      const text = await reply.text();
      expect(text).not.toContain(KEY);
      expect(JSON.parse(text)).toEqual({ proxy_error: "upstream answer withheld" });
    }
  });

  it("answers an upstream 204 or 304 with no body, and refuses an answer over the limit", async () => {
    for (const status of [204, 205, 304]) {
      const { handler } = proxy({}, upstream(() => new Response(null, { status })));
      const reply = await handler(post("/v1/end_session", "{}"));
      expect(reply.status).toBe(status);
      expect(reply.body).toBeNull();
    }
    const big = proxy({ maxResponseBytes: 100 }, upstream(() => new Response("x".repeat(101))));
    const reply = await big.handler(post("/v1/get_action", "{}"));
    expect(reply.status).toBe(502);
    expect(await reply.json()).toEqual({ proxy_error: "upstream answer too large" });
  });

  it("sends nothing for a caller that cancelled during sign-in, and stops the upstream call when the caller goes", async () => {
    const page = new AbortController();
    let release!: (value: boolean) => void;
    const signedIn = new Promise<boolean>((resolve) => (release = resolve));
    const slow = proxy({ authorize: () => signedIn });
    const pending = slow.handler(new Request("https://app.example.com/api/dream/v1/get_action", { method: "POST", body: "{}", signal: page.signal }));
    page.abort();
    release(true);
    expect((await pending).status).toBe(499);
    expect(slow.seen).toHaveLength(0);

    const caller = new AbortController();
    let upstreamSignal: AbortSignal | undefined;
    const hanging = createDreamProxy({
      apiKey: KEY,
      allowedHosts: "any",
      fetch: (_url, init) => {
        upstreamSignal = init.signal!;
        return new Promise<Response>((_, reject) => init.signal!.addEventListener("abort", () => reject(init.signal!.reason)));
      },
    });
    const call = hanging(new Request("http://127.0.0.1/v1/get_action", { method: "POST", body: "{}", signal: caller.signal }));
    for (let i = 0; i < 100 && !upstreamSignal; i++) await new Promise((resolve) => setTimeout(resolve, 5));
    expect(upstreamSignal?.aborted).toBe(false);
    caller.abort();
    expect((await call).status).toBe(499);
    expect(upstreamSignal!.aborted).toBe(true);
  });

  it("gives up on an upload that stalls, and maps a body that fails to read", async () => {
    const stalled = new ReadableStream<Uint8Array>({ start: (controller) => controller.enqueue(new Uint8Array(10)) });
    const { handler, seen } = proxy({ uploadTimeoutMs: 20 });
    const slow = await handler(new Request("https://app.example.com/api/dream/v1/get_action", { method: "POST", body: stalled, duplex: "half" } as RequestInit));
    expect(slow.status).toBe(408);
    const broken = new ReadableStream<Uint8Array>({ pull: (controller) => controller.error(new Error("connection reset")) });
    const failed = await handler(new Request("https://app.example.com/api/dream/v1/get_action", { method: "POST", body: broken, duplex: "half" } as RequestInit));
    expect(failed.status).toBe(400);
    expect(await failed.json()).toEqual({ proxy_error: "unreadable body" });
    expect(seen).toHaveLength(0);
  });

  it("passes the upstream status and body through, and never puts the key in an answer", async () => {
    const refusal = { status: { code: "DA-1000", reason: "unknown_key" } };
    const { handler } = proxy({}, upstream(() => Response.json(refusal, { status: 401, headers: { "Set-Cookie": "x=1", Location: "https://elsewhere" } })));
    const reply = await handler(post("/v1/get_action", "{}"));
    expect(reply.status).toBe(401);
    expect(await reply.json()).toEqual(refusal);
    expect(reply.headers.get("set-cookie")).toBeNull();
    expect(reply.headers.get("location")).toBeNull();
    const failing = proxy({}, upstream(() => Promise.reject(new TypeError(`fetch failed for Bearer ${KEY}`))));
    const unreachable = await failing.handler(post("/v1/get_action", "{}"));
    expect(unreachable.status).toBe(502);
    expect(await text(unreachable)).not.toContain(KEY);
  });

  it("answers 504 when Dream Labs does not answer in time", async () => {
    const hanging = (_url: string, init: RequestInit) =>
      new Promise<Response>((_, reject) => init.signal!.addEventListener("abort", () => reject(init.signal!.reason)));
    const handler = createDreamProxy({ apiKey: KEY, allowedHosts: "any", fetch: hanging, timeoutMs: 20 });
    const reply = await handler(new Request("http://127.0.0.1/v1/get_action", { method: "POST", body: "{}" }));
    expect(reply.status).toBe(504);
    expect(await reply.json()).toEqual({ proxy_error: "timeout" });
  });

  it("serves a DreamClient pointed at it with proxyUrl: no key in the client, the identity built upstream", async () => {
    const reply = { status: { code: "DA-0001", reason: "new" }, session_id: "ses_1", actions: { q_r_chunk: [[0.1, 0.2]] }, timing: { serve_infer_ms: 100, gateway_ms: 120 } };
    const model = { id: "dream-1-ft:0a1b2c3d-towels", owner: "mine", base_model: "dreamlabs/dream-1-beta", embodiment: "emb_1", servable: true, created_at: null, cameras: ["front"] };
    const mock = upstream((seen) => {
      if (seen.url.endsWith("/v1/models")) return Response.json([model]);
      if (seen.url.endsWith("/training_data")) return Response.json({ cameras: { front: [[640, 360]] }, prompts: ["wave hello"] });
      if (seen.url.endsWith("/v1/end_session")) return Response.json({ status: { code: "DA-0000" } });
      return Response.json(reply);
    });
    const { handler, seen } = proxy({ allowedOrigins: [] }, mock);
    const warnings: string[] = [];
    const client = new DreamClient({
      proxyUrl: "https://app.example.com/api/dream",
      fetch: (input, init) => handler(new Request(input, init)),
      onWarning: (warning) => void warnings.push(warning.message),
    });
    expect(client.baseUrl).toBe("https://app.example.com/api/dream");
    expect(client.platformUrl).toBe(client.baseUrl);
    expect(client.dataUrl).toBe(client.baseUrl);
    await client.policy({ checkpoint: model.id, strict: false });
    client.setObservation({ "video.front": "Zg==" }, { "state.right_arm": [0, 0, 0, 0, 0, 0], prompt: "wave" }, { timestamp: 1 });
    expect(await client.getAction()).toEqual({ q_r_chunk: [[0.1, 0.2]] });
    await client.close();
    expect(seen.map((s) => s.url)).toEqual([
      "https://data.test/v1/fine_tuning/models/dream-1-ft%3A0a1b2c3d-towels/training_data",
      "https://gateway.test/v1/get_action",
      "https://gateway.test/v1/end_session",
    ]);
    for (const s of seen) expect(s.headers.authorization).toBe(`Bearer ${KEY}`);
    expect(seen[0]!.headers["user-agent"]).toBe(`dream-client/${CLIENT_VERSION}`);
    expect(seen[1]!.headers["dream-client-version"]).toBe(CLIENT_VERSION);
    expect(seen[1]!.headers["user-agent"]).toMatch(new RegExp(`^dream-client/${CLIENT_VERSION.replace(/[.+]/g, "\\$&")} [0-9a-f]{8}$`));
    expect(warnings).toEqual([
      "the prompt 'wave' is none of the tasks dream-1-ft:0a1b2c3d-towels was trained on (1 tasks, such as 'wave hello'): a fine-tune follows the words of its training tasks best",
    ]);
    expect(() => new DreamClient({ proxyUrl: "/api/dream", apiKey: KEY })).toThrow(/leave apiKey out/);
  });
});

describe("nodeHandler", () => {
  let server: Server | null = null;
  afterEach(() => new Promise<void>((resolve) => (server ? server.close(() => resolve()) : resolve())));

  async function serve(options: Partial<DreamProxyOptions> = {}) {
    const mock = upstream();
    const handler = createDreamProxy({ apiKey: KEY, allowedHosts: "any", gatewayUrl: "https://gateway.test", fetch: mock.fetch, maxBodyBytes: 64 * 1024, ...options });
    server = createServer(nodeHandler(handler));
    await new Promise<void>((resolve) => server!.listen(0, "127.0.0.1", resolve));
    return { url: `http://127.0.0.1:${(server!.address() as AddressInfo).port}`, seen: mock.seen };
  }

  it("serves the proxy from node:http, body and headers included", async () => {
    const { url, seen } = await serve();
    const body = JSON.stringify({ observation: { prompt: "wave" } });
    const reply = await fetch(`${url}/v1/get_action`, { method: "POST", body, headers: { "Content-Type": "application/json", "Dream-Client-Version": "1.1.7", "Dream-Client-Instance": "0a1b2c3d" } });
    expect(reply.status).toBe(200);
    expect(reply.headers.get("content-type")).toBe("application/json");
    expect(await reply.json()).toEqual({ status: { code: "DA-0000" } });
    expect(seen[0]!.body).toBe(body);
    expect(seen[0]!.headers["user-agent"]).toBe(`dream-client/${CLIENT_VERSION} 0a1b2c3d`);
  });

  it("aborts the upstream call when the client disconnects", async () => {
    let upstreamSignal: AbortSignal | undefined;
    const handler = createDreamProxy({
      apiKey: KEY,
      allowedHosts: "any",
      fetch: (_url, init) => {
        upstreamSignal = init.signal!;
        return new Promise<Response>((_, reject) => init.signal!.addEventListener("abort", () => reject(init.signal!.reason)));
      },
    });
    server = createServer(nodeHandler(handler));
    await new Promise<void>((resolve) => server!.listen(0, "127.0.0.1", resolve));
    const url = `http://127.0.0.1:${(server!.address() as AddressInfo).port}`;
    const client = new AbortController();
    const call = fetch(`${url}/v1/get_action`, { method: "POST", body: "{}", signal: client.signal }).catch((error: Error) => error);
    for (let i = 0; i < 100 && !upstreamSignal; i++) await new Promise((resolve) => setTimeout(resolve, 10));
    expect(upstreamSignal?.aborted).toBe(false);
    client.abort();
    await call;
    for (let i = 0; i < 100 && !upstreamSignal!.aborted; i++) await new Promise((resolve) => setTimeout(resolve, 10));
    expect(upstreamSignal!.aborted).toBe(true);
  });

  it("answers 500, never an unhandled rejection, when the handler's answer fails to read", async () => {
    server = createServer(nodeHandler(async () => new Response(new ReadableStream({ pull: (controller) => controller.error(new Error("gone")) }))));
    await new Promise<void>((resolve) => server!.listen(0, "127.0.0.1", resolve));
    const reply = await fetch(`http://127.0.0.1:${(server!.address() as AddressInfo).port}/v1/models`);
    expect(reply.status).toBe(500);
  });

  it("answers 413 to a body over the limit, declared or streamed, and the client reads it", async () => {
    const { url, seen } = await serve();
    // Declared: refused before a byte is read. The rest is thrown away after the answer, so the client's upload
    // completes and it reads the 413 on every platform (closing at once made its write fail on macOS and Windows).
    for (let i = 0; i < 5; i++) {
      const reply = await fetch(`${url}/v1/get_action`, { method: "POST", body: new Uint8Array(1024 * 1024) });
      expect(reply.status).toBe(413);
      expect(await reply.json()).toEqual({ proxy_error: "too large" });
    }
    // Streamed with no length: the proxy stops reading at the limit.
    const chunks = Array.from({ length: 32 }, () => new Uint8Array(32 * 1024));
    const stream = new ReadableStream<Uint8Array>({ pull: (controller) => (chunks.length ? controller.enqueue(chunks.pop()!) : controller.close()) });
    const streamed = await fetch(`${url}/v1/get_action`, { method: "POST", body: stream, duplex: "half" } as RequestInit);
    expect(streamed.status).toBe(413);
    expect(seen).toHaveLength(0);
  });

  it("never buffers an unread body: past drainBytes the connection is closed", async () => {
    const handler = createDreamProxy({ apiKey: KEY, allowedHosts: "any", fetch: upstream().fetch, maxBodyBytes: 1024 });
    const listener = nodeHandler(handler, { drainBytes: 256 * 1024 });
    let received = -1;
    server = createServer((request, response) => {
      request.socket.once("close", () => (received = request.socket.bytesRead));
      void listener(request, response);
    });
    await new Promise<void>((resolve) => server!.listen(0, "127.0.0.1", resolve));
    const size = 16 * 1024 * 1024;
    // The client sees the 413 or, once the server closes mid-upload, a reset: either way it is not read in full.
    const outcome = await fetch(`http://127.0.0.1:${(server!.address() as AddressInfo).port}/v1/get_action`, { method: "POST", body: new Uint8Array(size) })
      .then((reply) => reply.status)
      .catch(() => "reset");
    expect([413, "reset"]).toContain(outcome);
    for (let i = 0; i < 200 && received < 0; i++) await new Promise((resolve) => setTimeout(resolve, 10));
    expect(received).toBeGreaterThan(0);
    expect(received).toBeLessThan(size / 2);
  });
});

describe("createDreamProxy, hardened", () => {
  const SECRET = "dk_SYNTHETIC_secret_7f3a9c";
  const b64 = (text: string) => btoa(text);
  const b64url = (text: string) => btoa(text).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  const hex = (text: string) => Array.from(text, (c) => c.charCodeAt(0).toString(16).padStart(2, "0")).join("");
  const escaped = (text: string) => Array.from(text, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`).join("");
  const answering = (body: string, extra: Partial<DreamProxyOptions> = {}) => {
    const mock = upstream(() => new Response(body, { status: 200 }));
    return { ...mock, handler: createDreamProxy({ apiKey: SECRET, allowedHosts: "any", fetch: mock.fetch, ...extra }) };
  };

  it("withholds an answer that carries the key in any encoding it checks", async () => {
    const carriers = [
      `{"status":{"code":"DA-0000","detail":"${SECRET}"}}`,
      `{"status":{"code":"DA-0000"},"actions":{"note":"${b64(SECRET)}"}}`,
      `{"status":{"code":"DA-0000"},"actions":{"note":"${b64(`Authorization: Bearer ${SECRET}`)}"}}`,
      `{"status":{"code":"DA-0000"},"actions":{"note":"xy${b64(`Bearer ${SECRET}`)}"}}`,
      `{"status":{"code":"DA-0000"},"actions":{"note":"${b64url(SECRET)}"}}`,
      `{"status":{"code":"DA-0000"},"actions":{"note":"${escaped(SECRET)}"}}`,
      `{"status":{"code":"DA-0000"},"actions":{"note":"{\\"k\\":\\"${escaped(SECRET).replace(/\\/g, "\\\\")}\\"}"}}`,
      `{"status":{"code":"DA-0000"},"actions":{"note":"${encodeURIComponent(SECRET).replace(/_/g, "%5F")}"}}`,
      `{"status":{"code":"DA-0000"},"actions":{"note":"${hex(SECRET)}"}}`,
      `{"status":{"code":"DA-0000"},"actions":{"${SECRET}":1}}`,
      `{"status":{"code":"DA-0000"},"actions":{"note":"${b64(b64(SECRET))}"}}`,
    ];
    for (const body of carriers) {
      const { handler } = answering(body);
      const reply = await handler(new Request("http://proxy.test/v1/get_action", { method: "POST", body: "{}" }));
      const text = await reply.text();
      expect(reply.status, body).toBe(502);
      expect(JSON.parse(text), body).toEqual({ proxy_error: "upstream answer withheld" });
      expect(text).not.toContain(SECRET);
    }
  });

  it("returns only answers of the route's shape, written anew", async () => {
    const gateway = answering(`{"status":{"code":"DA-0000","name":"ok","\\u0072eason":null,"extra":1},"session_id":"s","actions":{"q_r_chunk":[[0.5]]},"timing":{"gateway_ms":3},"leak":"x"}`);
    const reply = await gateway.handler(new Request("http://proxy.test/v1/get_action", { method: "POST", body: "{}" }));
    expect(reply.status).toBe(200);
    expect(await reply.text()).toBe('{"status":{"code":"DA-0000","name":"ok","reason":null},"session_id":"s","actions":{"q_r_chunk":[[0.5]]},"timing":{"gateway_ms":3}}');
    for (const [body, path] of [
      ["<html>bad gateway</html>", "/v1/get_action"],
      ['{"actions":{}}', "/v1/get_action"],
      ['{"status":{"code":1}}', "/v1/end_session"],
      ['{"status":{"code":"DA-0000"},"actions":"unexpected text"}', "/v1/get_action"],
      ['{"status":{"code":"DA-0000"},"actions":{"q_r_chunk":[["0.1"]]}}', "/v1/get_action"],
      ['[{"id":"m","servable":"yes"}]', "/v1/models"],
      ['{"id":"m"}', "/v1/models"],
      ['[{"name":"m"}]', "/v1/models"],
      ['"text"', "/v1/fine_tuning/models/dream-1-ft%3A0a1b2c3d-towels/training_data"],
    ] as const) {
      const { handler } = answering(body);
      const request = path.startsWith("/v1/get_action") || path === "/v1/end_session" ? new Request(`http://proxy.test${path}`, { method: "POST", body: "{}" }) : get(`http://proxy.test${path}`);
      expect((await handler(request)).status, `${path} ${body}`).toBe(502);
    }
    const models = answering('[{"id":"m","owner":"mine","cameras":null,"secret_field":"dropped"}]');
    expect(await (await models.handler(get("http://proxy.test/v1/models"))).json()).toEqual([{ id: "m", owner: "mine", cameras: null }]);
    const refusal = createDreamProxy({ apiKey: SECRET, allowedHosts: "any", fetch: async () => Response.json({ error: { code: "not_found", message: "no", trace: 1 } }, { status: 404 }) });
    const refused = await refusal(get("http://proxy.test/v1/fine_tuning/models/dream-1-ft%3A0a1b2c3d-towels/training_data"));
    expect(refused.status).toBe(404);
    expect(await refused.json()).toEqual({ error: { code: "not_found", message: "no" } });
  });

  it("refuses a browser's cross-site GET, which carries no Origin, and a GET without the client's header", async () => {
    const { handler, seen } = proxy({}, upstream(() => Response.json([])));
    const navigation = new Request("https://app.example.com/api/dream/v1/models", { headers: { "Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate" } });
    expect((await handler(navigation)).status).toBe(403);
    const forged = get("https://app.example.com/api/dream/v1/models", { "Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors" });
    expect((await handler(forged)).status).toBe(403);
    // A page of an allowed origin may still ask cross-site (CORS, with its Origin).
    expect((await handler(get("https://app.example.com/api/dream/v1/models", { "Sec-Fetch-Site": "cross-site", Origin: PAGE }))).status).toBe(200);
    const bare = await handler(new Request("https://app.example.com/api/dream/v1/models"));
    expect(bare.status).toBe(400);
    expect(seen).toHaveLength(1);
    // A native client (no Fetch Metadata) that sends the header is served.
    expect((await handler(get("https://app.example.com/api/dream/v1/models"))).status).toBe(200);
  });

  it("captures every option at construction", async () => {
    const options: DreamProxyOptions = { apiKey: KEY, allowedHosts: ["proxy.test"], authorize: () => false, fetch: upstream().fetch, timeoutMs: 70_000 };
    const handler = createDreamProxy(options);
    options.authorize = undefined;
    (options.allowedHosts as string[]).push("evil.test");
    options.apiKey = "dk_other";
    expect((await handler(new Request("http://proxy.test/v1/get_action", { method: "POST", body: "{}" }))).status).toBe(401);
    expect((await handler(new Request("http://evil.test/v1/get_action", { method: "POST", body: "{}" }))).status).toBe(421);
  });

  it("looks up only a published fine-tune id: anything else is not_found, with no upstream call", async () => {
    const { handler, seen } = proxy();
    for (const id of ["dream-1-ft:", "dream-1-ft:0a1b2c3d-", "dream-1-ft:0a1b2c3d", "dream-1-ft:0a1b2c3d-Towels", "dream-1-ft:0a1b2c-towels", `dream-1-ft:0a1b2c3d-${"x".repeat(41)}`, "dream-1-ft:0a1b2c3d-a b"]) {
      const reply = await handler(get(`https://app.example.com/api/dream/v1/fine_tuning/models/${encodeURIComponent(id)}/training_data`));
      expect(reply.status, id).toBe(404);
      expect(await reply.json()).toEqual({ error: { code: "not_found", message: "not a fine-tune id" } });
    }
    expect(seen).toHaveLength(0);
  });

  it("refuses durations and sizes that are not whole numbers in range", () => {
    for (const bad of [{ timeoutMs: 0.5 }, { uploadTimeoutMs: 1.5 }, { maxBodyBytes: 2 ** 31 }, { maxResponseBytes: -1 }, { timeoutMs: Number.NaN }]) {
      expect(() => createDreamProxy({ apiKey: KEY, allowedHosts: "any", ...bad }), JSON.stringify(bad)).toThrow(/whole number from 1 to/);
    }
  });

  it("never waits on a sign-in, a fetch or a stream that does not settle", async () => {
    const never = new Promise<never>(() => {});
    const page = new AbortController();
    const stuck = createDreamProxy({ apiKey: KEY, allowedHosts: "any", authorize: () => never, fetch: upstream().fetch });
    const pending = stuck(new Request("http://proxy.test/v1/get_action", { method: "POST", body: "{}", signal: page.signal }));
    setTimeout(() => page.abort(), 10);
    expect((await pending).status).toBe(499);
    const deaf = createDreamProxy({ apiKey: KEY, allowedHosts: "any", timeoutMs: 30, fetch: () => never });
    expect((await deaf(new Request("http://proxy.test/v1/get_action", { method: "POST", body: "{}" }))).status).toBe(504);
    const stubborn = (chunks: number) =>
      new ReadableStream<Uint8Array>({
        start(controller) {
          for (let i = 0; i < chunks; i++) controller.enqueue(new Uint8Array(400));
        },
        cancel: () => never,
      });
    const limited = createDreamProxy({ apiKey: KEY, allowedHosts: "any", maxBodyBytes: 1000, uploadTimeoutMs: 30, fetch: upstream().fetch });
    const body = stubborn(5);
    expect((await limited(new Request("http://proxy.test/v1/get_action", { method: "POST", body, duplex: "half" } as RequestInit))).status).toBe(413);
    expect((await limited(new Request("http://proxy.test/v1/get_action", { method: "POST", body: stubborn(1), duplex: "half" } as RequestInit))).status).toBe(408);
    // An answer whose body never ends is cut at the deadline too.
    const trickle = createDreamProxy({ apiKey: KEY, allowedHosts: "any", timeoutMs: 30, fetch: async () => new Response(stubborn(1)) });
    expect((await trickle(new Request("http://proxy.test/v1/get_action", { method: "POST", body: "{}" }))).status).toBe(504);
  });

  it("reads a fragmented body into one block, empty chunks skipped, and releases the reader", async () => {
    const parts: Uint8Array[] = [];
    for (let i = 0; i < 20_000; i++) parts.push(new Uint8Array(0), Uint8Array.of(i % 256));
    const stream = new ReadableStream<Uint8Array>({ pull: (controller) => (parts.length ? controller.enqueue(parts.shift()!) : controller.close()) });
    const bytes = await readAll(stream, 20_000, new AbortController().signal, 413);
    expect(bytes.length).toBe(20_000);
    expect(bytes[257]).toBe(1);
    expect(bytes.buffer.byteLength).toBeLessThanOrEqual(2 * 20_000);
    expect(stream.locked).toBe(false);
    const many = new ReadableStream<Uint8Array>({ start: (controller) => { for (let i = 0; i < 2001; i++) controller.enqueue(Uint8Array.of(1)); controller.close(); } });
    await expect(readAll(many, 2000, new AbortController().signal, 413)).rejects.toThrow("too large");
    expect(many.locked).toBe(false);
    const { handler, seen } = proxy({ maxBodyBytes: 64 * 1024 });
    const json = JSON.stringify({ observation: { prompt: "x".repeat(30_000) } });
    const pieces = Array.from(json, (c) => new TextEncoder().encode(c));
    const fragmented = new ReadableStream<Uint8Array>({ pull: (controller) => (pieces.length ? controller.enqueue(pieces.shift()!) : controller.close()) });
    expect((await handler(new Request("https://app.example.com/api/dream/v1/get_action", { method: "POST", body: fragmented, duplex: "half" } as RequestInit))).status).toBe(200);
    expect(seen[0]!.body).toBe(json);
  });
});

describe("nodeHandler, the Host", () => {
  let server: Server | null = null;
  afterEach(() => new Promise<void>((resolve) => (server ? server.close(() => resolve()) : resolve())));

  /** Sends `head` as written over a socket and returns the status line's code. */
  async function raw(port: number, head: string): Promise<number> {
    const { connect } = await import("node:net");
    return new Promise((resolve, reject) => {
      const socket = connect(port, "127.0.0.1", () => socket.write(head));
      let text = "";
      socket.on("data", (chunk) => (text += chunk));
      socket.on("end", () => resolve(Number(/^HTTP\/1\.[01] (\d{3})/.exec(text)?.[1] ?? 0)));
      socket.on("error", reject);
    });
  }

  it("refuses a missing, malformed or absolute-form host, and serves a port and an IPv6 literal it allows", async () => {
    const handler = createDreamProxy({ apiKey: KEY, allowedHosts: ["localhost", "localhost:8787", "[::1]:8787"], fetch: upstream(() => Response.json([])).fetch });
    server = createServer(nodeHandler(handler));
    await new Promise<void>((resolve) => server!.listen(0, "127.0.0.1", resolve));
    const port = (server!.address() as AddressInfo).port;
    const v = `Dream-Client-Version: ${CLIENT_VERSION}\r\n`;
    expect(await raw(port, `GET /v1/models HTTP/1.0\r\n${v}\r\n`)).toBe(400);
    expect(await raw(port, `GET /v1/models HTTP/1.1\r\nHost: local host\r\n${v}Connection: close\r\n\r\n`)).toBe(400);
    expect(await raw(port, `GET http://localhost/v1/models HTTP/1.1\r\nHost: localhost\r\n${v}Connection: close\r\n\r\n`)).toBe(400);
    expect(await raw(port, `GET /v1/models HTTP/1.1\r\nHost: localhost:8787\r\n${v}Connection: close\r\n\r\n`)).toBe(200);
    expect(await raw(port, `GET /v1/models HTTP/1.1\r\nHost: [::1]:8787\r\n${v}Connection: close\r\n\r\n`)).toBe(200);
    expect(await raw(port, `GET /v1/models HTTP/1.1\r\nHost: localhost:9999\r\n${v}Connection: close\r\n\r\n`)).toBe(421);
  });
});


describe("createDreamProxy, third review", () => {
  const SECRET = "dk_SYNTHETIC_secret_7f3a9c";
  const through = (body: string | (() => Response), extra: Partial<DreamProxyOptions> = {}) => {
    const mock = upstream(typeof body === "string" ? () => new Response(body) : body);
    return { ...mock, handler: createDreamProxy({ apiKey: SECRET, allowedHosts: "any", fetch: mock.fetch, ...extra }) };
  };
  const action = (handler: (r: Request) => Promise<Response>) => handler(new Request("http://proxy.test/v1/get_action", { method: "POST", body: "{}" }));

  it("takes only a canonical key: no surrounding spaces or controls, Dream's characters, at least 16 long", () => {
    for (const apiKey of [`${SECRET} `, ` ${SECRET}`, `${SECRET}\t`, `${SECRET}\n`, "status", "dk_short", `dk_${"x".repeat(14)} x`, `dk_${"é".repeat(14)}`]) {
      expect(() => createDreamProxy({ apiKey, allowedHosts: "any" }), JSON.stringify(apiKey)).toThrow(DreamConfigError);
    }
    // The client too refuses a key fetch would change.
    expect(() => new DreamClient({ apiKey: `${SECRET} ` })).toThrow(/starts or ends with a space or tab/);
    expect(() => new DreamClient({ apiKey: `${SECRET}\r` })).toThrow(DreamConfigError);
  });

  it("refuses any non-finite number, so JSON's 1e400 never reaches a caller as null", async () => {
    const reply = '{"status":{"code":"DA-0000"},"session_id":"s","actions":{"per_key":{"action.right_arm":[[0,0],[0,1e400]]}}}';
    const { handler } = through(reply);
    const answered = await action(handler);
    expect(answered.status).toBe(502);
    expect(await answered.json()).toEqual({ proxy_error: "upstream answer withheld" });
    // The existing regression, run through the proxy: no action reaches the caller.
    const client = new PolicyClient({ checkpoint: "c", baseUrl: "http://proxy.test", processing: "raw", proxyHeaders: true, fetch: (input, init) => handler(new Request(input, init)) });
    const error = await client.getAction({ prompt: "x" }).catch((e: Error) => e);
    expect(error).toBeInstanceOf(Error);
    expect((error as Error).name).toBe("DreamProtocolError");
    // And a null where a number belongs is refused by the client, whatever the path.
    const nulled = new PolicyClient({ checkpoint: "c", baseUrl: "http://gateway.test", processing: "raw", fetch: async () => new Response('{"status":{"code":"DA-0000"},"session_id":"s","actions":{"q_r_chunk":[[0,0],[0,null]]}}') });
    const refused = await nulled.getAction({ prompt: "x" }).catch((e: Error & { detail?: unknown }) => e);
    expect(refused.name).toBe("DreamActionError");
    expect((refused as { detail?: unknown }).detail).toEqual({ kind: "non_finite", key: "q_r_chunk", row: 1 });
  });

  it("keeps an answer whose timing is null, absent or unfamiliar, projecting the two timings it knows", async () => {
    const refusal = through('{"status":{"code":"DA-2000","reason":"pool_full","retryable":true,"retry_after_s":5},"timing":null}');
    const kept = await action(refusal.handler);
    expect(kept.status).toBe(200);
    expect(await kept.json()).toEqual({ status: { code: "DA-2000", reason: "pool_full", retryable: true, retry_after_s: 5 } });
    const odd = through('{"status":{"code":"DA-0000"},"session_id":"s","actions":{"q":[[1]]},"timing":{"gateway_ms":4,"serve_infer_ms":"slow","units":"ms"}}');
    expect(await (await action(odd.handler)).json()).toEqual({ status: { code: "DA-0000" }, session_id: "s", actions: { q: [[1]] }, timing: { gateway_ms: 4 } });
  });

  it("keeps a platform or data-service refusal's status, with a safe body when its own does not fit", async () => {
    for (const [status, body] of [[401, "{}"], [403, "<html>no</html>"], [404, '{"error":"gone"}'], [500, ""]] as const) {
      const { handler } = through(() => new Response(body, { status }));
      const reply = await handler(get("http://proxy.test/v1/models"));
      expect(reply.status, `${status} ${body}`).toBe(status);
      expect(await reply.json()).toEqual({ error: { code: "upstream_error", message: `the Dream service answered ${status}` } });
    }
    // So a client behind the proxy tells an authentication refusal from the rest, and does not retry it.
    let calls = 0;
    const mock = createDreamProxy({ apiKey: SECRET, allowedHosts: "any", fetch: async () => (calls++, new Response("{}", { status: 401 })) });
    const client = new DreamClient({ proxyUrl: "http://proxy.test", fetch: (input, init) => mock(new Request(input, init)) });
    await expect(client.getModels()).rejects.toMatchObject({ name: "DreamAuthError" });
    expect(calls).toBe(1);
  });

  it("checks answers within bounds: deep, long or costly ones are withheld quickly, never thrown", async () => {
    const deep = `{"status":{"code":"DA-0000"},"actions":{"a":${"[".repeat(5000)}1${"]".repeat(5000)}}}`;
    expect((await action(through(deep).handler)).status).toBe(502);
    const veryDeep = `{"status":{"code":"DA-0000"},"actions":{"a":${"[".repeat(200_000)}${"]".repeat(200_000)}}}`;
    expect((await action(through(veryDeep).handler)).status).toBe(502);
    // An envelope text field past ANSWER_LIMITS.string is left out; one within it is kept whole.
    const kept = "x".repeat(ANSWER_LIMITS.string);
    expect(await (await action(through(`{"status":{"code":"DA-2000","detail":"${kept}"}}`).handler)).json()).toEqual({ status: { code: "DA-2000", detail: kept } });
    expect(await (await action(through(`{"status":{"code":"DA-2000","detail":"${kept}x"}}`).handler)).json()).toEqual({ status: { code: "DA-2000" } });
    // The key scan's work is bounded by the text it reads: a hex diagnostic is decoded once per alignment, two deep.
    const hex = "0123456789abcdef".repeat(250);
    const work: ScanWork = { nodes: 0, decoded: 0 };
    expect(answerFor("gateway", 200, new TextEncoder().encode(`{"status":{"code":"DA-0000","detail":"${hex}"}}`), SECRET, work)).not.toBeNull();
    expect(work.decoded).toBeLessThanOrEqual(8 * hex.length);
  });

  it("counts values before it schedules them: a wide array stops at the limit, a real answer is cheap", () => {
    const wide = `{"status":{"code":"DA-0000"},"actions":{"extra":[${"0,".repeat(1_500_000)}0]}}`;
    const work: ScanWork = { nodes: 0, decoded: 0 };
    expect(answerFor("gateway", 200, new TextEncoder().encode(wide), SECRET, work)).toBeNull();
    expect(work.nodes).toBeLessThanOrEqual(ANSWER_LIMITS.nodes);
    // Eight heads of 48 rows of 7 values: every value visited once by the shape check and once by the key scan.
    const rows = Array.from({ length: 48 }, (_, r) => Array.from({ length: 7 }, (_, c) => Math.sin(r * 7 + c)));
    const per_key = Object.fromEntries(Array.from({ length: 6 }, (_, i) => [`action.part_${i}`, rows]));
    const answer = { status: { code: "DA-0000" }, session_id: "s", actions: { q_r_chunk: rows, q_l_chunk: rows, per_key } };
    const real: ScanWork = { nodes: 0, decoded: 0 };
    const written = answerFor("gateway", 200, new TextEncoder().encode(JSON.stringify(answer)), SECRET, real);
    expect(JSON.parse(written!)).toEqual(answer);
    expect(real.nodes).toBeLessThan(2 * 8 * 48 * 8 + 100);
  });

  it("matches a Host by its canonical form: one field, default ports, IPv6 spellings, a trailing dot", async () => {
    const answer = () => Response.json([]);
    const { handler } = through(answer, { allowedHosts: ["proxy.example", "[::1]:8787", "api.example:443"] });
    const at = (url: string, host?: string) => handler(get(url, host ? { Host: host } : {}));
    expect((await at("http://proxy.example:80/v1/models")).status).toBe(200);
    expect((await at("https://proxy.example:443/v1/models")).status).toBe(200);
    expect((await at("https://proxy.example./v1/models")).status).toBe(200);
    expect((await at("http://[0:0:0:0:0:0:0:1]:8787/v1/models")).status).toBe(200);
    expect((await at("https://api.example/v1/models")).status).toBe(200);
    expect((await at("http://api.example/v1/models")).status).toBe(421);
    expect((await at("http://proxy.example:8080/v1/models")).status).toBe(421);
    expect(() => createDreamProxy({ apiKey: SECRET, allowedHosts: ["a b"] })).toThrow(/allowedHosts/);
  });

  it("refuses a plain http:// upstream on another computer unless told the network is trusted", () => {
    expect(() => createDreamProxy({ apiKey: SECRET, allowedHosts: "any", gatewayUrl: "http://gateway.example" })).toThrow(/plain http/);
    expect(() => createDreamProxy({ apiKey: SECRET, allowedHosts: "any", gatewayUrl: "http://127.0.0.1:9000" })).not.toThrow();
    expect(() => createDreamProxy({ apiKey: SECRET, allowedHosts: "any", dataUrl: "http://data.lan", allowInsecureHttp: true })).not.toThrow();
    expect(() => new DreamClient({ apiKey: SECRET, baseUrl: "http://gateway.example" })).toThrow(/plain http/);
    expect(() => new DreamClient({ apiKey: SECRET, platformUrl: "http://[::1]:8000" })).not.toThrow();
    expect(() => new DreamClient({ baseUrl: "http://proxy.lan" })).not.toThrow();
    expect(() => new DreamClient({ apiKey: SECRET, dataUrl: "http://data.lan", allowInsecureHttp: true })).not.toThrow();
  });

  it("cancels a body that arrives after its deadline, and a sign-in answer the client left", async () => {
    let cancelled = 0;
    const body = () => new ReadableStream<Uint8Array>({ cancel: () => void cancelled++ });
    const slow = createDreamProxy({
      apiKey: SECRET,
      allowedHosts: "any",
      timeoutMs: 20,
      fetch: () => new Promise((resolve) => setTimeout(() => resolve(new Response(body())), 60)),
    });
    expect((await action(slow)).status).toBe(504);
    await new Promise((resolve) => setTimeout(resolve, 80));
    expect(cancelled).toBe(1);
    const page = new AbortController();
    const late = createDreamProxy({
      apiKey: SECRET,
      allowedHosts: "any",
      authorize: () => new Promise((resolve) => setTimeout(() => resolve(new Response(body(), { status: 401 })), 40)),
    });
    const pending = late(new Request("http://proxy.test/v1/get_action", { method: "POST", body: "{}", signal: page.signal }));
    page.abort();
    expect((await pending).status).toBe(499);
    await new Promise((resolve) => setTimeout(resolve, 60));
    expect(cancelled).toBe(2);
  });

  it("refuses the contradictory client configuration a proxy would refuse anyway", () => {
    expect(() => new DreamClient({ proxyUrl: "/api/dream", proxyHeaders: false })).toThrow(/leave proxyHeaders out/);
  });
});

describe("nodeHandler, third review", () => {
  let server: Server | null = null;
  afterEach(() => new Promise<void>((resolve) => (server ? server.close(() => resolve()) : resolve())));

  async function raw(port: number, head: string): Promise<number> {
    const { connect } = await import("node:net");
    return new Promise((resolve, reject) => {
      const socket = connect(port, "127.0.0.1", () => socket.write(head));
      let text = "";
      socket.on("data", (chunk) => (text += chunk));
      socket.on("end", () => resolve(Number(/^HTTP\/1\.[01] (\d{3})/.exec(text)?.[1] ?? 0)));
      socket.on("error", reject);
    });
  }

  it("refuses a request with two Host fields", async () => {
    const handler = createDreamProxy({ apiKey: KEY, allowedHosts: ["localhost"], fetch: upstream(() => Response.json([])).fetch });
    server = createServer(nodeHandler(handler));
    await new Promise<void>((resolve) => server!.listen(0, "127.0.0.1", resolve));
    const port = (server!.address() as AddressInfo).port;
    const v = `Dream-Client-Version: ${CLIENT_VERSION}\r\nConnection: close\r\n`;
    expect(await raw(port, `GET /v1/models HTTP/1.1\r\nHost: localhost\r\nHost: evil.test\r\n${v}\r\n`)).toBe(400);
    expect(await raw(port, `GET /v1/models HTTP/1.1\r\nHost: localhost\r\n${v}\r\n`)).toBe(200);
  });

  it("stops reading an open answer body when the client disconnects, and writes nothing after", async () => {
    let cancelled = false;
    let onPull!: () => void;
    let onCancel!: () => void;
    const pulled = new Promise<void>((resolve) => (onPull = resolve));
    const cancelledNow = new Promise<void>((resolve) => (onCancel = resolve));
    const handler = async () =>
      new Response(
        new ReadableStream<Uint8Array>({
          pull: () => onPull(),
          cancel: () => {
            cancelled = true;
            onCancel();
          },
        }),
        { status: 401 },
      );
    const errors: unknown[] = [];
    const listener = nodeHandler(handler);
    server = createServer((request, response) => void listener(request, response).catch((e) => errors.push(e)));
    await new Promise<void>((resolve) => server!.listen(0, "127.0.0.1", resolve));
    const client = new AbortController();
    const call = fetch(`http://127.0.0.1:${(server!.address() as AddressInfo).port}/v1/models`, { signal: client.signal }).catch((e: Error) => e);
    // Abort only once the adapter is reading the answer's body (its first pull), then wait for the cancel itself.
    await pulled;
    client.abort();
    await call;
    await cancelledNow;
    expect(cancelled).toBe(true);
    expect(errors).toEqual([]);
  });
});


describe("createDreamProxy, fourth review", () => {
  const SECRET = "dk_SYNTHETIC_secret_7f3a9c";
  const bytes = (text: string) => new TextEncoder().encode(text);
  const gateway = (body: unknown, status = 200) => answerFor("gateway", status, bytes(typeof body === "string" ? body : JSON.stringify(body)), SECRET);

  it("passes only finite numbers in actuator heads: no object, null or string where a joint value belongs", () => {
    for (const actions of ['{"q_r_chunk":[[{"x":0}]]}', '{"q_r_chunk":[[0,null]]}', '{"q_l_chunk":[["0.1"]]}', '{"per_key":{"action.right_arm":[[true]]}}', '{"per_key":{"action.right_arm":[{"0":1}]}}', '{"per_key":"x"}', '{"q_r_chunk":{"0":[1]}}']) {
      expect(gateway(`{"status":{"code":"DA-0000"},"actions":${actions}}`), actions).toBeNull();
    }
    // A whole head that is null stays allowed, as the protocol sends a joint head the serve does not answer.
    const nulled = { status: { code: "DA-0000" }, actions: { q_r_chunk: null, per_key: { "action.right_arm": null, "action.right_grip": [[0.5]] } } };
    expect(JSON.parse(gateway(nulled)!)).toEqual(nulled);
  });

  it("passes extensions through as plain JSON, strings and nulls included, as Python does", () => {
    const answer = { status: { code: "DA-0000" }, actions: { q_r_chunk: [[0.1, 0.2]], head_label: "right", extra: [null], note: { a: [1, "b"] } } };
    expect(JSON.parse(gateway(answer)!)).toEqual(answer);
  });

  it("keeps a session id exactly, up to ANSWER_LIMITS.key characters, and withholds the answer past that", () => {
    const id = "s".repeat(257);
    expect(JSON.parse(gateway({ status: { code: "DA-0001", reason: "new" }, session_id: id, actions: {} })!).session_id).toBe(id);
    expect(JSON.parse(gateway({ status: { code: "DA-0001", reason: "new" }, session_id: "s".repeat(ANSWER_LIMITS.key), actions: {} })!).session_id).toHaveLength(ANSWER_LIMITS.key);
    expect(gateway({ status: { code: "DA-0001", reason: "new" }, session_id: "s".repeat(ANSWER_LIMITS.key + 1), actions: { q_r_chunk: [[1]] } })).toBeNull();
  });

  it("never delivers a number it changed: inexact integers and infinities are withheld, -0 and doubles kept", () => {
    expect(gateway('{"status":{"code":"DA-0000"},"actions":{"q_r_chunk":[[9007199254740993]]}}')).toBeNull();
    expect(gateway('{"status":{"code":"DA-0000"},"actions":{"extra":[1e400]}}')).toBeNull();
    expect(gateway('{"status":{"code":"DA-0000"},"actions":{"q_r_chunk":[[9007199254740992, 1e21, 123456789012]]}}')).not.toBeNull();
    const zero = gateway('{"status":{"code":"DA-0000"},"actions":{"q_r_chunk":[[-0, 0.5]]}}')!;
    expect(zero).toContain("[[-0,0.5]]");
    expect(Object.is(JSON.parse(zero).actions.q_r_chunk[0][0], -0)).toBe(true);
    // Ordinary joint values round-trip bit for bit.
    let seed = 1;
    const rows = Array.from({ length: 48 }, () => Array.from({ length: 7 }, () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 6.283185307179586));
    const back = JSON.parse(gateway({ status: { code: "DA-0000" }, actions: { q_r_chunk: rows } })!).actions.q_r_chunk;
    expect(back).toEqual(rows);
  });

  it("withholds a key of digits echoed as a number", async () => {
    const digits = "1234567890123456";
    expect(answerFor("gateway", 200, bytes(`{"status":{"code":"DA-0000"},"actions":{"extra":${digits}}}`), digits)).toBeNull();
    expect(answerFor("gateway", 200, bytes(`{"status":{"code":"DA-0000"},"actions":{"extra":1}}`), digits)).not.toBeNull();
  });

  it("keeps a refusal's meaning: a gateway refusal's actions are ignored, a platform refusal that must be withheld keeps its status", async () => {
    expect(JSON.parse(gateway({ status: { code: "DA-2000", reason: "pool_full", retryable: true, retry_after_s: 2 }, actions: null })!)).toEqual({
      status: { code: "DA-2000", reason: "pool_full", retryable: true, retry_after_s: 2 },
    });
    expect(JSON.parse(gateway('{"status":{"code":"DA-4000"},"actions":"whatever"}')!)).toEqual({ status: { code: "DA-4000" } });
    let calls = 0;
    const handler = createDreamProxy({
      apiKey: SECRET,
      allowedHosts: "any",
      fetch: async () => (calls++, Response.json({ error: { code: "invalid_api_key", message: `bad key ${SECRET}` } }, { status: 401 })),
    });
    const reply = await handler(get("http://proxy.test/v1/models"));
    expect(reply.status).toBe(401);
    expect(await reply.json()).toEqual({ error: { code: "upstream_error", message: "the Dream service answered 401" } });
    const client = new DreamClient({ proxyUrl: "http://proxy.test", fetch: (input, init) => handler(new Request(input, init)) });
    await expect(client.getModels()).rejects.toMatchObject({ name: "DreamAuthError" });
    expect(calls).toBe(2);
  });

  it("keeps training prompts as given within ANSWER_LIMITS.string, and withholds rather than changes them", () => {
    const answer = (prompts: unknown) => answerFor("training", 200, bytes(JSON.stringify({ cameras: { front: [[640, 360]] }, prompts })), SECRET);
    const long = "p".repeat(4097);
    expect(JSON.parse(answer([long, "wave"])!)).toEqual({ cameras: { front: [[640, 360]] }, prompts: [long, "wave"] });
    expect(answer(["p".repeat(ANSWER_LIMITS.string + 1)])).toBeNull();
    expect(answer(["wave", 3])).toBeNull();
  });

  it("replays every reply of the recorded wire: the well-formed ones come back with the same content", () => {
    let replies = 0;
    let wellFormed = 0;
    for (const scenario of wire.scenarios) {
      for (const step of scenario.steps) {
        for (const request of step.requests) {
          replies++;
          const shape = request.path === "/v1/models" ? "models" : request.path.includes("/training_data") ? "training" : "gateway";
          const written = answerFor(shape, request.reply.status, bytes(request.reply.body), SECRET);
          let parsed: unknown = undefined;
          try {
            parsed = JSON.parse(request.reply.body);
          } catch {
            // not JSON
          }
          const fits = shape !== "gateway" || (typeof parsed === "object" && parsed !== null && typeof (parsed as { status?: { code?: unknown } }).status?.code === "string");
          if (!fits) {
            expect(written, request.reply.body).toBeNull();
            continue;
          }
          wellFormed++;
          expect(JSON.parse(written!), request.reply.body.slice(0, 80)).toEqual(parsed);
        }
      }
    }
    expect([replies, wellFormed]).toEqual([44, 42]);
  });
});
