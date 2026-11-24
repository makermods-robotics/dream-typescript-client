/**
 * createDreamProxy() against a mock Dream Labs: what it forwards and how, what it refuses, the CORS answers, the body
 * limit, that the key never comes back, and a DreamClient pointed at it with `proxyUrl`; nodeHandler() on node:http.
 */

import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import { CLIENT_VERSION, DreamClient, DreamConfigError } from "../src/index.js";
import { nodeHandler } from "../src/node/index.js";
import { createDreamProxy, type DreamProxyOptions } from "../src/proxy.js";

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
    const models = await handler(new Request("https://app.example.com/api/dream/v1/models"));
    expect(await models.json()).toEqual([{ id: "m" }]);
    expect(seen[0]).toEqual({
      url: "https://platform.test/v1/models",
      method: "GET",
      body: "",
      headers: { authorization: `Bearer ${KEY}`, accept: "application/json", "user-agent": `dream-client/${CLIENT_VERSION}` },
    });
    await handler(new Request("https://app.example.com/api/dream/v1/fine_tuning/models/dream-1-ft%3A0a1b2c3d-towels/training_data"));
    expect(seen[1]!.url).toBe("https://data.test/v1/fine_tuning/models/dream-1-ft%3A0a1b2c3d-towels/training_data");
    expect(seen[1]!.headers["user-agent"]).toBe(`dream-client/${CLIENT_VERSION}`);
  });

  it("refuses every other route and method, and a training-data path that is not a fine-tune's", async () => {
    const { handler, seen } = proxy();
    for (const path of ["/v1/start_session", "/v1/models/x", "/v2/get_action", "/v1/fine_tuning/models/dreamlabs%2Fdream-1-beta/training_data", "/v1/fine_tuning/models/dream-1-ft%3Aa%2F..%2Fb/training_data", "/v1/fine_tuning/jobs"]) {
      expect((await handler(new Request(`https://app.example.com/api/dream${path}`))).status, path).toBe(404);
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
    const { handler, seen } = proxy({ allowedHosts: ["app.example.com"] });
    expect((await handler(new Request("https://rebound.example/api/dream/v1/models"))).status).toBe(421);
    expect((await handler(new Request("https://APP.example.com/api/dream/v1/models"))).status).toBe(200);
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
