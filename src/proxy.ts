/**
 * `@makermods/dreamlabs/proxy`: a Dream proxy for browser apps, as a standard fetch handler `(request) => response`
 * (Node 22 through `@makermods/dreamlabs/node`'s nodeHandler(), Cloudflare Workers, Deno, Bun).
 *
 * A page cannot call Dream Labs itself: the gateway refuses cross-origin requests, a page cannot set `User-Agent`, and
 * the API key must never reach it. The page's client is pointed at this proxy (`new DreamClient({ proxyUrl })`); the
 * proxy forwards only what the client calls, with the key added:
 *
 *   POST <basePath>/v1/get_action, /v1/end_session                  -> the gateway (`gatewayUrl`)
 *   GET  <basePath>/v1/models                                        -> the platform (`platformUrl`)
 *   GET  <basePath>/v1/fine_tuning/models/<fine-tune>/training_data  -> the data service (`dataUrl`)
 *
 * Bodies pass through unchanged (DreamAPI reports every outcome in the body's status envelope). The client identity
 * sent upstream is the proxy's own, never the caller's: `Dream-Client-Version: <clientVersion>` and `User-Agent:
 * dream-client/<clientVersion> <instance>` on the POSTs (the instance from the client's `Dream-Client-Instance`), and
 * `User-Agent: dream-client/<clientVersion>` on the GETs. It is {@link CLIENT_VERSION} unless your server configures
 * another: the version decides whether the gateway serves a checkpoint whose robot has parts, which this package does
 * not convert, so a page must not be able to claim one. A request with an `Origin` must come from one of
 * `allowedOrigins` (none by default), which also get the CORS headers and their preflight answered; a `Host` must be
 * one of `allowedHosts`. Bodies over `maxBodyBytes` are refused, and an answer that carries the key is withheld; the
 * key is never logged or put in an error.
 *
 * Whoever reaches the proxy spends your key: deploy it behind your own sign-in (`authorize`). The origin and host
 * checks keep other web pages, and DNS-rebinding tricks, from using it through a visitor's browser; they are not a
 * sign-in.
 */

import { authorization, checkServiceUrl, DEFAULT_DATA_URL, DEFAULT_GATEWAY_URL, DEFAULT_PLATFORM_URL } from "./credentials.js";
import { DreamConfigError } from "./errors.js";
import { FINE_TUNE_PREFIX, pyQuote } from "./trained-inputs.js";
import { checkClientVersion, CLIENT_INSTANCE_HEADER, CLIENT_VERSION_HEADER, END_SESSION_PATH, GET_ACTION_PATH, USER_AGENT_PRODUCT } from "./transport.js";
import { CLIENT_VERSION } from "./version.js";

/** A standard fetch handler: what createDreamProxy() returns. */
export type FetchHandler = (request: Request) => Promise<Response>;

/** Options of {@link createDreamProxy}. */
export interface DreamProxyOptions {
  /** Your Dream API key, added to every forwarded request as `Authorization: Bearer`. Required. */
  apiKey: string;
  /**
   * The `Host` names the proxy answers to (`127.0.0.1:8787`, `api.example.com`); any other is refused (421), which
   * stops DNS rebinding (a page served from an attacker's name that resolves to your proxy, whose same-origin GETs carry
   * no `Origin`). Required: `"any"` turns the check off, for a deployment behind a router that already sends the proxy
   * only requests for your own names (a Cloudflare Worker on its routes, a reverse proxy matching `server_name`).
   */
  allowedHosts: readonly string[] | "any";
  /**
   * The page origins (`https://app.example.com`, scheme, host and port) whose requests are answered, with CORS headers
   * and preflight. A request with any other `Origin` is refused (403). Empty by default; there is no wildcard, and a
   * page the proxy itself serves must be listed too.
   */
  allowedOrigins?: readonly string[];
  /** Where the proxy is mounted: the path before `/v1/...`, e.g. `/api/dream`. Empty by default. */
  basePath?: string;
  /** The gateway (`DREAM_GATEWAY_URL`); production by default. */
  gatewayUrl?: string;
  /** The platform (`DREAM_API_URL`); production by default. */
  platformUrl?: string;
  /** The data service (`DREAM_DATA_URL`); production by default. */
  dataUrl?: string;
  /**
   * The client version sent upstream, {@link CLIENT_VERSION} by default. Set it only for clients that convert parts:
   * a `MAJOR.MINOR.PATCH` here lets the gateway serve them checkpoints whose robot has parts.
   */
  clientVersion?: string;
  /** The largest request body forwarded, bytes; 8 MiB by default (camera frames are most of a request). */
  maxBodyBytes?: number;
  /** How long a request body may take to arrive, milliseconds; 30 s by default (408 after it). */
  uploadTimeoutMs?: number;
  /** The largest answer passed back, bytes; 16 MiB by default (502 above it). */
  maxResponseBytes?: number;
  /** How long a forwarded request may take, answer included, milliseconds; 70 s by default (504 after it). */
  timeoutMs?: number;
  /**
   * Extra request headers a page may send, for the CORS preflight: your own sign-in header (the client's `headers`
   * option), which is how a page on another origin signs in to the proxy. Cookies are not sent cross-origin.
   */
  allowedHeaders?: readonly string[];
  /**
   * Your sign-in, called for every request but a preflight before anything is read or forwarded, with a copy of the
   * request that has its URL, method, headers and signal but no body: true lets it through, false answers 401, and a
   * Response is answered as it is (with the CORS headers of an allowed origin added).
   */
  authorize?: (request: Request) => boolean | Response | Promise<boolean | Response>;
  /** The `fetch` that reaches Dream Labs; the global one by default. */
  fetch?: (input: string, init: RequestInit) => Promise<Response>;
}

/** The default body limit, bytes. */
export const PROXY_MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The default answer limit, bytes. */
export const PROXY_MAX_RESPONSE_BYTES = 16 * 1024 * 1024;
/** The default upload deadline, milliseconds. */
export const PROXY_UPLOAD_TIMEOUT_MS = 30_000;
/** The default upstream timeout, milliseconds. */
export const PROXY_TIMEOUT_MS = 70_000;
/** The training_data path below `<basePath>/v1/fine_tuning/models/`. */
const TRAINING_DATA = /^\/v1\/fine_tuning\/models\/([^/]+)\/training_data$/;
const INSTANCE = /^[0-9a-f]{8}$/;
const AGENT_INSTANCE = new RegExp(`^${USER_AGENT_PRODUCT}/[!#$%&'*+.^_\`|~0-9A-Za-z-]{1,64} ([0-9a-f]{8})$`);
/** Statuses whose answer has no body. */
const NO_BODY = new Set([101, 204, 205, 304]);

interface Route {
  method: "GET" | "POST";
  target: string;
}

/** Why a request or answer body was not read whole. */
class BodyError extends Error {
  constructor(readonly status: number, readonly code: string) {
    super(code);
  }
}

/**
 * A fetch handler that forwards the Dream client's requests to Dream Labs with `options.apiKey` added; see the module
 * documentation for what it forwards and checks.
 *
 * @throws DreamConfigError An option is malformed: no `apiKey`, no `allowedHosts`, a URL that is not
 *   `http(s)://<host>[:<port>]`, an origin that is not one, a client version that is not an HTTP token, or a limit or
 *   timeout that is not a positive number.
 */
export function createDreamProxy(options: DreamProxyOptions): FetchHandler {
  if (typeof options !== "object" || options === null) throw new DreamConfigError("createDreamProxy() takes an options object");
  if (typeof options.apiKey !== "string" || !options.apiKey) throw new DreamConfigError("createDreamProxy() needs apiKey, your Dream API key");
  const secret = options.apiKey;
  const auth = authorization(secret);
  if (options.allowedHosts !== "any" && !Array.isArray(options.allowedHosts)) {
    throw new DreamConfigError(
      "createDreamProxy() needs allowedHosts: the Host names it answers to (e.g. ['127.0.0.1:8787']), or 'any' behind a router that only sends it your own names",
    );
  }
  const hosts = options.allowedHosts === "any" ? null : new Set(options.allowedHosts.map((host) => String(host).toLowerCase()));
  const gateway = checkServiceUrl(options.gatewayUrl ?? DEFAULT_GATEWAY_URL, "gatewayUrl", false);
  const platform = checkServiceUrl(options.platformUrl ?? DEFAULT_PLATFORM_URL, "platformUrl", false);
  const data = checkServiceUrl(options.dataUrl ?? DEFAULT_DATA_URL, "dataUrl", false);
  const version = checkClientVersion(options.clientVersion ?? CLIENT_VERSION);
  const origins = new Set((options.allowedOrigins ?? []).map(checkOrigin));
  const basePath = (options.basePath ?? "").replace(/\/+$/, "");
  if (basePath && !basePath.startsWith("/")) throw new DreamConfigError(`basePath must start with /, got ${JSON.stringify(options.basePath)}`);
  const maxBodyBytes = positive(options.maxBodyBytes ?? PROXY_MAX_BODY_BYTES, "maxBodyBytes");
  const maxResponseBytes = positive(options.maxResponseBytes ?? PROXY_MAX_RESPONSE_BYTES, "maxResponseBytes");
  const uploadTimeoutMs = positive(options.uploadTimeoutMs ?? PROXY_UPLOAD_TIMEOUT_MS, "uploadTimeoutMs");
  const timeoutMs = positive(options.timeoutMs ?? PROXY_TIMEOUT_MS, "timeoutMs");
  const allowHeaders = ["Content-Type", "Accept", CLIENT_VERSION_HEADER, CLIENT_INSTANCE_HEADER, ...(options.allowedHeaders ?? [])].join(", ");
  const upstream = options.fetch ?? ((input: string, init: RequestInit) => globalThis.fetch(input, init));
  /** The instance named for a caller that sends none: one per proxy, so such callers count as one robot. */
  const ownInstance = Array.from(crypto.getRandomValues(new Uint8Array(4)), (b) => b.toString(16).padStart(2, "0")).join("");
  /** The key as an answer could carry it: as is, and JSON-escaped. */
  const leaks = [...new Set([secret, JSON.stringify(secret).slice(1, -1), JSON.stringify(secret).slice(1, -1).replace(/\//g, "\\/")])];
  const routes: Record<string, Route> = {
    [GET_ACTION_PATH]: { method: "POST", target: `${gateway}${GET_ACTION_PATH}` },
    [END_SESSION_PATH]: { method: "POST", target: `${gateway}${END_SESSION_PATH}` },
    "/v1/models": { method: "GET", target: `${platform}/v1/models` },
  };

  const route = (path: string): Route | null => {
    if (Object.hasOwn(routes, path)) return routes[path]!;
    const training = TRAINING_DATA.exec(path);
    if (!training) return null;
    let checkpoint: string;
    try {
      checkpoint = decodeURIComponent(training[1]!);
    } catch {
      return null;
    }
    // Only a fine-tune's own record, as the client asks for it: never another path on the data service.
    if (!checkpoint.startsWith(FINE_TUNE_PREFIX) || checkpoint.length > 256 || /[\u0000-\u001f\u007f/?#\\]/.test(checkpoint)) return null;
    return { method: "GET", target: `${data}/v1/fine_tuning/models/${pyQuote(checkpoint)}/training_data` };
  };

  return async (request: Request): Promise<Response> => {
    const origin = request.headers.get("origin");
    const allowed = origin !== null && origins.has(origin);
    const cors: Record<string, string> = allowed ? { "Access-Control-Allow-Origin": origin, Vary: "Origin" } : origins.size ? { Vary: "Origin" } : {};
    const fail = (status: number, error: string, extra: Record<string, string> = {}) => json(status, { proxy_error: error }, { ...cors, ...extra });
    let url: URL;
    try {
      url = new URL(request.url);
    } catch {
      return fail(400, "bad url");
    }
    if (hosts && !hosts.has((request.headers.get("host") ?? url.host).toLowerCase())) return fail(421, "unknown host");
    if (origin !== null && !allowed) return json(403, { proxy_error: "origin not allowed" }, origins.size ? { Vary: "Origin" } : {});
    if (basePath && url.pathname !== basePath && !url.pathname.startsWith(`${basePath}/`)) return fail(404, "no such route");
    const target = route(url.pathname.slice(basePath.length));
    if (!target) return fail(404, "no such route");
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: { ...cors, "Access-Control-Allow-Methods": `${target.method}, OPTIONS`, "Access-Control-Allow-Headers": allowHeaders, "Access-Control-Max-Age": "600" },
      });
    }
    if (request.method !== target.method) return fail(405, "method not allowed", { Allow: `${target.method}, OPTIONS` });
    const signal = request.signal;
    if (options.authorize) {
      let verdict: boolean | Response;
      try {
        verdict = await options.authorize(new Request(request.url, { method: request.method, headers: request.headers, signal }));
      } catch {
        return fail(500, "authorize failed");
      }
      if (verdict instanceof Response) return withHeaders(verdict, cors);
      if (verdict !== true) return fail(401, "unauthorized");
    }
    // The caller gave up meanwhile (the page cancelled, the connection closed): nothing is sent on its behalf.
    if (signal.aborted) return fail(499, "cancelled");
    const headers: Record<string, string> = { ...auth, Accept: "application/json" };
    let body: Uint8Array<ArrayBuffer> | undefined;
    if (target.method === "POST") {
      const declared = Number(request.headers.get("content-length") ?? 0);
      if (declared > maxBodyBytes) return fail(413, "too large");
      try {
        body = request.body ? await readAll(request.body, maxBodyBytes, AbortSignal.any([signal, AbortSignal.timeout(uploadTimeoutMs)]), 413) : new Uint8Array(new ArrayBuffer(0));
      } catch (error) {
        if (signal.aborted) return fail(499, "cancelled");
        if (error instanceof BodyError) return fail(error.status, error.code);
        if (error instanceof Error && error.name === "TimeoutError") return fail(408, "upload timed out");
        return fail(400, "unreadable body");
      }
      if (signal.aborted) return fail(499, "cancelled");
      const instance = request.headers.get(CLIENT_INSTANCE_HEADER) ?? "";
      const agentInstance = AGENT_INSTANCE.exec(request.headers.get("user-agent") ?? "")?.[1];
      headers["Content-Type"] = "application/json";
      headers[CLIENT_VERSION_HEADER] = version;
      headers["User-Agent"] = `${USER_AGENT_PRODUCT}/${version} ${INSTANCE.test(instance) ? instance : (agentInstance ?? ownInstance)}`;
    } else {
      headers["User-Agent"] = `${USER_AGENT_PRODUCT}/${version}`;
    }
    // The caller's cancellation and the deadline end the exchange, answer included.
    const exchange = AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)]);
    let reply: Response;
    let payload: Uint8Array<ArrayBuffer> | null;
    try {
      reply = await upstream(target.target, { method: target.method, headers, body, redirect: "manual", signal: exchange });
      payload = NO_BODY.has(reply.status) || !reply.body ? null : await readAll(reply.body, maxResponseBytes, exchange, 502);
    } catch (error) {
      if (signal.aborted) return fail(499, "cancelled");
      if (error instanceof BodyError) return fail(502, "upstream answer too large");
      const name = error instanceof Error ? error.name : "";
      return name === "TimeoutError" ? fail(504, "timeout") : fail(502, "unreachable");
    }
    if (payload !== null) {
      const text = new TextDecoder().decode(payload);
      // An upstream (or a fetch) that echoes the request would hand the key to the page.
      if (leaks.some((leak) => text.includes(leak))) return fail(502, "upstream answer withheld");
    }
    const status = reply.status >= 200 && reply.status <= 599 ? reply.status : 502;
    return new Response(NO_BODY.has(status) ? null : (payload ?? new Uint8Array(0)), {
      status,
      headers: { ...cors, "Content-Type": "application/json", "Cache-Control": "no-store" },
    });
  };
}

/**
 * A stream read whole, or a {@link BodyError} with `tooLarge` once it passes `limit` bytes; `signal` cancels the read
 * (its reason is thrown). The stream is cancelled on every path that does not finish it.
 */
async function readAll(stream: ReadableStream<Uint8Array>, limit: number, signal: AbortSignal, tooLarge: number): Promise<Uint8Array<ArrayBuffer>> {
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  let stop: (() => void) | null = null;
  const stopped = new Promise<never>((_, reject) => {
    stop = () => reject(signal.reason);
    if (signal.aborted) stop();
    else signal.addEventListener("abort", stop, { once: true });
  });
  stopped.catch(() => undefined);
  try {
    for (;;) {
      const { done, value } = await Promise.race([reader.read(), stopped]);
      if (done) break;
      size += value.byteLength;
      if (size > limit) throw new BodyError(tooLarge, "too large");
      chunks.push(value);
    }
  } catch (error) {
    await reader.cancel().catch(() => undefined);
    throw error;
  } finally {
    signal.removeEventListener("abort", stop!);
  }
  const out = new Uint8Array(new ArrayBuffer(size));
  let at = 0;
  for (const chunk of chunks) {
    out.set(chunk, at);
    at += chunk.byteLength;
  }
  return out;
}

/** `response` with `extra` headers added (the CORS headers of an allowed origin). */
function withHeaders(response: Response, extra: Record<string, string>): Response {
  if (!Object.keys(extra).length) return response;
  const headers = new Headers(response.headers);
  for (const [name, value] of Object.entries(extra)) headers.set(name, value);
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

function json(status: number, body: unknown, headers: Record<string, string>): Response {
  return new Response(JSON.stringify(body), { status, headers: { ...headers, "Content-Type": "application/json", "Cache-Control": "no-store" } });
}

/** @throws DreamConfigError `origin` is not `scheme://host[:port]` as a browser sends it. */
function checkOrigin(origin: string): string {
  let parsed: URL | null = null;
  try {
    parsed = new URL(origin);
  } catch {
    // refused below
  }
  if (!parsed || !["http:", "https:"].includes(parsed.protocol) || parsed.origin !== origin) {
    throw new DreamConfigError(`allowedOrigins takes origins as a browser sends them, scheme://host[:port] with no path or trailing slash, got ${JSON.stringify(origin)}`);
  }
  return origin;
}

function positive(value: unknown, what: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) throw new DreamConfigError(`${what} must be a positive number, got ${JSON.stringify(value)}`);
  return value;
}
