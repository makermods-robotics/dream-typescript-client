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
 * Requests pass through unchanged; answers are parsed and written anew, in the shape their route produces, and withheld
 * when they carry the key (see {@link answerFor} and {@link carriesSecret}). The client identity
 * sent upstream is the proxy's own, never the caller's: `Dream-Client-Version: <clientVersion>` and `User-Agent:
 * dream-client/<clientVersion> <instance>` on the POSTs (the instance from the client's `Dream-Client-Instance`), and
 * `User-Agent: dream-client/<clientVersion>` on the GETs. It is {@link CLIENT_VERSION} unless your server configures
 * another: the version decides whether the gateway serves a checkpoint whose robot has parts, which this package does
 * not convert, so a page must not be able to claim one. A GET must carry `Dream-Client-Version` (the client always
 * sends it through a proxy), and a request whose Fetch Metadata says cross-site must carry an allowed `Origin`. A
 * request with an `Origin` must come from one of
 * `allowedOrigins` (none by default), which also get the CORS headers and their preflight answered; a `Host` must be
 * one of `allowedHosts`. Bodies over `maxBodyBytes` are refused, and an answer that carries the key is withheld; the
 * key is never logged or put in an error.
 *
 * Whoever reaches the proxy spends your key: deploy it behind your own sign-in (`authorize`). The origin and host
 * checks keep other web pages, and DNS-rebinding tricks, from using it through a visitor's browser; they are not a
 * sign-in.
 */

import { authorization, checkServiceUrl, DEFAULT_DATA_URL, DEFAULT_GATEWAY_URL, DEFAULT_PLATFORM_URL, PROXY_KEY, PROXY_KEY_MIN, requireHttps } from "./credentials.js";
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
  /**
   * Let the key go to a plain `http://` upstream on another computer, on a network you trust. Off by default: every
   * upstream must be `https://` or this computer's (`localhost`, 127.0.0.0/8, `[::1]`).
   */
  allowInsecureHttp?: boolean;
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
/** A published fine-tune id: `dream-1-ft:<uid8>-<suffix>`, the suffix `[a-z0-9-]{1,40}` (`dream.fine_tuning.create`). */
const FINE_TUNE_ID = /^dream-1-ft:[a-z0-9]{8}-[a-z0-9-]{1,40}$/;
/** An HTTP header name. */
const TOKEN = /^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/;
const AGENT_INSTANCE = new RegExp(`^${USER_AGENT_PRODUCT}/[!#$%&'*+.^_\`|~0-9A-Za-z-]{1,64} ([0-9a-f]{8})$`);
/** Statuses whose answer has no body. */
const NO_BODY = new Set([101, 204, 205, 304]);

interface Route {
  method: "GET" | "POST";
  target: string;
  answer: AnswerShape;
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
  // Every option is read and checked here, once: changing the object afterwards changes nothing.
  const {
    apiKey,
    allowedHosts,
    allowedOrigins = [],
    basePath: base = "",
    gatewayUrl = DEFAULT_GATEWAY_URL,
    platformUrl = DEFAULT_PLATFORM_URL,
    dataUrl = DEFAULT_DATA_URL,
    clientVersion = CLIENT_VERSION,
    maxBodyBytes: bodyLimit = PROXY_MAX_BODY_BYTES,
    uploadTimeoutMs: uploadLimit = PROXY_UPLOAD_TIMEOUT_MS,
    maxResponseBytes: answerLimit = PROXY_MAX_RESPONSE_BYTES,
    timeoutMs: timeLimit = PROXY_TIMEOUT_MS,
    allowedHeaders = [],
    authorize,
    fetch: fetchOption,
    allowInsecureHttp = false,
  } = options;
  if (typeof apiKey !== "string" || !apiKey) throw new DreamConfigError("createDreamProxy() needs apiKey, your Dream API key");
  // One canonical key, the one sent and the one every answer is checked for: no surrounding spaces (which fetch would
  // strip from the header), only the characters Dream keys are made of, and long enough never to match a field name.
  if (apiKey.length < PROXY_KEY_MIN || !PROXY_KEY.test(apiKey)) {
    throw new DreamConfigError(`createDreamProxy() apiKey must be at least ${PROXY_KEY_MIN} letters, digits or ._~+/=- with no spaces: copy the key again, whole`);
  }
  const secret = apiKey;
  const auth = authorization(secret);
  if (allowedHosts !== "any" && !Array.isArray(allowedHosts)) {
    throw new DreamConfigError(
      "createDreamProxy() needs allowedHosts: the Host names it answers to (e.g. ['127.0.0.1:8787']), or 'any' behind a router that only sends it your own names",
    );
  }
  // Each allowed host in its canonical form for either scheme: lower case, IPv6 compressed, the scheme's default port
  // dropped, a trailing dot dropped.
  const hosts =
    allowedHosts === "any"
      ? null
      : Object.fromEntries(
          (["http:", "https:"] as const).map((scheme) => [
            scheme,
            new Set(
              allowedHosts.map((host) => {
                const canonical = canonicalHost(String(host), scheme);
                if (canonical === null) throw new DreamConfigError(`allowedHosts takes host[:port] names as a Host header carries them, got ${JSON.stringify(String(host))}`);
                return canonical;
              }),
            ),
          ]),
        );
  const gateway = checkServiceUrl(gatewayUrl, "gatewayUrl", false);
  const platform = checkServiceUrl(platformUrl, "platformUrl", false);
  const data = checkServiceUrl(dataUrl, "dataUrl", false);
  for (const [what, url] of [["gatewayUrl", gateway], ["platformUrl", platform], ["dataUrl", data]] as const) requireHttps(url, what, allowInsecureHttp);
  const version = checkClientVersion(clientVersion);
  if (!Array.isArray(allowedOrigins)) throw new DreamConfigError("allowedOrigins must be a list of origins");
  const origins = new Set(allowedOrigins.map(checkOrigin));
  const basePath = String(base).replace(/\/+$/, "");
  if (basePath && !basePath.startsWith("/")) throw new DreamConfigError(`basePath must start with /, got ${JSON.stringify(base)}`);
  const maxBodyBytes = whole(bodyLimit, "maxBodyBytes");
  const maxResponseBytes = whole(answerLimit, "maxResponseBytes");
  const uploadTimeoutMs = whole(uploadLimit, "uploadTimeoutMs");
  const timeoutMs = whole(timeLimit, "timeoutMs");
  if (!Array.isArray(allowedHeaders) || !allowedHeaders.every((h) => typeof h === "string" && TOKEN.test(h))) {
    throw new DreamConfigError("allowedHeaders must be a list of header names");
  }
  const allowHeaders = ["Content-Type", "Accept", CLIENT_VERSION_HEADER, CLIENT_INSTANCE_HEADER, ...allowedHeaders].join(", ");
  if (authorize !== undefined && typeof authorize !== "function") throw new DreamConfigError("authorize must be a function");
  if (fetchOption !== undefined && typeof fetchOption !== "function") throw new DreamConfigError("fetch must be a function");
  const upstream = fetchOption ?? ((input: string, init: RequestInit) => globalThis.fetch(input, init));
  /** The instance named for a caller that sends none: one per proxy, so such callers count as one robot. */
  const ownInstance = Array.from(crypto.getRandomValues(new Uint8Array(4)), (b) => b.toString(16).padStart(2, "0")).join("");
  const routes: Record<string, Route> = {
    [GET_ACTION_PATH]: { method: "POST", target: `${gateway}${GET_ACTION_PATH}`, answer: "gateway" },
    [END_SESSION_PATH]: { method: "POST", target: `${gateway}${END_SESSION_PATH}`, answer: "gateway" },
    "/v1/models": { method: "GET", target: `${platform}/v1/models`, answer: "models" },
  };

  /** The route, null for none, or "not_fine_tune" for a training-data path whose id cannot be a fine-tune's. */
  const route = (path: string): Route | "not_fine_tune" | null => {
    if (Object.hasOwn(routes, path)) return routes[path]!;
    const training = TRAINING_DATA.exec(path);
    if (!training) return null;
    let checkpoint: string;
    try {
      checkpoint = decodeURIComponent(training[1]!);
    } catch {
      return null;
    }
    if (!checkpoint.startsWith(FINE_TUNE_PREFIX)) return null;
    // Only a published fine-tune id (`dream-1-ft:<uid8>-<suffix>`): anything else is not looked up with the key.
    if (!FINE_TUNE_ID.test(checkpoint)) return "not_fine_tune";
    return { method: "GET", target: `${data}/v1/fine_tuning/models/${pyQuote(checkpoint)}/training_data`, answer: "training" };
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
    if (hosts) {
      const host = canonicalHost(request.headers.get("host") ?? url.host, url.protocol === "https:" ? "https:" : "http:");
      if (host === null || !hosts[url.protocol === "https:" ? "https:" : "http:"]!.has(host)) return fail(421, "unknown host");
    }
    if (origin !== null && !allowed) return json(403, { proxy_error: "origin not allowed" }, origins.size ? { Vary: "Origin" } : {});
    // A navigation or a no-CORS load from another site carries no Origin; Fetch Metadata names it.
    if (!allowed && request.headers.get("sec-fetch-site") === "cross-site") return json(403, { proxy_error: "cross-site request" }, origins.size ? { Vary: "Origin" } : {});
    if (basePath && url.pathname !== basePath && !url.pathname.startsWith(`${basePath}/`)) return fail(404, "no such route");
    const target = route(url.pathname.slice(basePath.length));
    if (!target) return fail(404, "no such route");
    const method = target === "not_fine_tune" ? "GET" : target.method;
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: { ...cors, "Access-Control-Allow-Methods": `${method}, OPTIONS`, "Access-Control-Allow-Headers": allowHeaders, "Access-Control-Max-Age": "600" },
      });
    }
    if (request.method !== method) return fail(405, "method not allowed", { Allow: `${method}, OPTIONS` });
    // The client always sends Dream-Client-Version through a proxy; a page cannot add it to a navigation, an image or a
    // form, and from another origin only after a preflight the origin check answers.
    if (method === "GET" && !request.headers.has(CLIENT_VERSION_HEADER)) return fail(400, "send Dream-Client-Version, as DreamClient({ proxyUrl }) does");
    if (target === "not_fine_tune") return json(404, { error: { code: "not_found", message: "not a fine-tune id" } }, cors);
    const signal = request.signal;
    // The caller gave up (the page cancelled, the connection closed): nothing more is done or sent on its behalf.
    if (signal.aborted) return fail(499, "cancelled");
    if (authorize) {
      let verdict: boolean | Response;
      try {
        verdict = await untilAborted(Promise.resolve(authorize(new Request(request.url, { method: request.method, headers: request.headers, signal }))), signal, discardBody);
      } catch {
        return signal.aborted ? fail(499, "cancelled") : fail(500, "authorize failed");
      }
      if (verdict instanceof Response) return withHeaders(verdict, cors);
      if (verdict !== true) return fail(401, "unauthorized");
    }
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
    // The caller's cancellation and the deadline end the exchange, answer included, even with a fetch that ignores
    // its signal.
    const exchange = AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)]);
    let reply: Response;
    let payload: Uint8Array<ArrayBuffer> | null;
    try {
      reply = await untilAborted(upstream(target.target, { method: target.method, headers, body, redirect: "manual", signal: exchange }), exchange, discardBody);
      payload = NO_BODY.has(reply.status) || !reply.body ? null : await readAll(reply.body, maxResponseBytes, exchange, 502);
    } catch (error) {
      if (signal.aborted) return fail(499, "cancelled");
      if (error instanceof BodyError) return fail(502, "upstream answer too large");
      const name = error instanceof Error ? error.name : "";
      return name === "TimeoutError" || exchange.aborted ? fail(504, "timeout") : fail(502, "unreachable");
    }
    const status = reply.status >= 200 && reply.status <= 599 ? reply.status : 502;
    if (NO_BODY.has(status)) {
      discardBody(reply);
      return new Response(null, { status, headers: { ...cors, "Cache-Control": "no-store" } });
    }
    // Only an answer of the route's own shape goes back, parsed and written anew, and only when no key or string in
    // it carries the key (see answerFor).
    const answer = answerFor(target.answer, status, payload, secret);
    if (answer === null) return fail(502, "upstream answer withheld");
    return new Response(answer, { status, headers: { ...cors, "Content-Type": "application/json", "Cache-Control": "no-store" } });
  };
}

/** What each route's answer may be, besides the key check (see {@link answerFor}). */
type AnswerShape = "gateway" | "models" | "training";

/** The bounds an answer is checked within; past any of them it is withheld. */
export const ANSWER_LIMITS = {
  /** Nesting, arrays and objects. */
  depth: 32,
  /** Characters of one string value. */
  string: 4096,
  /** Characters of one object key. */
  key: 256,
  /** Values, all told. */
  nodes: 1_000_000,
  /** Characters decoded while looking for the key, all told. */
  decoded: 4 * 1024 * 1024,
} as const;

/** An answer outside {@link ANSWER_LIMITS} or its route's shape. */
class Unfit extends Error {}

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

/**
 * Walks `value` without recursion, within {@link ANSWER_LIMITS}, calling `visit` on every value with its depth and its
 * object key, if any.
 *
 * @throws Unfit A limit is passed.
 */
function walk(value: unknown, visit: (node: unknown, key: string | null) => void): void {
  const stack: [unknown, number, string | null][] = [[value, 0, null]];
  let nodes = 0;
  while (stack.length) {
    const [node, depth, key] = stack.pop()!;
    if (++nodes > ANSWER_LIMITS.nodes || depth > ANSWER_LIMITS.depth) throw new Unfit();
    visit(node, key);
    if (Array.isArray(node)) for (let i = node.length - 1; i >= 0; i--) stack.push([node[i], depth + 1, null]);
    else if (typeof node === "object" && node !== null) for (const [name, item] of Object.entries(node)) stack.push([item, depth + 1, name]);
  }
}

const record = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);
const text = (value: unknown, limit: number = ANSWER_LIMITS.string): value is string => typeof value === "string" && value.length <= limit;
/** A finite number that JSON writes back as itself (an infinity is written as null). */
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

/**
 * JSON of the plain kinds: every string within its limit, every number finite, every key within its limit; arrays and
 * objects within the depth. `strings` false refuses strings and booleans too (an action holds numbers, null, arrays and
 * objects only).
 *
 * @throws Unfit It is not.
 */
function plain(value: unknown, strings: boolean): Json {
  walk(value, (node, key) => {
    if (key !== null && key.length > ANSWER_LIMITS.key) throw new Unfit();
    if (node === null || Array.isArray(node) || record(node) || finite(node)) return;
    if (strings && (typeof node === "boolean" || text(node))) return;
    throw new Unfit();
  });
  return value as Json;
}

/** An optional field of a projection: its value when `test` passes, else left out. */
function optional(from: Record<string, unknown>, out: Record<string, unknown>, key: string, test: (value: unknown) => boolean): void {
  if (Object.hasOwn(from, key) && test(from[key])) out[key] = from[key];
}

/**
 * The answer to pass back, re-serialised, or null to withhold it. It must parse as UTF-8 JSON and fit its route:
 *
 * - get_action and end_session: an object with a `status` envelope whose `code` is a string; of the envelope, `name`,
 *   `reason`, `detail` and `trace_id` are kept when strings or null, `retryable` when a boolean or null and
 *   `retry_after_s` when a finite number or null; `session_id` when a string or null; `actions` must be an object of
 *   numbers, null, arrays and objects only (no strings: these become joint targets), every number finite; of `timing`,
 *   `gateway_ms` and `serve_infer_ms` are kept when finite numbers, and the rest dropped. Nothing else is kept.
 * - models: a list of model objects, each a string `id`, and `owner`, `base_model`, `embodiment`, `created_at` (strings
 *   or null), `servable`, `parts_withheld` (booleans), `cameras` (strings, or null) and `embodiment_config` (plain JSON
 *   or null) when present; another type for one of them withholds the answer.
 * - training_data: `cameras` (per role, the sizes that are two finite numbers) and `prompts` (the strings).
 * - a refusal of the platform or the data service (status 400 or more): `{ error: { code, message } }` with the
 *   strings it has, its status kept; one of another shape is answered `{ error: { code: "upstream_error", message } }`
 *   with the same status, so the client still tells an authentication refusal from the rest.
 *
 * Every string, key, depth and count is within {@link ANSWER_LIMITS}. Then every key and string kept is checked for
 * the key ({@link carriesSecret}).
 */
export function answerFor(shape: AnswerShape, status: number, payload: Uint8Array | null, secret: string): string | null {
  let decoded: unknown;
  try {
    decoded = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(payload ?? new Uint8Array(0)));
  } catch {
    decoded = undefined;
  }
  let kept: Json;
  try {
    kept = project(shape, status, decoded);
    return carriesSecret(kept, secret) ? null : JSON.stringify(kept);
  } catch {
    return null;
  }
}

/** @throws Unfit The answer does not fit its route. */
function project(shape: AnswerShape, status: number, decoded: unknown): Json {
  if (shape !== "gateway" && status >= 400) {
    const error = record(decoded) && record(decoded.error) ? decoded.error : {};
    const out: Record<string, string> = {};
    if (text(error.code, ANSWER_LIMITS.key)) out.code = error.code;
    if (text(error.message)) out.message = error.message;
    return { error: Object.keys(out).length ? out : { code: "upstream_error", message: `the Dream service answered ${status}` } };
  }
  if (decoded === undefined) throw new Unfit();
  if (shape === "gateway") {
    if (!record(decoded) || !record(decoded.status) || !text(decoded.status.code, ANSWER_LIMITS.key)) throw new Unfit();
    const from = decoded.status;
    const envelope: Record<string, unknown> = { code: from.code };
    for (const key of ["name", "reason", "detail", "trace_id"]) optional(from, envelope, key, (v) => v === null || text(v));
    optional(from, envelope, "retryable", (v) => v === null || typeof v === "boolean");
    optional(from, envelope, "retry_after_s", (v) => v === null || finite(v));
    const out: Record<string, unknown> = { status: envelope };
    optional(decoded, out, "session_id", (v) => v === null || text(v, ANSWER_LIMITS.key));
    if (Object.hasOwn(decoded, "actions")) {
      if (!record(decoded.actions)) throw new Unfit();
      out.actions = plain(decoded.actions, false);
    }
    if (record(decoded.timing)) {
      const timing: Record<string, unknown> = {};
      optional(decoded.timing, timing, "gateway_ms", finite);
      optional(decoded.timing, timing, "serve_infer_ms", finite);
      out.timing = timing;
    }
    return out as Json;
  }
  if (shape === "models") {
    if (!Array.isArray(decoded) || decoded.length > 10_000) throw new Unfit();
    const nullableText = (v: unknown) => v === null || text(v);
    const fields: Record<string, (v: unknown) => boolean> = {
      owner: nullableText,
      base_model: nullableText,
      embodiment: nullableText,
      created_at: nullableText,
      servable: (v) => typeof v === "boolean",
      parts_withheld: (v) => typeof v === "boolean",
      cameras: (v) => v === null || (Array.isArray(v) && v.every((c) => text(c, ANSWER_LIMITS.key))),
      embodiment_config: (v) => v === null || record(v),
    };
    return decoded.map((model) => {
      if (!record(model) || !text(model.id, ANSWER_LIMITS.key)) throw new Unfit();
      const out: Record<string, unknown> = { id: model.id };
      for (const [key, test] of Object.entries(fields)) {
        if (!Object.hasOwn(model, key)) continue;
        if (!test(model[key])) throw new Unfit();
        out[key] = key === "embodiment_config" && model[key] !== null ? plain(model[key], true) : model[key];
      }
      return out as Json;
    });
  }
  if (!record(decoded)) throw new Unfit();
  const out: Record<string, Json> = {};
  if (Object.hasOwn(decoded, "cameras")) {
    if (!record(decoded.cameras)) throw new Unfit();
    const cameras: Record<string, Json> = Object.create(null);
    for (const [role, sizes] of Object.entries(decoded.cameras)) {
      if (role.length > ANSWER_LIMITS.key || !Array.isArray(sizes)) throw new Unfit();
      cameras[role] = sizes.filter((size): size is number[] => Array.isArray(size) && size.length === 2 && size.every(finite));
    }
    out.cameras = cameras;
  }
  if (Object.hasOwn(decoded, "prompts")) {
    if (!Array.isArray(decoded.prompts)) throw new Unfit();
    out.prompts = decoded.prompts.filter((prompt): prompt is string => text(prompt));
  }
  return out;
}

/**
 * Whether `value` carries `secret` (and so `Bearer <secret>`), by exact match of that key, in any object key or string,
 * as written or decoded: from `\uXXXX` escapes (JSON text inside a string), from percent-encoding, and from every run of
 * base64, base64url or hex characters, each decoded at every alignment, to two levels. Numbers carry nothing. Any
 * other encoding (a key split across strings, reversed, encrypted) is not detected. The work is bounded: past
 * {@link ANSWER_LIMITS} (`decoded` characters all told) it answers true, and the answer is withheld.
 */
export function carriesSecret(value: unknown, secret: string): boolean {
  let budget = ANSWER_LIMITS.decoded;
  const seen = (found: string, depth: number): boolean => {
    budget -= found.length;
    if (budget < 0) throw new Unfit();
    if (found.includes(secret)) return true;
    if (depth >= 2 || found.length < secret.length) return false;
    for (const decoded of decodings(found)) if (decoded !== found && seen(decoded, depth + 1)) return true;
    return false;
  };
  let carries = false;
  try {
    walk(value, (node, key) => {
      if (carries) return;
      if ((key !== null && seen(key, 0)) || (typeof node === "string" && seen(node, 0))) carries = true;
    });
  } catch {
    return true;
  }
  return carries;
}

/** The texts a string may hide another in: unescaped, percent-decoded, and its base64/base64url/hex runs decoded. */
function decodings(found: string): string[] {
  const out: string[] = [];
  if (found.includes("\\u")) out.push(found.replace(/\\u([0-9a-fA-F]{4})/g, (_, hex: string) => String.fromCharCode(parseInt(hex, 16))));
  if (found.includes("%")) out.push(found.replace(/%([0-9a-fA-F]{2})/g, (_, hex: string) => String.fromCharCode(parseInt(hex, 16))));
  for (const [run] of found.matchAll(/[A-Za-z0-9+/_-]{16,}={0,2}/g)) {
    const standard = run.replace(/-/g, "+").replace(/_/g, "/").replace(/=+$/, "");
    for (let skip = 0; skip < 4 && skip < standard.length; skip++) {
      let part = standard.slice(skip);
      part = part.slice(0, part.length - (part.length % 4 === 1 ? 1 : 0));
      try {
        out.push(atob(part + "=".repeat((4 - (part.length % 4)) % 4)));
      } catch {
        // not base64 at this alignment
      }
    }
  }
  for (const [run] of found.matchAll(/[0-9a-fA-F]{32,}/g)) {
    for (const skip of [0, 1]) {
      let decoded = "";
      for (let i = skip; i + 1 < run.length; i += 2) decoded += String.fromCharCode(parseInt(run.slice(i, i + 2), 16));
      out.push(decoded);
    }
  }
  return out;
}

/** The canonical `host[:port]` of a Host header for `scheme`, or null when it is not one; see `allowedHosts`. */
export function canonicalHost(host: string, scheme: "http:" | "https:"): string | null {
  if (!host || /[\s/?#@\\]/.test(host)) return null;
  let parsed: URL;
  try {
    parsed = new URL(`${scheme}//${host}`);
  } catch {
    return null;
  }
  if (parsed.pathname !== "/" || parsed.username || parsed.password) return null;
  // A trailing dot names the same DNS host: dropped, so `example.com.` and `example.com` match.
  return parsed.host.replace(/\.(?=(:\d+)?$)/, "");
}

/** A Response's body cancelled without waiting: an answer no one will read. */
function discardBody(value: unknown): void {
  if (value instanceof Response && value.body && !value.bodyUsed) void value.body.cancel().catch(() => undefined);
}

/**
 * `promise`, or the signal's reason as soon as `signal` fires; one listener, removed when either settles. A value that
 * arrives after the signal won is handed to `late` (a body to cancel).
 */
function untilAborted<T>(promise: Promise<T>, signal: AbortSignal, late: (value: T) => void = () => undefined): Promise<T> {
  let settled = false;
  promise.then(
    (value) => settled && late(value),
    () => undefined,
  );
  if (signal.aborted) {
    settled = true;
    return Promise.reject(signal.reason);
  }
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => {
      settled = true;
      reject(signal.reason);
    };
    signal.addEventListener("abort", onAbort, { once: true });
    promise
      .then(
        (value) => {
          if (!settled) {
            settled = true;
            resolve(value);
          }
        },
        (error) => {
          if (!settled) {
            settled = true;
            reject(error);
          }
        },
      )
      .finally(() => signal.removeEventListener("abort", onAbort));
  });
}

/** The largest block readAll() starts with; it grows by doubling, never past the limit. */
const FIRST_BLOCK = 64 * 1024;

/**
 * A stream read whole, or a {@link BodyError} with `tooLarge` once it passes `limit` bytes. `signal` cancels the read
 * (its reason is thrown) without waiting for the stream's own cancel to settle. The bytes are kept in one block, grown
 * by doubling (so at most twice what was read), empty chunks skipped; the reader is released on every path.
 */
export async function readAll(stream: ReadableStream<Uint8Array>, limit: number, signal: AbortSignal, tooLarge: number): Promise<Uint8Array<ArrayBuffer>> {
  const reader = stream.getReader();
  let block = new Uint8Array(new ArrayBuffer(Math.min(limit, FIRST_BLOCK)));
  let size = 0;
  // Cancelling the stream ends a pending read at once, whatever its source does.
  const stop = () => void reader.cancel(signal.reason).catch(() => undefined);
  signal.addEventListener("abort", stop, { once: true });
  try {
    if (signal.aborted) throw signal.reason;
    for (;;) {
      const { done, value } = await reader.read();
      if (signal.aborted) throw signal.reason;
      if (done) break;
      if (!(value instanceof Uint8Array)) throw new TypeError("a body chunk is not bytes");
      if (!value.byteLength) continue;
      if (size + value.byteLength > limit) throw new BodyError(tooLarge, "too large");
      if (size + value.byteLength > block.length) {
        const grown = new Uint8Array(new ArrayBuffer(Math.min(limit, Math.max(block.length * 2, size + value.byteLength))));
        grown.set(block.subarray(0, size));
        block = grown;
      }
      block.set(value, size);
      size += value.byteLength;
    }
  } catch (error) {
    void reader.cancel().catch(() => undefined);
    throw error;
  } finally {
    signal.removeEventListener("abort", stop);
    try {
      reader.releaseLock();
    } catch {
      // a runtime that refuses to release with a read pending: the stream is cancelled anyway
    }
  }
  return block.subarray(0, size);
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

/** The largest duration (setTimeout's) and size taken. */
export const PROXY_MAX_OPTION = 2 ** 31 - 1;

/** @throws DreamConfigError `value` is not a whole number in 1..{@link PROXY_MAX_OPTION}. */
function whole(value: unknown, what: string): number {
  if (!Number.isInteger(value) || (value as number) < 1 || (value as number) > PROXY_MAX_OPTION) {
    throw new DreamConfigError(`${what} must be a whole number from 1 to ${PROXY_MAX_OPTION}, got ${JSON.stringify(value)}`);
  }
  return value as number;
}
