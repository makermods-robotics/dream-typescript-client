/**
 * The HTTP channel to the gateway and the JSON wire encoding (`dream.transport`), over `fetch`.
 *
 * DreamAPI is plain request/reply JSON: `POST <gateway>/v1/get_action` per chunk and `POST <gateway>/v1/end_session`
 * to release a session. The HTTP status is not interpreted here: DreamAPI reports every outcome in the body's status
 * envelope, success and refusal alike. A timeout aborts the request, so a late reply can never be read as the answer
 * to the next one.
 */

import { authorization, requestUrl, requireHttps, safeCause } from "./credentials.js";
import { parseRaw, type RawJson } from "./json-raw.js";
import { DreamCancelled, DreamConfigError, DreamError, DreamProtocolError, DreamTimeout, detail } from "./errors.js";
import { CLIENT_VERSION } from "./version.js";

/** The one robot-facing endpoint of DreamAPI. */
export const GET_ACTION_PATH = "/v1/get_action";
/** The gateway's session release: frees the seat now instead of on idle eviction. */
export const END_SESSION_PATH = "/v1/end_session";
/** The header every request carries with the client version (`transport.CLIENT_VERSION_HEADER`). */
export const CLIENT_VERSION_HEADER = "Dream-Client-Version";
/**
 * The header a request names its instance in when `proxyHeaders` is on (8 lowercase hex digits): the proxy builds the
 * `User-Agent` upstream as `dream-client/<version> <instance>`, as the MakerMods console's proxies do.
 */
export const CLIENT_INSTANCE_HEADER = "Dream-Client-Instance";
/** The product every request's `User-Agent` names, `dream-client/<version> <8 hex>`, outside browsers. */
export const USER_AGENT_PRODUCT = "dream-client";

/** What one request cost on the wire, whether or not a reply came back (`transport.Exchange`). */
export interface Exchange {
  rttMs: number;
  requestBytes: number;
  responseBytes: number;
}

/** The `fetch` the client calls: the global one by default. */
export type FetchLike = (input: string, init: RequestInit) => Promise<Response>;

/**
 * Serializes a request body to JSON (`transport.encode`): typed arrays become lists, as numpy arrays do. Unlike
 * `JSON.stringify` it refuses, rather than silently changes, what JSON cannot carry: a non-finite number, `undefined`,
 * a bigint, a function, binary data (`ArrayBuffer`, `Blob`, a `DataView`), a `Map` or a `Set`, a class instance.
 *
 * @throws TypeError naming what cannot be encoded.
 */
export function encode(value: unknown): string {
  return JSON.stringify(wire(value, new Set()));
}

function wire(value: unknown, seen: Set<object>): unknown {
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new TypeError(`cannot encode the number ${value} as JSON`);
    return value;
  }
  if (typeof value !== "object") throw new TypeError(`cannot encode a value of type ${typeof value} for the wire`);
  if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
    if (value instanceof BigInt64Array || value instanceof BigUint64Array) throw new TypeError("cannot encode a bigint array for the wire");
    return Array.from(value as unknown as ArrayLike<number>, (item) => wire(item, seen));
  }
  if (seen.has(value)) throw new TypeError("cannot encode a circular structure for the wire");
  seen.add(value);
  try {
    if (Array.isArray(value)) {
      // Indices, not map(): map() keeps a hole, which JSON then sends as null, a number the model never got.
      const out: unknown[] = [];
      for (let i = 0; i < value.length; i++) {
        if (!(i in value)) throw new TypeError(`cannot encode a sparse array (a hole at index ${i}) for the wire`);
        out.push(wire(value[i], seen));
      }
      return out;
    }
    const proto = Object.getPrototypeOf(value);
    if (proto !== Object.prototype && proto !== null) {
      throw new TypeError(`cannot encode an object of type ${value.constructor?.name ?? "unknown"} for the wire`);
    }
    const out: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value)) out[key] = wire(item, seen);
    return out;
  } finally {
    seen.delete(value);
  }
}

/** Options of a {@link Transport}. */
export interface TransportOptions {
  /** Where the gateway (or your proxy) is: requests go to `${baseUrl}/v1/...`. */
  baseUrl: string;
  /** Per-request timeout, milliseconds. */
  timeoutMs: number;
  /** Sent as `Authorization: Bearer`; leave it out behind a proxy that adds it. */
  apiKey?: string | null;
  fetch?: FetchLike;
  /** Extra headers on every request (a proxy's own auth, say). */
  headers?: Record<string, string>;
  /** The clock, milliseconds; `performance.now` by default. */
  now?: () => number;
  /** `Dream-Client-Version`; {@link CLIENT_VERSION} by default. */
  clientVersion?: string;
  /** Let `apiKey` go to a plain `http://` gateway on another computer: only literally `true`. */
  allowInsecureHttp?: boolean;
  /**
   * `baseUrl` is your proxy, which builds the `User-Agent` upstream: send the instance as `Dream-Client-Instance`
   * instead of in a `User-Agent` (which a browser cannot set anyway).
   */
  proxyHeaders?: boolean;
}

/** True where `User-Agent` cannot be set on a request (a browser page or worker); read when a request is built. */
export function inBrowser(): boolean {
  return typeof (globalThis as { document?: unknown }).document !== "undefined" ||
    typeof (globalThis as { WorkerGlobalScope?: unknown }).WorkerGlobalScope !== "undefined";
}

/** Request/reply channel to one DreamAPI gateway (`transport.Transport`). One request at a time; the client serialises. */
export class Transport {
  readonly baseUrl: string;
  readonly timeoutMs: number;
  /** 8 hex digits drawn once for this channel, the last word of the `User-Agent`; random, so it names no robot. */
  readonly instance: string;
  /** The {@link Exchange} of the most recent request, failed or not; null before the first. */
  lastExchange: Exchange | null = null;
  private closed = false;
  private readonly fetchImpl: FetchLike;
  /** Per request in flight, what ends it: the first call latches its error and aborts the fetch. */
  private readonly active = new Set<(error: DreamError) => void>();
  // A `#` field, so the key it carries never shows in console.log or JSON.stringify of the client.
  readonly #headers: Record<string, string>;
  readonly #secret: string | null;
  readonly #auth: Record<string, string>;
  readonly #allowInsecureHttp: unknown;
  private readonly now: () => number;

  constructor(options: TransportOptions) {
    this.baseUrl = options.baseUrl;
    this.timeoutMs = options.timeoutMs;
    this.fetchImpl = options.fetch ?? ((input, init) => globalThis.fetch(input, init));
    this.now = options.now ?? (() => performance.now());
    this.instance = Array.from(crypto.getRandomValues(new Uint8Array(4)), (b) => b.toString(16).padStart(2, "0")).join("");
    this.#secret = options.apiKey || null;
    // Where the bearer header is attached: the destination it will really go to must be https (or this computer).
    if (this.#secret) requireHttps(this.baseUrl, "baseUrl", options.allowInsecureHttp);
    const version = checkClientVersion(options.clientVersion ?? CLIENT_VERSION);
    this.#headers = {
      "Content-Type": "application/json",
      [CLIENT_VERSION_HEADER]: version,
      ...(options.proxyHeaders ? { [CLIENT_INSTANCE_HEADER]: this.instance } : inBrowser() ? {} : { "User-Agent": `${USER_AGENT_PRODUCT}/${version} ${this.instance}` }),
      ...options.headers,
    };
    this.#auth = options.apiKey ? authorization(options.apiKey) : {};
    this.#allowInsecureHttp = options.allowInsecureHttp;
  }

  /**
   * Posts one JSON body and returns the decoded reply, whatever its shape, each number keeping its source text
   * ({@link parseRaw}); `lastExchange` is set on every path out.
   *
   * @param options.signal Cancels the request; composed with the timeout, and the error says which fired.
   * @throws DreamError The channel is closed.
   * @throws DreamCancelled `signal` fired, or the channel was closed meanwhile; the request was aborted.
   * @throws DreamTimeout No reply within the timeout; the request was aborted.
   * @throws DreamProtocolError The gateway cannot be reached, or answered something that is not JSON.
   * @throws TypeError The payload holds something {@link encode} refuses (nothing was sent).
   */
  async request(payload: unknown, path: string = GET_ACTION_PATH, options: { timeoutMs?: number; signal?: AbortSignal } = {}): Promise<RawJson> {
    if (this.closed) throw new DreamError(`inference request on a closed channel to ${this.baseUrl}`);
    const { timeoutMs = this.timeoutMs, signal } = options;
    if (signal?.aborted) throw cancelled(signal, "the request was cancelled before it was sent");
    const body = encode(payload);
    const requestBytes = new TextEncoder().encode(body).length;
    const abort = new AbortController();
    // The first of the timer, the caller's signal and close() decides what the request ended as, latched here with
    // its cause: a later one (the timer after a cancel, close() after a cancel, a cancel after close()) changes nothing.
    let ended: DreamError | null = null;
    const end = (error: DreamError) => {
      if (ended) return;
      ended = error;
      abort.abort(error);
    };
    const timer = setTimeout(() => end(new DreamTimeout(`inference request to ${this.baseUrl} timed out after ${timeoutMs} ms`)), timeoutMs);
    const onAbort = () => end(cancelled(signal, "the request was cancelled"));
    signal?.addEventListener("abort", onAbort, { once: true });
    this.active.add(end);
    const started = this.now();
    let raw = "";
    let status = 0;
    try {
      // The one URL the request goes to, checked where the key is attached, and given to fetch as it is.
      const url = requestUrl(this.baseUrl, path, "baseUrl", this.#secret !== null, this.#allowInsecureHttp);
      const reply = await this.fetchImpl(url, {
        method: "POST",
        headers: { ...this.#headers, ...this.#auth },
        body,
        signal: abort.signal,
        // A redirect is not followed: the key never goes to another host.
        redirect: "manual",
      });
      status = reply.status;
      raw = await reply.text();
    } catch (error) {
      if (ended) throw this.aborted(ended, error);
      // A fetch that refuses a header names its value: never let the key into a message.
      // One safe copy of the error, its cause read once under a guard (a getter may throw), for the message and the cause.
      const safe = this.scrubbed(error);
      const why = safe.cause instanceof Error ? safe.cause.message : safe.message;
      throw new DreamProtocolError(`could not reach Dream Labs at ${this.baseUrl}: ${why}; check this computer's internet connection, or the baseUrl`, { cause: safe });
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
      this.active.delete(end);
      this.lastExchange = { rttMs: this.now() - started, requestBytes, responseBytes: new TextEncoder().encode(raw).length };
    }
    // A reply read after the abort fired (a fetch that ignores its signal) is not the answer: the caller has moved on.
    if (ended) throw this.aborted(ended);
    try {
      // Read keeping each number's text (an integer a double cannot hold is then seen before it is changed).
      return parseRaw(raw);
    } catch {
      // No cause: the parser's message quotes the reply.
      throw new DreamProtocolError(`inference reply from ${this.baseUrl} is not JSON (HTTP ${status}); is the correct DreamAPI gateway running there?`);
    }
  }

  /** Refuses further requests and aborts the ones in flight (they throw {@link DreamCancelled}). Idempotent. */
  close(): void {
    this.closed = true;
    for (const end of this.active) end(new DreamCancelled("the request was cancelled: the client was closed"));
  }

  /**
   * What an aborted request throws: the error its first abort latched (the timeout, or the caller's or close()'s
   * cancellation), never the state at the time it is read. A timeout keeps the fetch's own error as its cause.
   */
  private aborted(ended: DreamError, error?: unknown): DreamError {
    if (ended instanceof DreamTimeout && error !== undefined && error !== ended) return new DreamTimeout(ended.message, { cause: this.scrubbed(error) });
    return ended;
  }

  /**
   * A new error standing for `error` as a cause: only its name, its message and its `code`, `errno` and `syscall`
   * (strings or numbers), the key cut out, and so for its own causes, three deep. Never the original object, which may
   * carry the request (its headers, the key among them).
   */
  private scrubbed(error: unknown): Error {
    return safeCause(error, this.#secret);
  }
}

/**
 * The {@link DreamCancelled} for an aborted call. An aborted `signal` decides, by the reason its first abort latched:
 * its own DreamCancelled, else the caller's, with that reason as `cause`. `closed` (the client closed meanwhile)
 * names the cause only when the signal never fired.
 */
export function cancelled(signal: AbortSignal | undefined, what: string, closed = false): DreamCancelled {
  if (signal?.aborted) {
    const reason: unknown = signal.reason;
    if (reason instanceof DreamCancelled) return reason;
    return new DreamCancelled(`${what} by the caller's signal`, reason === undefined ? undefined : { cause: reason });
  }
  if (closed) return new DreamCancelled(`${what}: the client was closed`);
  return new DreamCancelled(`${what} by the caller's signal`);
}

/** A `Dream-Client-Version` an HTTP header and a `User-Agent` product token can carry. */
export function checkClientVersion(version: unknown): string {
  if (typeof version !== "string" || !/^[!#$%&'*+.^_`|~0-9A-Za-z-]{1,64}$/.test(version)) {
    throw new DreamConfigError(
      `clientVersion must be 1 to 64 characters of an HTTP token (letters, digits and !#$%&'*+.^_\`|~-), got ${JSON.stringify(version)}`,
      detail("range", "clientVersion"),
    );
  }
  return version;
}
