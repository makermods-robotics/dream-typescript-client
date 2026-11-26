/**
 * Models on the Dream Labs platform (`dream.models`): the checkpoints an account may run, `GET <platform>/v1/models`,
 * plus the JSON call the platform and the data service share (`dream.datasets._call`).
 */

import { API_KEY_ENV, authorization, KEY_PER_SITE, requestUrl, safeMessage, scrub, scrubJson } from "./credentials.js";

export { scrub };
import { DreamAuthError, DreamConfigError, DreamProtocolError, DreamServerError } from "./errors.js";
import { BodyError, cancelled, inBrowser, MAX_REPLY_BYTES, readAll, type FetchLike } from "./transport.js";
import { CLIENT_VERSION } from "./version.js";

/** The public Dream-1 checkpoints, as the platform lists them in 2026-10. */
export const DREAM_1_BETA = "dreamlabs/dream-1-beta";
export const DREAM_1_BETA_FLASH = "dreamlabs/dream-1-beta-flash";
export const PUBLIC_CHECKPOINTS = [DREAM_1_BETA, DREAM_1_BETA_FLASH] as const;
export type PublicCheckpoint = (typeof PUBLIC_CHECKPOINTS)[number];
/** A fine-tuning job's `fine_tuned_model`: `dream-1-ft:<uid8>-<suffix>`. */
export type FineTunedCheckpoint = `dream-1-ft:${string}`;
/** A checkpoint id, what `policy()` takes: a public model, a fine-tune, or any other id the platform lists. */
export type CheckpointId = PublicCheckpoint | FineTunedCheckpoint | (string & {});

/** One checkpoint your account may run (`dream.models.Model`); pass its `id` to `policy()` as `checkpoint`. */
export interface Model {
  id: CheckpointId;
  /** `public` for a checkpoint every account may run, `mine` for one of your fine-tunes. */
  owner: "public" | "mine" | (string & {});
  baseModel: string;
  /** The body it was trained on: `yam` for the Dream-1 base models, an `emb_...` id for a fine-tune; null while not available. */
  embodiment: string | null;
  /** Whether it can be started; a servable model's first call may still be refused `pool_warming` while it starts. */
  servable: boolean;
  createdAt: Date | null;
  /** The camera roles the model reads, each sent as `video.<role>`; null when the platform does not know them. */
  cameras: string[] | null;
  /** For a fine-tune of a custom body, the embodiment config it was trained on; null for a builtin body. */
  embodimentConfig: Record<string, unknown> | null;
  /** True when that body's parts are listed only to the account that registered it. */
  partsWithheld: boolean;
}

/** Where a platform call goes and how. */
export interface ServiceOptions {
  /** The service's URL, or your proxy's; requests go to `${url}/v1/...`. */
  url: string;
  apiKey?: string | null;
  fetch?: FetchLike;
  headers?: Record<string, string>;
  /** Cancels the call, retries and backoff included ({@link DreamCancelled}). */
  signal?: AbortSignal;
  /** The version the `User-Agent` names, `dream-client/<version>`; {@link CLIENT_VERSION} by default. */
  clientVersion?: string;
  /** `url` is your proxy: send the version as `Dream-Client-Version` too, for it to build the `User-Agent` upstream. */
  proxyHeaders?: boolean;
  /** Let `apiKey` go to a plain `http://` service on another computer: only literally `true`. */
  allowInsecureHttp?: boolean;
}

const TIMEOUT_MS = 60_000;
const ATTEMPTS = 8;
const BACKOFF_BASE_S = 0.5;
const BACKOFF_MAX_S = 30.0;

/**
 * One JSON request to the platform or the data service (`datasets._call`); `retry` repeats one that is safe to repeat
 * on a 5xx or a lost connection, with exponential backoff and full jitter.
 */
export async function callService(method: string, path: string, options: ServiceOptions, body?: unknown, retry = false): Promise<unknown> {
  const url = options.url.replace(/\/+$/, "");
  if (!options.apiKey && /(^|\.)dreamlabs\.ai(:\d+)?$/.test(hostOf(url))) {
    throw new DreamConfigError(`no api key: pass apiKey, or set ${API_KEY_ENV} (Node: @makermods/dreamlabs/node loadCredentials())`);
  }
  // The one URL the request goes to, checked where the key is attached, and given to fetch as it is.
  const target = requestUrl(url, path, "the service URL", Boolean(options.apiKey), options.allowInsecureHttp);
  const version = options.clientVersion ?? CLIENT_VERSION;
  const headers: Record<string, string> = {
    Accept: "application/json",
    // A page or a worker cannot set it, and behind a proxy the proxy names the client.
    ...(inBrowser() || options.proxyHeaders ? {} : { "User-Agent": `dream-client/${version}` }),
    ...(options.proxyHeaders ? { "Dream-Client-Version": version } : {}),
    ...(options.apiKey ? authorization(options.apiKey) : {}),
    ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    ...options.headers,
  };
  const fetchImpl = options.fetch ?? ((input: string, init: RequestInit) => globalThis.fetch(input, init));
  const attempts = retry ? ATTEMPTS : 1;
  const { signal } = options;
  const what = `${method} ${path}`;
  if (signal?.aborted) throw cancelled(signal, `${what} was cancelled`);
  for (let attempt = 0; ; attempt++) {
    let status: number;
    let text: string;
    const timeout = AbortSignal.timeout(TIMEOUT_MS);
    const ends = signal ? AbortSignal.any([signal, timeout]) : timeout;
    try {
      const reply = await fetchImpl(target, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: ends,
        redirect: "manual",
      });
      status = reply.status;
      // Counted as it streams and cancelled at the limit: an answer with no Content-Length is never buffered whole.
      text = reply.body ? new TextDecoder().decode(await readAll(reply.body, MAX_REPLY_BYTES, ends, 0)) : "";
    } catch (error) {
      if (signal?.aborted) throw cancelled(signal, `${what} was cancelled`);
      // Not retried: the same service would answer the same.
      if (error instanceof BodyError) throw new DreamProtocolError(`the Dream service at ${url} answered more than ${MAX_REPLY_BYTES} bytes`);
      if (attempt + 1 >= attempts) {
        // The guarded formatter: a message getter or a string conversion that throws never escapes.
        const why = timeout.aborted ? `no answer within ${TIMEOUT_MS / 1000} s` : safeMessage(error);
        throw new DreamProtocolError(`cannot reach the Dream service at ${url}: ${scrub(why, options.apiKey)}`);
      }
      await sleep(backoffS(attempt), signal, what);
      continue;
    }
    if (signal?.aborted) throw cancelled(signal, `${what} was cancelled`);
    if (status >= 500 && attempt + 1 < attempts) {
      await sleep(backoffS(attempt), signal, what);
      continue;
    }
    return answer(status, text, url, options.apiKey);
  }
}


function answer(status: number, text: string, url: string, secret: string | null | undefined): unknown {
  let decoded: unknown = null;
  try {
    // Every key and string of the answer without the key: a service that echoes the request must not hand it to a
    // field of an error, a model record or a log.
    decoded = text ? scrubJson(JSON.parse(text), secret) : null;
  } catch {
    // No cause: the parser's message quotes the answer.
    throw new DreamProtocolError(`the Dream service at ${url} answered ${status}, not with JSON`);
  }
  if (status < 400) return decoded;
  const envelope = isRecord(decoded) && isRecord(decoded.error) ? decoded.error : null;
  const code = typeof envelope?.code === "string" ? envelope.code : null;
  const message = scrub((typeof envelope?.message === "string" && envelope.message) || `the Dream service answered ${status}`, secret);
  if (status === 401 || status === 403) throw new DreamAuthError(`${message}; called ${url}; ${KEY_PER_SITE}`, { code });
  throw new DreamServerError(message, { code, retryable: status >= 500 });
}

/** The checkpoints your account may run, the public ones first (`dream.models.list_models`). */
export async function listModels(options: ServiceOptions): Promise<Model[]> {
  const listed = await callService("GET", "/v1/models", options, undefined, true);
  if (!Array.isArray(listed)) throw new DreamProtocolError(`the Dream service answered ${typeof listed}, not a JSON list`);
  return listed.map(toModel);
}

/** One model as the platform answers it (`models._model`). */
export function toModel(found: unknown): Model {
  if (isRecord(found) && "id" in found && !("cameras" in found)) {
    throw new DreamProtocolError(
      `the Dream platform answered model '${String(found.id)}' without its camera roles ('cameras'): the platform is older than this library`,
    );
  }
  const fail = (why: string): never => {
    throw new DreamProtocolError(`the Dream service answered a model the client cannot read (${why}); is the URL the Dream service's?`);
  };
  if (!isRecord(found)) return fail("not an object");
  for (const key of ["id", "owner", "base_model", "embodiment", "servable", "cameras"]) if (!(key in found)) fail(`no '${key}'`);
  const createdAt = found.created_at == null ? null : new Date(String(found.created_at));
  if (createdAt && Number.isNaN(createdAt.getTime())) fail(`created_at ${JSON.stringify(found.created_at)}`);
  return {
    id: String(found.id),
    owner: String(found.owner),
    baseModel: String(found.base_model),
    embodiment: found.embodiment == null ? null : String(found.embodiment),
    servable: Boolean(found.servable),
    createdAt,
    cameras: found.cameras == null ? null : (found.cameras as unknown[]).map(String),
    embodimentConfig: isRecord(found.embodiment_config) ? found.embodiment_config : null,
    partsWithheld: found.parts_withheld === true,
  };
}

/** One checkpoint's record, for a client that must know what it was trained on (`models._record`). */
export async function modelRecord(checkpoint: string, options: ServiceOptions): Promise<Model> {
  const listed = await listModels(options);
  const found = listed.find((model) => model.id === checkpoint);
  if (!found) {
    throw new DreamConfigError(
      `checkpoint '${checkpoint}' is not a model your key may run; the Models page and getModels() list those: ${JSON.stringify(listed.map((m) => m.id))}`,
    );
  }
  if (found.embodiment === null) throw new DreamConfigError(`checkpoint '${checkpoint}' is not available to run yet: it has no embodiment`);
  return found;
}

const backoffS = (attempt: number) => Math.random() * Math.min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2 ** attempt);
/** Waits `s` seconds, or throws {@link DreamCancelled} as soon as `signal` fires. */
function sleep(s: number, signal: AbortSignal | undefined, what: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const done = () => {
      clearTimeout(timer);
      reject(cancelled(signal, `${what} was cancelled`));
    };
    const timer = setTimeout(() => (signal?.removeEventListener("abort", done), resolve()), s * 1000);
    signal?.addEventListener("abort", done, { once: true });
  });
}

function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return "";
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
