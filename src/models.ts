/**
 * Models on the Dream Labs platform (`dream.models`): the checkpoints an account may run, `GET <platform>/v1/models`,
 * plus the JSON call the platform and the data service share (`dream.datasets._call`).
 */

import { API_KEY_ENV, authorization, KEY_PER_SITE } from "./credentials.js";
import { DreamAuthError, DreamConfigError, DreamProtocolError, DreamServerError } from "./errors.js";
import { cancelled, type FetchLike } from "./transport.js";
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
  const headers: Record<string, string> = {
    Accept: "application/json",
    ...(typeof (globalThis as { document?: unknown }).document === "undefined" ? { "User-Agent": `dream-client/${CLIENT_VERSION}` } : {}),
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
    try {
      const reply = await fetchImpl(`${url}${path}`, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
        redirect: "manual",
      });
      status = reply.status;
      text = await reply.text();
    } catch (error) {
      if (signal?.aborted) throw cancelled(signal, `${what} was cancelled`);
      if (attempt + 1 >= attempts) {
        const why = timeout.aborted ? `no answer within ${TIMEOUT_MS / 1000} s` : error instanceof Error ? error.message : String(error);
        throw new DreamProtocolError(`cannot reach the Dream service at ${url}: ${why}`, { cause: error });
      }
      await sleep(backoffS(attempt), signal, what);
      continue;
    }
    if (signal?.aborted) throw cancelled(signal, `${what} was cancelled`);
    if (status >= 500 && attempt + 1 < attempts) {
      await sleep(backoffS(attempt), signal, what);
      continue;
    }
    return answer(status, text, url);
  }
}

function answer(status: number, text: string, url: string): unknown {
  let decoded: unknown = null;
  try {
    decoded = text ? JSON.parse(text) : null;
  } catch (error) {
    throw new DreamProtocolError(`the Dream service at ${url} answered ${status}, not with JSON`, { cause: error });
  }
  if (status < 400) return decoded;
  const envelope = isRecord(decoded) && isRecord(decoded.error) ? decoded.error : null;
  const code = typeof envelope?.code === "string" ? envelope.code : null;
  const message = (typeof envelope?.message === "string" && envelope.message) || `the Dream service answered ${status}`;
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
