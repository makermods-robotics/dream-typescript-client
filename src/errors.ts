/**
 * The errors the client throws (`dream.errors`). Every one derives from {@link DreamError}, so `instanceof DreamError`
 * catches all of them. They fall in three families:
 *
 * - The call was wrong, and making it again gets the same error: {@link DreamConfigError} (the config, the key, a URL),
 *   {@link DreamObservationError} (the observation) and {@link DreamActionError} (the model's reply did not match).
 * - No answer the client could read: {@link DreamTimeout} and {@link DreamProtocolError}. Check the URL and the
 *   network, then make the same call again.
 * - A refusal: {@link DreamServerError} and its subclasses {@link DreamAuthError}, {@link DreamBadRequest} and
 *   {@link DreamSessionExpired}, which carry the answer's `code`, `reason`, `retryable` and `retryAfterS`.
 *
 * And one that is none of these: {@link DreamCancelled}, the call was cancelled (your `signal`, close(), or a newer
 * policy()); never retryable, and not a network error.
 *
 * {@link DreamConfigError} and {@link DreamActionError} carry a machine-readable {@link DreamErrorDetail} (`detail`) for
 * the checks that name a kind, the key and the row.
 *
 * Python's `DreamBadRequest` is both a `DreamConfigError` and a `DreamServerError`; JavaScript has single
 * inheritance, so here it extends `DreamServerError` and `instanceof DreamConfigError` is also true for it.
 */

/** What a check found, for code to read: e.g. `{ kind: "non_finite", key: "per_key.action.right_eef_xyz", row: 3 }`. */
export interface DreamErrorDetail {
  /**
   * `non_finite` (a NaN or an infinity), `shape` (wrong width, length or rows), `range` (out of range: a rate, a
   * cap, a horizon), `degenerate_rotation` (a rotation_6d with a near-zero or parallel column), `unsupported` (a
   * feature this package does not implement), `model_mismatch` (the model was not trained for the arm).
   */
  kind: "non_finite" | "shape" | "range" | "degenerate_rotation" | "unsupported" | "model_mismatch" | (string & {});
  /** The option, observation or action key; null when the check is about the whole value. */
  key: string | null;
  /** The row of a chunk or a target list; null when the value has no rows. */
  row: number | null;
}

/** The detail of a check, `key` and `row` null unless given. */
export function detail(kind: DreamErrorDetail["kind"], key: string | null = null, row: number | null = null): DreamErrorDetail {
  return { kind, key, row };
}

/** Base of every error the client throws; also thrown as itself when a client is used wrongly (closed, or busy). */
export class DreamError extends Error {
  override name = "DreamError";
}

/** Something the caller passed or configured is wrong; the message says what, and the same call fails again. */
export class DreamConfigError extends DreamError {
  override name = "DreamConfigError";
  /** What the check found, when it names it; null otherwise (and on a {@link DreamBadRequest}). */
  readonly detail: DreamErrorDetail | null;

  constructor(message: string, detail: DreamErrorDetail | null = null, options?: ErrorOptions) {
    super(message, options);
    this.detail = detail;
  }

  /** A {@link DreamBadRequest} is a config error too, as in Python. */
  static override [Symbol.hasInstance](value: unknown): boolean {
    return (
      Function.prototype[Symbol.hasInstance].call(this, value) ||
      (this === DreamConfigError && value instanceof DreamBadRequest)
    );
  }
}

/** An observation that cannot be sent; thrown before anything is sent, with the key named in the message. */
export class DreamObservationError extends DreamError {
  override name = "DreamObservationError";
}

/** The model's reply is not an action the client can return; nothing is returned, and the message names the key. */
export class DreamActionError extends DreamError {
  override name = "DreamActionError";
  /** What the check found, when it names it; null otherwise. */
  readonly detail: DreamErrorDetail | null;

  constructor(message: string, detail: DreamErrorDetail | null = null, options?: ErrorOptions) {
    super(message, options);
    this.detail = detail;
  }
}

/**
 * The call was cancelled before it returned: by your `signal`, by close(), or by a newer policy(). Nothing was
 * returned or opened, and no state changed. Not a network error, and never retryable as such: call again if you still
 * want the answer. `cause` is the signal's reason when your signal fired.
 */
export class DreamCancelled extends DreamError {
  override name = "DreamCancelled";
  readonly retryable = false;
}

/** The model did not answer within `timeoutMs`; the request was aborted, so calling again is safe. */
export class DreamTimeout extends DreamError {
  override name = "DreamTimeout";
}

/** No answer the client could read: the host could not be reached, or what answered is not the Dream API. */
export class DreamProtocolError extends DreamError {
  override name = "DreamProtocolError";
}

/** The fields of a refusal. */
export interface DreamServerErrorFields {
  code?: string | null;
  reason?: string | null;
  retryable?: boolean;
  retryAfterS?: number | null;
  /** The status envelope as the gateway sent it (its `trace_id` included), when there was one. */
  envelope?: Record<string, unknown> | null;
}

/**
 * A refusal, carried with the answer's code and reason; whether to call again is `retryable`. When it is true, wait
 * `retryAfterS` seconds (a moment when it is null) and make the same call again.
 */
export class DreamServerError extends DreamError {
  override name = "DreamServerError";
  /** e.g. `"DA-2000"` from getAction(), `"not_found"` from the platform; null when the answer carried none. */
  readonly code: string | null;
  /** e.g. `"pool_warming"`; null when the answer carried none. */
  readonly reason: string | null;
  readonly retryable: boolean;
  readonly retryAfterS: number | null;
  readonly envelope: Record<string, unknown> | null;

  constructor(message: string, fields: DreamServerErrorFields = {}, options?: ErrorOptions) {
    super(message, options);
    this.code = fields.code ?? null;
    this.reason = fields.reason ?? null;
    this.retryable = fields.retryable ?? false;
    this.retryAfterS = fields.retryAfterS ?? null;
    this.envelope = fields.envelope ?? null;
  }

  /** The envelope's `trace_id`, which Dream Labs support needs to look a refusal up; null when absent. */
  get traceId(): string | null {
    const id = this.envelope?.trace_id;
    return typeof id === "string" ? id : null;
  }
}

/** The API key was refused (`DA-1000`, or HTTP 401/403 from the platform); never retryable. */
export class DreamAuthError extends DreamServerError {
  override name = "DreamAuthError";
}

/** The model refused the request as malformed (`DA-4000`); never retryable. Also `instanceof DreamConfigError`. */
export class DreamBadRequest extends DreamServerError {
  override name = "DreamBadRequest";
}

/**
 * The model replaced the session in the middle of an episode (`DA-0001` with a reason other than `new`). The new
 * session is already in use and the action that came with it is not returned: call getAction() again to continue on
 * it, or resetEpisode() to start the episode over.
 */
export class DreamSessionExpired extends DreamServerError {
  override name = "DreamSessionExpired";
}

/**
 * Not an error: the client's recent round trips are slow (`dream.DreamSlowNetworkWarning`). Handed to the client's
 * `onWarning` (console.warn by default), never thrown.
 */
export class DreamSlowNetworkWarning extends Error {
  override name = "DreamSlowNetworkWarning";
}

/**
 * Not an error: the frames or the prompt a fine-tune is sent differ from what it was trained on, or what it was trained
 * on could not be read (`dream.trained_inputs`, a `UserWarning` in Python). Handed to the client's `onWarning`, each
 * line once a policy, never thrown.
 */
export class DreamTrainingWarning extends Error {
  override name = "DreamTrainingWarning";
}

/** What a client hands its `onWarning`. */
export type DreamWarning = DreamSlowNetworkWarning | DreamTrainingWarning;
