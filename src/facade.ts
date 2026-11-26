/**
 * DreamClient (`dream.facade.DreamClient`): the one object for running a Dream model.
 *
 * Running a model is three calls: policy() chooses the checkpoint and starts an empty frame window, setObservation()
 * hands the client every camera tick (one frame per camera and the latest state), and getAction() sends the frames
 * the window kept with the newest state and returns the next chunk of actions.
 *
 * Where the services are is the constructor's `baseUrl` (the gateway, `DREAM_GATEWAY_URL`), `platformUrl` (the
 * platform, `DREAM_API_URL`) and `dataUrl` (the data service, `DREAM_DATA_URL`), production by default. In a browser,
 * point all three at your own server with `proxyUrl` and leave `apiKey` out: your server adds the key, which never
 * reaches the page (`@makermods/dreamlabs/proxy`).
 */

import {
  ACTION_HZ,
  EXECUTION_HORIZON,
  SYNC_ABOVE_MS,
  type PacingMode,
} from "./pacing.js";
import {
  checkIkLead,
  checkProcessing,
  checkTimeout,
  PolicyClient,
  ROBOTS,
  type ActionDict,
  type Commanded,
  type Diagnostics,
  type GetActionOptions,
  type IkLead,
  type PolicyIk,
  type Processing,
  type Robot,
  type Timing,
} from "./client.js";
import { authorization, checkServiceUrl, safeMessage, DEFAULT_DATA_URL, DEFAULT_GATEWAY_URL, DEFAULT_PLATFORM_URL, requireHttps } from "./credentials.js";
import { parseEmbodiment, type EmbodimentConfig, type ShapeSpec } from "./embodiment.js";
import { DreamCallbackWarning, DreamCancelled, DreamConfigError, DreamError, DreamProtocolError, DreamTrainingWarning, detail, type DreamWarning } from "./errors.js";
import { listModels, scrub, type CheckpointId, type Model, type ServiceOptions } from "./models.js";
import { FINE_TUNE_PREFIX, fetchTrainedInputs, trainedNotes, type TrainedInputs } from "./trained-inputs.js";
import type { Kinematics } from "./seam-join.js";
import { ARM_POSE_KEYS } from "./smoothing.js";
import { cancelled, checkClientVersion, type FetchLike } from "./transport.js";
import { OBS_FPS, OBS_WINDOW, ObservationWindow, pyList } from "./window.js";
import type { IkConfig } from "./ik/config.js";

/** Options of a {@link DreamClient}. */
export interface DreamClientOptions {
  /**
   * Your API key, sent as `Authorization: Bearer`. Never ship one to a browser: leave it out and point `baseUrl` and
   * `platformUrl` at a server of yours that adds it. In Node, `@makermods/dreamlabs/node` reads it from
   * `~/.dreamlabs/credentials` and `DREAM_API_KEY`.
   */
  apiKey?: string | null;
  /** The gateway (`DREAM_GATEWAY_URL`), or your proxy for it, e.g. `/api/dream`; requests go to `${baseUrl}/v1/...`. */
  baseUrl?: string;
  /** The platform (`DREAM_API_URL`), or your proxy for it; getModels() calls `${platformUrl}/v1/models`. */
  platformUrl?: string;
  /**
   * The data service (`DREAM_DATA_URL`), or your proxy for it: policy() reads what one of your fine-tunes was trained
   * on from `${dataUrl}/v1/fine_tuning/models/<id>/training_data`.
   */
  dataUrl?: string;
  /**
   * Your proxy for all three services (`@makermods/dreamlabs/proxy`'s createDreamProxy(), mounted at this URL), e.g.
   * `/api/dream`: the default of `baseUrl`, `platformUrl` and `dataUrl`, and turns `proxyHeaders` on. The proxy adds
   * the key, so `apiKey` is refused with it.
   */
  proxyUrl?: string;
  /** The `fetch` to call; the global one by default. */
  fetch?: FetchLike;
  /** Extra headers on every request (your proxy's own auth, say). */
  headers?: Record<string, string>;
  /**
   * Receives a {@link DreamSlowNetworkWarning} when the round trips stay slow, and a {@link DreamTrainingWarning} for
   * each way a fine-tune's frames or prompt differ from what it was trained on; `console.warn` by default.
   */
  onWarning?: (warning: DreamWarning) => void;
  /** Receives the {@link Diagnostics} of every successful getAction() (the raw reply, the IK, the timings). */
  onDiagnostics?: (diagnostics: Diagnostics) => void;
  /**
   * `Dream-Client-Version` (and the `User-Agent` product's version); {@link CLIENT_VERSION} by default, which is
   * deliberately not a semver: a semver no older than the first parts-converting library would let the gateway serve
   * this client checkpoints whose robot has parts, which it cannot convert.
   */
  clientVersion?: string;
  /**
   * `baseUrl` is your proxy, which builds the gateway's `User-Agent` (`dream-client/<version> <instance>`) from what
   * the client sends: `Dream-Client-Version`, and `Dream-Client-Instance`, 8 random hex digits per policy. Turn it on
   * behind a proxy that forwards both, as the MakerMods console's do; a browser cannot set `User-Agent` itself. Always
   * on with `proxyUrl`; set it when you point `baseUrl`, `platformUrl` or `dataUrl` at a createDreamProxy() one by one,
   * which refuses a GET without `Dream-Client-Version`.
   */
  proxyHeaders?: boolean;
  /** The clock, milliseconds on a monotonic scale; `performance.now` by default (tests replace it). */
  now?: () => number;
  /**
   * Let `apiKey` go to a plain `http://` service on another computer, on a network you trust. Off by default: with a
   * key, every absolute service URL must be `https://` or this computer's (`localhost`, 127.0.0.0/8, `[::1]`).
   */
  allowInsecureHttp?: boolean;
}

/** The config of {@link DreamClient.policy}; Python's snake_case keys in camelCase. */
export interface PolicyConfig {
  /** The model to run: a {@link Model}'s `id` from getModels(), or a fine-tuning job's `fine_tuned_model`. */
  checkpoint: CheckpointId;
  /** What strict mode checks against: the body's embodiment config, or a shape spec. Loaded for a custom body's fine-tune. */
  embodiment?: EmbodimentConfig | ShapeSpec;
  /** Check observations and replies against `embodiment` (default true). */
  strict?: boolean;
  /** How long one call waits for the model, milliseconds (default 15000). */
  timeoutMs?: number;
  /** Frames kept per frame key (default 9). */
  obsWindow?: number;
  /** Frames kept per second of setObservation()'s timestamps (default 5). */
  obsFps?: number;
  /** Rows of each chunk the loop runs before the next takes over, in async (default 30). */
  executionHorizon?: number;
  /** The loop's row rate, rows per second (default 20). */
  actionHz?: number;
  /** Median round trip, milliseconds, past which the pacing is sync (default 700). */
  syncAboveMs?: number;
  /** Per chunk key, the kinematics of the arm it drives: its seams are joined on the end effector's pose. */
  kinematics?: Readonly<Record<string, Kinematics>>;
  /** The robot the model runs on, as a model trained on YAM arms reads it; sent as the observation's `embodiment`. */
  robot?: Robot;
  /** Client-side IK for a model that answers hand poses: `{ arm: "metal", head: "eef" }` (`@makermods/dreamlabs/ik`). */
  ik?: IkConfig;
  /**
   * Not in Python: the platform's record of the checkpoint (from getModels(), perhaps fetched by your server), which
   * skips the platform lookup policy() otherwise makes without `embodiment`, and for `ik`.
   */
  model?: Model;
  /** Not in Python: `default` (IK, seam join, final projection) or `raw` (the checked reply only); see {@link Processing}. */
  processing?: Processing;
  /**
   * Not in Python: with `ik`, the IK solve time the seam adds to its time in flight, since the loop runs on while the
   * client solves: `auto` (default, from the last solves' `ikMs`), milliseconds, or 0 for the round trip only.
   */
  ikLeadMs?: IkLead;
}

const CONFIG_KEYS = new Set([
  "checkpoint", "embodiment", "strict", "timeoutMs", "obsWindow", "obsFps", "executionHorizon", "actionHz",
  "syncAboveMs", "kinematics", "robot", "ik", "model", "processing",
  "ikLeadMs",
]);

/**
 * One getAction(): what the model was sent and what came back (`dream.RoundTrip`); {@link DreamClient.onRoundTrip}
 * hands it over. Frozen.
 */
export interface RoundTrip {
  /** When the request went out, seconds since the epoch (`Date.now() / 1000`). */
  readonly sentS: number;
  /** Per frame key (`video.<role>`), the base64 JPEGs sent, oldest first: the frames the model saw. */
  readonly frames: Readonly<Record<string, readonly string[]>>;
  /** Every other key sent: the state and the prompt, as setObservation() was given them. */
  readonly observation: Readonly<Record<string, unknown>>;
  /** What getAction() returned; null when it threw. */
  readonly action: ActionDict | null;
  /** {@link DreamClient.lastTiming} after the call; null when nothing has reached the network yet. */
  readonly timing: Timing | null;
  /** What getAction() threw, a {@link DreamError}; null when it returned. */
  readonly error: DreamError | null;
  /**
   * Where the episode's frames or prompt differ from what the fine-tune was trained on, one line each; empty when they
   * match, and for a model that is not one of your fine-tunes.
   */
  readonly trained: readonly string[];
}

/** One client for models and inference, with the key bound once (`dream.DreamClient`). */
export class DreamClient {
  readonly baseUrl: string;
  readonly platformUrl: string;
  readonly dataUrl: string;
  // `#` fields: unlike TypeScript's `private`, they never show in console.log(client) or JSON.stringify(client).
  readonly #apiKey: string | null;
  readonly #options: Omit<DreamClientOptions, "apiKey">;
  private policyClient: PolicyClient | null = null;
  private window: ObservationWindow | null = null;
  private readonly embodimentConfigs = new Map<string, Record<string, unknown> | null>();
  /**
   * The lifecycle: bumped, synchronously, by close() and by every policy() that gets past its checks. A policy() whose
   * generation is no longer current after an await stops there, opening nothing ({@link DreamCancelled}).
   */
  #generation = 0;
  /** Aborts the network work of the policy() in progress (its model lookup) when it is superseded or closed. */
  #pending: AbortController | null = null;
  #closing: Promise<void> | null = null;
  #releasing = new Set<Promise<void>>();
  readonly #proxyHeaders: boolean;
  /** One entry per registration (a function added twice is two), removed by identity. */
  readonly #roundTripFns: { fn: (trip: RoundTrip) => unknown }[] = [];
  /** Per fine-tune, what it was trained on (null: not one of yours), read once per checkpoint. */
  readonly #trainedFor = new Map<string, TrainedInputs | null>();
  /** The open policy's fine-tune and what it was trained on; null for any other model. */
  #trained: { checkpoint: string; inputs: TrainedInputs } | null = null;
  /** The episode's lines, from its first getAction(); null until then. */
  #trainedNotes: string[] | null = null;
  /** The lines the open policy has warned about, each once. */
  #trainedWarned = new Set<string>();

  /** Names the class only; never the key. */
  toString(): string {
    return "DreamClient";
  }

  /** Binds the key and the URLs; nothing is sent. */
  constructor(options: DreamClientOptions = {}) {
    const { apiKey, ...rest } = options;
    this.#options = rest;
    this.#apiKey = apiKey ?? null;
    const proxy = options.proxyUrl !== undefined ? checkServiceUrl(options.proxyUrl, "proxyUrl", true) : null;
    if (proxy !== null && this.#apiKey) throw new DreamConfigError("proxyUrl is your proxy, which adds the API key: leave apiKey out", detail("unsupported", "apiKey"));
    // A proxy requires the client's marker (Dream-Client-Version) on every request: with proxyUrl it is always sent.
    if (proxy !== null && options.proxyHeaders === false) {
      throw new DreamConfigError("proxyUrl sends the proxy headers the proxy requires: leave proxyHeaders out (or true)", detail("unsupported", "proxyHeaders"));
    }
    if (this.#apiKey) authorization(this.#apiKey);
    this.baseUrl = checkServiceUrl(options.baseUrl ?? proxy ?? DEFAULT_GATEWAY_URL, "baseUrl", true);
    this.platformUrl = checkServiceUrl(options.platformUrl ?? proxy ?? DEFAULT_PLATFORM_URL, "platformUrl", true);
    this.dataUrl = checkServiceUrl(options.dataUrl ?? proxy ?? DEFAULT_DATA_URL, "dataUrl", true);
    if (this.#apiKey) {
      for (const [what, url] of [["baseUrl", this.baseUrl], ["platformUrl", this.platformUrl], ["dataUrl", this.dataUrl]] as const) requireHttps(url, what, options.allowInsecureHttp);
    }
    this.#proxyHeaders = proxy !== null || (options.proxyHeaders ?? false);
    if (options.clientVersion !== undefined) checkClientVersion(options.clientVersion);
  }

  /**
   * The checkpoints the account may run: the public ones and its own fine-tunes (`GET /v1/models`).
   *
   * @param options.signal Cancels the call, its retries included ({@link DreamCancelled}).
   */
  async getModels(options: { signal?: AbortSignal } = {}): Promise<Model[]> {
    return listModels(this.service(this.platformUrl, options.signal));
  }

  /**
   * Chooses the model and opens an empty frame window; nothing is sent to the model until the first getAction().
   * Opening a second policy closes the first, which ends its session and empties its window.
   *
   * For a fine-tune of a custom body, when `config` names no `embodiment` and `strict` is on, the embodiment config
   * the body was trained on is loaded from getModels() (or `config.model`), and every getAction() then checks the
   * observation before anything is sent and the reply before it is returned.
   *
   * A checkpoint this package cannot run (a robot with parts) is refused whenever its record is known (looked up, or
   * `config.model`), whatever `strict` says; otherwise the gateway refuses it to this client's version
   * ({@link CLIENT_VERSION}). One policy() at a time: a newer call, or close(), cancels one still in progress, which
   * then opens nothing and throws {@link DreamCancelled}.
   *
   * @param options.signal Cancels this call while it is in progress (the model lookup included).
   * @throws DreamConfigError A key is missing, of the wrong type or range, or not one the client takes; `ik` cannot
   *   be served, or not on this model; the checkpoint is not one getModels() lists (when it is looked up), or its robot
   *   has parts.
   * @throws DreamProtocolError The model lookup failed; nothing opened, and a policy already open stays open.
   * @throws DreamCancelled `signal` fired, close() was called, or a newer policy() started, before this one opened.
   */
  async policy(config: PolicyConfig, options: { signal?: AbortSignal } = {}): Promise<void> {
    if (typeof config !== "object" || config === null) throw new DreamConfigError(`policy() takes a config object, got ${String(config)}`);
    const unknown = Object.keys(config).filter((key) => !CONFIG_KEYS.has(key)).sort();
    if (unknown.length) {
      throw new DreamConfigError(
        `config keys ${pyList(unknown)} are not taken; policy() takes ${pyList([...CONFIG_KEYS])} (where the model is comes from the client's baseUrl)`,
      );
    }
    const obsWindow = config.obsWindow ?? OBS_WINDOW;
    const obsFps = config.obsFps ?? OBS_FPS;
    if (!Number.isInteger(obsWindow) || obsWindow < 1) throw new DreamConfigError(`config key 'obsWindow' must be an int of frames, at least 1, got ${JSON.stringify(obsWindow)}`);
    if (typeof obsFps !== "number" || !(obsFps > 0) || !Number.isFinite(obsFps)) {
      throw new DreamConfigError(`config key 'obsFps' must be a positive, finite number of frames per second, got ${JSON.stringify(obsFps)}`);
    }
    if (config.timeoutMs !== undefined) checkTimeout(config.timeoutMs);
    if (config.strict !== undefined && typeof config.strict !== "boolean") throw new DreamConfigError(`config key 'strict' must be a bool, got ${JSON.stringify(config.strict)}`);
    checkPacing(config);
    checkKinematics(config.kinematics);
    if (config.robot !== undefined && !ROBOTS.includes(config.robot)) {
      throw new DreamConfigError(`config key 'robot' must be one of ${pyList([...ROBOTS])}, got ${JSON.stringify(config.robot)}`);
    }
    const checkpoint = config.checkpoint;
    if (typeof checkpoint !== "string" || !checkpoint) {
      throw new DreamConfigError(`config key 'checkpoint' must be a non-empty str naming the checkpoint, got ${JSON.stringify(checkpoint)}`);
    }
    const processing = checkProcessing(config.processing ?? "default", config.ik !== undefined, config.kinematics !== undefined);
    const ikLeadMs = checkIkLead(config.ikLeadMs ?? "auto");
    if (options.signal?.aborted) throw cancelled(options.signal, "policy() was cancelled");
    // From here this call is the current one: an older policy() still in progress stops at its next await.
    const generation = ++this.#generation;
    this.#pending?.abort(new DreamCancelled("policy() was cancelled: a newer policy() call started"));
    const controller = new AbortController();
    this.#pending = controller;
    const signal = options.signal ? AbortSignal.any([controller.signal, options.signal]) : controller.signal;
    const current = () => {
      if (generation !== this.#generation || signal.aborted) throw cancelled(signal, "policy() was cancelled", generation !== this.#generation);
    };
    try {
      const ikModule = config.ik !== undefined ? await import("./ik/client-ik.js") : null;
      current();
      ikModule?.checkIkConfig(config.ik);
      const strict = config.strict ?? true;
      let models: Model[] | null = config.model ? [config.model] : null;
      const lookup = async (): Promise<Model[]> => (models ??= await this.getModels({ signal }));
      let embodiment: unknown = config.embodiment;
      if (embodiment === undefined && strict) {
        const loaded = await this.embodimentConfigFor(checkpoint, lookup, config.model !== undefined);
        current();
        if (loaded !== null) embodiment = loaded;
      }
      const trained = await this.trainedInputsFor(checkpoint, signal);
      current();
      const actionHz = config.actionHz ?? ACTION_HZ;
      let model: Model | undefined;
      if (ikModule && config.ik) {
        model = (await lookup()).find((m) => m.id === checkpoint);
        current();
      } else if (config.model?.id === checkpoint) model = config.model;
      // Whatever `strict` says: a checkpoint this package cannot run is refused whenever its record is known.
      if (model) refuseUnsupported(model);
      // As in Python: from here on, the policy already open is closed, whether or not the new one opens.
      await this.closePolicy();
      current();
      let ik: PolicyIk | null = null;
      if (ikModule && config.ik) {
        if (!model) {
          throw new DreamConfigError(`checkpoint '${checkpoint}' is not a model your key may run; the Models page and getModels() list those`);
        }
        if (model.embodiment === null) throw new DreamConfigError(`checkpoint '${checkpoint}' is not available to run yet: it has no embodiment`);
        ik = new ikModule.ClientIK(config.ik, actionHz, embodiment, model);
      }
      const next = new PolicyClient({
        checkpoint,
        baseUrl: this.baseUrl,
        apiKey: this.#apiKey,
        timeoutMs: config.timeoutMs,
        embodiment,
        strict,
        robot: config.robot,
        executionHorizon: config.executionHorizon,
        actionHz,
        syncAboveMs: config.syncAboveMs,
        kinematics: config.kinematics,
        ik,
        processing,
        ikLeadMs,
        fetch: this.#options.fetch,
        headers: this.#options.headers,
        clientVersion: this.#options.clientVersion,
        proxyHeaders: this.#proxyHeaders,
        allowInsecureHttp: this.#options.allowInsecureHttp,
        onWarning: this.#options.onWarning,
        onDiagnostics: this.#options.onDiagnostics,
        now: this.#options.now,
      });
      // Nothing awaits between the check above and here; the policy opens only as the current call.
      this.policyClient = next;
      this.window = new ObservationWindow(obsWindow, 1 / obsFps);
      this.#trained = trained;
      this.#trainedNotes = null;
      this.#trainedWarned = new Set();
    } finally {
      if (this.#pending === controller) this.#pending = null;
    }
  }

  /**
   * Hands the client one camera tick: its frames go into the window when due, its observation replaces the last.
   * Call it on every camera tick; it never waits on the network.
   *
   * @param frames Per window key (`video.<role>`), one frame: a non-empty base64 JPEG string, resized and encoded by you.
   * @param observation Every other key the checkpoint reads (`state.<part>`, `prompt`); never `embodiment`.
   * @param options.timestamp When the frames were captured, seconds on a monotonic clock; now when left out.
   * @throws DreamConfigError No policy is open.
   * @throws DreamObservationError The tick is not valid; the message names the key, and the window is unchanged.
   */
  setObservation(frames: Readonly<Record<string, string>>, observation: Readonly<Record<string, unknown>>, options: { timestamp?: number } = {}): void {
    const window = this.window;
    if (!this.policyClient || !window) throw noPolicy("setObservation()");
    window.add(frames, observation, options.timestamp ?? this.now() / 1000);
  }

  /**
   * Sends the window and the latest observation to the model and returns its action chunk. One call at a time.
   *
   * @param options.commanded Per chunk key (`q_r_chunk`, `q_l_chunk`), the joints your loop last commanded that arm
   *   before the observation; never sent. With them each chunk takes over where the loop is, not where the arm trails.
   * @param options.signal Cancels the call; composed with `timeoutMs` (a timeout is a {@link DreamTimeout}, your signal a
   *   {@link DreamCancelled}).
   * @param options.execution The loop's own timing for the seam: `observationAgeMs`, `elapsedExecutionS`.
   * @returns The action dict ({@link ActionDict}), each arm's chunk joined onto the previous one; with `ik`, every row
   *   within the arm's limits and speed caps (the final projection); with `processing: "raw"`, the checked reply.
   * @throws DreamConfigError No policy is open.
   * @throws DreamObservationError Before anything is sent: no setObservation() this episode, the observation carries
   *   `embodiment`, does not match the embodiment (strict), or holds a value JSON cannot carry.
   * @throws DreamActionError The reply is not a usable action.
   * @throws DreamTimeout No answer within `timeoutMs`; calling again is safe.
   * @throws DreamProtocolError The model could not be reached, or what answered is not the Dream API.
   * @throws DreamAuthError The key was refused (`DA-1000`); never retryable.
   * @throws DreamBadRequest The request was refused as malformed (`DA-4000`); never retryable.
   * @throws DreamSessionExpired The model replaced the session mid-episode; call again to continue on the new one.
   * @throws DreamServerError Any other refusal; when `retryable`, wait `retryAfterS` and call again.
   * @throws DreamCancelled `signal` fired, or the policy was closed or replaced while the call was in flight: nothing
   *   is returned, even a reply that had already arrived.
   * @throws DreamError The client is closed, or another getAction() is in flight.
   */
  async getAction(options: GetActionOptions = {}): Promise<ActionDict> {
    const policy = this.policyClient;
    const window = this.window;
    if (!policy || !window) throw noPolicy("getAction()");
    const sent = window.buildObservation();
    const frameKeys = window.frameKeys();
    this.#trainedNotes ??= this.heldToTraining(sent, frameKeys);
    const trained = this.#trainedNotes;
    const sentS = Date.now() / 1000;
    let action: ActionDict;
    try {
      action = await policy.getAction(sent, options);
      // The policy closes (and aborts the call) when replaced; this holds even for a fetch that ignores its signal.
      if (this.policyClient !== policy) throw new DreamCancelled("getAction() was cancelled: its policy was closed or replaced meanwhile");
    } catch (error) {
      if (error instanceof DreamError) this.handOver(roundTrip(sentS, sent, frameKeys, null, policy.lastTiming, error, trained));
      throw error;
    }
    this.handOver(roundTrip(sentS, sent, frameKeys, action, policy.lastTiming, null, trained));
    // A callback may have closed or replaced the policy, or fired the caller's signal: the action is no one's answer.
    if (this.policyClient !== policy) throw new DreamCancelled("getAction() was cancelled: its policy was closed or replaced meanwhile");
    if (options.signal?.aborted) throw cancelled(options.signal, "getAction() was cancelled");
    return action;
  }

  /**
   * Calls `fn` after every getAction() that built its observation from the window, with what the model was sent and
   * what came back, a {@link RoundTrip}, whether the call returned or threw, a refusal before anything was sent
   * included (`DreamClient.on_round_trip`): for a page of what the model sees, or your own logs.
   * Each `fn` runs synchronously, in the order added, before getAction() returns; as in Python, what `fn` throws is
   * thrown out of getAction() (after the call's own outcome is settled), so keep it from throwing. A promise `fn`
   * returns is not waited for: its rejection is handed to `onWarning` as a {@link DreamCallbackWarning}.
   *
   * @returns Not in Python: a function that removes this registration of `fn` (another registration of the same
   *   function stays); calling it again does nothing.
   */
  onRoundTrip(fn: (trip: RoundTrip) => unknown): () => void {
    if (typeof fn !== "function") throw new DreamConfigError(`onRoundTrip() takes a function, got ${typeof fn}`);
    const registration = { fn };
    this.#roundTripFns.push(registration);
    return () => {
      const at = this.#roundTripFns.indexOf(registration);
      if (at >= 0) this.#roundTripFns.splice(at, 1);
    };
  }

  /**
   * Waits until the IK policy() built (its `ik` key) is ready to run every solve (`DreamClient.wait_for_ik`). Python
   * compiles its `traj` solve on a thread for up to 20 s, so a robot program waits before it powers arms that would
   * starve meanwhile; this package has nothing to compile, so it is ready at once.
   *
   * @param timeoutS Seconds to wait at most.
   * @returns True: ready (also without `ik`).
   * @throws DreamConfigError No policy is open.
   */
  async waitForIk(timeoutS: number): Promise<boolean> {
    void timeoutS;
    if (!this.policyClient) throw noPolicy("waitForIk()");
    return true;
  }

  /**
   * Not in Python: `action` re-anchored at the row the loop installs it at; see {@link PolicyClient.takeoverAt}. Call
   * it as the chunk is installed, with the chunk's row due now (rows run since the request) and the joints the loop
   * commanded last: the step into the chunk then stays within the arm's speed caps, however late the install.
   *
   * @throws DreamConfigError No policy is open, or `row` is not an int, at least 0.
   * @throws DreamObservationError `commanded` is not the arms' joints.
   * @throws DreamActionError An arm's chunk in `action` (named in `commanded` or not) is not rows of finite numbers
   *   with no holes, of the shape the policy returned; no plan changes.
   */
  takeoverAt(action: ActionDict, row: number, commanded: Commanded | null): ActionDict {
    if (!this.policyClient) throw noPolicy("takeoverAt()");
    return this.policyClient.takeoverAt(action, row, commanded);
  }

  /**
   * Ends the episode: ends the session on the model, then empties the window. Call it between episodes.
   *
   * @throws DreamConfigError No policy is open.
   * @throws DreamError A getAction() is in flight; nothing changed.
   */
  async resetEpisode(): Promise<void> {
    const policy = this.policyClient;
    const window = this.window;
    if (!policy || !window) throw noPolicy("resetEpisode()");
    await policy.resetEpisode();
    window.clear();
    this.#trainedNotes = null;
  }

  /**
   * Whether your loop should call getAction() now (`dream.pacing`); local and free. `executed`: rows of the current
   * chunk run since it was installed (not its row counted from the observation); `remaining`: its rows still queued.
   */
  refillDue(executed: number, remaining: number): boolean {
    return this.open("refillDue()").refillDue(executed, remaining);
  }

  /** Rows of each chunk your loop runs before the next takes over, the open policy's. */
  get executionHorizon(): number {
    return this.open("executionHorizon").executionHorizon;
  }

  /** Rows per second your loop runs, the open policy's. */
  get actionHz(): number {
    return this.open("actionHz").actionHz;
  }

  /** How the open policy paces getAction() now: `async` or `sync`. */
  get mode(): PacingMode {
    return this.open("mode").mode;
  }

  /** Where the last getAction()'s time went; null before the first, and when no policy is open. */
  get lastTiming(): Timing | null {
    return this.policyClient?.lastTiming ?? null;
  }

  /** What the last successful getAction() did (raw reply, IK, timings); null before the first, and with no policy. */
  get lastDiagnostics(): Diagnostics | null {
    return this.policyClient?.lastDiagnostics ?? null;
  }

  /** The model the open policy runs; null when none is open. */
  get checkpoint(): string | null {
    return this.policyClient?.checkpoint ?? null;
  }

  /** The session the model answers the episode under, once getAction() opened it; else null. */
  get sessionId(): string | null {
    return this.policyClient?.sessionId ?? null;
  }

  /**
   * Empties the window and releases the open session, if any; calling it again does nothing. Synchronously, before it
   * returns: a policy() in progress is cancelled (it opens nothing), and a getAction() in flight is aborted (it
   * returns nothing). Calls made while one is closing share its promise. The client stays usable: policy() opens a
   * new policy.
   */
  close(): Promise<void> {
    this.#generation++;
    this.#pending?.abort(new DreamCancelled("policy() was cancelled: the client was closed"));
    this.#pending = null;
    return (this.#closing ??= this.closePolicy().finally(() => (this.#closing = null)));
  }

  /** Closes the open policy, then waits for every policy still closing (one a concurrent call began) to finish. */
  private async closePolicy(): Promise<void> {
    const policy = this.policyClient;
    this.window?.clear();
    this.window = null;
    this.policyClient = null;
    if (policy) {
      const release = policy.close();
      this.#releasing.add(release);
      void release.finally(() => this.#releasing.delete(release));
    }
    await Promise.all(this.#releasing);
  }

  private open(what: string): PolicyClient {
    if (!this.policyClient) throw noPolicy(what);
    return this.policyClient;
  }

  private now(): number {
    return (this.#options.now ?? (() => performance.now()))();
  }

  /** A platform or data service call's options, with this client's key, fetch, headers and identity. */
  private service(url: string, signal?: AbortSignal): ServiceOptions {
    return {
      url,
      apiKey: this.#apiKey,
      fetch: this.#options.fetch,
      headers: this.#options.headers,
      signal,
      clientVersion: this.#options.clientVersion,
      proxyHeaders: this.#proxyHeaders,
      allowInsecureHttp: this.#options.allowInsecureHttp,
    };
  }

  /** Hands `warning` to `onWarning` (console.warn by default); never throws. */
  private warn(warning: DreamWarning): void {
    try {
      (this.#options.onWarning ?? ((w: DreamWarning) => console.warn(`${w.name}: ${w.message}`)))(warning);
    } catch {
      // A warning never throws out of a call.
    }
  }

  private handOver(trip: RoundTrip): void {
    for (const { fn } of [...this.#roundTripFns]) {
      const result = fn(trip);
      // Synchronous by contract; a promise it returns is not waited for, and its rejection is a warning, never an
      // unhandled rejection.
      if (typeof (result as PromiseLike<unknown> | undefined)?.then === "function") {
        // The reporter never throws itself: any rejection value is formatted safely, and the chain ends here.
        Promise.resolve(result).then(undefined, (error: unknown) => {
          try {
            this.warn(new DreamCallbackWarning(scrub(`an onRoundTrip() callback's promise was rejected: ${safeMessage(error)}`, this.#apiKey)));
          } catch {
            // nothing left to report to
          }
        });
      }
    }
  }

  /**
   * What a fine-tune was trained on, read once per checkpoint (`facade._trained_inputs_for`); null for any other model,
   * and when it could not be read, which is warned about and never refuses.
   */
  private async trainedInputsFor(checkpoint: string, signal: AbortSignal): Promise<{ checkpoint: string; inputs: TrainedInputs } | null> {
    if (!checkpoint.startsWith(FINE_TUNE_PREFIX)) return null;
    if (!this.#trainedFor.has(checkpoint)) {
      try {
        this.#trainedFor.set(checkpoint, await fetchTrainedInputs(checkpoint, this.service(this.dataUrl, signal)));
      } catch (error) {
        if (!(error instanceof DreamError) || error instanceof DreamCancelled) throw error;
        const why = scrub(error.message, this.#apiKey);
        this.warn(new DreamTrainingWarning(`could not read what ${checkpoint} was trained on, so this policy's cameras and prompt are not compared with it: ${why}`));
        return null;
      }
    }
    const inputs = this.#trainedFor.get(checkpoint)!;
    return inputs === null ? null : { checkpoint, inputs };
  }

  /** The episode's lines against what the fine-tune was trained on, each warned once a policy (`_held_to_training`). */
  private heldToTraining(sent: Readonly<Record<string, unknown>>, frameKeys: ReadonlySet<string>): string[] {
    if (!this.#trained) return [];
    const frames = Object.fromEntries([...frameKeys].map((key) => [key, sent[key] as string[]]));
    let lines: string[];
    try {
      // The lines quote the training record: never with the key in them, in a warning or a round trip.
      lines = trainedNotes(this.#trained.inputs, this.#trained.checkpoint, frames, sent.prompt).map((line) => scrub(line, this.#apiKey));
    } catch {
      // Only ever a warning: whatever the record holds, the call goes on.
      return [];
    }
    for (const line of lines) {
      if (this.#trainedWarned.has(line)) continue;
      this.#trainedWarned.add(line);
      this.warn(new DreamTrainingWarning(line));
    }
    return lines;
  }

  /** The embodiment config a fine-tune was trained on, looked up once per checkpoint (`facade._embodiment_config_for`). */
  private async embodimentConfigFor(checkpoint: string, lookup: () => Promise<Model[]>, given: boolean): Promise<Record<string, unknown> | null> {
    if (!given && this.embodimentConfigs.has(checkpoint)) return this.embodimentConfigs.get(checkpoint)!;
    let model: Model | undefined;
    try {
      model = (await lookup()).find((m) => m.id === checkpoint);
    } catch (error) {
      if (!(error instanceof DreamError) || error instanceof DreamCancelled) throw error;
      throw new DreamProtocolError(
        `could not look up checkpoint '${checkpoint}' on the platform (GET /v1/models), so its embodiment config and the ` +
          "check of its calls cannot run and nothing is sent; call policy() again once the platform answers, pass the " +
          `robot's embodiment config as config.embodiment, or the model record as config.model: ${error.message}`,
        { cause: error },
      );
    }
    if (!model) {
      throw new DreamConfigError(
        `checkpoint '${checkpoint}' is not among the models this account may run (getModels(), GET /v1/models), so its ` +
          "embodiment config could not be looked up and nothing is sent; check the name, or pass config.embodiment",
      );
    }
    refuseUnsupported(model);
    const found = model.embodimentConfig === null ? null : { ...model.embodimentConfig };
    // Every listed config is parsed, as Python does: one this client cannot read must not turn the checks off.
    if (found !== null) parseEmbodiment(found);
    if (!given) this.embodimentConfigs.set(checkpoint, found);
    return found;
  }
}

/** @throws DreamConfigError The model's robot has parts, which this package does not convert. */
function refuseUnsupported(model: Model): void {
  if (model.partsWithheld || (model.embodimentConfig !== null && "parts" in model.embodimentConfig)) {
    throw new DreamConfigError(
      `checkpoint '${model.id}' was trained on robot '${model.embodiment}', registered with parts` +
        (model.partsWithheld ? " the platform lists only to the account that registered it" : "") +
        "; @makermods/dreamlabs does not convert parts yet",
      detail("unsupported", "checkpoint"),
    );
  }
}

function checkPacing(config: PolicyConfig): void {
  const horizon = config.executionHorizon ?? EXECUTION_HORIZON;
  if (!Number.isInteger(horizon) || horizon < 1) {
    throw new DreamConfigError(`config key 'executionHorizon' must be an int of rows, at least 1, got ${JSON.stringify(horizon)}`);
  }
  for (const [key, value] of [["actionHz", config.actionHz ?? ACTION_HZ], ["syncAboveMs", config.syncAboveMs ?? SYNC_ABOVE_MS]] as const) {
    if (typeof value !== "number" || !(value > 0)) throw new DreamConfigError(`config key '${key}' must be a positive number, got ${JSON.stringify(value)}`, detail("range", key));
  }
  const actionHz = config.actionHz ?? ACTION_HZ;
  if (!Number.isFinite(actionHz)) throw new DreamConfigError(`config key 'actionHz' must be a finite number of rows per second, got ${actionHz}`, detail("range", "actionHz"));
}

function checkKinematics(kinematics: unknown): void {
  if (kinematics === undefined) return;
  if (typeof kinematics !== "object" || kinematics === null) {
    throw new DreamConfigError(`config key 'kinematics' must be an object of chunk key to kinematics, got ${String(kinematics)}`);
  }
  for (const [key, arm] of Object.entries(kinematics)) {
    if (!(key in ARM_POSE_KEYS)) throw new DreamConfigError(`config key 'kinematics' names '${key}'; it takes the chunk keys ${pyList(Object.keys(ARM_POSE_KEYS).sort())}`);
    if (typeof arm?.pose !== "function" || typeof arm?.jacobian !== "function") {
      throw new DreamConfigError(`config key 'kinematics' entry '${key}' must have pose() and jacobian() methods`);
    }
  }
}

function roundTrip(
  sentS: number,
  sent: Record<string, unknown>,
  frameKeys: ReadonlySet<string>,
  action: ActionDict | null,
  timing: Timing | null,
  error: DreamError | null,
  trained: readonly string[],
): RoundTrip {
  const frames: Record<string, string[]> = {};
  const observation: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(sent)) {
    if (frameKeys.has(key)) frames[key] = value as string[];
    else observation[key] = value;
  }
  return Object.freeze({ sentS, frames, observation, action, timing, error, trained: [...trained] });
}

function noPolicy(call: string): DreamConfigError {
  return new DreamConfigError(`no policy is open: call policy({ checkpoint: ... }) before ${call}`);
}
