/**
 * The inference client (`dream.client.PolicyClient`): checks an observation against the configured embodiment, sends
 * it to the model, reads the status envelope of the answer, checks the returned action, and joins each arm's chunk onto
 * the motion it is on. It keeps the session: none before the first answer and after resetEpisode(), then the one the
 * model answered. {@link DreamClient} (`facade.ts`) is the public front over it, adding the frame window.
 */

import { checkServiceUrl, KEY_PER_SITE } from "./credentials.js";
import { check, checkPerKey, compileChecks, isEmbodimentConfig, isObject, type Checks, type ShapeEntry } from "./embodiment.js";
import {
  detail,
  DreamActionError,
  DreamAuthError,
  DreamBadRequest,
  DreamCancelled,
  DreamConfigError,
  DreamError,
  DreamObservationError,
  DreamProtocolError,
  DreamServerError,
  DreamSessionExpired,
  DreamSlowNetworkWarning,
  type DreamWarning,
} from "./errors.js";
import { Pacer, type PacingMode } from "./pacing.js";
import type { Kinematics } from "./seam-join.js";
import { Watch } from "./slow-network.js";
import { checkFiniteAction, every, firstHole } from "./finite.js";
import { ARM_POSE_KEYS, SeamSmoother, SIDE_POSE_KEYS, type Fence, type Projection } from "./smoothing.js";
import { cancelled, encode, END_SESSION_PATH, GET_ACTION_PATH, Transport, type FetchLike } from "./transport.js";
import { pyList, snapshot } from "./window.js";

/** The robot a model trained on YAM arms runs on, as the serve reads it. */
export const YAM = "yam";
/** Any other robot: the serve may ignore the arm's state, and answers no joint head (only each hand's pose). */
export const OTHER_EMBODIMENT = "other_embodiment";
export const ROBOTS = [YAM, OTHER_EMBODIMENT] as const;
export type Robot = (typeof ROBOTS)[number];

/** Envelope codes (`shared.contracts.envelope`). */
export const OK = "DA-0000";
export const OK_NEW_SESSION = "DA-0001";
export const UNAUTHORIZED = "DA-1000";
export const UNAVAILABLE = "DA-2000";
export const BAD_REQUEST = "DA-4000";
export const SERVE_FAILED = "DA-5000";
export const INTERNAL = "DA-6000";
/**
 * Per envelope code, the reasons the gateway answers it with (`shared.contracts.envelope`, dreamlabs 1.0.42): what
 * {@link DreamServerError.reason} holds. `DA-4000 client_outdated` is also what a checkpoint whose robot has parts
 * answers this package ({@link CLIENT_VERSION}).
 */
export const REASONS = {
  [OK_NEW_SESSION]: ["new", "expired", "unknown", "serve_died"],
  [UNAUTHORIZED]: ["missing", "unknown_key", "disabled", "over_cap", "not_entitled", "region_not_supported"],
  [UNAVAILABLE]: ["pool_full", "pool_warming", "rate_limited", "queue_timeout", "max_serves"],
  [BAD_REQUEST]: ["malformed", "too_large", "unknown_op", "invalid_session_id", "global_op_rejected", "unknown_checkpoint", "client_outdated"],
  [SERVE_FAILED]: ["timeout", "error", "unreachable", "capacity"],
  [INTERNAL]: ["internal", "store_write_failed", "cannot_scale", "service_unavailable", "connection_timed_out"],
} as const;
/** A reason the gateway answers a code with ({@link REASONS}). */
export type DreamReason = (typeof REASONS)[keyof typeof REASONS][number];
/** Timeout of the session release resetEpisode() and close() send: a release, best effort. */
export const END_SESSION_TIMEOUT_MS = 2000;
/** Default per-request timeout, milliseconds. */
export const TIMEOUT_MS = 15000;
/** Largest `timeoutMs` accepted. */
export const MAX_TIMEOUT_MS = 2 ** 31 - 1;
/** Recent client-IK solves whose time `ikLeadMs: "auto"` takes the largest of (an episode's first, cold, solve left out). */
export const IK_LEAD_SOLVES = 3;

const JOINT_HEAD_KEYS = ["action.right_arm", "action.left_arm"];

/** Where the milliseconds of the most recent getAction() went (`PolicyClient.last_timing`). */
export interface Timing {
  /** The client's own round trip around the HTTP exchange. */
  rttMs: number;
  /** The model's own inference time, as its answer reports it; null without timing (a refusal). */
  serveInferMs: number | null;
  /** The time on the Dream Labs side, from receiving the request to sending the answer. */
  gatewayMs: number | null;
  /** `rttMs - gatewayMs`, the part spent on the network. */
  networkMs: number | null;
  /** The client IK's solve of the reply, milliseconds; null without client IK (and before the reply). */
  ikMs: number | null;
  /** The whole getAction(), from the call to its return or throw, milliseconds: checks, the round trip, IK, the join. */
  totalMs: number | null;
  requestBytes: number;
  /** 0 when no reply was read. */
  responseBytes: number;
}

/** A status envelope as DreamAPI sends it. */
export interface StatusEnvelope {
  code: string;
  name?: string;
  reason?: string | null;
  retryable?: boolean;
  retry_after_s?: number | null;
  detail?: string | null;
  trace_id?: string;
  [key: string]: unknown;
}

/** The `/v1/get_action` request body. */
export interface GetActionRequest {
  checkpoint: string;
  /** Always sent, null to mint a session. */
  session_id: string | null;
  reset: false;
  observation: Record<string, unknown>;
}

/** The `/v1/get_action` reply body. */
export interface GetActionReply {
  status: StatusEnvelope;
  session_id?: string;
  actions?: ActionDict;
  timing?: { serve_infer_ms?: number; gateway_ms?: number };
}

/**
 * The action dict a model answers. A YAM body's chunks are `q_r_chunk` and `q_l_chunk`, `[H][joints + 1]` (the arm's
 * joints, then the gripper), joined onto the previous chunk; a fine-tuned model's parts are under `per_key`, one
 * `action.<part>` per action part, as rows; any other key the model adds is passed through.
 */
export interface ActionDict {
  q_r_chunk?: number[][];
  q_l_chunk?: number[][];
  per_key?: Record<string, number[][] | number[] | null>;
  [key: string]: unknown;
}

/** One arm's IK solve of the most recent chunk, for {@link Diagnostics}. */
export interface IkTrace {
  /** Rows of the chunk before its clock row, which continue the previous plan instead of being solved. */
  clock: number;
  /** The per-joint speed cap the solve ran under, rad/s (Infinity: the arm's own caps only). */
  speedCap: number;
  /** Per solved row, metres between the commanded tool position and the model's target. */
  posMiss: number[];
  /** Per solved row, radians between the commanded tool rotation and the model's target. */
  angMiss: number[];
  /** Per solved row, whether the speed cap shortened the step. */
  atCap: boolean[];
}

/** Client-side IK as the client drives it; built by `@makermods/dreamlabs/ik`'s `ClientIK`. */
export interface PolicyIk {
  /** Per chunk key, the joints of the arm it drives, as the loop reads and writes them. */
  readonly joints: Readonly<Record<string, number>>;
  /** Per side, the joint slots of a model trained on YAM arms (sent as zeros), or null. */
  readonly yamJoints: number | null;
  /** The reach area's hold, or null without one. */
  readonly fence: Fence | null;
  /** The head the reply is run from. */
  readonly head: "eef" | "joint";
  observe(observation: Readonly<Record<string, unknown>>): Record<string, unknown>;
  solve(
    action: Record<string, unknown>,
    observation: Readonly<Record<string, unknown>>,
    smoother: SeamSmoother,
    inFlightS: number,
    commanded: Readonly<Record<string, number[]>> | null,
    trace?: Record<string, IkTrace> | null,
  ): Record<string, unknown>;
  /** The final step on every returned chunk: the arm's limits and speed caps (see {@link Projection}). */
  project: Projection;
  sentChecks(checks: Readonly<Record<string, ShapeEntry>>): Record<string, ShapeEntry>;
  replyWidths(widths: Readonly<Record<string, number>>): Record<string, number>;
}

/**
 * What getAction() does with the reply. `default`: the client IK, the seam join onto the previous chunk and the final
 * projection, with the pacing measured. `raw`: the reply as the model answered it, checked (strict mode, finite numbers)
 * with the session and the errors handled, but no IK, no seam join, no projection and no pacing state: for an app that
 * runs its own post-processing and adopts the transport first.
 */
export type Processing = "raw" | "default";
export const PROCESSING: readonly Processing[] = ["default", "raw"];

/** When the loop's clock and the round trip disagree: what the seam takes as the time the loop ran meanwhile. */
export interface ExecutionTiming {
  /** How old the observation (and `commanded`) was when getAction() was called, ms; added to the round trip. */
  observationAgeMs?: number;
  /**
   * Seconds of the previous chunk the loop runs past `commanded` before it installs this reply; replaces the round
   * trip, the age and the IK lead (`ikLeadMs`) as the seam's time in flight. The pacing still records the round trip.
   */
  elapsedExecutionS?: number;
}

/** Options of getAction(). */
export interface GetActionOptions {
  /** Per chunk key, the joints the loop last commanded that arm before the observation; never sent. */
  commanded?: Commanded | null;
  /** Cancels the call ({@link DreamCancelled}); composed with `timeoutMs`, and the error says which fired. */
  signal?: AbortSignal;
  execution?: ExecutionTiming;
}

/** What the most recent successful getAction() did, for logs and tuning, so an app need not intercept `fetch`. */
export interface Diagnostics {
  /** The reply's action dict as the model answered it, checked, before IK and the seam join. Do not mutate it. */
  raw: Record<string, unknown>;
  /** Per side, the IK solve; null without client IK, and for head `joint`. */
  ik: Record<string, IkTrace> | null;
  timing: Timing;
  /**
   * The seconds the seam took as run between the observation and the chunk's install (see {@link ExecutionTiming}):
   * the round trip, the observation's age and `ikLeadS`, unless `elapsedExecutionS` replaced them.
   */
  inFlightS: number;
  /** The part of `inFlightS` that is the client IK's own solve, as `ikLeadMs` predicted it; 0 without it. */
  ikLeadS: number;
  processing: Processing;
}

/** Options of a {@link PolicyClient}; {@link DreamClient.policy} builds them from its config. */
export interface PolicyClientOptions {
  checkpoint: string;
  /** The gateway, or your proxy; requests go to `${baseUrl}/v1/get_action`. */
  baseUrl: string;
  apiKey?: string | null;
  timeoutMs?: number;
  /** A shape spec or an embodiment config, which strict mode checks every observation and reply against. */
  embodiment?: unknown;
  strict?: boolean;
  robot?: Robot;
  executionHorizon?: number;
  actionHz?: number;
  syncAboveMs?: number;
  /** Per chunk key (`q_r_chunk`, `q_l_chunk`), the kinematics of the arm it drives: joined in end-effector space. */
  kinematics?: Readonly<Record<string, Kinematics>>;
  ik?: PolicyIk | null;
  /** `default` (IK, seam join, projection) or `raw`: see {@link Processing}. */
  processing?: Processing;
  /**
   * Not in Python: the time the client IK's solve takes, which the loop keeps running through, added to the seam's
   * time in flight so the chunk takes over at the row the loop installs it at. `auto` (the default): the largest of
   * the last {@link IK_LEAD_SOLVES} solves' `ikMs`, leaving out each episode's first solve (cold: no plan to continue,
   * and the first in a fresh process), which is used only until a later one is measured; a number: that many
   * milliseconds; 0: the round trip only, as upstream (whose solve is fast). Without client IK, nothing is added.
   */
  ikLeadMs?: IkLead;
  fetch?: FetchLike;
  headers?: Record<string, string>;
  /** `Dream-Client-Version`; {@link CLIENT_VERSION} by default (deliberately not a semver: see there). */
  clientVersion?: string;
  /** `baseUrl` is your proxy: send `Dream-Client-Instance` (8 hex) for it to build the `User-Agent` upstream. */
  proxyHeaders?: boolean;
  /** Receives a {@link DreamSlowNetworkWarning}; `console.warn` by default. */
  onWarning?: (warning: DreamWarning) => void;
  /** Receives the {@link Diagnostics} of every successful getAction(); never throws out of it. */
  onDiagnostics?: (diagnostics: Diagnostics) => void;
  /** The clock, milliseconds on a monotonic scale; `performance.now` by default. */
  now?: () => number;
}

/** Per chunk key, the joints the loop last commanded that arm: the arm's joints, or its whole written row. */
export type Commanded = Readonly<Record<string, ArrayLike<number>>>;

/** `ikLeadMs`: `auto`, or a fixed number of milliseconds (0 for none). */
export type IkLead = "auto" | number;

/** The inference client for one checkpoint (`dream.client.PolicyClient`). One call at a time per client. */
export class PolicyClient {
  readonly checkpoint: string;
  readonly baseUrl: string;
  readonly timeoutMs: number;
  readonly processing: Processing;
  private sessionIdValue: string | null = null;
  private readonly checks: Checks | null;
  private readonly robot: Robot | null;
  private readonly ik: PolicyIk | null;
  private readonly smoother: SeamSmoother;
  private readonly pacer: Pacer;
  private readonly transport: Transport;
  private readonly watch = new Watch();
  private readonly onWarning: (warning: DreamWarning) => void;
  private readonly onDiagnostics: ((diagnostics: Diagnostics) => void) | null;
  private readonly now: () => number;
  private timing: Timing | null = null;
  private diagnostics: Diagnostics | null = null;
  private inFlight: Promise<unknown> | null = null;
  private readonly ikLead: IkLead;
  /** The most recent solves' `ikMs`, for `ikLeadMs: "auto"`, each episode's first left out. */
  private readonly ikTimes: number[] = [];
  /** The latest episode-first solve's `ikMs`: the lead until a later solve is measured. */
  private coldIkMs: number | null = null;
  /** The most recent result, and the chunk keys it holds the first chunk of the episode for, for takeoverAt(). */
  private returned: { result: ActionDict; first: string[] } | null = null;
  /** A session whose release was aborted (close() during resetEpisode()): close() releases it again. */
  private unreleased: string | null = null;
  /** Aborts the call in flight (getAction() or resetEpisode()'s release) when the client closes. */
  private active: AbortController | null = null;
  private closing: Promise<void> | null = null;
  private closed = false;

  /** @throws DreamConfigError A config value is wrong: see {@link DreamClient.policy}. */
  constructor(options: PolicyClientOptions) {
    const { checkpoint } = options;
    if (typeof checkpoint !== "string" || !checkpoint) {
      throw new DreamConfigError(`config key 'checkpoint' must be a non-empty str naming the checkpoint, got ${JSON.stringify(checkpoint)}`);
    }
    this.checkpoint = checkpoint;
    this.baseUrl = checkServiceUrl(options.baseUrl, "baseUrl", true);
    if (this.baseUrl.startsWith("/") || this.baseUrl === "") {
      if (typeof (globalThis as { location?: unknown }).location === "undefined") {
        throw new DreamConfigError(`baseUrl ${JSON.stringify(options.baseUrl)} is relative, which only a page or a worker can resolve; give an absolute URL`);
      }
    }
    this.timeoutMs = options.timeoutMs ?? TIMEOUT_MS;
    checkTimeout(this.timeoutMs);
    if (options.robot !== undefined && !ROBOTS.includes(options.robot)) {
      throw new DreamConfigError(`config key 'robot' must be one of ${pyList([...ROBOTS])}, got ${JSON.stringify(options.robot)}`);
    }
    this.processing = checkProcessing(options.processing ?? "default", options.ik != null, options.kinematics !== undefined);
    this.ikLead = checkIkLead(options.ikLeadMs ?? "auto");
    const ik = options.ik ?? null;
    if (options.robot === OTHER_EMBODIMENT && ik?.head === "joint") {
      throw new DreamConfigError(
        "config key 'robot' is 'other_embodiment', on which the model answers no joints, so config key 'ik' with the head " +
          "'joint' has nothing to run: use the head 'eef'",
      );
    }
    const embodiment = options.embodiment;
    // Refused whatever `strict` says: the client cannot convert parts, so it must never run such a body unchecked.
    if (isEmbodimentConfig(embodiment) && "parts" in embodiment) {
      throw new DreamConfigError(
        "config key 'embodiment' is an embodiment config with parts; @makermods/dreamlabs does not convert parts yet: use " +
          "the Python client for this model",
        detail("unsupported", "embodiment"),
      );
    }
    let checks = (options.strict ?? true) && embodiment !== undefined ? compileChecks(embodiment) : null;
    this.ik = ik;
    if (ik && checks) {
      checks = { ...checks, observation: ik.sentChecks(checks.observation), perKeyWidths: ik.replyWidths(checks.perKeyWidths) };
    }
    this.robot = options.robot ?? (ik && ik.yamJoints !== null ? OTHER_EMBODIMENT : null);
    if (checks && this.robot === OTHER_EMBODIMENT) {
      checks = { ...checks, perKeyWidths: Object.fromEntries(Object.entries(checks.perKeyWidths).filter(([key]) => !JOINT_HEAD_KEYS.includes(key))) };
    }
    this.checks = checks;
    const poseKeys = ik || isEmbodimentConfig(embodiment) ? SIDE_POSE_KEYS : ARM_POSE_KEYS;
    const actionHz = options.actionHz ?? 20;
    this.smoother = new SeamSmoother(actionHz, options.kinematics ?? null, poseKeys, ik?.fence ?? null, ik ? (key, rows, from, first, observation) => ik.project(key, rows, from, first, observation) : null);
    this.pacer = new Pacer(options.executionHorizon, actionHz, options.syncAboveMs);
    this.now = options.now ?? (() => performance.now());
    this.transport = new Transport({
      baseUrl: this.baseUrl,
      timeoutMs: this.timeoutMs,
      apiKey: options.apiKey ?? null,
      fetch: options.fetch,
      headers: options.headers,
      now: this.now,
      clientVersion: options.clientVersion,
      proxyHeaders: options.proxyHeaders,
    });
    this.onWarning = options.onWarning ?? ((warning) => console.warn(`${warning.name}: ${warning.message}`));
    this.onDiagnostics = options.onDiagnostics ?? null;
  }

  /** The session the model answers the episode under; null before the first reply and after resetEpisode(). */
  get sessionId(): string | null {
    return this.sessionIdValue;
  }

  /** Where the milliseconds of the most recent request went; null before the first. A fresh object on every read. */
  get lastTiming(): Timing | null {
    return this.timing ? { ...this.timing } : null;
  }

  /** What the most recent successful getAction() did; null before the first. */
  get lastDiagnostics(): Diagnostics | null {
    return this.diagnostics;
  }

  /** Always `async` with `processing: "raw"`, which records no round trips. */
  get mode(): PacingMode {
    return this.pacer.mode;
  }

  get executionHorizon(): number {
    return this.pacer.executionHorizon;
  }

  get actionHz(): number {
    return this.pacer.actionHz;
  }

  /** Whether the loop should send the next getAction() now, by the pacing (`dream.pacing`); local and free. */
  refillDue(executed: number, remaining: number): boolean {
    return this.pacer.refillDue(executed, remaining);
  }

  /**
   * Sends one observation to the model and returns its action chunk, each arm's chunk joined onto the previous one.
   * The observation's numbers are copied first, so a buffer the caller reuses cannot change the join after the fact.
   * See {@link DreamClient.getAction} for what is thrown.
   */
  async getAction(observation: Record<string, unknown>, options: GetActionOptions = {}): Promise<ActionDict> {
    if (this.closed) throw new DreamError("inference request on a closed client");
    if (this.inFlight) {
      throw new DreamError("PolicyClient is not reentrant: another inference request is in flight; await it, or use one client per loop");
    }
    const execution = checkExecution(options.execution);
    if (options.signal?.aborted) throw cancelled(options.signal, "getAction() was cancelled before it was sent");
    const controller = new AbortController();
    const signal = options.signal ? AbortSignal.any([controller.signal, options.signal]) : controller.signal;
    const run = this.run(snapshot(observation), options.commanded ?? null, execution, signal);
    this.active = controller;
    this.inFlight = run;
    try {
      return await run;
    } finally {
      this.inFlight = null;
      this.active = null;
    }
  }

  private async run(observation: Record<string, unknown>, commanded: Commanded | null, execution: ExecutionTiming, signal: AbortSignal): Promise<ActionDict> {
    const started = this.now();
    let sent = false;
    let ikMs: number | null = null;
    try {
      if (isObject(observation) && "embodiment" in observation) {
        throw new DreamObservationError(
          "observation key 'embodiment' is the client's own: it sends the robot the model runs on there, from config key " +
            "'robot' (one of ['yam', 'other_embodiment']); leave it out of the observation and set 'robot' in the policy() config instead",
        );
      }
      const armJoints = this.ik ? { ...this.ik.joints } : this.smoother.armJoints(observation);
      const last = commandedOf(commanded, armJoints);
      const out = this.ik ? this.ik.observe(observation) : observation;
      if (this.checks) check(out, this.checks.observation, "observation", { frameWindowGroup: this.checks.frameWindowGroup });
      const sentAt = this.now();
      let action: unknown;
      sent = true;
      try {
        action = await this.post(out, signal);
      } catch (error) {
        // The episode restarted server-side: the plan the arms were on is gone with it.
        if (error instanceof DreamSessionExpired) {
          this.smoother.reset();
          this.returned = null;
        }
        throw error;
      }
      // Closed, replaced or cancelled while the reply was on its way: it is no longer anyone's answer.
      if (signal.aborted || this.closed) throw cancelled(signal, "getAction() was cancelled", this.closed);
      if (this.checks) {
        // Checked before returning: a mis-shaped action that reaches the control loop is one that reaches hardware.
        check(action, this.checks.action, "action", { allowExtra: true });
        checkPerKey(action as Record<string, unknown>, this.checks.perKeyWidths);
      }
      if (!isObject(action)) throw new DreamActionError(`the model's reply carries no action dict, got ${action === null ? "None" : typeof action}`);
      checkFiniteAction(action);
      const rttS = (this.now() - sentAt) / 1000;
      // The loop keeps running while the IK solves, so the chunk is installed that much later than the reply arrives.
      const ikLeadS = this.ik && execution.elapsedExecutionS === undefined ? this.predictIkMs() / 1000 : 0;
      const inFlightS = execution.elapsedExecutionS ?? rttS + (execution.observationAgeMs ?? 0) / 1000 + ikLeadS;
      if (this.processing === "raw") {
        this.finish(started, null, { raw: action, ik: null, inFlightS, ikLeadS, processing: "raw" });
        this.live(signal);
        return action as ActionDict;
      }
      // As upstream: the pacing measures the round trip, before the IK.
      this.pacer.record(rttS);
      const slow = this.watch.record(this.timing, this.now() / 1000);
      if (slow !== null) {
        try {
          this.onWarning(new DreamSlowNetworkWarning(slow));
        } catch {
          // A warning never throws out of getAction(): the action is already here.
        }
      }
      let solved: Record<string, unknown> = action;
      let trace: Record<string, IkTrace> | null = null;
      if (this.ik) {
        const solveStarted = this.now();
        trace = this.ik.head === "eef" ? {} : null;
        // Cold: the episode's first chunk on every arm (no plan to continue), and the process's first solve.
        const cold = !Object.keys(this.ik.joints).some((key) => this.smoother.has(key));
        solved = this.ik.solve(action, observation, this.smoother, inFlightS, last, trace);
        ikMs = this.now() - solveStarted;
        if (cold) this.coldIkMs = ikMs;
        else {
          this.ikTimes.push(ikMs);
          if (this.ikTimes.length > IK_LEAD_SOLVES) this.ikTimes.shift();
        }
      }
      const joined = this.smoother.prepare(observation, solved, inFlightS, last);
      this.finish(started, ikMs, { raw: action, ik: trace, inFlightS, ikLeadS, processing: "default" });
      // A callback (onDiagnostics) may have cancelled or closed: then nothing is returned and no plan is kept.
      this.live(signal);
      joined.commit();
      const result = joined.result as ActionDict;
      this.returned = { result, first: joined.first };
      return result;
    } finally {
      if (sent && this.timing && this.timing.totalMs === null) this.timing = { ...this.timing, ikMs, totalMs: this.now() - started };
    }
  }

  /** @throws DreamCancelled The call was cancelled, or the client closed, meanwhile: the reply is no one's answer. */
  private live(signal: AbortSignal): void {
    if (signal.aborted || this.closed) throw cancelled(signal, "getAction() was cancelled", this.closed);
  }

  /** The IK solve time the seam takes the loop to run through, ms (`ikLeadMs`). */
  private predictIkMs(): number {
    if (this.ikLead !== "auto") return this.ikLead;
    // Before any later solve is measured, the cold one: slower, so the old plan is continued a few rows longer.
    return this.ikTimes.length ? Math.max(...this.ikTimes) : (this.coldIkMs ?? 0);
  }

  /**
   * Not in Python: `action` re-anchored at the row the loop installs it at, for when that is later than the seam
   * assumed (an IK slower than its lead, a hand-off from a worker). From `row` on, each arm's rows are run through the
   * final projection again, starting from `commanded` (the joints the loop commanded last, on the old plan): the step
   * from the old plan into the chunk then stays within the arm's limits and speed caps, where playing the chunk from
   * `row` as is could jump (the grip too, within `ik.gripSpeedCap` when set, from `commanded`'s grip when it is the
   * written row). Rows before `row` are returned as they were. When `action` is the latest getAction() returned, the
   * re-anchored chunks become the plans the next seam continues from.
   *
   * As getAction(), it checks before it keeps: every arm's chunk in `action`, whether `commanded` names the arm or
   * not, must be rows of finite numbers with no holes (the latest reply's still in the shape it was returned in, any
   * other at least the arm's joints wide), before and after the projection, and every arm is re-anchored before any
   * plan is kept.
   *
   * Without client IK (no projection) or with `commanded` null, `action` itself is returned. An arm `commanded` leaves
   * out is returned (and its plan kept) with its rows as they were, checked.
   *
   * @param action What getAction() returned.
   * @param row The chunk's row due at install, counted from the observation (rows the loop ran since the request).
   * @param commanded Per chunk key, the joints (or the written row) the loop commanded last.
   * @throws DreamConfigError `row` is not an int, at least 0.
   * @throws DreamActionError An arm's chunk is not rows of finite numbers of the shape above, or the latest reply lost
   *   an arm's chunk (the caller changed it); no plan changes.
   */
  takeoverAt(action: ActionDict, row: number, commanded: Commanded | null): ActionDict {
    if (!(Number.isInteger(row) && row >= 0)) throw new DreamConfigError(`takeoverAt() row must be an int, at least 0, got ${String(row)}`, detail("range", "row"));
    const ik = this.ik;
    if (!ik || commanded === null || this.processing === "raw") return action;
    const from = commandedOf(commanded, { ...ik.joints });
    const latest = this.returned?.result === action ? this.returned : null;
    // Every arm the action carries is checked, re-anchored or not (an arm left out of `commanded` is returned, and its
    // plan kept, as it is): the latest reply's must still be the plan the smoother kept, row for row; any other chunk
    // the arm's rows. Then every arm is re-anchored before any plan is kept: when one throws, nothing changes.
    const checked = new Map<string, number[][]>();
    for (const key of Object.keys(ik.joints)) {
      const shape = latest ? this.smoother.planShape(key) : null;
      if (!(key in action)) {
        if (shape) throw new DreamActionError(`takeoverAt() action key '${key}' is gone; the chunk getAction() returned has it`, detail("shape", key));
        continue;
      }
      checked.set(key, takeoverRows(action[key], key, shape?.rows ?? null, shape?.width ?? null, shape ? null : ik.joints[key]!, "before"));
    }
    const chunks = new Map(checked);
    for (const [key, joints] of Object.entries(from ?? {})) {
      const rows = checked.get(key);
      if (!rows || row >= rows.length) continue;
      const tail = ik.project(key, rows.slice(row), joints, latest?.first.includes(key) ?? false);
      chunks.set(key, takeoverRows([...rows.slice(0, row), ...tail], key, rows.length, rows[0]!.length, null, "after"));
    }
    const out: ActionDict = { ...action };
    for (const [key, chunk] of chunks) {
      out[key] = chunk.map((r) => [...r]);
      if (latest) this.smoother.adopt(key, chunk);
    }
    if (latest) this.returned = { result: out, first: latest.first };
    return out;
  }

  /** Completes the timing and the diagnostics of a successful call, and hands them to `onDiagnostics`. */
  private finish(started: number, ikMs: number | null, diagnostics: Omit<Diagnostics, "timing">): void {
    if (this.timing) this.timing = { ...this.timing, ikMs, totalMs: this.now() - started };
    if (!this.timing) return;
    this.diagnostics = { ...diagnostics, timing: { ...this.timing } };
    try {
      this.onDiagnostics?.(this.diagnostics);
    } catch {
      // As a warning: never thrown out of getAction().
    }
  }

  /**
   * Ends the episode: releases the held session (best effort, at most two seconds, never throwing) and forgets the
   * previous chunk, so the next getAction() starts a new session and its first chunk is not joined onto the old plan.
   *
   * @throws DreamError The client is closed, or a getAction() is in flight.
   */
  async resetEpisode(): Promise<void> {
    if (this.closed) throw new DreamError("episode reset on a closed client");
    if (this.inFlight) throw new DreamError("PolicyClient is not reentrant: an inference request is in flight; reset the episode between requests");
    const controller = new AbortController();
    const release = this.endSession(controller.signal);
    this.active = controller;
    this.inFlight = release;
    try {
      await release;
    } finally {
      this.inFlight = null;
      this.active = null;
    }
    this.smoother.reset();
    this.returned = null;
  }

  /**
   * Closes the client: aborts a getAction() in flight (it throws {@link DreamCancelled} and returns nothing, even when
   * its reply was already on the way), then releases the session, if any, and the channel. Calls made meanwhile share
   * one promise; calling it again does nothing.
   */
  close(): Promise<void> {
    return (this.closing ??= this.shutdown());
  }

  private async shutdown(): Promise<void> {
    this.closed = true;
    this.active?.abort(new DreamCancelled("the call was cancelled: the client was closed"));
    if (this.inFlight) {
      // A fetch that ignores its signal is not waited for long; its late reply is refused on arrival.
      let timer: ReturnType<typeof setTimeout> | undefined;
      await Promise.race([this.inFlight.catch(() => undefined), new Promise((resolve) => (timer = setTimeout(resolve, END_SESSION_TIMEOUT_MS)))]);
      clearTimeout(timer);
    }
    await this.endSession();
    this.transport.close();
  }

  private async endSession(signal?: AbortSignal): Promise<void> {
    const sessionId = this.sessionIdValue ?? this.unreleased;
    if (sessionId === null) return;
    this.sessionIdValue = null;
    // Kept until the release is answered: close() aborting resetEpisode()'s release sends it again on its own signal.
    this.unreleased = sessionId;
    try {
      await this.transport.request({ checkpoint: this.checkpoint, session_id: sessionId }, END_SESSION_PATH, { timeoutMs: END_SESSION_TIMEOUT_MS, signal });
      if (this.unreleased === sessionId) this.unreleased = null;
    } catch {
      // The seat frees on idle eviction anyway; a stop or a reset must never fail on it.
      if (!signal?.aborted && this.unreleased === sessionId) this.unreleased = null;
    }
  }

  private async post(observation: Record<string, unknown>, signal: AbortSignal): Promise<unknown> {
    const request: GetActionRequest = {
      checkpoint: this.checkpoint,
      // Always sent, even when null: the gateway reads a null id as "mint me a session".
      session_id: this.sessionIdValue,
      reset: false,
      observation: this.robot === null ? observation : { ...observation, embodiment: this.robot },
    };
    let response: unknown;
    try {
      response = await this.transport.request(request, GET_ACTION_PATH, { signal });
    } catch (error) {
      if (error instanceof DreamError) {
        this.timing = timingOf(this.transport, null);
        throw error;
      }
      if (error instanceof TypeError) throw new DreamObservationError(unencodable(request.observation, error));
      throw error;
    }
    const reply = isObject(response) ? response : null;
    this.timing = timingOf(this.transport, reply?.timing);
    const status = reply?.status;
    if (!isObject(status) || typeof status.code !== "string") {
      throw new DreamProtocolError("inference reply carries no status envelope; is the correct DreamAPI gateway running?");
    }
    const code = status.code;
    const sent = this.sessionIdValue;
    // Adopt whatever session the gateway says we are in: learnt on the first call, an echo after, unless replaced.
    const returned = reply!.session_id;
    if (typeof returned === "string" && returned) this.sessionIdValue = returned;
    if (code === OK) return reply!.actions;
    if (code === OK_NEW_SESSION) {
      if (sent === null || status.reason === "new") return reply!.actions;
      throw new DreamSessionExpired(
        `inference session ${sent} was replaced by ${String(returned)} (${String(status.reason)}): the episode restarted server-side`,
        { code, reason: text(status.reason), envelope: status },
      );
    }
    throw errorFor(status as StatusEnvelope, this.baseUrl, this.checkpoint);
  }
}

/** The exception for a non-OK status envelope, keeping DreamAPI's wording (`client._error_for`). */
export function errorFor(status: StatusEnvelope, called: string, checkpoint: string): DreamServerError {
  const code = String(status.code);
  const reason = text(status.reason);
  const detail = text(status.detail);
  const name = text(status.name) || code;
  let message = `DreamAPI refused the inference request: ${code} ${name}`;
  if (reason) message += ` (${reason})`;
  if (detail) message += `: ${detail}`;
  if (code === UNAUTHORIZED) return new DreamAuthError(`${message}; called ${called}; ${KEY_PER_SITE}`, { code, reason, envelope: status });
  if (code === BAD_REQUEST) {
    if (reason === "unknown_checkpoint") {
      message += `; checkpoint '${checkpoint}' is not a model your key may run: use an id from the Models page or DreamClient.getModels() (GET /v1/models)`;
    } else if (reason === "client_outdated") {
      message += "; update @makermods/dreamlabs to the release that tracks the gateway's current dreamlabs-client";
    }
    return new DreamBadRequest(message, { code, reason, envelope: status });
  }
  const retryAfter = status.retry_after_s;
  return new DreamServerError(message, {
    code,
    reason,
    retryable: Boolean(status.retryable ?? false),
    retryAfterS: typeof retryAfter === "number" ? retryAfter : typeof retryAfter === "boolean" ? Number(retryAfter) : null,
    envelope: status,
  });
}

function timingOf(transport: Transport, block: unknown): Timing | null {
  const exchange = transport.lastExchange;
  if (exchange === null) return null;
  const gatewayMs = isObject(block) ? wholeMs(block.gateway_ms) : null;
  return {
    rttMs: exchange.rttMs,
    serveInferMs: isObject(block) ? wholeMs(block.serve_infer_ms) : null,
    gatewayMs,
    networkMs: gatewayMs !== null ? exchange.rttMs - gatewayMs : null,
    ikMs: null,
    totalMs: null,
    requestBytes: exchange.requestBytes,
    responseBytes: exchange.responseBytes,
  };
}

/** The message for an observation JSON cannot carry, naming the first key whose value is the cause. */
function unencodable(observation: Record<string, unknown>, error: Error): string {
  const advice = "send numbers, strings, arrays or typed arrays, and each camera frame as a base64 string, not bytes";
  for (const [key, value] of Object.entries(observation)) {
    try {
      encode({ [key]: value });
    } catch (cause) {
      return `observation key '${key}' cannot be sent as JSON (${(cause as Error).message}); ${advice}`;
    }
  }
  return `the observation cannot be sent as JSON (${error.message}); ${advice}`;
}

/** getAction()'s `commanded` as flat number arrays per chunk key (`client._commanded`). */
function commandedOf(commanded: Commanded | null, armJoints: Readonly<Record<string, number | null>>): Record<string, number[]> | null {
  if (commanded === null) return null;
  const out: Record<string, number[]> = {};
  for (const [key, value] of Object.entries(commanded)) {
    if (!(key in armJoints)) throw new DreamObservationError(`commanded key '${key}' is not a chunk key; it takes ${pyList(Object.keys(armJoints).sort())}`);
    if (!(Array.isArray(value) || ArrayBuffer.isView(value))) throw new DreamObservationError(`commanded '${key}' is not numbers`);
    const joints = Array.from(value as ArrayLike<unknown>);
    if (!joints.every((v) => typeof v === "number" && Number.isFinite(v))) {
      throw new DreamObservationError(`commanded '${key}' must be a flat run of finite numbers; got ${JSON.stringify(joints)}`);
    }
    const count = armJoints[key];
    if (count !== null && count !== undefined && joints.length !== count && joints.length !== count + 1) {
      throw new DreamObservationError(
        `commanded '${key}' must be the arm's ${count} joint angles, or its written row of ${count + 1} (the gripper last), all finite; got ${joints.length}`,
      );
    }
    out[key] = joints as number[];
  }
  return out;
}

/**
 * One arm's chunk as takeoverAt() takes it: rows of finite numbers, `count` rows of `width` when given (the plan the
 * smoother kept for the latest reply), else rows of one width, at least `minWidth` (the arm's joints). Rows may be
 * arrays or typed arrays; a hole (a sparse array's missing row or number) is refused like a non-number. Returns the
 * rows as new arrays.
 *
 * @throws DreamActionError It is not; `stage` says whether before or after the projection.
 */
function takeoverRows(value: unknown, key: string, count: number | null, width: number | null, minWidth: number | null, stage: "before" | "after"): number[][] {
  const what = stage === "before" ? `takeoverAt() action key '${key}'` : `takeoverAt() re-anchored '${key}'`;
  if (!Array.isArray(value) || !value.length) throw new DreamActionError(`${what} is not rows of numbers`, detail("shape", key));
  const isRow = (r: unknown): r is ArrayLike<unknown> => Array.isArray(r) || (ArrayBuffer.isView(r) && !(r instanceof DataView));
  // Indices, not every()/forEach(): those skip a hole, which would be returned and kept as `undefined`.
  for (let i = 0; i < value.length; i++) {
    if (!(i in value)) throw new DreamActionError(`${what} row ${i} is a hole (a sparse array's missing row), not a row of numbers`, detail("shape", key, i));
    if (!isRow(value[i])) throw new DreamActionError(`${what} is not rows of numbers: row ${i} is not a row`, detail("shape", key, i));
  }
  const rows = value as ArrayLike<unknown>[];
  const wide = width ?? rows[0]!.length;
  if (count !== null && rows.length !== count) throw new DreamActionError(`${what} has ${rows.length} rows; the chunk getAction() returned has ${count}`, detail("shape", key));
  if (minWidth !== null && wide < minWidth) throw new DreamActionError(`${what} rows are ${wide} wide; the arm has ${minWidth} joints`, detail("shape", key, 0));
  return rows.map((r, i) => {
    if (r.length !== wide) throw new DreamActionError(`${what} row ${i} is ${r.length} wide, not ${wide}`, detail("shape", key, i));
    const hole = firstHole(r);
    if (hole >= 0) throw new DreamActionError(`${what} row ${i} has a hole at column ${hole} (a sparse array's missing number)`, detail("shape", key, i));
    if (!every(r, (v) => typeof v === "number")) throw new DreamActionError(`${what} row ${i} holds a non-number`, detail("shape", key, i));
    if (!every(r as ArrayLike<number>, Number.isFinite)) throw new DreamActionError(`${what} row ${i} holds a non-finite number (NaN or an infinity)`, detail("non_finite", key, i));
    return Array.from(r as ArrayLike<number>);
  });
}

/** @throws DreamConfigError `processing` is not one the client takes, or is `raw` with IK or the EEF seam join. */
export function checkProcessing(processing: unknown, ik: boolean, kinematics: boolean): Processing {
  if (!PROCESSING.includes(processing as Processing)) {
    throw new DreamConfigError(`config key 'processing' must be one of ${pyList([...PROCESSING])}, got ${JSON.stringify(processing)}`, detail("range", "processing"));
  }
  if (processing === "raw" && (ik || kinematics)) {
    throw new DreamConfigError(
      `config key 'processing' is 'raw', which runs no post-processing, so config key '${ik ? "ik" : "kinematics"}' has nothing to run: leave it out, or use 'default'`,
      detail("range", "processing"),
    );
  }
  return processing as Processing;
}

/** @throws DreamConfigError `ikLeadMs` is not `auto` nor a finite number of milliseconds, at least 0. */
export function checkIkLead(lead: unknown): IkLead {
  if (lead === "auto" || (typeof lead === "number" && Number.isFinite(lead) && lead >= 0)) return lead;
  throw new DreamConfigError(`config key 'ikLeadMs' must be 'auto' or a finite number of milliseconds, at least 0, got ${JSON.stringify(lead)}`, detail("range", "ikLeadMs"));
}

/** getAction()'s `execution`, checked. */
function checkExecution(execution: unknown): ExecutionTiming {
  if (execution === undefined || execution === null) return {};
  if (!isObject(execution)) throw new DreamConfigError(`getAction() option 'execution' must be an object, got ${String(execution)}`, detail("shape", "execution"));
  for (const [key, value] of Object.entries(execution)) {
    if (key !== "observationAgeMs" && key !== "elapsedExecutionS") {
      throw new DreamConfigError(`getAction() option 'execution' takes 'observationAgeMs' and 'elapsedExecutionS', not '${key}'`, detail("shape", `execution.${key}`));
    }
    if (value !== undefined && !(typeof value === "number" && Number.isFinite(value) && value >= 0)) {
      throw new DreamConfigError(`getAction() option 'execution.${key}' must be a finite number, at least 0, got ${String(value)}`, detail("range", `execution.${key}`));
    }
  }
  return execution as ExecutionTiming;
}

export function checkTimeout(timeoutMs: unknown): void {
  if (!(Number.isInteger(timeoutMs) && (timeoutMs as number) >= 1 && (timeoutMs as number) <= MAX_TIMEOUT_MS)) {
    throw new DreamConfigError(`config key 'timeoutMs' must be an int of milliseconds in 1..${MAX_TIMEOUT_MS}, got ${JSON.stringify(timeoutMs)}`);
  }
}

/** Python's `str()` of an envelope field, or null when absent. */
function text(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "boolean") return value ? "True" : "False";
  return typeof value === "string" ? value : JSON.stringify(value);
}

/** An int millisecond count as the contract sends it, or null for anything else. */
function wholeMs(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) ? value : null;
}

