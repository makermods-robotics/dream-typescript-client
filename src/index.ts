/**
 * `@makermods/dreamlabs`: the TypeScript client for Dream Labs' Dream API, the counterpart of the Python
 * `dreamlabs-client` (module `dream`). Runtime-agnostic: browsers, Node 22+, Bun and Deno, standard Web APIs only.
 *
 * @example
 * ```ts
 * import { DreamClient } from "@makermods/dreamlabs";
 *
 * const client = new DreamClient({ proxyUrl: "/api/dream" }); // createDreamProxy() from "@makermods/dreamlabs/proxy"
 * await client.policy({ checkpoint: "dreamlabs/dream-1-beta", ik: { arm: "metal", head: "eef" }, embodiment });
 * client.setObservation(frames, observation); // every camera tick
 * const chunk = await client.getAction();     // when refillDue() says so
 * ```
 *
 * The 0.x public API is this entry, `./ik` (client-side IK and FK), `./node` (credentials, and a `node:http` adapter
 * for the proxy), `./browser` (camera frames) and `./proxy` (a server-side proxy for browser apps). `./internal` holds the building blocks (the wire, the strict checks, the seam join's helpers, the solver),
 * exported for tests and experiments, with no stability promise.
 */

export { DreamClient, type DreamClientOptions, type PolicyConfig, type RoundTrip } from "./facade.js";
export {
  IK_LEAD_SOLVES,
  PolicyClient,
  REASONS,
  type ActionDict,
  type Commanded,
  type Diagnostics,
  type DreamReason,
  type ExecutionTiming,
  type GetActionOptions,
  type IkLead,
  type IkTrace,
  type PolicyClientOptions,
  type Processing,
  type Robot,
  type Timing,
} from "./client.js";
export {
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
  DreamTimeout,
  DreamTrainingWarning,
  type DreamErrorDetail,
  type DreamServerErrorFields,
  type DreamWarning,
} from "./errors.js";
export { DREAM_1_BETA, DREAM_1_BETA_FLASH, type CheckpointId, type Model } from "./models.js";
export { FINE_TUNE_PREFIX, type TrainedInputs } from "./trained-inputs.js";
export { ObservationWindow } from "./window.js";
export { Pacer, type PacingMode } from "./pacing.js";
export { SeamSmoother } from "./smoothing.js";
export type { Kinematics } from "./seam-join.js";
export type { EmbodimentConfig, ShapeSpec } from "./embodiment.js";
export type { FetchLike } from "./transport.js";
export { CLIENT_VERSION, UPSTREAM, VERSION } from "./version.js";
export type { IkConfig, IkHead, IkSolve } from "./ik/config.js";
