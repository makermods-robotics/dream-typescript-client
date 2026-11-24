/**
 * `@makermods/dreamlabs/internal`: the building blocks under the public API, for tests, tools and experiments. No
 * stability promise: anything here may change in any 0.x release. Importing it pulls in the IK.
 */

export {
  BAD_REQUEST,
  checkProcessing,
  checkTimeout,
  END_SESSION_TIMEOUT_MS,
  errorFor,
  INTERNAL,
  MAX_TIMEOUT_MS,
  OK,
  OK_NEW_SESSION,
  OTHER_EMBODIMENT,
  PROCESSING,
  ROBOTS,
  SERVE_FAILED,
  TIMEOUT_MS,
  UNAUTHORIZED,
  UNAVAILABLE,
  YAM,
  type GetActionReply,
  type GetActionRequest,
  type PolicyIk,
  type StatusEnvelope,
} from "./client.js";
export { detail } from "./errors.js";
export {
  ASPECT_TOLERANCE,
  b64decode,
  casefold,
  checkedTrainedInputs,
  CLOSE_PROMPT,
  closestMatch,
  fetchTrainedInputs,
  fixed2,
  jpegSize,
  plain as plainPrompt,
  pyQuote,
  pyRepr,
  sequenceRatio,
  trainedNotes,
  trainingDataPath,
} from "./trained-inputs.js";
export {
  API_KEY_ENV,
  authorization,
  checkServiceUrl,
  DATA_URL_ENV,
  DEFAULT_DATA_URL,
  DEFAULT_FILE,
  DEFAULT_GATEWAY_URL,
  DEFAULT_PLATFORM_URL,
  FILE_ENV,
  GATEWAY_URL_ENV,
  NAMES,
  parseCredentialsFile,
  PLATFORM_URL_ENV,
  resolveCredentials,
  type Credentials,
} from "./credentials.js";
export { callService, listModels, modelRecord, PUBLIC_CHECKPOINTS, toModel, type FineTunedCheckpoint, type PublicCheckpoint, type ServiceOptions } from "./models.js";
export {
  check,
  checkPerKey,
  compileChecks,
  EEF_PARTS,
  isEmbodimentConfig,
  parseEmbodiment,
  WRIST_ROLES,
  YAM_BODY,
  type Body,
  type Checks,
  type ShapeEntry,
} from "./embodiment.js";
export { median, OBS_FPS, OBS_WINDOW, SLACK_CALLS, snapshot } from "./window.js";
export { ACTION_HZ, EXECUTION_HORIZON, RTT_WINDOW, RUNWAY_MARGIN_S, SYNC_ABOVE_MS } from "./pacing.js";
export {
  anchorRow,
  ARM_POSE_KEYS,
  FILTER_CONTEXT_ROWS,
  filterFrom,
  flatPose,
  MAX_OFFSET_RAD,
  nearestRow,
  oldMotion,
  savgolFilter,
  SAVGOL_POLYORDER,
  SAVGOL_WINDOW,
  SIDE_POSE_KEYS,
  UPSAMPLE,
  type Fence,
  type Projection,
  type Takeover,
} from "./smoothing.js";
export { blendedSteps, eefWindow, fitMotion, OLD_PATH_TIMES, RAMP_ROWS, smoothstep5, solve as seamSolve, velocityBlend, type Rows } from "./seam-join.js";
export { message as slowNetworkMessage, Watch as SlowNetworkWatch } from "./slow-network.js";
export {
  cancelled,
  checkClientVersion,
  CLIENT_INSTANCE_HEADER,
  CLIENT_VERSION_HEADER,
  encode,
  END_SESSION_PATH,
  GET_ACTION_PATH,
  Transport,
  USER_AGENT_PRODUCT,
  type Exchange,
  type TransportOptions,
} from "./transport.js";
export { every, firstFailing, firstHole, firstNonFinite, isNumbers } from "./finite.js";
export { actuatorHeads, checkAction, CHUNK_HEADS, type DeclaredHeads } from "./actions.js";
export { firstInexact, parseRaw, RawNumber, setOwn, unwrap, writeRaw, type RawJson } from "./json-raw.js";
export { invert, mul3, quaternion, ROT6_MIN_NORM, rotationMatrix, rotvec, rpyMatrix, solveLinear, transpose3 } from "./linalg.js";
export { extendPoses, headPoses, MAX_SOLVE_ROWS, SERVE, seamBlend, smoothPoses } from "./ik/decode.js";
export { capStep, maxStep, solve, solveTraj, TRAJ_FEW_JOINTS_ORI_SCALE, TRAJ_WEIGHTS } from "./ik/solver.js";
export { cameraSlots, checkIkConfig, clampToReach, EEF_STATE_PARTS, MODEL_JOINT_ROWS, REACH_RADIUS_M, REACH_SLACK_M, reachFence, SIDES, Wire } from "./ik/client-ik.js";
export { ARM_FILE_KEYS, ARM_FILE_OPTIONAL_KEYS, MAX_SPEED, parseXml, toolAxes, type XmlElement } from "./ik/urdf.js";
