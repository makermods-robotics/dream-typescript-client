/**
 * `@makermods/dreamlabs/ik`: client-side IK and FK for the arms dream_ik ships (`dream_ik`, `dream.client_ik`,
 * `dream.arm_file`), runtime-agnostic and dependency-free. The arm data is bundled, so nothing is fetched.
 *
 * - {@link arm} / {@link embodiment}: a built-in arm (`metal`, `makers`, `so101_faas`, ...) with `tool()` (FK),
 *   `jacobian()`, `frames()` and `kinematics()` (for the seam join's `kinematics` config key). Its data is frozen.
 * - {@link solveChunk}: dream_ik's chunk solve, tool poses to joints (tick or traj).
 * - {@link ClientIK}: what `policy({ ik })` runs; {@link defineArm}: an arm of your own from its URDF.
 * - {@link matToRot6} / {@link rot6ToMat}: the rotation_6d the poses travel in, for `tool()`'s matrices.
 */

export { ARM_NAMES, Arm, arm, Embodiment, embodiment, type ArmData, type ArmName, type EmbodimentData, type JointData } from "./arms.js";
export type { IkConfig, IkHead, IkSolve } from "./config.js";
export { solveChunk, type ServeSpec, type SolveChunkOptions } from "./decode.js";
export type { IkResult, Pose, TrajWeights } from "./solver.js";
export { ClientIK } from "./client-ik.js";
export { defineArm, type ArmFile } from "./urdf.js";
export { matToRot6, rot6ToMat, type Mat3, type Vec3 } from "../linalg.js";
