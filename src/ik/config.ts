/** The `ik` config key of `policy()` (`dream.client_ik`), as a type; no code, so the main entry imports it for free. */

import type { ArmName, Embodiment } from "./arms.js";

/** `eef` solves the reply's tool poses to joints; `joint` keeps the model's joint answer. */
export type IkHead = "eef" | "joint";
/** `traj` (the default) solves every row of the chunk in one problem; `tick` one row at a time. */
export type IkSolve = "traj" | "tick";

/**
 * Client-side IK for a model that answers hand poses (`dream.client_ik`): the client adds each arm's tool pose to the
 * observation, from forward kinematics of `state.<side>_arm` in radians, and with head `eef` turns the model's tool
 * poses into `q_r_chunk` and `q_l_chunk` (the arm's joints in radians, then the grip from 0 closed to 1 open).
 */
export interface IkConfig {
  /** A built-in arm by its dream_ik name (`metal`, `makers`, `so101_faas`, ...), or one {@link defineArm} built. */
  arm: ArmName | Embodiment | (string & {});
  head: IkHead;
  solve?: IkSolve;
  /** Per side, the reach area: `[x, y]` points (metres, that arm's base frame) the hand stays within 4 cm of. */
  reach?: Partial<Record<"right" | "left", [number, number][]>>;
  /**
   * Not in Python: every chunk's per-joint speed cap, rad/s (m/s for a prismatic joint), on top of the arm's own;
   * Infinity (the default) for the arm's caps only. The solve and the final projection both hold it.
   */
  speedCap?: number;
  /**
   * Not in Python: the cap of each arm's first chunk of an episode, rad/s, below `speedCap` (the MakerMods console runs
   * 0.5); `speedCap` by default.
   */
  startupSpeedCap?: number;
  /**
   * Not in Python: the grip's speed cap, grip units (0 closed to 1 open) per second, held after the [0, 1] clamp by
   * the final projection and by takeoverAt(). Each starts from the grip commanded last when `commanded` is the written
   * row (the grip its last value); otherwise getAction() starts from the measured `state.<side>_grip`, and takeoverAt()
   * from the row it re-anchors at. Infinity (the default) for none, as upstream, whose grip steps as the model answers.
   */
  gripSpeedCap?: number;
  /**
   * Not in Python: the platform's embodiment id of this arm (`emb_...`). Required by the head `joint`, which runs only
   * a model trained on exactly that embodiment (matching widths cannot tell another robot's joints from this arm's).
   * The head `eef` runs across embodiments and ignores it.
   */
  embodimentId?: string;
}
