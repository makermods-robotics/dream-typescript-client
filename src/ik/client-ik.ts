/**
 * Client-side IK for an EEF checkpoint (`dream.client_ik.ClientIK`): forward kinematics on the way out, the chunk
 * solve on the way back.
 *
 * A model that reads and answers tool poses runs no IK for the caller, so with the `ik` config key the client does
 * both ends, with dream_ik's arm models (the same that computed the training poses). The loop speaks the arm's URDF
 * units: `state.<side>_arm` is the arm's joints in radians, and each chunk the client returns is those joints, then the
 * grip (0 closed, 1 open). A FaaS beta robot's checkpoint reads its joints and gripper in the robot's recorded units,
 * and its arm spec says how: the client converts both ways. A model trained on YAM arms (the Dream-1 base models) is
 * sent zeros in its joint slots, the grip as `state.<side>_gripper` and the body's cameras in the model's roles, and
 * each request names the robot `other_embodiment`.
 */

import type { IkTrace } from "../client.js";
import { isEmbodimentConfig, WRIST_ROLES, YAM_BODY, type ShapeEntry } from "../embodiment.js";
import { detail, DreamActionError, DreamConfigError, DreamObservationError } from "../errors.js";
import { every, firstHole, isNumbers } from "../finite.js";
import { matToRot6 } from "../linalg.js";
import type { Model } from "../models.js";
import type { Rows } from "../seam-join.js";
import type { SeamSmoother } from "../smoothing.js";
import { pyList } from "../window.js";
import { ARM_NAMES, Embodiment, embodiment as builtIn, type Arm } from "./arms.js";
import type { IkConfig, IkHead, IkSolve } from "./config.js";
import { headPoses, MAX_SOLVE_ROWS, smoothPoses, solveChunk, extendPoses, SERVE } from "./decode.js";
import { capStep, maxStep, type Pose } from "./solver.js";

/** Per side: the chunk key its joints are returned under, and the observation key of its measured joints. */
export const SIDES: Readonly<Record<string, readonly [string, string]>> = {
  right: ["q_r_chunk", "state.right_arm"],
  left: ["q_l_chunk", "state.left_arm"],
};
/** The model's own first row of joints, which a solved reply drops. */
export const MODEL_JOINT_ROWS = ["q_r", "q_l"];
/** The state parts of a side, beside its gripper drive, that an EEF checkpoint reads. */
export const EEF_STATE_PARTS = ["arm", "eef_xyz", "eef_rot6", "grip"];
export const HEADS: readonly IkHead[] = ["eef", "joint"];
export const SOLVES: readonly IkSolve[] = ["tick", "traj"];
/** The reach area's brush, metres. */
export const REACH_RADIUS_M = 0.04;
/** How far past the reach area's edge a solved hand may go before the arm holds instead, metres. */
export const REACH_SLACK_M = 0.01;
/** YAM's joints: the slots a model trained on YAM arms reads. */
export const YAM_JOINTS = 6;
const EEF_PARTS: readonly [string, number][] = [["eef_xyz", 3], ["eef_rot6", 6], ["grip", 1]];
const IK_KEYS = ["arm", "head", "solve", "reach", "speedCap", "startupSpeedCap", "gripSpeedCap", "embodimentId"];

/**
 * Checks the `ik` config key's shape before anything is looked up for it (`client_ik.check_config`).
 *
 * @throws DreamConfigError It is not `{arm, head}` with optional `solve` and `reach`, or names an arm, a head or a
 *   solve the client does not take.
 */
export function checkIkConfig(config: unknown): asserts config is IkConfig {
  const keys = typeof config === "object" && config !== null ? Object.keys(config) : null;
  const unsupported = keys?.find((key) => key === "keep_out" || key === "region");
  if (unsupported) {
    throw new DreamConfigError(
      "config key 'ik' names 'keep_out' or 'region', which @makermods/dreamlabs does not implement (no collision cost): use the Python client for them",
      detail("unsupported", `ik.${unsupported}`),
    );
  }
  if (!keys || !["arm", "head"].every((key) => keys.includes(key)) || !keys.every((key) => IK_KEYS.includes(key))) {
    throw new DreamConfigError(`config key 'ik' must be an object with the keys 'arm', 'head' and optionally ${pyList(IK_KEYS.slice(2))}, got ${JSON.stringify(config)}`, detail("shape", "ik"));
  }
  const { arm, head, solve, speedCap, startupSpeedCap, gripSpeedCap, embodimentId } = config as IkConfig;
  if (solve !== undefined && !SOLVES.includes(solve)) throw new DreamConfigError(`config key 'ik' names the solve ${JSON.stringify(solve)}; it takes ${pyList([...SOLVES])}`, detail("range", "ik.solve"));
  for (const [key, cap] of [["speedCap", speedCap], ["startupSpeedCap", startupSpeedCap], ["gripSpeedCap", gripSpeedCap]] as const) {
    if (cap !== undefined && !(typeof cap === "number" && cap > 0)) {
      const unit = key === "gripSpeedCap" ? "grip units (0 closed to 1 open) per second" : "rad/s";
      throw new DreamConfigError(`config key 'ik.${key}' must be a positive number of ${unit} (Infinity for none), got ${String(cap)}`, detail("range", `ik.${key}`));
    }
  }
  if (embodimentId !== undefined && (typeof embodimentId !== "string" || !embodimentId)) {
    throw new DreamConfigError(`config key 'ik.embodimentId' must be the arm's embodiment id on the platform, a non-empty string, got ${JSON.stringify(embodimentId)}`, detail("shape", "ik.embodimentId"));
  }
  if (!(arm instanceof Embodiment)) {
    if (arm === "yam") {
      throw new DreamConfigError(
        "config key 'ik' names the arm 'yam', which @makermods/dreamlabs does not solve (its camera-pole obstacle and two-arm cost are not implemented)",
        detail("unsupported", "ik.arm"),
      );
    }
    if (typeof arm === "string" && arm.endsWith(".json")) throw new DreamConfigError("config key 'ik' names an arm file; in TypeScript, build the arm with defineArm(file, urdfText) from @makermods/dreamlabs/ik and pass it as 'arm'");
    if (typeof arm !== "string" || !ARM_NAMES.includes(arm)) {
      throw new DreamConfigError(`config key 'ik' names the arm ${JSON.stringify(arm)}; it takes ${pyList(ARM_NAMES)} or an arm defineArm() built`, detail("range", "ik.arm"));
    }
  }
  if (!HEADS.includes(head)) throw new DreamConfigError(`config key 'ik' names the head ${JSON.stringify(head)}; it takes ${pyList([...HEADS])}`, detail("range", "ik.head"));
}

/** One arm's map between the loop's units and the checkpoint's (`client_ik._Wire`). */
export class Wire {
  /** Spec indices of the joints the loop drives. */
  readonly free: readonly number[];
  /** Joints the loop drives. */
  readonly joints: number;
  /** Width of the checkpoint's `<side>_arm`. */
  readonly width: number;
  /** Whether the checkpoint reads and writes the loop's own units. */
  readonly own: boolean;
  private readonly dof: number;
  private readonly scale: readonly number[];
  private readonly offset: readonly number[];
  private readonly jaw: readonly number[] | null;

  constructor(arm: Arm, private readonly jawInArm: boolean, readonly driveKey: string | null) {
    this.free = arm.free;
    this.joints = this.free.length;
    this.dof = arm.dof;
    const units = arm.data.jointUnits;
    this.scale = units ? units[0]! : new Array<number>(this.dof).fill(1);
    this.offset = units ? units[1]! : new Array<number>(this.dof).fill(0);
    this.jaw = arm.data.jaw;
    this.width = jawInArm ? this.dof + 1 : this.dof;
    this.own = this.jaw === null && units === null && arm.data.heldJoints.length === 0 && !jawInArm;
  }

  /** The loop's joints as the spec's: the held joints put back, at 0. */
  spec(joints: readonly number[]): number[] {
    const out = new Array<number>(this.dof).fill(0);
    this.free.forEach((i, k) => (out[i] = joints[k]!));
    return out;
  }

  /** The gripper drive the checkpoint reads, from the grip (0 closed, 1 open). */
  drive(grip: number): number {
    const [closed, opened] = this.jaw!;
    return closed! + grip * (opened! - closed!);
  }

  /** The checkpoint's `<side>_arm`: every spec joint in the recorded units, then the drive when it rides there. */
  arm(joints: readonly number[], grip: number): number[] {
    const units = this.spec(joints).map((v, j) => (v - this.offset[j]!) / this.scale[j]!);
    return this.jawInArm ? [...units, this.drive(grip)] : units;
  }

  /** The loop's joints from the checkpoint's joint head rows. */
  jointsOf(rows: Rows): Rows {
    return rows.map((row) => this.free.map((i) => this.offset[i]! + this.scale[i]! * row[i]!));
  }
}

/** Adds the tool poses to each observation and, for head `eef`, solves each reply's tool poses to joint chunks. */
export class ClientIK {
  readonly head: IkHead;
  /** Per side, YAM's joint slots for a model trained on YAM arms (sent as zeros); null for any other model. */
  readonly yamJoints: number | null;
  /** Per body camera role, the model's role it is sent under. */
  readonly cameras: Record<string, string> = {};
  readonly wires: Record<string, Wire> = {};
  readonly embodiment: Embodiment;
  /** Per chunk key, the joints of the arm it drives. */
  readonly joints: Record<string, number> = {};
  readonly reach: Record<string, number[][]> = {};
  readonly fence: ((key: string, rows: Rows) => Rows) | null;
  /** The per-joint speed caps, rad/s: every chunk's, and an arm's first chunk of an episode's. */
  readonly speedCap: number;
  readonly startupSpeedCap: number;
  /** The grip's speed cap, grip units per second (Infinity: none). */
  readonly gripSpeedCap: number;
  private readonly solveMode: IkSolve;
  private readonly actionHz: number;

  /**
   * @throws DreamConfigError The config is not valid ({@link checkIkConfig}), `actionHz` is not a positive, finite
   *   number, or the model cannot run on this arm: a joint head of a model not trained on it (a YAM-trained model, one
   *   of another embodiment id, one whose embodiment config answers other joints, or one it cannot tell), or an EEF
   *   head of a fine-tune whose embodiment config answers no tool pose for one of the arm's sides.
   */
  constructor(config: IkConfig, actionHz: number, embodimentBlock: unknown, model: Model) {
    checkIkConfig(config);
    if (!(typeof actionHz === "number" && actionHz > 0 && Number.isFinite(actionHz))) {
      throw new DreamConfigError(`actionHz must be a positive, finite number of rows per second, got ${String(actionHz)}`, detail("range", "actionHz"));
    }
    if (config.head === "joint" && model.embodiment === YAM_BODY) {
      throw new DreamConfigError(
        `config key 'ik' names the head 'joint', but '${model.id}' was trained on YAM arms, so its joint answer is another ` +
          "robot's joints and is not run on this arm, for safety: use the head 'eef', or a fine-tune of this robot",
        detail("model_mismatch", "ik.head"),
      );
    }
    this.head = config.head;
    this.actionHz = actionHz;
    this.solveMode = config.solve ?? "traj";
    this.speedCap = config.speedCap ?? Infinity;
    this.startupSpeedCap = Math.min(config.startupSpeedCap ?? Infinity, this.speedCap);
    this.gripSpeedCap = config.gripSpeedCap ?? Infinity;
    this.embodiment = config.arm instanceof Embodiment ? config.arm : builtIn(config.arm);
    this.yamJoints = model.embodiment === YAM_BODY ? YAM_JOINTS : null;
    const roles = videoRoles(embodimentBlock);
    if (roles === null && this.yamJoints !== null) {
      throw new DreamConfigError(
        "config key 'ik' with a model trained on YAM arms needs the body's embodiment config, whose video names the " +
          "cameras the client sends in the model's slots: pass `embodiment`",
      );
    }
    if (roles !== null && model.cameras !== null) Object.assign(this.cameras, cameraSlots(roles, model.cameras, model.id));
    for (const side of this.embodiment.sides) {
      const arm = this.embodiment.sideArm(side);
      this.wires[side] = new Wire(arm, this.embodiment.jawInArm, driveKey(side, arm, this.embodiment.jawInArm, embodimentBlock, this.yamJoints === null));
      this.joints[SIDES[side]![0]] = this.wires[side]!.joints;
    }
    checkModelFits(config, model, embodimentBlock, this.wires);
    if (config.reach !== undefined) Object.assign(this.reach, parseReach(config.reach, this.embodiment.sides));
    this.fence = Object.keys(this.reach).length ? (key, rows) => this.reachHold(key, rows) : null;
  }

  /**
   * The final step on each arm's returned chunk (the client's {@link Projection}): every joint clamped to the arm's
   * margined limits (a joint `from` already holds past them may stay there, never go further) and each row kept within
   * one row's speed cap of the row before, from `from`, the direction of the step kept. The seam join and the reach
   * hold run after the solve and could otherwise step past what the solve was capped to.
   *
   * Not in Python, deliberately: the grip is clamped to [0, 1] too. The model's grip can leave it (-0.024 was seen
   * on an SO-101), and the loop maps it straight onto a gripper drive's range. With `gripSpeedCap`, each row's grip
   * also stays within one row's grip cap of the row before, from the grip in `from` (a written row's last value),
   * else the measured `state.<side>_grip` of `observation`, else from the first row.
   */
  readonly project = (key: string, rows: Rows, from: readonly number[] | null, first: boolean, observation?: Readonly<Record<string, unknown>> | null): Rows => {
    const side = Object.keys(SIDES).find((s) => SIDES[s]![0] === key);
    const wire = side ? this.wires[side] : undefined;
    if (!side || !wire) return rows;
    const arm = this.embodiment.sideArm(side);
    const n = wire.joints;
    const anchor = from && from.length >= n ? from.slice(0, n) : null;
    const lo = wire.free.map((i, j) => Math.min(arm.lo[i]!, anchor?.[j] ?? Infinity));
    const hi = wire.free.map((i, j) => Math.max(arm.hi[i]!, anchor?.[j] ?? -Infinity));
    const all = maxStep(arm, first ? this.startupSpeedCap : this.speedCap, this.actionHz);
    const step = wire.free.map((i) => all[i]!);
    let prev = anchor;
    const gripStep = this.gripSpeedCap / this.actionHz;
    let prevGrip = Number.isFinite(gripStep) ? gripFrom(from, n, observation, side) : null;
    return rows.map((row) => {
      const clamped = row.slice(0, n).map((v, j) => Math.min(hi[j]!, Math.max(lo[j]!, v)));
      const joints = prev ? capStep(prev, clamped, step).q : clamped;
      prev = joints;
      const grips = row.slice(n).map((g, c) => {
        const g01 = Math.min(1, Math.max(0, g));
        if (!Number.isFinite(gripStep) || c > 0) return g01;
        const out = prevGrip === null ? g01 : prevGrip + Math.min(gripStep, Math.max(-gripStep, g01 - prevGrip));
        prevGrip = out;
        return out;
      });
      return [...joints, ...grips];
    });
  };

  /** The strict checks of the body's observation, made to read what {@link observe} sends. */
  sentChecks(checks: Readonly<Record<string, ShapeEntry>>): Record<string, ShapeEntry> {
    const out: Record<string, ShapeEntry> = {};
    for (const [key, entry] of Object.entries(checks)) {
      const [group, role] = splitKey(key);
      out[group === "video" && role in this.cameras ? `video.${this.cameras[role]}` : key] = entry;
    }
    if (this.yamJoints === null) return out;
    for (const [side, wire] of Object.entries(this.wires)) {
      if (wire.driveKey !== null) delete out[wire.driveKey];
      out[SIDES[side]![1]] = { shape: [this.yamJoints] };
      out[`state.${side}_gripper`] = { shape: [1] };
    }
    return out;
  }

  /** The strict widths of the reply's `per_key`, made to read what the model answers. */
  replyWidths(widths: Readonly<Record<string, number>>): Record<string, number> {
    const out = { ...widths };
    if (this.yamJoints === null) return out;
    for (const [side, wire] of Object.entries(this.wires)) {
      if (wire.driveKey !== null) delete out[`action.${splitKey(wire.driveKey)[1]}`];
      if (`action.${side}_arm` in out) out[`action.${side}_arm`] = this.yamJoints;
    }
    return out;
  }

  /**
   * The observation with each arm's tool pose added (`state.<side>_eef_xyz`, `state.<side>_eef_rot6`), from forward
   * kinematics of `state.<side>_arm`; a beta robot's joints and gripper drive in the checkpoint's units; for a model
   * trained on YAM arms, zeros in its joint slots, the grip as `state.<side>_gripper` and the cameras in its roles.
   *
   * @throws DreamObservationError The observation carries a tool pose or a gripper drive itself, or lacks a side's
   *   joints or grip, or holds them as anything other than the arm's joints and one number.
   */
  observe(observation: Readonly<Record<string, unknown>>): Record<string, unknown> {
    const sides = this.embodiment.sides;
    const sent = sides.flatMap((side) => ["xyz", "rot6"].map((part) => `state.${side}_eef_${part}`)).filter((key) => key in observation).sort();
    if (sent.length) {
      throw new DreamObservationError(`observation keys ${pyList(sent)} are computed by the client from state.<side>_arm (config key 'ik'); leave them out`);
    }
    const drives = Object.values(this.wires).map((w) => w.driveKey).filter((key): key is string => key !== null && key in observation).sort();
    if (drives.length) {
      throw new DreamObservationError(`observation keys ${pyList(drives)} are the gripper drives the client computes from state.<side>_grip (config key 'ik'); leave them out`);
    }
    const out: Record<string, unknown> = { ...observation };
    for (const side of sides) {
      const wire = this.wires[side]!;
      const joints = measured(observation, SIDES[side]![1], wire.joints);
      const pose = this.embodiment.sideArm(side).tool(wire.spec(joints));
      out[`state.${side}_eef_xyz`] = [...pose.position];
      out[`state.${side}_eef_rot6`] = matToRot6(pose.rotation);
      if (this.yamJoints !== null) {
        out[SIDES[side]![1]] = new Array<number>(this.yamJoints).fill(0);
        out[`state.${side}_gripper`] = [grip(observation, `state.${side}_grip`)];
        continue;
      }
      if (wire.own) continue;
      const g = grip(observation, `state.${side}_grip`);
      out[SIDES[side]![1]] = wire.arm(joints, g);
      if (wire.driveKey !== null) out[wire.driveKey] = [wire.drive(g)];
    }
    if (Object.keys(this.cameras).length) renameCameras(out, this.cameras);
    return out;
  }

  /**
   * The reply with `q_r_chunk` and `q_l_chunk` solved from its tool poses (head `eef`), or built from a beta robot's
   * joint head (head `joint`): `[H][joints + 1]`, the arm's joints in radians, then the grip; the model's own `q_r`
   * and `q_l` are dropped. Each arm's solve starts from the joints it is commanded to when the loop installs the chunk
   * (the seam join's takeover), or the measured joints for an episode's first chunk and an arm off its plan.
   *
   * @throws DreamActionError The reply's `per_key` lacks a part the client solves from, or holds one of the wrong shape.
   */
  solve(
    action: Record<string, unknown>,
    observation: Readonly<Record<string, unknown>>,
    smoother: SeamSmoother,
    inFlightS: number,
    commanded: Readonly<Record<string, number[]>> | null,
    trace: Record<string, IkTrace> | null = null,
  ): Record<string, unknown> {
    if (this.head === "joint") return Object.values(this.wires).every((wire) => wire.own) ? action : this.jointHead(action);
    const sides = this.embodiment.sides;
    const widths: Record<string, number> = {};
    for (const side of sides) for (const [part, width] of EEF_PARTS) widths[`${side}_${part}`] = width;
    const parts = replyParts(action, widths);
    const count = parts[`${sides[0]}_grip`]!.length;
    const out = armReply(action);
    for (const side of sides) {
      const [chunkKey, poseKey] = SIDES[side]!;
      const wire = this.wires[side]!;
      const arm = this.embodiment.sideArm(side);
      const now = measured(observation, poseKey, wire.joints);
      const takeover = smoother.takeoverHistory(chunkKey, count, wire.joints + 1, now, inFlightS, commanded?.[chunkKey] ?? null);
      const skip = takeover?.clock ?? 0;
      const start = takeover?.joints ?? now;
      const before = takeover?.before ?? [now, now];
      const speedCap = smoother.has(chunkKey) ? this.speedCap : this.startupSpeedCap;
      // dream's filter over the whole chunk first: it is centred, so filtering after the cut would bend the last rows.
      const targets = headPoses(parts[`${side}_eef_xyz`]!, parts[`${side}_eef_rot6`]!, [`action.${side}_eef_xyz`, `action.${side}_eef_rot6`]);
      const poses = smoothPoses(targets, SERVE.smoothWindow, SERVE.smoothPolyorder);
      // The rows from the clock row on, padded back to the full length: `traj` continues the last motion, `tick` holds.
      const padded = pad(poses, skip, this.solveMode === "traj");
      const xyz = padded.map((p) => [...p.position]);
      if (this.reach[side]) {
        const clamped = clampToReach(xyz.map(([x, y]) => [x!, y!]), this.reach[side]);
        clamped.forEach(([x, y], k) => ((xyz[k]![0] = x!), (xyz[k]![1] = y!)));
      }
      const solved = solveChunk(arm, xyz, padded.map((p) => matToRot6(p.rotation)), wire.spec(start), {
        hz: this.actionHz,
        speedCap,
        serve: { smoothWindow: 0, seamBlendTicks: 0, solve: this.solveMode },
        qBefore: before.map((row) => wire.spec(row)),
      });
      const kept = count - skip;
      if (trace) {
        trace[side] = { clock: skip, speedCap, posMiss: solved.posMiss.slice(0, kept), angMiss: solved.angMiss.slice(0, kept), atCap: solved.atCap.slice(0, kept) };
      }
      const rows = solved.q.slice(0, kept).map((row) => wire.free.map((i) => row[i]!));
      const joints = [...Array.from({ length: skip }, () => [...start]), ...rows];
      out[chunkKey] = joints.map((row, k) => [...row, parts[`${side}_grip`]![k]![0]!]);
    }
    return out;
  }

  /** A beta robot's reply with `q_r_chunk` and `q_l_chunk` built from its joint head, in the loop's units. */
  jointHead(action: Record<string, unknown>): Record<string, unknown> {
    const widths: Record<string, number> = {};
    for (const [side, wire] of Object.entries(this.wires)) widths[`${side}_arm`] = wire.width;
    for (const side of Object.keys(this.wires)) widths[`${side}_grip`] = 1;
    const parts = replyParts(action, widths);
    const out = armReply(action);
    for (const [side, wire] of Object.entries(this.wires)) {
      const grips = parts[`${side}_grip`]!;
      out[SIDES[side]![0]] = wire.jointsOf(parts[`${side}_arm`]!).map((row, k) => [...row, grips[k]![0]!]);
    }
    return out;
  }

  /** One arm's joined chunk held where its hand would leave its reach area (`ClientIK.fence`). */
  private reachHold(key: string, rows: Rows): Rows {
    const side = Object.keys(SIDES).find((s) => SIDES[s]![0] === key);
    const points = side ? this.reach[side] : undefined;
    if (!side || !points) return rows;
    const arm = this.embodiment.sideArm(side);
    const wire = this.wires[side]!;
    const hands = rows.map((row) => {
      const p = arm.tool(wire.spec(row.slice(0, -1))).position;
      return [p[0], p[1]];
    });
    const row = reachFence(hands, points);
    if (row === null) return rows;
    return rows.map((r, k) => (k <= row ? r : [...rows[row]!.slice(0, -1), r.at(-1)!]));
  }
}

/** Which of the model's camera roles each of the body's cameras fills (`client_ik.camera_slots`). */
export function cameraSlots(roles: readonly string[], modelRoles: readonly string[], checkpoint: string): Record<string, string> {
  const wrists: readonly string[] = WRIST_ROLES;
  const scene = roles.filter((role) => !wrists.includes(role));
  const modelScene = modelRoles.filter((role) => !wrists.includes(role));
  const slots: Record<string, string> =
    scene.length === 1 && modelScene.length === 1
      ? { [scene[0]!]: modelScene[0]!, ...Object.fromEntries(roles.filter((r) => wrists.includes(r)).map((r) => [r, r])) }
      : Object.fromEntries(roles.map((r) => [r, r]));
  const missing = modelRoles.filter((role) => !Object.values(slots).includes(role)).sort();
  const extra = Object.entries(slots).filter(([, role]) => !modelRoles.includes(role)).map(([role]) => role).sort();
  if (missing.length || extra.length) {
    throw new DreamConfigError(
      `the body's cameras ${pyList([...roles])} cannot be sent the way '${checkpoint}' reads them, which is ${pyList([...modelRoles])} ` +
        `(one scene camera, then ${wrists.join(" and ")}, matched by name): missing ${pyList(missing)}, extra ${pyList(extra)}. Name the body's cameras after those roles`,
    );
  }
  return slots;
}

/**
 * Hand targets kept inside a reach area seen from above, the disks of {@link REACH_RADIUS_M} around its points: a row
 * outside moves straight toward its nearest point until it is exactly the radius away (`client_ik.clamp_to_reach`).
 */
export function clampToReach(xy: readonly (readonly number[])[], points: readonly (readonly number[])[]): number[][] {
  return xy.map(([x, y]) => {
    let best = 0;
    let bestScore = -Infinity;
    points.forEach(([px, py], i) => {
      const score = x! * px! + y! * py! - (px! * px! + py! * py!) / 2;
      if (score > bestScore) [best, bestScore] = [i, score];
    });
    const [px, py] = points[best]!;
    const ax = x! - px!;
    const ay = y! - py!;
    const distance = Math.hypot(ax, ay);
    return distance > REACH_RADIUS_M ? [px! + (ax * REACH_RADIUS_M) / distance, py! + (ay * REACH_RADIUS_M) / distance] : [x!, y!];
  });
}

/** The row an arm holds at so its hand stays in its reach area, or null when no row leaves (`client_ik.reach_fence`). */
export function reachFence(hands: readonly (readonly number[])[], points: readonly (readonly number[])[]): number | null {
  const distance = hands.map(([x, y]) => Math.min(...points.map(([px, py]) => Math.hypot(x! - px!, y! - py!))));
  let closest = Infinity;
  for (let i = 1; i < distance.length; i++) {
    closest = Math.min(closest, distance[i - 1]!);
    if (distance[i]! > Math.max(REACH_RADIUS_M + REACH_SLACK_M, closest + 1e-3)) return i - 1;
  }
  return null;
}

function parseReach(reach: unknown, sides: readonly string[]): Record<string, number[][]> {
  if (typeof reach !== "object" || reach === null || !Object.keys(reach).length || !Object.keys(reach).every((s) => sides.includes(s))) {
    throw new DreamConfigError(`config key 'ik' has 'reach' ${JSON.stringify(reach)}: it must be an object with some of the keys ${pyList([...sides].sort())}, each a list of [x, y] points in metres`);
  }
  const out: Record<string, number[][]> = {};
  for (const [side, points] of Object.entries(reach as Record<string, unknown>)) {
    if (!Array.isArray(points) || !points.length) throw new DreamConfigError(`config key 'ik' has 'reach.${side}' ${JSON.stringify(points)}: it must be a list of [x, y] points, at least one`);
    for (const point of points) {
      if (!(Array.isArray(point) && point.length === 2 && isNumbers(point, true))) {
        throw new DreamConfigError(`config key 'ik' has a bad point ${JSON.stringify(point)} in 'reach.${side}': it must be [x, y], two finite numbers in metres`);
      }
    }
    out[side] = points.map((p: number[]) => [p[0]!, p[1]!]);
  }
  return out;
}

/** The observation key a side's gripper drive goes out under (`client_ik._drive_key`). */
function driveKey(side: string, arm: Arm, jawInArm: boolean, block: unknown, read: boolean): string | null {
  if (arm.data.jaw === null || jawInArm) return null;
  const state = typeof block === "object" && block !== null ? (block as Record<string, unknown>).state : undefined;
  if (typeof state !== "object" || state === null || Array.isArray(state)) {
    throw new DreamConfigError(
      "config key 'ik' needs the body's embodiment config, whose state names each arm's gripper part: pass `embodiment` " +
        "(DreamClient.policy() loads it for your own fine-tunes with `strict` on)",
    );
  }
  const known = EEF_STATE_PARTS.map((part) => `${side}_${part}`);
  const others = Object.keys(state).filter((part) => part.startsWith(`${side}_`) && !known.includes(part)).sort();
  if (!others.length && !read) return null;
  if (others.length !== 1) {
    throw new DreamConfigError(
      `config key 'embodiment' must name one gripper part of the ${side} arm beside ${pyList([...known].sort())}, got ${pyList(others)}` +
        (others.length ? "" : "; a fine-tune reads the gripper, so train it on an embodiment that names one"),
    );
  }
  return `state.${others[0]}`;
}

/**
 * Whether the model may run on this arm (`ClientIK`'s constructor): the two heads take different rules. A joint head
 * runs the model's joints as this arm's, so the model must have been trained on this arm: `ik.embodimentId` is
 * required and must be the model's embodiment, and an embodiment config, when known (the model's record first, then
 * the `embodiment` key), must answer a `<side>_arm` action as wide as each side's wire. An EEF head is cross-embodiment by design (tool poses, solved here): a fine-tune's embodiment
 * config, when known, must answer each side's tool pose and grip.
 */
function checkModelFits(config: IkConfig, model: Model, block: unknown, wires: Readonly<Record<string, Wire>>): void {
  if (model.embodiment === YAM_BODY) return;
  const body = model.embodimentConfig ?? (isEmbodimentConfig(block) ? block : null);
  const action = body !== null && typeof body.action === "object" && body.action !== null ? (body.action as Record<string, unknown>) : null;
  const width = (part: string): number | null => {
    const span = action?.[part];
    return Array.isArray(span) && span.length === 2 && every(span, Number.isInteger) ? span[1] - span[0] : null;
  };
  const refuse = (why: string): never => {
    throw new DreamConfigError(`config key 'ik' cannot run '${model.id}' (embodiment ${JSON.stringify(model.embodiment)}) on this arm: ${why}`, detail("model_mismatch", "ik"));
  };
  if (config.head === "joint") {
    // Matching widths say nothing of joint order, units or which robot: only the arm's own id vouches for its joints.
    if (config.embodimentId === undefined) {
      refuse("the client cannot tell that it was trained on this arm (an embodiment config's widths cannot say which robot, nor its joint order or units): pass ik.embodimentId, the arm's embodiment id on the platform, or use the head 'eef'");
    }
    if (model.embodiment !== config.embodimentId) {
      refuse(`it was trained on another embodiment than this arm's '${config.embodimentId}', so its joint answer is another robot's joints, not run here for safety; use the head 'eef'`);
    }
    if (action === null) return;
    for (const [side, wire] of Object.entries(wires)) {
      const found = width(`${side}_arm`);
      if (found !== wire.width) {
        refuse(`its embodiment config's '${side}_arm' action is ${found === null ? "missing" : `${found} wide`}, where this arm's ${side} side takes ${wire.width} joint values; its joint answer is not this arm's`);
      }
    }
    return;
  }
  if (action === null) return;
  for (const side of Object.keys(wires)) {
    for (const [part, size] of EEF_PARTS) {
      if (width(`${side}_${part}`) !== size) refuse(`its embodiment config answers no '${side}_${part}' (${size} wide), a tool pose the head 'eef' solves from for the ${side} arm`);
    }
  }
}

function videoRoles(block: unknown): string[] | null {
  const video = typeof block === "object" && block !== null ? (block as Record<string, unknown>).video : undefined;
  return typeof video === "object" && video !== null && !Array.isArray(video) ? Object.keys(video) : null;
}

function renameCameras(observation: Record<string, unknown>, cameras: Readonly<Record<string, string>>): void {
  const sent = Object.keys(observation).filter((key) => key.startsWith("video.")).map((key) => splitKey(key)[1]);
  const missing = Object.keys(cameras).filter((role) => !sent.includes(role)).sort();
  const extra = sent.filter((role) => !(role in cameras)).sort();
  if (missing.length || extra.length) {
    throw new DreamObservationError(`the observation's cameras must be the body's ${pyList(Object.keys(cameras).sort())}: missing ${pyList(missing)}, extra ${pyList(extra)}`);
  }
  const windows = Object.keys(cameras).map((role) => {
    const window = observation[`video.${role}`];
    delete observation[`video.${role}`];
    return [role, window] as const;
  });
  for (const [role, window] of windows) observation[`video.${cameras[role]}`] = window;
}

const splitKey = (key: string): [string, string] => {
  const dot = key.indexOf(".");
  return dot < 0 ? [key, ""] : [key.slice(0, dot), key.slice(dot + 1)];
};

/** `value` as numbers: a typed array, or a plain array of numbers with no hole; else null. */
function numbers(value: unknown): number[] | null {
  if (ArrayBuffer.isView(value)) return Array.from(value as unknown as ArrayLike<number>, Number);
  if (Array.isArray(value) && isNumbers(value)) return value as number[];
  return null;
}

/** One arm's measured joints (`client_ik._measured`). */
function measured(observation: Readonly<Record<string, unknown>>, key: string, joints: number): number[] {
  if (!(key in observation)) throw new DreamObservationError(`observation has no '${key}', the arm's joints the tool pose is computed from`);
  const values = numbers(observation[key]);
  if (values === null) throw new DreamObservationError(`observation key '${key}' is not numbers`);
  if (values.length !== joints) throw new DreamObservationError(`observation key '${key}' has shape (${values.length},); the arm has ${joints} joints`);
  if (!values.every(Number.isFinite)) throw new DreamObservationError(`observation key '${key}' holds a non-finite number; the arm's joints must be finite radians`);
  return values;
}

/** The grip the grip cap steps from: `from`'s past the arm's joints, else the measured one, else null (none known). */
function gripFrom(from: readonly number[] | null, joints: number, observation: Readonly<Record<string, unknown>> | null | undefined, side: string): number | null {
  const given = from && from.length > joints ? from[joints]! : (observation?.[`state.${side}_grip`] as unknown);
  const value = Array.isArray(given) || ArrayBuffer.isView(given) ? (given as ArrayLike<unknown>)[0] : given;
  return typeof value === "number" && Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : null;
}

/** One arm's grip, 0 closed to 1 open (`client_ik._grip`): one number, in a list of one. */
function grip(observation: Readonly<Record<string, unknown>>, key: string): number {
  if (!(key in observation)) throw new DreamObservationError(`observation has no '${key}', the gripper (0 closed, 1 open) the drive is made from`);
  const values = numbers(observation[key]);
  if (values === null) throw new DreamObservationError(`observation key '${key}' is not numbers`);
  if (values.length !== 1) throw new DreamObservationError(`observation key '${key}' has shape (${values.length},); it needs (1,)`);
  if (!Number.isFinite(values[0])) throw new DreamObservationError(`observation key '${key}' holds a non-finite number; the grip is 0 closed to 1 open`);
  return values[0]!;
}

function armReply(action: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(action).filter(([key]) => !MODEL_JOINT_ROWS.includes(key)));
}

/** Parts of the reply's `per_key`, each as rows of its width, all with the same number of rows (`client_ik._parts`). */
function replyParts(action: Record<string, unknown>, widths: Readonly<Record<string, number>>): Record<string, Rows> {
  const perKey = action.per_key;
  if (typeof perKey !== "object" || perKey === null || Array.isArray(perKey)) {
    throw new DreamActionError("action has no 'per_key' dict, where the model returns the parts the client solves from", detail("shape", "per_key"));
  }
  const parts: Record<string, Rows> = {};
  for (const [part, width] of Object.entries(widths)) {
    const key = `action.${part}`;
    if (!(key in perKey)) throw new DreamActionError(`action 'per_key' lacks '${key}', which config key 'ik' solves from`, detail("shape", `per_key.${key}`));
    let rows = (perKey as Record<string, unknown>)[key];
    if (!Array.isArray(rows)) throw new DreamActionError(`action 'per_key' entry '${key}' is not rows of numbers`, detail("shape", `per_key.${key}`));
    // Holes first: every(), forEach() and map() skip them, so a sparse entry would pass and be solved as `undefined`.
    const hole = firstHole(rows);
    if (hole >= 0) throw new DreamActionError(`action 'per_key' entry '${key}' row ${hole} is a hole (a sparse array's missing row)`, detail("shape", `per_key.${key}`, hole));
    if (width === 1 && isNumbers(rows)) rows = rows.map((v) => [v]);
    if (!((rows as unknown[]).length > 0 && (rows as unknown[]).length <= MAX_SOLVE_ROWS)) {
      throw new DreamActionError(`action 'per_key' entry '${key}' has ${(rows as unknown[]).length} rows; the client solves 1 to ${MAX_SOLVE_ROWS}`, detail("range", `per_key.${key}`));
    }
    (rows as unknown[]).forEach((row, i) => {
      if (!(Array.isArray(row) && row.length === width && isNumbers(row))) {
        throw new DreamActionError(`action 'per_key' entry '${key}' is not rows of ${width} numbers; it needs (H, ${width})`, detail("shape", `per_key.${key}`, i));
      }
      if (!row.every(Number.isFinite)) {
        throw new DreamActionError(`action 'per_key' entry '${key}' row ${i} holds a non-finite number (NaN or an infinity)`, detail("non_finite", `per_key.${key}`, i));
      }
    });
    parts[part] = rows as Rows;
  }
  const counts = Object.fromEntries(Object.entries(parts).map(([part, rows]) => [`action.${part}`, rows.length]));
  if (new Set(Object.values(counts)).size !== 1) throw new DreamActionError(`action 'per_key' parts disagree on the number of rows: ${JSON.stringify(counts)}`, detail("shape", "per_key"));
  return parts;
}

/** The poses from `skip` on, padded back to the same length: the last rows' motion continued, or a hold. */
function pad(poses: readonly Pose[], skip: number, continueMotion: boolean): Pose[] {
  const kept = poses.slice(skip);
  if (continueMotion) return extendPoses(kept, skip);
  return [...kept, ...Array.from({ length: skip }, () => poses.at(-1)!)];
}

