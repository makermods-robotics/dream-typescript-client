/**
 * The arms the client solves IK for (`dream_ik.specs`), and their kinematics: every arm is data, the solver is generic.
 *
 * Each arm is its URDF chain from the root link to the flange, evaluated as URDF does (per joint, its origin xyz and
 * fixed-axis roll-pitch-yaw in the parent link, then a turn about, or a slide along, its axis), and the tool frame on
 * the flange (`ToolPose`: tool = flange @ [rotation | offset]). Joint vectors are in the spec's joint order; a joint the
 * spec lists that is not on the path to the flange (SO-101's joint6 pad) moves nothing on the tool path.
 *
 * The built-in arms are `dream.client_ik.ARMS` less `yam` (its camera-pole obstacle and two-arm cost are not
 * implemented here); {@link defineArm} (`urdf.ts`) adds an arm of your own from its URDF, as an arm file does.
 */

import { detail, DreamConfigError } from "../errors.js";
import { add3, apply3, axisAngle, cross3, IDENTITY3, mul3, rpyMatrix, type Mat3, type Vec3 } from "../linalg.js";
import type { Kinematics } from "../seam-join.js";
import { EMBODIMENT_DATA, type ArmName } from "./arms.data.js";

/** One URDF joint on the chain, as written in the URDF. */
export interface JointData {
  readonly name: string;
  type: "fixed" | "revolute" | "prismatic";
  readonly xyz: readonly number[];
  readonly rpy: readonly number[];
  axis?: readonly number[];
}

/** One arm type as dream_ik's `ArmSpec` holds it. An {@link Arm}'s is a frozen copy: changing it changes no arm. */
export interface ArmData {
  readonly urdf: string;
  /** Actuated joints, command order. */
  readonly joints: readonly string[];
  /** Joints the IK keeps at 0 (margined limits 0..0): a five-joint arm's dummy joint6 on a six-joint wire. */
  readonly heldJoints: readonly string[];
  readonly flange: string;
  readonly root: string;
  /** The URDF joints from the root link to the flange. */
  readonly chain: readonly JointData[];
  /** Per joint (spec order), its URDF `[lower, upper]`. */
  readonly limits: readonly (readonly number[])[];
  readonly tool: { readonly offset: readonly number[]; readonly rotation: readonly (readonly number[])[] };
  readonly restPose: readonly number[];
  readonly restWeight: readonly number[];
  /** Per joint, rad/s (m/s for a prismatic joint): the direction-preserving cap on the command. */
  readonly maxSpeed: readonly number[];
  readonly limitMargin: number;
  /** A FaaS bake's gripper drive, `[closed, open]`; null for none. */
  readonly jaw: readonly number[] | null;
  /** Recorded joint units, `[scale[], offset[]]` with radians = offset + scale * unit; null for radians. */
  readonly jointUnits: readonly (readonly number[])[] | null;
  /** IKWeights: pos, ori, limit, last_pose (collision weights are unused here: no obstacles). */
  readonly weights: Readonly<Record<string, number>>;
}

/** A robot: one arm spec for every side, or one per side. */
export interface EmbodimentData {
  readonly sides: readonly string[];
  /** Trossen's FaaS layout: the checkpoint's `<side>_arm` is the arm joints, then the jaw drive. */
  readonly jawInArm: boolean;
  readonly arm: ArmData | Readonly<Record<string, ArmData>>;
}

/** `value` and everything in it frozen. */
function frozen<T>(value: T): T {
  if (typeof value === "object" && value !== null && !Object.isFrozen(value)) {
    for (const item of Object.values(value)) frozen(item);
    Object.freeze(value);
  }
  return value;
}

/** The names of the built-in arms. */
export const ARM_NAMES: readonly string[] = Object.freeze(Object.keys(EMBODIMENT_DATA));
export type { ArmName };

interface CompiledJoint {
  type: "fixed" | "revolute" | "prismatic";
  xyz: Vec3;
  rotation: Mat3;
  axis: Vec3;
  /** The spec joint it is driven by, or -1 for a fixed joint. */
  index: number;
}

/** One arm, compiled: its kinematics and the solver's view of it (`solver._Arm`), joint vectors in spec order. */
export class Arm {
  /** The embodiment's name. */
  readonly name: string;
  readonly side: string;
  readonly data: ArmData;
  readonly dof: number;
  /** Indices of the joints the loop drives (every joint but the held ones). */
  readonly free: readonly number[];
  /** Margined limits; a held joint's are 0..0. */
  readonly lo: readonly number[];
  readonly hi: readonly number[];
  readonly toolOffset: Readonly<Vec3>;
  readonly toolRotation: Readonly<Mat3>;
  private readonly chain: CompiledJoint[];

  /** Keeps a frozen copy of `data`. @throws DreamConfigError `data` is inconsistent. */
  constructor(name: string, side: string, data: ArmData) {
    this.name = name;
    this.side = side;
    data = frozen(structuredClone(data));
    this.data = data;
    this.dof = data.joints.length;
    for (const key of ["restPose", "restWeight", "maxSpeed", "limits"] as const) {
      if (data[key].length !== this.dof) throw new DreamConfigError(`arm ${name}: ${key} has ${data[key].length} values for ${this.dof} joints`);
    }
    const held = new Set(data.heldJoints);
    this.free = Object.freeze(data.joints.map((_, i) => i).filter((i) => !held.has(data.joints[i]!)));
    this.lo = Object.freeze(data.limits.map(([lower], i) => (held.has(data.joints[i]!) ? 0 : lower! + data.limitMargin)));
    this.hi = Object.freeze(data.limits.map(([, upper], i) => (held.has(data.joints[i]!) ? 0 : upper! - data.limitMargin)));
    this.toolOffset = data.tool.offset as Vec3;
    this.toolRotation = Object.freeze(data.tool.rotation.flat()) as Mat3;
    this.chain = data.chain.map((joint) => {
      const index = joint.type === "fixed" ? -1 : data.joints.indexOf(joint.name);
      if (joint.type !== "fixed" && index < 0) throw new DreamConfigError(`arm ${name}: chain joint ${joint.name} is not one of its joints`);
      // A URDF axis a hair off unit length (bracketbot's, by ~5e-10) is normalized, as URDF readers do.
      const axis = (joint.axis ?? [0, 0, 1]) as Vec3;
      const length = Math.hypot(...axis);
      if (!(length > 0) || !Number.isFinite(length)) throw new DreamConfigError(`arm ${name}: chain joint ${joint.name} has a zero axis`, detail("shape", "axis"));
      return { type: joint.type, xyz: joint.xyz as Vec3, rotation: rpyMatrix(joint.rpy as Vec3), axis: axis.map((v) => v / length) as Vec3, index };
    });
  }

  /** The tool pose in the arm's base frame: position (m) and row-major rotation. `q`: every spec joint. */
  tool(q: ArrayLike<number>): { position: Vec3; rotation: Mat3 } {
    return this.walk(q, null);
  }

  /** The tool's `[6][dof]` Jacobian in the base frame: linear velocity rows, then angular. */
  jacobian(q: ArrayLike<number>): number[][] {
    const columns: { joint: Vec3; axis: Vec3; type: CompiledJoint["type"]; index: number }[] = [];
    const tool = this.walk(q, columns);
    const out = Array.from({ length: 6 }, () => new Array<number>(this.dof).fill(0));
    for (const { joint, axis, type, index } of columns) {
      const linear = type === "prismatic" ? axis : cross3(axis, [tool.position[0] - joint[0], tool.position[1] - joint[1], tool.position[2] - joint[2]]);
      for (let r = 0; r < 3; r++) {
        out[r]![index]! += linear[r]!;
        if (type === "revolute") out[r + 3]![index]! += axis[r]!;
      }
    }
    return out;
  }

  /** Every frame of the chain, base to tool, for drawing the arm: the base, each joint's child link, the tool. */
  frames(q: ArrayLike<number>): { name: string; position: Vec3; rotation: Mat3 }[] {
    const out: { name: string; position: Vec3; rotation: Mat3 }[] = [{ name: this.data.root, position: [0, 0, 0], rotation: [...IDENTITY3] as Mat3 }];
    const tool = this.walk(q, null, (i, position, rotation) => out.push({ name: this.data.chain[i]!.name, position, rotation }));
    out.push({ name: "tool", ...tool });
    return out;
  }

  /** {@link Kinematics} of this arm for the seam join, over the joints the loop drives. */
  kinematics(): Kinematics {
    const full = (joints: readonly number[]) => {
      const q = new Array<number>(this.dof).fill(0);
      this.free.forEach((i, k) => (q[i] = joints[k]!));
      return q;
    };
    return {
      pose: (joints) => this.tool(full(joints)),
      jacobian: (joints) => this.jacobian(full(joints)).map((row) => this.free.map((i) => row[i]!)),
    };
  }

  private walk(
    q: ArrayLike<number>,
    columns: { joint: Vec3; axis: Vec3; type: CompiledJoint["type"]; index: number }[] | null,
    visit?: (i: number, position: Vec3, rotation: Mat3) => void,
  ): { position: Vec3; rotation: Mat3 } {
    if (q?.length !== this.dof) throw new DreamConfigError(`arm ${this.name} takes ${this.dof} joints, got ${q?.length}`, detail("shape", "q"));
    for (let i = 0; i < this.dof; i++) if (!Number.isFinite(q[i])) throw new DreamConfigError(`arm ${this.name}: joint ${i} is ${q[i]}, not a finite number`, detail("non_finite", "q", i));
    let rotation: Mat3 = [...IDENTITY3] as Mat3;
    let position: Vec3 = [0, 0, 0];
    this.chain.forEach((joint, i) => {
      position = add3(position, apply3(rotation, joint.xyz));
      rotation = mul3(rotation, joint.rotation);
      if (joint.type !== "fixed") {
        const value = Number(q[joint.index]);
        columns?.push({ joint: position, axis: apply3(rotation, joint.axis), type: joint.type, index: joint.index });
        if (joint.type === "revolute") rotation = mul3(rotation, axisAngle(joint.axis, value));
        else position = add3(position, apply3(rotation, [joint.axis[0] * value, joint.axis[1] * value, joint.axis[2] * value]));
      }
      visit?.(i, position, rotation);
    });
    return { position: add3(position, apply3(rotation, this.toolOffset)), rotation: mul3(rotation, this.toolRotation) };
  }
}

/** A robot: its sides and each side's {@link Arm} (`dream_ik.specs.Embodiment`). */
export class Embodiment {
  readonly sides: readonly string[];
  readonly jawInArm: boolean;
  private readonly arms: Map<string, Arm>;

  constructor(readonly name: string, data: EmbodimentData) {
    this.sides = Object.freeze([...data.sides]);
    this.jawInArm = data.jawInArm;
    const shared = "urdf" in data.arm;
    this.arms = new Map(this.sides.map((side) => [side, new Arm(name, side, shared ? (data.arm as ArmData) : (data.arm as Record<string, ArmData>)[side]!)]));
  }

  /** The arm of one side. */
  sideArm(side: string): Arm {
    const arm = this.arms.get(side);
    if (!arm) throw new DreamConfigError(`${this.name} has no ${side} arm; its sides are ${this.sides.join(", ")}`, detail("range", "side"));
    return arm;
  }
}

const built = new Map<string, Embodiment>();

/** A built-in robot by its dream_ik name (`metal`, `makers`, `so101_faas`, ...). */
export function embodiment(name: ArmName | (string & {})): Embodiment {
  let found = built.get(name);
  if (found) return found;
  const data = EMBODIMENT_DATA[name];
  if (!data) throw new DreamConfigError(`no built-in arm named '${name}'; the built-in arms are ${ARM_NAMES.join(", ")}`);
  built.set(name, (found = new Embodiment(name, data)));
  return found;
}

/** One side's arm of a built-in robot: `arm("metal")`, `arm("so101_faas", "left")`. */
export function arm(name: ArmName | (string & {}), side = "right"): Arm {
  return embodiment(name).sideArm(side);
}
