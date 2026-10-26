import { describe, expect, it } from "vitest";
import { arm, embodiment } from "../src/ik/index.js";
import { cameraSlots, clampToReach, reachFence, Wire } from "../src/internal.js";
import { DreamConfigError } from "../src/index.js";
import { expectClose, fixture } from "./helpers.js";

const f = fixture("client_ik.json");

describe("dream.client_ik helpers", () => {
  it("clamp_to_reach", () => {
    for (const c of f.clamp) expectClose(clampToReach(c.xy, c.points), c.expected, 1e-12);
  });

  it("reach_fence", () => {
    for (const c of f.fence) expect(reachFence(c.hands, c.points)).toBe(c.expected);
  });

  it("camera_slots", () => {
    for (const c of f.slots) {
      if (c.error) expect(() => cameraSlots(c.roles, c.model, "ckpt")).toThrow(DreamConfigError);
      else expect(cameraSlots(c.roles, c.model, "ckpt")).toEqual(c.expected);
    }
  });

  it("_Wire: joint units, the gripper drive, held joints", () => {
    for (const c of f.wires) {
      const emb = embodiment(c.arm);
      const spec = arm(c.arm, c.side);
      const wire = new Wire(spec, emb.jawInArm, spec.data.jaw === null || emb.jawInArm ? null : `state.${c.side}_gripper`);
      expect([wire.own, wire.width], `${c.arm} ${c.side}`).toEqual([c.own, c.width]);
      expectClose(wire.arm(c.joints, c.grip), c.arm_units, 1e-12, `${c.arm} ${c.side} arm`);
      expectClose(wire.jointsOf(c.rows), c.joints_of, 1e-12, `${c.arm} ${c.side} joints_of`);
      if (c.drive) expectClose([wire.drive(c.grip)], c.drive, 1e-12);
    }
  });
});
