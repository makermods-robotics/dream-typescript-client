import { describe, expect, it } from "vitest";
import {
  ARM_POSE_KEYS,
  anchorRow,
  filterFrom,
  fitMotion,
  nearestRow,
  oldMotion,
  savgolFilter,
  velocityBlend,
  MAX_OFFSET_RAD,
} from "../src/internal.js";
import { SeamSmoother } from "../src/index.js";
import { arm } from "../src/ik/index.js";
import { rotationMatrix, rotvec, type Vec3 } from "../src/linalg.js";
import { expectClose, fixture } from "./helpers.js";

const f = fixture("smoothing.json");

describe("dream.smoothing and dream.seam_join helpers", () => {
  it("savgol_filter", () => {
    for (const c of f.savgol) expectClose(savgolFilter(c.rows, c.window, c.polyorder), c.expected, 1e-10);
  });

  it("filter_from", () => {
    for (const c of f.filter_from) expectClose(filterFrom(c.chunk, c.clock), c.expected, 1e-10);
  });

  it("fit_motion", () => {
    for (const c of f.fit_motion) {
      const { now, velocity } = fitMotion(c.rows, c.times);
      expectClose(now, c.now, 1e-10);
      expectClose(velocity, c.velocity, 1e-10);
    }
  });

  it("velocity_blend", () => {
    for (const c of f.velocity_blend) expectClose(velocityBlend(c.now, c.velocity, c.plan), c.expected, 1e-12);
  });

  it("rotation_matrix and rotvec", () => {
    for (const c of f.rotations) {
      expectClose(rotationMatrix(c.vector as Vec3), c.matrix.flat(), 1e-12);
      expectClose(rotvec(c.matrix.flat()), c.rotvec, 1e-9);
    }
  });

  it("nearest_row, anchor_row and old_motion", () => {
    for (const c of f.nearest_row) expect(nearestRow(c.chunk, c.pose, MAX_OFFSET_RAD)).toBe(c.expected);
    for (const c of f.anchor_row) expect(anchorRow(c.previous, c.pose, c.commanded)).toBe(c.expected);
    for (const c of f.old_motion) {
      const { clock, rows, times } = oldMotion(c.previous, c.anchor, c.rows_in_flight);
      expect(clock).toBe(c.clock);
      expectClose(rows, c.rows);
      expectClose(times, c.times);
    }
  });
});

function runSequence(steps: any[], smoother: SeamSmoother, tolerance: number) {
  for (const [i, step] of steps.entries()) {
    const commanded = step.commanded?.q_r_chunk ?? null;
    const takeover = smoother.takeoverHistory("q_r_chunk", step.action.q_r_chunk.length, step.action.q_r_chunk[0].length, step.observation["state.arm_r.joint_pos"], step.rtt_s, commanded);
    if (step.takeover === null) expect(takeover, `step ${i} takeover`).toBeNull();
    else expectClose(takeover, step.takeover, tolerance, `step ${i} takeover`);
    const result = smoother.smooth(step.observation, step.action, step.rtt_s, step.commanded);
    expectClose(result, step.expected, tolerance, `step ${i}`);
  }
}

describe("SeamSmoother vs dream.smoothing.SeamSmoother", () => {
  it("joins in joint space: first chunk, joins, a hold, an arm off its plan, a new shape, commanded", () => {
    runSequence(f.smoother_joint, new SeamSmoother(20, null, ARM_POSE_KEYS), 1e-10);
  });

  it("joins in end-effector space with the Metal arm's kinematics", () => {
    // Python's reference kinematics are float64 through yourdfpy with a numeric Jacobian; ours analytic.
    runSequence(f.smoother_eef, new SeamSmoother(20, { q_r_chunk: arm("metal").kinematics() }, ARM_POSE_KEYS), 2e-6);
  });

  it("forgets the previous chunk on reset", () => {
    const smoother = new SeamSmoother(20, null, ARM_POSE_KEYS);
    const step = f.smoother_joint[0];
    smoother.smooth(step.observation, step.action, 0.3);
    smoother.reset();
    expect(smoother.takeoverHistory("q_r_chunk", 48, 7, step.observation["state.arm_r.joint_pos"], 0.3, null)).toBeNull();
  });
});
