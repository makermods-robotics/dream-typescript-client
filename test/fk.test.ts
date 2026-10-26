import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { arm, ARM_NAMES, defineArm, embodiment } from "../src/ik/index.js";
import { DreamConfigError } from "../src/index.js";
import { CEILINGS } from "../scripts/fidelity.js";
import { fixture } from "./helpers.js";

const fk = fixture("fk.json");

describe("forward kinematics vs the URDFs (float64, yourdfpy)", () => {
  for (const c of fk.cases) {
    it(`${c.arm} ${c.side}`, () => {
      const model = arm(c.arm, c.side);
      c.q.forEach((q: number[], k: number) => {
        const { position, rotation } = model.tool(q);
        const expected = c.poses[k];
        position.forEach((v, i) => expect(Math.abs(v - expected[i][3])).toBeLessThan(CEILINGS.fk.positionVsUrdf));
        rotation.forEach((v, i) => expect(Math.abs(v - expected[Math.floor(i / 3)][i % 3])).toBeLessThan(CEILINGS.fk.rotationVsUrdf));
        // dream_ik's own forward kinematics (pyroki, float32) agree to float32 precision.
        position.forEach((v, i) => expect(Math.abs(v - c.dream_ik_fk[k][i][3])).toBeLessThan(CEILINGS.fk.positionVsDreamIk));
      });
    });
  }

  it("covers every built-in arm and side", () => {
    const covered = new Set(fk.cases.map((c: any) => `${c.arm}:${c.side}`));
    for (const name of ARM_NAMES) for (const side of embodiment(name).sides) expect(covered.has(`${name}:${side}`), `${name} ${side}`).toBe(true);
  });

  it("has a Jacobian that matches finite differences of the tool pose", () => {
    for (const name of ["metal", "bracketbot_faas", "ai_worker_faas"]) {
      const model = arm(name);
      const q = model.data.restPose.map((v, j) => v + 0.1 * Math.sin(j + 1));
      const jac = model.jacobian(q);
      const h = 1e-6;
      for (let j = 0; j < model.dof; j++) {
        const up = [...q];
        const down = [...q];
        up[j]! += h;
        down[j]! -= h;
        const a = model.tool(up).position;
        const b = model.tool(down).position;
        for (let r = 0; r < 3; r++) expect(Math.abs((a[r]! - b[r]!) / (2 * h) - jac[r]![j]!)).toBeLessThan(CEILINGS.fk.jacobianVsFiniteDifferences);
      }
    }
  });
});

describe("defineArm (dream.arm_file)", () => {
  const metalUrdf = readFileSync(new URL("./fixtures/metal.urdf", import.meta.url), "utf8");

  it("builds the Metal arm from its URDF and tool axes as dream_ik's own spec", () => {
    const custom = defineArm(
      { base: "base_link", flange: "link6", tcp: [0.135, 0, 0], approach: [1, 0, 0], jaw: [0, 1, 0], rest: [0, -1.2, 1.5, 0, 0, 0] },
      metalUrdf,
    ).sideArm("right");
    const builtIn = arm("metal");
    for (const q of [builtIn.data.restPose, [0.3, -1, 1.2, 0.4, -0.5, 1]]) {
      const a = custom.tool(q);
      const b = builtIn.tool(q);
      a.position.forEach((v, i) => expect(Math.abs(v - b.position[i]!)).toBeLessThan(1e-12));
      a.rotation.forEach((v, i) => expect(Math.abs(v - b.rotation[i]!)).toBeLessThan(1e-12));
    }
    expect(custom.data.maxSpeed.every((v) => v <= 3)).toBe(true);
  });

  it("refuses an arm file it cannot solve", () => {
    const file = { base: "base_link", flange: "link6", tcp: [0, 0, 0.1], approach: [0, 0, 1], jaw: [0, 1, 0], rest: [0, 0, 0, 0, 0, 0] };
    expect(() => defineArm({ ...file, flange: "nowhere" }, metalUrdf)).toThrow(DreamConfigError);
    expect(() => defineArm({ ...file, rest: [0] }, metalUrdf)).toThrow(/rest has 1 joints/);
    expect(() => defineArm({ ...file, jaw: [0, 0, 2] }, metalUrdf)).toThrow(/not parallel/);
    expect(() => defineArm(file, "<robot><link name='a'>")).toThrow(DreamConfigError);
  });
});
