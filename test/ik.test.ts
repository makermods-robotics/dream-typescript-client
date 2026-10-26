import { describe, expect, it } from "vitest";
import { arm, solveChunk } from "../src/ik/index.js";
import { CEILINGS, ikCases, measureIkCase, solveLimitStop } from "../scripts/fidelity.js";
import { fixture } from "./helpers.js";

const ik = fixture("ik.json");
const limitStop = fixture("limit_stop.json");

describe("solveChunk vs dream_ik.decode.solve_chunk", () => {
  // Two independent optimizers (pyroki/jaxls float32, ours float64), held to dream_ik's answer case by case: the
  // commanded tool path, the joints, the rotation reached, and our orientation miss of the target row by row against
  // dream_ik's own (dream_ik misses the target rotation itself: it weighs orientation 10 to position's 50, and the
  // joint step bounds each tick). Each ceiling is a margin over what `npm run fidelity` measures (scripts/fidelity.ts,
  // the README's Fidelity table); the one wide exception is ai_worker_faas's traj, whose seven-joint null space the
  // two settle apart in (joints ~0.07 rad apart; capped, the tool path then differs by 6.7 mm).
  for (const c of ikCases()) {
    const cap = c.speed_cap ?? Infinity;
    it(`${c.arm} ${c.variant} ${c.solve} cap ${c.speed_cap ?? "none"}`, () => {
      const m = measureIkCase(c);
      expect(m.tool, "tool path vs dream_ik").toBeLessThan(m.ceilings.tool);
      expect(m.jointSpread, "joints vs dream_ik").toBeLessThan(m.ceilings.joints);
      expect(m.rotation, "rotation reached vs dream_ik's").toBeLessThan(m.ceilings.rotation);
      expect(m.orientationMissExcess, "orientation miss vs dream_ik's").toBeLessThan(m.ceilings.orientationMissExcess);
      if (m.ceilings.orientationMiss !== null) expect(m.orientationMiss, "orientation miss").toBeLessThan(m.ceilings.orientationMiss);
      m.sides.forEach(({ side, model, q }, i) => {
        expect(q, side).toHaveLength(c.xyz[i].length);
        const step = model.data.maxSpeed.map((v) => Math.min(v, cap) / c.hz);
        let prev: number[] = c.q_from[i];
        for (const row of q) {
          row.forEach((v, j) => {
            expect(Number.isFinite(v)).toBe(true);
            expect(Math.abs(v - prev[j]!)).toBeLessThanOrEqual(step[j]! + 1e-9);
            expect(v).toBeGreaterThanOrEqual(Math.min(model.lo[j]!, c.q_from[i][j]) - 1e-9);
            expect(v).toBeLessThanOrEqual(Math.max(model.hi[j]!, c.q_from[i][j]) + 1e-9);
          });
          prev = row;
        }
      });
    });
  }
});

describe("solveChunk at a joint limit (a recorded two-arm SO-101 chunk)", () => {
  // dream_ik's own traj solve stopped early here, 0.02 rad from its converged answer (jaxls's relative cost change
  // fell under 1e-5 while the stiff limit cost kept each step short; where it stops turns on float rounding). Ours is
  // held to the converged answer, the same solve run to convergence in float64.
  const spread = (a: number[][], b: number[][]) => Math.max(...a.flatMap((row, k) => row.map((v, j) => Math.abs(v - b[k]![j]!))));
  for (const c of limitStop.cases) {
    it(`${limitStop.arm} ${c.side}`, () => {
      const { model, q } = solveLimitStop(limitStop, c);
      expect(spread(q, c.converged), "joints vs dream_ik converged").toBeLessThan(CEILINGS.ik.jointsAtLimit);
      const step = c.free.map((i: number) => model.data.maxSpeed[i]! / limitStop.hz);
      let prev: number[] = c.free.map((i: number) => c.q_from[i]);
      for (const row of q) {
        row.forEach((v, j) => {
          expect(v).toBeGreaterThanOrEqual(c.lo[j] - 1e-6);
          expect(v).toBeLessThanOrEqual(c.hi[j] + 1e-6);
          expect(Math.abs(v - prev[j]!)).toBeLessThanOrEqual(step[j]! + 1e-9);
        });
        prev = row;
      }
    });
  }

  it("is a chunk dream_ik's own solve stops early on, at the limit", () => {
    const left = limitStop.cases.find((c: any) => c.side === "left");
    expect(spread(left.dream, left.converged)).toBeGreaterThan(10 * CEILINGS.ik.jointsAtLimit);
    // the left lift (joint 1) at its margined lower limit in dream's answer
    expect(Math.min(...left.dream.map((row: number[]) => row[1]!))).toBeLessThan(left.lo[1] + 1e-3);
  });
});

describe("solveChunk options", () => {
  it("refuses ServeSpec.fences, which it does not implement", () => {
    const c = ik.cases[0];
    expect(() => solveChunk(arm(c.arm), c.xyz[0], c.rot6[0], c.q_from[0], { hz: 20, serve: { fences: true } as never })).toThrow(/fences/);
  });

  it("takes one history row (qFrom) by default, as dream_ik does", () => {
    const c = ik.cases.find((x: any) => x.arm === "metal" && x.solve === "traj")!;
    const model = arm(c.arm);
    const options = { hz: c.hz, serve: { solve: "traj" as const } };
    const byDefault = solveChunk(model, c.xyz[0], c.rot6[0], c.q_from[0], options).q;
    const oneRow = solveChunk(model, c.xyz[0], c.rot6[0], c.q_from[0], { ...options, qBefore: [c.q_from[0]] }).q;
    expect(byDefault).toEqual(oneRow);
  });
});
