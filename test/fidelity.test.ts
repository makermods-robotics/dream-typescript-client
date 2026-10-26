/**
 * The README's Fidelity numbers are what scripts/fidelity.ts measures: its table verbatim, and the numbers its prose
 * names. Each measured value is under the ceiling the tests hold it to (the tests take those from the same file).
 */

import { readFileSync } from "node:fs";
import { beforeAll, describe, expect, it } from "vitest";
import { measureFidelity, over, show, table, type Fidelity } from "../scripts/fidelity.js";

const readme = readFileSync(new URL("../README.md", import.meta.url), "utf8");
let measured: Fidelity;

// The whole measurement: every ik.json case solved, the wire replayed (some 10 s; more beside the other test files).
beforeAll(async () => {
  measured = await measureFidelity();
}, 180_000);

describe("the README's Fidelity section", () => {
  it("measures every number under its ceiling", () => {
    expect(over(measured).map((r) => `${r.quantity} (${r.conditions}): ${r.value} >= ${r.ceiling}`)).toEqual([]);
  });

  it("holds the table `npm run fidelity` prints, verbatim", () => {
    const block = /<!-- fidelity:start \(npm run fidelity\) -->\n([\s\S]*?)\n<!-- fidelity:end -->/.exec(readme);
    expect(block, "the README's fidelity block").not.toBeNull();
    expect(block![1], "README.md's table: paste `npm run fidelity`'s output").toBe(table(measured));
  });

  it("names the measured values in its prose", () => {
    const aiWorker = measured.ik.sevenJointTraj.ai_worker_faas!;
    const named = [
      show(aiWorker.uncapped.joints, ""),
      show(aiWorker.capped.joints, ""),
      show(aiWorker.uncapped.tool, "m"),
      show(aiWorker.capped.tool, "m"),
      show(measured.ik.sevenJointTraj.bracketbot_faas!.capped.tool, "m"),
      show(measured.invariants.trajPosition, "m"),
      show(measured.unpatched.max.dq_rad, "rad"),
      show(measured.unpatched.max.tool_m, "m"),
      show(measured.unpatched.max.tool_m_redundant_capped, "m"),
      show(measured.limitStop.joints, "rad"),
      show(measured.limitStop.dream, "rad"),
    ];
    const prose = readme.replace(/<!-- fidelity:start[\s\S]*?<!-- fidelity:end -->/, "");
    for (const value of named) expect(prose, value).toContain(value);
    // Nothing left of the loose ceilings and claims the script replaced.
    for (const stale of ["2 cm", "0.44 mm", "0.071 rad", "6.7 mm", "2.8e-4"]) expect(prose, stale).not.toContain(stale);
  });
});
