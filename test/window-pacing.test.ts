import { describe, expect, it } from "vitest";
import { DreamObservationError, ObservationWindow, Pacer } from "../src/index.js";
import { expectClose, fixture } from "./helpers.js";

const windows = fixture("window.json");
const pacing = fixture("pacing.json");

describe("ObservationWindow vs dream._window.Window", () => {
  for (const c of windows.cases) {
    it(c.name, () => {
      const window = new ObservationWindow(c.size, c.period_s);
      for (const step of c.steps) {
        if (step.op === "clear") {
          window.clear();
          continue;
        }
        const timestamp = step.timestamp === null ? NaN : step.timestamp;
        if (step.error !== undefined) {
          expect(() => window.add(step.frames, step.observation, timestamp)).toThrow(DreamObservationError);
        } else {
          expect(window.add(step.frames, step.observation, timestamp)).toBe(step.kept);
        }
        if (step.built_error !== undefined) expect(() => window.buildObservation()).toThrow(DreamObservationError);
        else expectClose(window.buildObservation(), step.built);
      }
    });
  }

  it("keeps a copy of the observation, not the caller's object", () => {
    const window = new ObservationWindow();
    const observation = { prompt: "a" };
    window.add({ "video.overhead": "Zg==" }, observation, 0);
    observation.prompt = "b";
    expect(window.buildObservation().prompt).toBe("a");
  });

  it("refuses bytes for a frame", () => {
    const window = new ObservationWindow();
    expect(() => window.add({ "video.overhead": new Uint8Array(3) as unknown as string }, {}, 0)).toThrow(/bytes/);
  });
});

describe("Pacer vs dream.pacing.Pacer", () => {
  for (const c of pacing.cases) {
    it(c.name, () => {
      const pacer = new Pacer(c.execution_horizon, c.action_hz, c.sync_above_ms);
      for (const step of c.steps) {
        if (step.record !== undefined) pacer.record(step.record);
        expect(pacer.mode).toBe(step.mode);
        expect(c.queries.map(([executed, remaining]: [number, number]) => pacer.refillDue(executed, remaining))).toEqual(step.due);
      }
    });
  }
});

describe("refillDue()'s `executed`: rows run since the install, as robots.control passes it", () => {
  /**
   * robots.control's loop (dreamlabs 1.0.17) on a fake clock: a reply lands `rttRows` after its request and is installed
   * past the rows run meanwhile (`chunk.actions[skipped:]`); each tick asks `refill_due(installed_rows - remaining,
   * remaining)`. `executedOf` is what the loop passes as `executed`. Returns the rows each chunk ran.
   */
  function loop(executedOf: (row: number, installedAt: number) => number, rttRows = 10, ticks = 400): number[] {
    const pacer = new Pacer(30, 20, 700);
    const ran: number[] = [];
    let chunk = 0; // rows of the current chunk; 0 before the first
    let row = 0;
    let installedAt = 0;
    let requestAt: number | null = null;
    for (let tick = 0; tick < ticks; tick++) {
      if (requestAt !== null && tick - requestAt === rttRows) {
        pacer.record(rttRows / 20);
        if (chunk) ran.push(row - installedAt);
        chunk = 48;
        row = installedAt = rttRows; // the rows run during the round trip are skipped
        requestAt = null;
      }
      if (requestAt === null && pacer.refillDue(executedOf(row, installedAt), chunk - row)) requestAt = tick;
      if (row < chunk) row++;
    }
    return ran.slice(2);
  }

  it("each chunk runs the execution horizon: the next request goes out executionHorizon - round trip rows after the install", () => {
    const ran = loop((row, installedAt) => row - installedAt);
    expect(new Set(ran)).toEqual(new Set([30]));
  });

  it("the chunk's row counted from the observation is not it: that refills a round trip early, 1.5x the requests", () => {
    // The mistake examples/node-basic.ts made (found on an SO-101): `executed` set to the install row.
    const ran = loop((row) => row);
    expect(new Set(ran)).toEqual(new Set([20]));
  });

  it("pinned against upstream's numbers: 10 rows of round trip, horizon 30, a chunk installed at its row 10", () => {
    const pacer = new Pacer(30, 20, 700);
    pacer.record(0.5);
    // Installed 38 rows (48 - 10 skipped); robots.control asks with installed_rows - remaining.
    expect(pacer.refillDue(38 - 19, 19)).toBe(false);
    expect(pacer.refillDue(38 - 18, 18)).toBe(true);
  });
});
