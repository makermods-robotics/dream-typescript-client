/**
 * Numerical validation, the final projection, the model-to-arm check, and the IK's invariants (orientation error,
 * limits, unreachable targets, short horizons, seams).
 */

import { describe, expect, it } from "vitest";
import { arm, ClientIK, embodiment, solveChunk } from "../src/ik/index.js";
import { extendPoses } from "../src/internal.js";
import { DreamActionError, DreamClient, DreamConfigError, ObservationWindow, SeamSmoother, type ActionDict, type Diagnostics, type Model } from "../src/index.js";
import { matToRot6, rotationMatrix, rotvec, mul3, transpose3 } from "../src/linalg.js";
import { CEILINGS } from "../scripts/fidelity.js";

const HZ = 20;
const metal = arm("metal");

/** The tool poses of a joint path on one arm, as a reply's per_key parts. */
function poses(model: typeof metal, path: number[][]): { xyz: number[][]; rot6: number[][] } {
  const tools = path.map((q) => model.tool(q));
  return { xyz: tools.map((t) => [...t.position]), rot6: tools.map((t) => matToRot6(t.rotation)) };
}

/** A joint path from `start` swinging every joint by `amplitude` rad, fast. */
function swing(start: readonly number[], amplitude: number, phase = 0, rows = 48, perRow = 0.4): number[][] {
  return Array.from({ length: rows }, (_, k) => start.map((v, j) => v + amplitude * (Math.sin(perRow * k + phase + j) - Math.sin(phase + j))));
}

const yamModel: Model = { id: "dreamlabs/dream-1-beta", owner: "public", baseModel: "dreamlabs/dream-1-beta", embodiment: "yam", servable: true, createdAt: null, cameras: ["overhead", "wrist_l", "wrist_r"], embodimentConfig: null, partsWithheld: false };
const body = { state: { right_arm: [0, 6], right_grip: [6, 7], left_arm: [7, 13], left_grip: [13, 14] }, action: { right_arm: [0, 6], left_arm: [6, 12] }, video: { front: "observation.images.front", wrist_l: "observation.images.wrist_l", wrist_r: "observation.images.wrist_r" } };
const frames = { "video.front": "Zg==", "video.wrist_l": "Zg==", "video.wrist_r": "Zg==" };

/** A DreamClient on Metal with client IK, against a stub gateway answering the swing of `phase` from where the arm is. */
async function metalClient(ik: Record<string, unknown>, onDiagnostics?: (d: Diagnostics) => void) {
  let call = 0;
  let clock = 0;
  let from: number[] = [...metal.data.restPose];
  const fetch = async (url: string): Promise<Response> => {
    clock += 300; // every round trip 0.3 s: 6 rows in flight
    if (url.endsWith("/v1/end_session")) return new Response('{"status":{"code":"DA-0000"}}');
    const path = swing(from, 0.6, call++);
    const { xyz, rot6 } = poses(metal, path);
    const perKey: Record<string, unknown> = {};
    for (const side of ["right", "left"]) {
      perKey[`action.${side}_eef_xyz`] = xyz;
      perKey[`action.${side}_eef_rot6`] = rot6;
      perKey[`action.${side}_grip`] = path.map(() => [0.5]);
    }
    return new Response(JSON.stringify({ status: { code: "DA-0000" }, session_id: "s", actions: { per_key: perKey } }));
  };
  const client = new DreamClient({ baseUrl: "http://gateway.test", platformUrl: "http://platform.test", fetch, now: () => clock, onWarning: () => {}, onDiagnostics });
  await client.policy({ checkpoint: yamModel.id, model: yamModel, embodiment: body as never, strict: false, ik: { arm: "metal", head: "eef", ...ik } as never });
  const observe = (q: readonly number[]) => {
    from = [...q];
    client.setObservation(frames, { "state.right_arm": [...q], "state.right_grip": [0.5], "state.left_arm": [...q], "state.left_grip": [0.5] }, { timestamp: clock / 1000 });
  };
  return { client, observe };
}

/** Every row within the margined limits (or where `from` already was) and one capped step from the row before. */
function expectProjected(rows: number[][], from: readonly number[], capRadS: number) {
  const step = metal.data.maxSpeed.map((v) => Math.min(v, capRadS) / HZ);
  let prev = from;
  rows.forEach((row, k) => {
    row.slice(0, 6).forEach((v, j) => {
      expect(Number.isFinite(v)).toBe(true);
      expect(Math.abs(v - prev[j]!), `row ${k} joint ${j} step`).toBeLessThanOrEqual(step[j]! + 1e-12);
      expect(v, `row ${k} joint ${j} low`).toBeGreaterThanOrEqual(Math.min(metal.lo[j]!, from[j]!) - 1e-12);
      expect(v, `row ${k} joint ${j} high`).toBeLessThanOrEqual(Math.max(metal.hi[j]!, from[j]!) + 1e-12);
    });
    prev = row.slice(0, 6);
  });
}

describe("the final projection", () => {
  it("holds the speed cap and the limits on every returned row, across seams, and the next seam continues what was returned", async () => {
    const diagnostics: Diagnostics[] = [];
    const { client, observe } = await metalClient({ speedCap: 0.5, startupSpeedCap: 0.2 }, (d) => diagnostics.push(d));
    let q = [...metal.data.restPose];
    observe(q);
    let chunk = (await client.getAction()).q_r_chunk as number[][];
    expectProjected(chunk, q, 0.2); // the first chunk under the startup cap
    for (let call = 0; call < 2; call++) {
      const commanded = chunk[5]!;
      q = commanded.slice(0, 6);
      observe(q);
      const next = (await client.getAction({ commanded: { q_r_chunk: commanded, q_l_chunk: commanded } })).q_r_chunk as number[][];
      expectProjected(next, commanded, 0.5);
      // 6 rows in flight: the rows before the clock row are the previous returned chunk's, continued.
      for (let k = 0; k < 6; k++) next[k]!.slice(0, 6).forEach((v, j) => expect(Math.abs(v - chunk[6 + k]![j]!)).toBeLessThan(1e-12));
      chunk = next;
    }
    expect(diagnostics.at(-1)!.ik!.right).toMatchObject({ clock: 6, speedCap: 0.5 });
    expect(diagnostics[0]!.ik!.right!.speedCap).toBe(0.2);
    const timing = client.lastTiming!;
    // The stub clock moves only on round trips: the solve takes no time on it.
    expect(timing.ikMs).toBe(0);
    expect(timing.totalMs).toBe(timing.rttMs);
    await client.close();
  });

  it("clamps the grip to [0, 1] (not in Python: the model's grip can leave it)", async () => {
    const { client, observe } = await metalClient({});
    observe(metal.data.restPose);
    const solved = await client.getAction();
    const ik = new ClientIK({ arm: "metal", head: "eef" } as never, HZ, body, yamModel);
    const rows = Array.from({ length: 4 }, (_, k) => [...metal.data.restPose, [-0.024, 1.3, 0.4, 0][k]!]);
    expect(ik.project("q_r_chunk", rows, metal.data.restPose, false).map((row) => row[6])).toEqual([0, 1, 0.4, 0]);
    expect((solved.q_r_chunk as number[][]).every((row) => row[6]! >= 0 && row[6]! <= 1)).toBe(true);
    await client.close();
  });

  it("takes the loop's own timing for the seam when given", async () => {
    const diagnostics: Diagnostics[] = [];
    const { client, observe } = await metalClient({}, (d) => diagnostics.push(d));
    observe(metal.data.restPose);
    const chunk = (await client.getAction()).q_r_chunk as number[][];
    observe(chunk[5]!.slice(0, 6));
    await client.getAction({ commanded: { q_r_chunk: chunk[5]! }, execution: { elapsedExecutionS: 0.5 } });
    expect(diagnostics.at(-1)!.inFlightS).toBe(0.5);
    expect(diagnostics.at(-1)!.ik!.right!.clock).toBe(10);
    await expect(client.getAction({ execution: { observationAgeMs: -1 } })).rejects.toThrow(DreamConfigError);
    await client.close();
  });
});

describe("numerical validation", () => {
  const c = { xyz: [[0.3, 0, 0.2]], rot6: [[1, 0, 0, 0, 1, 0]] };
  const q0 = [...metal.data.restPose];

  it("refuses a degenerate rotation_6d instead of solving it to NaN", () => {
    for (const rot6 of [[0, 0, 0, 0, 0, 0], [1, 0, 0, 2, 0, 0], [1e-9, 0, 0, 0, 1, 0]]) {
      const error = (() => {
        try {
          solveChunk(metal, c.xyz, [rot6], q0, { hz: HZ });
        } catch (e) {
          return e as DreamActionError;
        }
      })();
      expect(error, JSON.stringify(rot6)).toBeInstanceOf(DreamActionError);
      expect(error!.detail).toEqual({ kind: "degenerate_rotation", key: "rot6", row: 0 });
    }
  });

  it("refuses bad options as DreamConfigError and bad targets as DreamActionError, never a RangeError", () => {
    const bad: [() => unknown, typeof DreamConfigError | typeof DreamActionError, string | null][] = [
      [() => solveChunk(metal, c.xyz, c.rot6, q0, { hz: 0 }), DreamConfigError, "hz"],
      [() => solveChunk(metal, c.xyz, c.rot6, q0, { hz: Infinity }), DreamConfigError, "hz"],
      [() => solveChunk(metal, c.xyz, c.rot6, q0, { hz: HZ, speedCap: 0 }), DreamConfigError, "speedCap"],
      [() => solveChunk(metal, c.xyz, c.rot6, q0, { hz: HZ, speedCap: NaN }), DreamConfigError, "speedCap"],
      [() => solveChunk(metal, c.xyz, c.rot6, [...q0.slice(0, 5), NaN], { hz: HZ }), DreamConfigError, "qFrom"],
      [() => solveChunk(metal, c.xyz, c.rot6, q0.slice(0, 5), { hz: HZ }), DreamConfigError, "qFrom"],
      [() => solveChunk(metal, c.xyz, c.rot6, q0, { hz: HZ, qBefore: [q0.slice(0, 4)] }), DreamConfigError, "qBefore"],
      [() => solveChunk(metal, c.xyz, c.rot6, q0, { hz: HZ, qBefore: [q0, q0, q0] }), DreamConfigError, "qBefore"],
      [() => solveChunk(metal, [], [], q0, { hz: HZ }), DreamActionError, "xyz"],
      [() => solveChunk(metal, Array(513).fill(c.xyz[0]), Array(513).fill(c.rot6[0]), q0, { hz: HZ }), DreamActionError, "xyz"],
      [() => solveChunk(metal, [[0.3, Infinity, 0.2]], c.rot6, q0, { hz: HZ }), DreamActionError, "xyz"],
      [() => metal.tool([0, 0]), DreamConfigError, "q"],
      [() => metal.tool([0, 0, 0, 0, 0, NaN]), DreamConfigError, "q"],
      [() => embodiment("metal").sideArm("middle"), DreamConfigError, "side"],
    ];
    for (const [run, type, key] of bad) {
      let error: unknown;
      try {
        run();
      } catch (e) {
        error = e;
      }
      expect(error, String(run)).toBeInstanceOf(type);
      expect((error as DreamConfigError).detail?.key, String(run)).toBe(key);
    }
  });

  it("extends a single remaining pose as a hold, not NaN", () => {
    const pose = { position: [0.3, 0, 0.2] as [number, number, number], rotation: rotationMatrix([0.1, 0.2, 0.3]) };
    const out = extendPoses([pose], 5);
    expect(out).toHaveLength(6);
    for (const p of out) {
      expect(p.position).toEqual(pose.position);
      expect(p.rotation).toEqual(pose.rotation);
    }
  });

  it("refuses a non-finite number in a joint chunk or an IK part, naming the key and row", () => {
    const smoother = new SeamSmoother(HZ);
    const rows = Array.from({ length: 48 }, () => [0, 0, 0, 0, 0, 0, 0.5]);
    rows[7] = [0, 0, Infinity, 0, 0, 0, 0.5];
    const error = (() => {
      try {
        smoother.smooth({ "state.arm_r.joint_pos": [0, 0, 0, 0, 0, 0] }, { q_r_chunk: rows }, 0.3);
      } catch (e) {
        return e as DreamActionError;
      }
    })();
    expect(error?.detail).toEqual({ kind: "non_finite", key: "q_r_chunk", row: 7 });
    expect(() => new SeamSmoother(0)).toThrow(DreamConfigError);
  });

  it("commits no arm's plan when another arm's chunk is refused", () => {
    const smoother = new SeamSmoother(HZ);
    const good = Array.from({ length: 48 }, (_, k) => [k * 0.001, 0, 0, 0, 0, 0, 0.5]);
    const obs = { "state.arm_r.joint_pos": [0, 0, 0, 0, 0, 0], "state.arm_l.joint_pos": [0, 0, 0, 0, 0, 0] };
    expect(() => smoother.smooth(obs, { q_r_chunk: good, q_l_chunk: [[0, 0], [0]] }, 0.3)).toThrow(DreamActionError);
    expect(smoother.has("q_r_chunk")).toBe(false);
    smoother.smooth(obs, { q_r_chunk: good, q_l_chunk: good }, 0.3);
    expect(smoother.has("q_r_chunk") && smoother.has("q_l_chunk")).toBe(true);
  });

  it("refuses finite rows the filter overflows (alternating ±1.7e308), keeping no arm's plan", () => {
    const smoother = new SeamSmoother(HZ);
    const good = Array.from({ length: 48 }, (_, k) => [k * 0.001, 0.5]);
    const huge = Array.from({ length: 48 }, (_, k) => [k % 2 ? 1.7e308 : -1.7e308, 0.5]);
    const obs = { "state.arm_r.joint_pos": [0], "state.arm_l.joint_pos": [0] };
    const error = (() => {
      try {
        smoother.smooth(obs, { q_l_chunk: good, q_r_chunk: huge }, 0.3);
      } catch (e) {
        return e as DreamActionError;
      }
    })();
    expect(error).toBeInstanceOf(DreamActionError);
    expect(error!.detail).toMatchObject({ kind: "non_finite", key: "q_r_chunk" });
    expect(smoother.has("q_l_chunk") || smoother.has("q_r_chunk")).toBe(false);
  });
});

describe("snapshots and frozen arm data", () => {
  it("copies an observation's numbers when it is added, so a reused buffer changes nothing sent", () => {
    const window = new ObservationWindow();
    const buffer = Float64Array.of(1, 2, 3);
    const nested = [[1, 2]];
    window.add({ "video.top": "Zg==" }, { "state.a": buffer, "state.b": nested }, 0);
    buffer[0] = 99;
    nested[0]![0] = 99;
    const built = window.buildObservation();
    expect(Array.from(built["state.a"] as Float64Array)).toEqual([1, 2, 3]);
    expect(built["state.b"]).toEqual([[1, 2]]);
  });

  it("copies a Node Buffer's bytes, whose own slice() is a view of the caller's memory", () => {
    const window = new ObservationWindow();
    const buffer = Buffer.from([1, 2, 3]);
    window.add({ "video.top": "Zg==" }, { "state.a": buffer }, 0);
    buffer[0] = 99;
    expect(Array.from(window.buildObservation()["state.a"] as Uint8Array)).toEqual([1, 2, 3]);
  });

  it("keeps the built-in arm data frozen: no caller can change another's arm", () => {
    const a = arm("metal");
    expect(() => ((a.data.maxSpeed as number[])[0] = 100)).toThrow(TypeError);
    expect(() => ((a.lo as number[])[0] = 100)).toThrow(TypeError);
    expect(arm("metal").data.maxSpeed[0]).toBe(4.5);
  });
});

describe("the model-to-arm check", () => {
  const so101 = (head: "eef" | "joint", embodimentId?: string) => ({ arm: "so101_faas", head, ...(embodimentId ? { embodimentId } : {}) }) as never;
  // Metal has no gripper drive, so it needs no embodiment config to be built: the check alone decides.
  const metalArm = (head: "eef" | "joint", embodimentId?: string) => ({ arm: "metal", head, ...(embodimentId ? { embodimentId } : {}) }) as never;
  const fineTune = (action: Record<string, number[]> | null, id = "emb_so101"): Model => ({
    id: "dream-1-ft:x-y",
    owner: "mine",
    baseModel: "b",
    embodiment: id,
    servable: true,
    createdAt: null,
    cameras: ["front"],
    embodimentConfig: action === null ? null : { state: { right_arm: [0, 6], right_gripper: [6, 7], left_arm: [7, 13], left_gripper: [13, 14] }, action, video: { front: "observation.images.front" } },
    partsWithheld: false,
  });
  const joints = { right_arm: [0, 6], right_gripper: [6, 7], left_arm: [7, 13], left_gripper: [13, 14] };
  const eef = { right_eef_xyz: [0, 3], right_eef_rot6: [3, 9], right_grip: [9, 10], left_eef_xyz: [10, 13], left_eef_rot6: [13, 19], left_grip: [19, 20] };
  const fits = (config: unknown, model: Model) => () => new ClientIK(config as never, HZ, model.embodimentConfig, model);
  const mismatch = (run: () => unknown) => {
    try {
      run();
    } catch (error) {
      return (error as DreamConfigError).detail?.kind;
    }
    return "fits";
  };

  it("runs a joint head only from a model trained on this arm", () => {
    // Same widths, another robot: widths alone never vouch for a joint answer (Codex review round 2, P1).
    expect(mismatch(fits(so101("joint"), fineTune(joints)))).toBe("model_mismatch");
    expect(mismatch(fits(metalArm("joint"), fineTune(joints, "emb_makers")))).toBe("model_mismatch");
    expect(mismatch(fits(metalArm("joint", "emb_metal"), fineTune(joints, "emb_makers")))).toBe("model_mismatch");
    expect(mismatch(fits(so101("joint", "emb_so101"), fineTune(joints)))).toBe("fits");
    expect(mismatch(fits(so101("joint", "emb_so101"), fineTune({ ...joints, right_arm: [0, 7] })))).toBe("model_mismatch");
    expect(mismatch(fits(so101("joint", "emb_other"), fineTune(joints)))).toBe("model_mismatch");
    expect(mismatch(fits(so101("joint"), fineTune({ ...joints, right_arm: [0, 7] })))).toBe("model_mismatch");
    expect(mismatch(fits(metalArm("joint"), fineTune(null, "emb_metal")))).toBe("model_mismatch");
    expect(mismatch(fits(metalArm("joint", "emb_metal"), fineTune(null, "emb_metal")))).toBe("fits");
    expect(mismatch(fits(so101("joint"), yamModel))).toBe("model_mismatch");
  });

  it("runs an EEF head across embodiments, from a model that answers each side's tool pose", () => {
    expect(mismatch(fits(so101("eef", "emb_other"), fineTune({ ...joints, ...eef })))).toBe("fits");
    expect(mismatch(fits(so101("eef"), fineTune(joints)))).toBe("model_mismatch");
    expect(mismatch(fits(metalArm("eef"), fineTune(null, "emb_metal")))).toBe("fits");
  });
});

describe("IK invariants", () => {
  const q0 = [...metal.data.restPose];
  const angle = (a: number[], b: number[]) => {
    const rotation = (r: number[]) => {
      const x = [r[0]!, r[1]!, r[2]!];
      const n = Math.hypot(...x);
      const b1 = x.map((v) => v / n);
      const d = b1[0]! * r[3]! + b1[1]! * r[4]! + b1[2]! * r[5]!;
      const u = [r[3]! - d * b1[0]!, r[4]! - d * b1[1]!, r[5]! - d * b1[2]!];
      const m = Math.hypot(...u);
      const b2 = u.map((v) => v / m);
      const b3 = [b1[1]! * b2[2]! - b1[2]! * b2[1]!, b1[2]! * b2[0]! - b1[0]! * b2[2]!, b1[0]! * b2[1]! - b1[1]! * b2[0]!];
      return [b1[0]!, b2[0]!, b3[0]!, b1[1]!, b2[1]!, b3[1]!, b1[2]!, b2[2]!, b3[2]!] as Parameters<typeof transpose3>[0];
    };
    return Math.hypot(...rotvec(mul3(rotation(a), transpose3(rotation(b)))));
  };

  // The tick solve tracks each row; the traj solve trades tracking for smoothness (its velocity, acceleration and jerk
  // costs against the arm's history), so on a path moving ~0.3 rad/s from rest it trails by up to ~8 mm, as dream_ik's.
  // The ceilings are a margin over what `npm run fidelity` measures (scripts/fidelity.ts measureReach, this same path).
  const i = CEILINGS.invariants;
  const reach = { tick: { position: i.tickPosition, angle: i.tickAngle }, traj: { position: i.trajPosition, angle: i.trajAngle } };
  for (const solve of ["tick", "traj"] as const) {
    it(`${solve}: reaches a reachable path, position and orientation`, () => {
      for (const rows of [24, 2]) {
        const { xyz, rot6 } = poses(metal, swing(q0, 0.15, 0, rows, 0.1));
        const res = solveChunk(metal, xyz, rot6, q0, { hz: HZ, serve: { solve, smoothWindow: 0 } });
        expect(res.q).toHaveLength(rows);
        res.q.forEach((row, k) => {
          const reached = poses(metal, [row]);
          expect(Math.hypot(...reached.xyz[0]!.map((v, i) => v - xyz[k]![i]!)), `${rows} rows: row ${k} position`).toBeLessThan(reach[solve].position);
          expect(angle(reached.rot6[0]!, rot6[k]!), `${rows} rows: row ${k} orientation`).toBeLessThan(reach[solve].angle);
        });
      }
    });

    it(`${solve}: never leaves the margined limits near a limit, nor for an unreachable target`, () => {
      const near = [...q0];
      near[1] = metal.hi[1]! - 0.01; // joint2 just inside its upper margin
      const beyond = swing(near, 0.3, 0, 16).map((row) => row.map((v, j) => (j === 1 ? metal.hi[1]! + 0.4 : v)));
      const far = { xyz: Array.from({ length: 16 }, () => [3, 3, 3]), rot6: Array.from({ length: 16 }, () => [1, 0, 0, 0, 1, 0]) };
      for (const [name, targets, start] of [["near a limit", poses(metal, beyond), near], ["unreachable", far, q0]] as const) {
        const res = solveChunk(metal, targets.xyz, targets.rot6, start, { hz: HZ, speedCap: 1, serve: { solve } });
        const step = metal.data.maxSpeed.map((v) => Math.min(v, 1) / HZ);
        let prev: readonly number[] = start;
        for (const row of res.q) {
          row.forEach((v, j) => {
            expect(Number.isFinite(v), name).toBe(true);
            expect(v, `${name} joint ${j}`).toBeLessThanOrEqual(Math.max(metal.hi[j]!, start[j]!) + 1e-9);
            expect(v, `${name} joint ${j}`).toBeGreaterThanOrEqual(Math.min(metal.lo[j]!, start[j]!) - 1e-9);
            expect(Math.abs(v - prev[j]!)).toBeLessThanOrEqual(step[j]! + 1e-9);
          });
          prev = row;
        }
        expect(res.posMiss.every(Number.isFinite) && res.angMiss.every(Number.isFinite), name).toBe(true);
      }
    });

    it(`${solve}: solves a horizon of one row, and holds a one-row target from anywhere`, () => {
      const { xyz, rot6 } = poses(metal, [q0.map((v, j) => v + 0.05 * Math.sin(j + 1))]);
      const res = solveChunk(metal, xyz, rot6, q0, { hz: HZ, serve: { solve } });
      expect(res.q).toHaveLength(1);
      expect(res.q[0]!.every(Number.isFinite) && res.posMiss.every(Number.isFinite)).toBe(true);
      const still = solveChunk(metal, ...Object.values(poses(metal, [q0])) as [number[][], number[][]], q0, { hz: HZ, serve: { solve } });
      expect(Math.max(...still.q[0]!.map((v, j) => Math.abs(v - q0[j]!)))).toBeLessThan(1e-3);
    });
  }

  it("holds a one-row chunk after a seam (every row skipped but one)", async () => {
    const { client, observe } = await metalClient({});
    observe(q0);
    const chunk = (await client.getAction()).q_r_chunk as number[][];
    observe(chunk[0]!.slice(0, 6));
    // Far more time in flight than the plan has rows: the clock row is the plan's last, one pose is left to extend.
    const next = (await client.getAction({ commanded: { q_r_chunk: chunk[0]! }, execution: { elapsedExecutionS: 60 } })) as ActionDict;
    expect((next.q_r_chunk as number[][]).flat().every(Number.isFinite)).toBe(true);
    await client.close();
  });
});
