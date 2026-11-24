/**
 * Holes: a sparse array's missing element (`[1, , 3]`, `delete a[1]`). `every`, `some`, `forEach`, `map`, `filter`
 * and `flat` skip holes, so a check written with them passes an array with a hole in it, which then reads as
 * `undefined` (or is dropped, shifting the rest). Codex's final check found that in takeoverAt() (test/takeover.test.ts);
 * these are the same check everywhere else the client takes arrays. Typed arrays have no holes and are taken as before.
 */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { arm, ClientIK, defineArm, solveChunk } from "../src/ik/index.js";
import { DreamActionError, DreamConfigError, DreamObservationError, PolicyClient, SeamSmoother, type Model } from "../src/index.js";
import {
  check,
  checkFiniteAction,
  compileChecks,
  encode,
  every,
  firstFailing,
  firstHole,
  firstNonFinite,
  flatPose,
  headPoses,
  isNumbers,
  parseEmbodiment,
  SIDE_POSE_KEYS,
} from "../src/internal.js";
import { matToRot6 } from "../src/linalg.js";

/** `values` with the element at each of `at` deleted: a sparse array. */
function holed<T>(values: readonly T[], ...at: number[]): T[] {
  const out = [...values];
  for (const i of at) delete out[i];
  return out;
}

const metal = arm("metal");
const q = [...metal.data.restPose];
const yamModel: Model = { id: "dreamlabs/dream-1-beta", owner: "public", baseModel: "dreamlabs/dream-1-beta", embodiment: "yam", servable: true, createdAt: null, cameras: ["front"], embodimentConfig: null, partsWithheld: false };
const body = { state: { right_arm: [0, 6], right_grip: [6, 7], left_arm: [7, 13], left_grip: [13, 14] }, action: { right_arm: [0, 6], left_arm: [6, 12] }, video: { front: "observation.images.front" } };

describe("the hole-aware checks", () => {
  it("find a hole that every(), some() and forEach() skip, in plain arrays only", () => {
    const sparse = holed([1, 2, 3], 1);
    expect(sparse.every((v) => typeof v === "number")).toBe(true); // the pattern that let holes through
    expect(every(sparse, (v) => typeof v === "number")).toBe(false);
    expect(firstHole(sparse)).toBe(1);
    expect(firstHole([1, , 3])).toBe(1);
    expect(firstHole(Float64Array.of(1, 2, 3))).toBe(-1);
    expect(firstFailing([1, , 3], () => true)).toBe(1);
    expect(isNumbers(sparse)).toBe(false);
    expect(isNumbers(Float32Array.of(1, 2))).toBe(true);
    expect(isNumbers(Float64Array.of(1, NaN), true)).toBe(false);
    expect(isNumbers(BigInt64Array.of(1n))).toBe(false);
  });

  it("refuse a hole anywhere in the model's reply, which no JSON reply has but a changed one may", () => {
    // From the wire, JSON.parse never makes a hole; the reply checks still refuse one, as getAction() would.
    const reply = { q_r_chunk: [[0, 0], holed([0, 0], 1)], per_key: { "action.right_grip": [[0.5], [0.5]] } };
    expect(firstNonFinite(reply)).toEqual({ key: "q_r_chunk", row: 1, hole: true });
    expect(() => checkFiniteAction(reply)).toThrow(/has a hole \(a missing element of a sparse array\) at key 'q_r_chunk' row 1/);
    try {
      checkFiniteAction({ per_key: { "action.right_grip": holed([[0.5], [0.5], [0.5]], 2) } });
      expect.unreachable();
    } catch (error) {
      expect(error).toBeInstanceOf(DreamActionError);
      expect((error as DreamActionError).detail).toEqual({ kind: "shape", key: "per_key.action.right_grip", row: 2 });
    }
    expect(firstNonFinite({ q: [Float64Array.of(1, 2)] })).toBeNull();
  });
});

describe("the seam smoother", () => {
  const chunk = () => Array.from({ length: 24 }, (_, k) => [0.01 * k, 0, 0, 0, 0, 0, 0.5]);
  const observation = { "state.right_arm": [0, 0, 0, 0, 0, 0] };

  it("refuses a chunk with a missing row or number, before or after the clock row", () => {
    const smoother = new SeamSmoother(20, null, SIDE_POSE_KEYS);
    smoother.smooth(observation, { q_r_chunk: chunk() }, 0.1);
    for (const [name, mutate, message] of [
      ["a row early", (c: number[][]) => delete c[1], /row 1 is a hole/],
      ["a row late", (c: number[][]) => delete c[20], /row 20 is a hole/],
      ["a number early", (c: number[][]) => delete c[1]![2], /row 1 holds a non-number or a hole/],
      ["a number late", (c: number[][]) => delete c[20]![6], /row 20 holds a non-number or a hole/],
    ] as const) {
      const c = chunk();
      mutate(c);
      expect(() => smoother.smooth(observation, { q_r_chunk: c }, 0.1), name).toThrow(DreamActionError);
      expect(() => smoother.smooth(observation, { q_r_chunk: c }, 0.1), name).toThrow(message);
    }
    // Typed rows are still taken.
    expect(() => smoother.smooth(observation, { q_r_chunk: chunk().map((r) => Float64Array.from(r)) }, 0.1)).not.toThrow();
  });

  it("refuses a sparse plan to adopt, and a sparse measured pose (flat() dropped the hole and shifted the joints)", () => {
    const smoother = new SeamSmoother(20, null, SIDE_POSE_KEYS);
    smoother.smooth(observation, { q_r_chunk: chunk() }, 0.1);
    const plan = chunk();
    delete plan[3]![1];
    expect(() => smoother.adopt("q_r_chunk", plan)).toThrow(/is not 24 rows of 7/);
    expect(() => smoother.adopt("q_r_chunk", holed(chunk(), 5))).toThrow(/is not 24 rows of 7/);
    expect(flatPose(holed([1, 2, 3], 1))).toBeNull();
    expect(flatPose([[1, 2], holed([3, 4], 0)])).toBeNull();
    expect(flatPose([[1, 2], [3, 4]])).toEqual([1, 2, 3, 4]);
    expect(flatPose(Float32Array.of(1, 2))).toEqual([1, 2]);
    expect(smoother.armJoints({ "state.right_arm": holed([0, 0, 0], 1) }).q_r_chunk).toBeNull();
    expect(smoother.armJoints({ "state.right_arm": Float64Array.of(0, 0, 0) }).q_r_chunk).toBe(3);
  });
});

describe("client IK", () => {
  const tools = Array.from({ length: 12 }, () => metal.tool(q));
  const reply = () => {
    const perKey: Record<string, unknown> = {};
    for (const side of ["right", "left"]) {
      perKey[`action.${side}_eef_xyz`] = tools.map((t) => [...t.position]);
      perKey[`action.${side}_eef_rot6`] = tools.map((t) => matToRot6(t.rotation));
      perKey[`action.${side}_grip`] = tools.map(() => [0.5]);
    }
    return { per_key: perKey } as { per_key: Record<string, unknown[]> };
  };
  const observation = { "state.right_arm": q, "state.right_grip": [0.5], "state.left_arm": q, "state.left_grip": [0.5], "video.front": ["Zg=="] };

  it("refuses a sparse per_key part, a sparse measured pose and a sparse reach point", () => {
    const ik = new ClientIK({ arm: "metal", head: "eef", speedCap: 0.5 }, 20, body, yamModel);
    const solve = (action: Record<string, unknown>, obs: Record<string, unknown> = observation) => ik.solve(action, obs, new SeamSmoother(20, null, SIDE_POSE_KEYS), 0.1, null);
    expect(() => solve(reply())).not.toThrow();
    for (const [name, mutate, message] of [
      ["a missing xyz row", (a: ReturnType<typeof reply>) => delete a.per_key["action.right_eef_xyz"]![4], /'action.right_eef_xyz' row 4 is a hole/],
      ["a missing rot6 number", (a: ReturnType<typeof reply>) => delete (a.per_key["action.left_eef_rot6"]![2] as number[])[5], /'action.left_eef_rot6' is not rows of 6 numbers/],
      ["a missing grip row", (a: ReturnType<typeof reply>) => delete a.per_key["action.right_grip"]![0], /'action.right_grip' row 0 is a hole/],
      ["a sparse flat grip list", (a: ReturnType<typeof reply>) => (a.per_key["action.right_grip"] = holed(tools.map(() => 0.5), 7)), /'action.right_grip' row 7 is a hole/],
    ] as const) {
      const action = reply();
      mutate(action);
      expect(() => solve(action), name).toThrow(DreamActionError);
      expect(() => solve(action), name).toThrow(message);
    }
    expect(() => solve(reply(), { ...observation, "state.left_arm": holed(q, 3) })).toThrow(/observation key 'state.left_arm' is not numbers/);
    expect(() => solve(reply(), { ...observation, "state.left_arm": Float64Array.from(q) })).not.toThrow();
    expect(() => new ClientIK({ arm: "metal", head: "eef", reach: { right: [[0.2, 0], holed([0.3, 0.1], 1)] as never } }, 20, body, yamModel)).toThrow(/bad point/);
  });

  it("the decode helpers refuse sparse rows: headPoses (map() skipped the hole) and solveChunk's qBefore (forEach() did)", () => {
    const xyz = tools.map((t) => [...t.position]);
    const rot6 = tools.map((t) => matToRot6(t.rotation));
    expect(headPoses(xyz, rot6)).toHaveLength(12);
    expect(() => headPoses(holed(xyz, 3), rot6)).toThrow(/xyz row 3 is not 3 numbers/);
    expect(() => headPoses(xyz, rot6.map((r, k) => (k === 9 ? holed(r, 0) : r)))).toThrow(/rot6 row 9 is not 6 numbers/);
    expect(() => solveChunk(metal, xyz, rot6, q, { hz: 20, qBefore: holed([q, q], 1) })).toThrow(/qBefore row 1 must be/);
    expect(() => solveChunk(metal, xyz, rot6, holed(q, 2), { hz: 20 })).toThrow(/qFrom holds a non-finite joint/);
    expect(() => solveChunk(metal, xyz, rot6, Float64Array.from(q) as never, { hz: 20 })).not.toThrow();
  });
});

describe("observations and configs", () => {
  it("the wire refuses a hole instead of sending it as null, and getAction() names the key before anything is sent", async () => {
    expect(() => encode({ a: holed([1, 2, 3], 1) })).toThrow(/sparse array \(a hole at index 1\)/);
    expect(encode({ a: Float32Array.of(1, 2) })).toBe('{"a":[1,2]}');
    const sent: string[] = [];
    const fetch = async (url: string): Promise<Response> => (sent.push(url), new Response('{"status":{"code":"DA-0000"}}'));
    const client = new PolicyClient({ checkpoint: "c", baseUrl: "http://gateway.test", fetch });
    const observation = { "state.arm_r.joint_pos": holed([0, 0, 0, 0, 0, 0], 2), "video.front": ["Zg=="] };
    await expect(client.getAction(observation)).rejects.toThrow(DreamObservationError);
    await expect(client.getAction(observation)).rejects.toThrow(/observation key 'state.arm_r.joint_pos' cannot be sent as JSON \(cannot encode a sparse array/);
    expect(sent.filter((url) => url.endsWith("/v1/get_action"))).toEqual([]);
  });

  it("the strict checks see a hole as no shape, and refuse sparse spans and shapes", () => {
    const config = { state: { right_arm: [0, 6], right_gripper: [6, 7] }, action: { right_arm: [0, 6], right_gripper: [6, 7] }, video: { overhead: "observation.images.top" } };
    const checks = compileChecks(config)!;
    const good = { "state.right_arm": [0, 0, 0, 0, 0, 0], "state.right_gripper": [1], "video.overhead": ["Zg=="] };
    check(good, checks.observation, "observation", { frameWindowGroup: checks.frameWindowGroup });
    expect(() => check({ ...good, "state.right_arm": holed(good["state.right_arm"], 3) }, checks.observation, "observation")).toThrow(
      /'state.right_arm' is not an array of one shape \(a hole, a sparse array's missing element, at index 3\)/,
    );
    expect(() => check({ ...good, "video.overhead": holed(["Zg==", "Zg=="], 0) }, checks.observation, "observation", { frameWindowGroup: "video" })).toThrow(DreamObservationError);
    expect(() => parseEmbodiment({ ...config, state: { right_arm: holed([0, 6], 1), right_gripper: [6, 7] } })).toThrow(/must be \[start, end\], two whole numbers/);
    expect(() => compileChecks({ state: { "arm_r.joint_pos": { shape: holed([6, 1], 1) } } })).toThrow(/expected a list of ints/);
  });

  it("an arm file with a sparse rest or tcp is refused", () => {
    const urdf = readFileSync(new URL("./fixtures/metal.urdf", import.meta.url), "utf8");
    const file = { base: "base_link", flange: "link6", tcp: [0.135, 0, 0], approach: [1, 0, 0], jaw: [0, 1, 0], rest: [0, -1.2, 1.5, 0, 0, 0] };
    expect(() => defineArm(file, urdf)).not.toThrow();
    expect(() => defineArm({ ...file, rest: holed(file.rest, 4) }, urdf)).toThrow(DreamConfigError);
    expect(() => defineArm({ ...file, tcp: holed(file.tcp, 1) }, urdf)).toThrow(/tcp must be 3 numbers/);
  });
});
