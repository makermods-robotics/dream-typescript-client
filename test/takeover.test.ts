/**
 * The IK's own time in flight (`ikLeadMs`) and the install-time takeover (`takeoverAt`), on a fake clock: a loop that
 * keeps playing rows while the client solves installs each chunk that many rows after the round trip, and the step
 * into it must stay within the speed cap. On a real SO-101 a 220-800 ms solve made seams jump up to 0.377 rad.
 */

import { describe, expect, it } from "vitest";
import { arm, ClientIK } from "../src/ik/index.js";
import type { PolicyIk } from "../src/client.js";
import { DreamActionError, DreamClient, DreamConfigError, PolicyClient, type ActionDict, type Diagnostics, type Model } from "../src/index.js";
import { matToRot6 } from "../src/linalg.js";

const HZ = 20;
const ROW_MS = 1000 / HZ;
const RTT_ROWS = 6;
const CAP = 0.5;
const metal = arm("metal");
const STEP = metal.data.maxSpeed.map((v) => Math.min(v, CAP) / HZ);

const yamModel: Model = { id: "dreamlabs/dream-1-beta", owner: "public", baseModel: "dreamlabs/dream-1-beta", embodiment: "yam", servable: true, createdAt: null, cameras: ["front"], embodimentConfig: null, partsWithheld: false };
const body = { state: { right_arm: [0, 6], right_grip: [6, 7], left_arm: [7, 13], left_grip: [13, 14] }, action: { right_arm: [0, 6], left_arm: [6, 12] }, video: { front: "observation.images.front" } };

/** A Metal client with client IK whose solve takes `ikRows(call)` rows on the fake clock, against a stub gateway. */
function slowIkClient(ikRows: (call: number) => number, ikLeadMs?: "auto" | number, grips?: (call: number, k: number) => number, gripSpeedCap?: number) {
  const clock = { ms: 0 };
  let call = 0;
  let from = [...metal.data.restPose];
  const fetch = async (url: string): Promise<Response> => {
    if (url.endsWith("/v1/end_session")) return new Response('{"status":{"code":"DA-0000"}}');
    clock.ms += RTT_ROWS * ROW_MS;
    // A fresh path from where the arm is, its phase moving every call: each chunk disagrees with the one before.
    const path = Array.from({ length: 48 }, (_, k) => from.map((v, j) => v + 0.6 * (Math.sin(0.4 * k + call + j) - Math.sin(call + j))));
    call++;
    const tools = path.map((q) => metal.tool(q));
    const perKey: Record<string, unknown> = {};
    for (const side of ["right", "left"]) {
      perKey[`action.${side}_eef_xyz`] = tools.map((t) => [...t.position]);
      perKey[`action.${side}_eef_rot6`] = tools.map((t) => matToRot6(t.rotation));
      perKey[`action.${side}_grip`] = path.map((_, k) => [grips ? grips(call - 1, k) : 0.5]);
    }
    return new Response(JSON.stringify({ status: { code: "DA-0000" }, session_id: "s", actions: { per_key: perKey } }));
  };
  const ik = new ClientIK({ arm: "metal", head: "eef", speedCap: CAP, ...(gripSpeedCap === undefined ? {} : { gripSpeedCap }) }, HZ, body, yamModel);
  let solves = 0;
  // The solve itself, but taking time on the loop's clock.
  const slow: PolicyIk = Object.create(ik, {
    solve: { value: (...args: Parameters<ClientIK["solve"]>) => ((clock.ms += ikRows(solves++) * ROW_MS), ik.solve(...args)) },
  });
  const diagnostics: Diagnostics[] = [];
  const client = new PolicyClient({ checkpoint: yamModel.id, baseUrl: "http://gateway.test", fetch, ik: slow, embodiment: body, strict: false, actionHz: HZ, now: () => clock.ms, onWarning: () => {}, onDiagnostics: (d) => diagnostics.push(d), ...(ikLeadMs === undefined ? {} : { ikLeadMs }) });
  const observation = (q: readonly number[], grip = 0.5) => {
    from = [...q];
    return { "state.right_arm": [...q], "state.right_grip": [grip], "state.left_arm": [...q], "state.left_grip": [grip], "video.front": ["Zg=="] };
  };
  return { client, clock, observation, diagnostics };
}

/** Largest per-joint step between consecutive commands, as a multiple of the speed cap's step (1: at the cap). */
function worstStep(commands: readonly (readonly number[])[]): number {
  let worst = 0;
  for (let i = 1; i < commands.length; i++) for (let j = 0; j < 6; j++) worst = Math.max(worst, Math.abs(commands[i]![j]! - commands[i - 1]![j]!) / STEP[j]!);
  return worst;
}

/**
 * The loop of the control-loop doc on the fake clock: it plays one row per 50 ms while a request is in flight, and
 * installs each reply at the row due by then (rows run since the request). Returns every command it sent from the
 * first seam on, and the row each chunk was installed at.
 */
async function run(
  setup: ReturnType<typeof slowIkClient>,
  install: (client: PolicyClient, action: ActionDict, row: number, last: number[]) => ActionDict = (_c, action) => action,
  chunks = 5,
): Promise<{ commands: number[][]; rows: number[] }> {
  const { client, clock, observation } = setup;
  let q = [...metal.data.restPose];
  let chunk = (await client.getAction(observation(q))).q_r_chunk as number[][];
  let at = 19; // the row of `chunk` last commanded: the first chunk ran 20 rows
  const commands: number[][] = [chunk[at]!.slice(0, 6)];
  const rows: number[] = [];
  for (let n = 1; n < chunks; n++) {
    let last = chunk[at]!;
    q = last.slice(0, 6);
    const sentAt = clock.ms;
    const reply = await client.getAction(observation(q), { commanded: { q_r_chunk: last, q_l_chunk: last } });
    // The rows the loop ran meanwhile, on the old chunk: the round trip and the solve.
    const row = Math.floor((clock.ms - sentAt) / ROW_MS);
    for (let k = 1; k <= row; k++) commands.push((last = chunk[at + k]!).slice(0, 6));
    chunk = install(client, reply, row, last).q_r_chunk as number[][];
    rows.push(row);
    // Then 12 rows of the new chunk from the row due, and the next request.
    for (let k = row; k < row + 12; k++) commands.push(chunk[k]!.slice(0, 6));
    at = row + 11;
  }
  return { commands, rows };
}

describe("the IK's own time in flight", () => {
  it("auto: the seam takes the solve time as run, so a chunk installed after an 8-row solve steps in within the cap", async () => {
    const setup = slowIkClient(() => 8);
    const { commands, rows } = await run(setup);
    expect(rows).toEqual([RTT_ROWS + 8, RTT_ROWS + 8, RTT_ROWS + 8, RTT_ROWS + 8]);
    expect(worstStep(commands)).toBeLessThanOrEqual(1 + 1e-9);
    // The first solve has no history; every later seam leads by the solves before it.
    expect(setup.diagnostics.map((d) => d.ikLeadS)).toEqual([0, 0.4, 0.4, 0.4, 0.4]);
    expect(setup.diagnostics.slice(1).map((d) => d.ik!.right!.clock)).toEqual([14, 14, 14, 14]);
  });

  it("0 (upstream's round trip only): the same loop jumps at every seam, the failure seen on hardware", async () => {
    const { commands } = await run(slowIkClient(() => 8, 0));
    expect(worstStep(commands)).toBeGreaterThan(2);
  });

  it("auto takes the slowest of the last three solves, so a slower one after faster ones is covered", async () => {
    const times = [4, 4, 12, 4, 4, 4, 4];
    const setup = slowIkClient((n) => times[n]!);
    await run(setup, undefined, 7);
    expect(setup.diagnostics.map((d) => Math.round(d.ikLeadS * 1000))).toEqual([0, 200, 200, 600, 600, 600, 200]);
  });

  it("auto leaves out each episode's first (cold) solve once a later one is measured, and uses it only until then", async () => {
    // The first solve of an episode (no plan to continue, the process's first) is the slow one, as on hardware.
    const times = [12, 4, 8, 4, 4, 4, 4, 12, 4, 4];
    const setup = slowIkClient((n) => times[n]!);
    await run(setup, undefined, 7);
    expect(setup.diagnostics.map((d) => Math.round(d.ikLeadS * 1000))).toEqual([0, 600, 200, 400, 400, 400, 200]);
    // A new episode's cold solve leaves the lead on the warm ones before it.
    await setup.client.resetEpisode();
    await run(setup, undefined, 3);
    expect(setup.diagnostics.slice(7).map((d) => Math.round(d.ikLeadS * 1000))).toEqual([200, 200, 200]);
    // Deterministic: the same solve times give the same leads.
    const again = slowIkClient((n) => times[n]!);
    await run(again, undefined, 7);
    expect(again.diagnostics.map((d) => d.ikLeadS)).toEqual(setup.diagnostics.slice(0, 7).map((d) => d.ikLeadS));
  });

  it("takeoverAt(): re-anchors a chunk installed later than the seam assumed, within the cap, and the next seam continues it", async () => {
    // The lead is off and the solve takes 10 rows: every install is 10 rows past the seam's clock row.
    const setup = slowIkClient(() => 10, 0);
    const adopted: number[][][] = [];
    const replies: number[][][] = [];
    const { commands, rows } = await run(setup, (client, action, row, last) => {
      replies.push(action.q_r_chunk as number[][]);
      const out = client.takeoverAt(action, row, { q_r_chunk: last, q_l_chunk: last });
      adopted.push(out.q_r_chunk as number[][]);
      // Rows before the install row are as returned; the reply itself is not changed.
      expect((out.q_r_chunk as number[][]).slice(0, row)).toEqual((action.q_r_chunk as number[][]).slice(0, row));
      return out;
    });
    expect(rows.every((row) => row === RTT_ROWS + 10)).toBe(true);
    expect(worstStep(commands)).toBeLessThanOrEqual(1 + 1e-9);
    // The plan the next seam continues is the re-anchored one: the next reply's rows before its clock row are the
    // re-anchored chunk's, from the row after the last command (row + 11) on.
    expect(setup.diagnostics.at(-1)!.ik!.right!.clock).toBe(RTT_ROWS);
    for (let n = 1; n < replies.length; n++) {
      for (let k = 0; k < RTT_ROWS; k++) {
        const want = adopted[n - 1]![rows[n - 1]! + 12 + k]!;
        replies[n]![k]!.slice(0, 6).forEach((v, j) => expect(Math.abs(v - want[j]!), `chunk ${n} row ${k} joint ${j}`).toBeLessThan(1e-12));
      }
    }
  });

  it("takeoverAt(): the first chunk installed late ramps from the measured pose under the startup cap", async () => {
    const setup = slowIkClient(() => 10);
    const q = [...metal.data.restPose];
    const first = await setup.client.getAction(setup.observation(q));
    const late = setup.client.takeoverAt(first, 12, { q_r_chunk: q, q_l_chunk: q });
    const commands = [q, ...(late.q_r_chunk as number[][]).slice(12).map((row) => row.slice(0, 6))];
    expect(worstStep(commands)).toBeLessThanOrEqual(1 + 1e-9);
    // Played as is from row 12, the step from the pose would be 12 rows' worth.
    expect(worstStep([q, (first.q_r_chunk as number[][])[12]!.slice(0, 6)])).toBeGreaterThan(2);
  });

  it("takeoverAt() returns the chunk as is without client IK or without commanded, and refuses a bad row", async () => {
    const client = new DreamClient({ baseUrl: "http://gateway.test", platformUrl: "http://platform.test", fetch: async () => new Response("{}") });
    expect(() => client.takeoverAt({}, 0, null)).toThrow(/no policy is open/);
    const { client: policy } = slowIkClient(() => 1);
    const action = { q_r_chunk: [[0, 0, 0, 0, 0, 0, 0.5]] };
    expect(policy.takeoverAt(action, 0, null)).toBe(action);
    expect(() => policy.takeoverAt(action, -1, null)).toThrow(/row must be an int/);
    await expect(new DreamClient({ baseUrl: "http://gateway.test", platformUrl: "http://platform.test" }).policy({ checkpoint: "c", ikLeadMs: -5 })).rejects.toThrow(/ikLeadMs/);
  });
});

describe("takeoverAt() checks what it is given", () => {
  /** The plan the smoother keeps per arm, for the test to see it is unchanged. */
  const plans = (client: PolicyClient) => {
    const previous = (client as unknown as { smoother: { previous: Map<string, number[][]> } }).smoother.previous;
    return Object.fromEntries([...previous].map(([key, rows]) => [key, rows.map((row) => [...row])]));
  };
  const copy = (action: ActionDict): ActionDict => ({ ...action, q_r_chunk: (action.q_r_chunk as number[][]).map((r) => [...r]), q_l_chunk: (action.q_l_chunk as number[][]).map((r) => [...r]) });

  it("refuses a latest action the caller changed to hold NaN, an infinity, a wrong width or row count, and keeps no arm's plan", async () => {
    // Codex final check: a caller-modified latest action was projected and adopted, NaN and all.
    const { client, observation } = slowIkClient(() => 1);
    const q = [...metal.data.restPose];
    const latest = await client.getAction(observation(q));
    const kept = plans(client);
    const commanded = { q_r_chunk: q, q_l_chunk: q };
    const cases: [string, (a: ActionDict) => void, RegExp][] = [
      ["NaN after the install row", (a) => ((a.q_l_chunk as number[][])[30]![2] = NaN), /row 30 holds a non-finite/],
      ["NaN before the install row", (a) => ((a.q_l_chunk as number[][])[3]![0] = NaN), /row 3 holds a non-finite/],
      ["an infinite joint", (a) => ((a.q_l_chunk as number[][])[20]![1] = Infinity), /row 20 holds a non-finite/],
      ["an infinite grip", (a) => ((a.q_l_chunk as number[][])[40]![6] = -Infinity), /row 40 holds a non-finite/],
      ["a short row", (a) => (a.q_l_chunk as number[][])[15]!.pop(), /row 15 is 6 wide, not 7/],
      ["narrower rows", (a) => (a.q_l_chunk = (a.q_l_chunk as number[][]).map((r) => r.slice(0, 5))), /rows|wide/],
      ["a missing row", (a) => (a.q_l_chunk as number[][]).pop(), /has 47 rows; the chunk getAction\(\) returned has 48/],
      ["not rows", (a) => (a.q_l_chunk = 3 as never), /is not rows of numbers/],
    ];
    for (const [name, mutate, message] of cases) {
      mutate(latest);
      // The right arm is checked first and is fine: it must not be adopted either.
      expect(() => client.takeoverAt(latest, 12, commanded), name).toThrow(DreamActionError);
      expect(() => client.takeoverAt(latest, 12, commanded), name).toThrow(message);
      expect(plans(client), name).toEqual(kept);
      Object.assign(latest, { q_l_chunk: (kept.q_l_chunk as number[][]).map((r) => [...r]) });
    }
    // The same checks on a chunk that is not the latest (nothing to adopt): refused, never returned.
    const other = copy(latest);
    (other.q_r_chunk as number[][])[25]![4] = NaN;
    expect(() => client.takeoverAt(other, 12, commanded)).toThrow(DreamActionError);
    const narrow = copy(latest);
    narrow.q_r_chunk = (narrow.q_r_chunk as number[][]).map((r) => r.slice(0, 4));
    expect(() => client.takeoverAt(narrow, 12, commanded)).toThrow(/the arm has 6 joints/);
    // A finite change the caller made is still taken, and becomes the plan.
    (latest.q_r_chunk as number[][])[30]![0]! += 0.01;
    const out = client.takeoverAt(latest, 12, commanded);
    expect((out.q_r_chunk as number[][]).every((r) => r.every(Number.isFinite))).toBe(true);
    expect(plans(client).q_r_chunk).toEqual(out.q_r_chunk);
  });
});

describe("takeoverAt() checks every arm, holes included", () => {
  const plans = (client: PolicyClient) => {
    const previous = (client as unknown as { smoother: { previous: Map<string, number[][]> } }).smoother.previous;
    return Object.fromEntries([...previous].map(([key, rows]) => [key, rows.map((row) => [...row])]));
  };

  it("refuses a bad arm `commanded` leaves out, and a hole before or after the install row, and keeps no arm's plan", async () => {
    // Codex final check: an arm left out of `commanded` escaped the checks; a hole (`delete row[2]`, `[1, , 3]`) passed
    // the every()/forEach() checks, which skip holes, and was returned and kept as `undefined`.
    const { client, observation } = slowIkClient(() => 1);
    const q = [...metal.data.restPose];
    const latest = await client.getAction(observation(q));
    const kept = plans(client);
    const both = { q_r_chunk: q, q_l_chunk: q };
    const rightOnly = { q_r_chunk: q };
    const cases: [string, (a: ActionDict) => void, RegExp][] = [
      ["NaN after the install row", (a) => ((a.q_l_chunk as number[][])[30]![2] = NaN), /'q_l_chunk' row 30 holds a non-finite/],
      ["NaN before the install row", (a) => ((a.q_l_chunk as number[][])[3]![0] = NaN), /'q_l_chunk' row 3 holds a non-finite/],
      ["a number deleted before the install row", (a) => delete (a.q_l_chunk as number[][])[3]![2], /'q_l_chunk' row 3 has a hole at column 2/],
      ["a number deleted after the install row", (a) => delete (a.q_l_chunk as number[][])[30]![2], /'q_l_chunk' row 30 has a hole at column 2/],
      ["a sparse row literal before the install row", (a) => ((a.q_l_chunk as number[][])[5] = [0, , 0, 0, 0, 0, 0.5] as never), /'q_l_chunk' row 5 has a hole at column 1/],
      ["the grip deleted before the install row", (a) => delete (a.q_l_chunk as number[][])[2]![6], /'q_l_chunk' row 2 has a hole at column 6/],
      ["a row deleted before the install row", (a) => delete (a.q_l_chunk as number[][])[4], /'q_l_chunk' row 4 is a hole/],
      ["a row deleted after the install row", (a) => delete (a.q_l_chunk as number[][])[40], /'q_l_chunk' row 40 is a hole/],
      ["a string", (a) => (((a.q_l_chunk as number[][])[6]![1] as unknown) = "1"), /'q_l_chunk' row 6 holds a non-number/],
      ["a short row", (a) => (a.q_l_chunk as number[][])[15]!.pop(), /'q_l_chunk' row 15 is 6 wide, not 7/],
      ["the arm's chunk gone", (a) => delete a.q_l_chunk, /'q_l_chunk' is gone/],
    ];
    for (const [name, mutate, message] of cases) {
      for (const [which, commanded] of [["both commanded", both], ["left arm left out", rightOnly]] as const) {
        mutate(latest);
        expect(() => client.takeoverAt(latest, 12, commanded), `${name}, ${which}`).toThrow(DreamActionError);
        expect(() => client.takeoverAt(latest, 12, commanded), `${name}, ${which}`).toThrow(message);
        expect(plans(client), `${name}, ${which}`).toEqual(kept);
        Object.assign(latest, { q_l_chunk: (kept.q_l_chunk as number[][]).map((r) => [...r]) });
      }
    }
    // The same on the arm that is commanded, while the other is left out.
    delete (latest.q_r_chunk as number[][])[3]![2];
    expect(() => client.takeoverAt(latest, 12, rightOnly)).toThrow(/'q_r_chunk' row 3 has a hole at column 2/);
    expect(plans(client)).toEqual(kept);
    (latest.q_r_chunk as number[][])[3]![2] = (kept.q_r_chunk as number[][])[3]![2]!;
    // Valid, the arm left out is returned as it was (dense, new arrays), its plan unchanged; the other re-anchored.
    const out = client.takeoverAt(latest, 12, rightOnly);
    expect(out.q_l_chunk).toEqual(kept.q_l_chunk);
    expect(out.q_l_chunk).not.toBe(latest.q_l_chunk);
    expect(plans(client).q_l_chunk).toEqual(kept.q_l_chunk);
    expect(plans(client).q_r_chunk).toEqual(out.q_r_chunk);
  });

  it("takes typed-array rows and commanded (they have no holes), refuses the sparse plain ones, on any chunk", async () => {
    const { client, observation } = slowIkClient(() => 1);
    const q = [...metal.data.restPose];
    const latest = await client.getAction(observation(q));
    const kept = plans(client);
    // Not the latest: nothing kept, but a bad arm left out is still never returned.
    const other: ActionDict = { q_r_chunk: (kept.q_r_chunk as number[][]).map((r) => [...r]), q_l_chunk: (kept.q_l_chunk as number[][]).map((r) => [...r]) };
    delete (other.q_l_chunk as number[][])[1]![0];
    expect(() => client.takeoverAt(other, 12, { q_r_chunk: q })).toThrow(/'q_l_chunk' row 1 has a hole at column 0/);
    const typed = { q_r_chunk: (kept.q_r_chunk as number[][]).map((r) => Float64Array.from(r)), q_l_chunk: (kept.q_l_chunk as number[][]).map((r) => Float64Array.from(r)) } as unknown as ActionDict;
    const fromTyped = client.takeoverAt(typed, 12, { q_r_chunk: Float64Array.from(q) as never });
    expect(fromTyped.q_l_chunk).toEqual(kept.q_l_chunk);
    expect((fromTyped.q_l_chunk as number[][]).every(Array.isArray)).toBe(true);
    // A sparse commanded is refused before anything; a typed one is the same as the plain one.
    const sparse = [...q];
    delete sparse[2];
    expect(() => client.takeoverAt(latest, 12, { q_r_chunk: sparse })).toThrow(/commanded 'q_r_chunk' must be a flat run of finite numbers/);
    expect(plans(client)).toEqual(kept);
    const plain = client.takeoverAt({ ...other, q_l_chunk: (kept.q_l_chunk as number[][]) }, 12, { q_r_chunk: q });
    expect(client.takeoverAt({ ...other, q_l_chunk: (kept.q_l_chunk as number[][]) }, 12, { q_r_chunk: Float64Array.from(q) as never })).toEqual(plain);
    expect(plans(client)).toEqual(kept);
  });

  it("getAction() refuses a sparse commanded or observation before anything is sent", async () => {
    const { client, observation } = slowIkClient(() => 1);
    const q = [...metal.data.restPose];
    await client.getAction(observation(q));
    const kept = plans(client);
    const sparse = [...q, 0.5];
    delete sparse[1];
    await expect(client.getAction(observation(q), { commanded: { q_r_chunk: sparse } })).rejects.toThrow(/commanded 'q_r_chunk' must be a flat run of finite numbers/);
    const obs = observation(q);
    delete (obs["state.right_arm"] as number[])[4];
    await expect(client.getAction(obs)).rejects.toThrow(/observation key 'state.right_arm' is not numbers/);
    const typed = { ...observation(q), "state.right_arm": Float64Array.from(q) };
    await expect(client.getAction(typed, { commanded: { q_r_chunk: Float32Array.from(q) as never } })).resolves.toBeDefined();
    expect(plans(client)).not.toEqual(kept);
  });
});

describe("ik.gripSpeedCap", () => {
  const GRIP_CAP = 0.5; // grip units per second: 0.025 per row at 20 Hz
  const worstGrip = (rows: readonly (readonly number[])[]) => Math.max(...rows.slice(1).map((r, k) => Math.abs(r[6]! - rows[k]![6]!)));

  it("is off by default (as upstream): the grip steps as the model answers it, clamped to [0, 1]", async () => {
    const { client, observation } = slowIkClient(() => 1, undefined, (_call, k) => (k % 2 ? 1.3 : -0.2));
    const chunk = (await client.getAction(observation([...metal.data.restPose], 0))).q_r_chunk as number[][];
    expect(chunk.every((r) => r[6]! >= 0 && r[6]! <= 1)).toBe(true);
    expect(worstGrip(chunk)).toBeGreaterThan(0.5);
  });

  it("holds every row's grip step within the cap, from the measured grip, across seams and in takeoverAt()", async () => {
    // Each chunk flips the grip between open and closed: without the cap, a seam and a late install step it whole.
    const setup = slowIkClient(() => 2, undefined, (call, k) => ((call + Math.floor(k / 8)) % 2 ? 1 : 0), GRIP_CAP);
    const step = GRIP_CAP / HZ;
    const { commands } = await (async () => {
      const { client, observation } = setup;
      const q = [...metal.data.restPose];
      let chunk = (await client.getAction(observation(q, 0.9))).q_r_chunk as number[][];
      // The first row steps from the measured grip, 0.9.
      expect(Math.abs(chunk[0]![6]! - 0.9)).toBeLessThanOrEqual(step + 1e-12);
      const commands: number[][] = chunk.slice(0, 20);
      for (let n = 0; n < 4; n++) {
        const last = commands.at(-1)!;
        const reply = await client.getAction(observation(last.slice(0, 6), last[6]!), { commanded: { q_r_chunk: last, q_l_chunk: last } });
        // Installed late, at row 9, through takeoverAt() from the written row (the grip its last value).
        chunk = client.takeoverAt(reply, 9, { q_r_chunk: last, q_l_chunk: last }).q_r_chunk as number[][];
        commands.push(...chunk.slice(9, 21));
      }
      return { commands };
    })();
    expect(worstGrip(commands)).toBeLessThanOrEqual(step + 1e-12);
    // It moved, at the cap: the cap shapes the grip, it does not freeze it.
    expect(Math.max(...commands.map((r) => r[6]!)) - Math.min(...commands.map((r) => r[6]!))).toBeGreaterThan(0.2);
  });

  it("is checked like the other caps", async () => {
    expect(() => new ClientIK({ arm: "metal", head: "eef", gripSpeedCap: 0 }, HZ, body, yamModel)).toThrow(DreamConfigError);
    expect(() => new ClientIK({ arm: "metal", head: "eef", gripSpeedCap: -1 }, HZ, body, yamModel)).toThrow(/ik.gripSpeedCap' must be a positive number of grip units/);
    expect(new ClientIK({ arm: "metal", head: "eef", gripSpeedCap: Infinity }, HZ, body, yamModel).gripSpeedCap).toBe(Infinity);
  });
});
