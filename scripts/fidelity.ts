/**
 * Measures every number the README's Fidelity section states, from the committed fixtures, and holds each to its
 * ceiling. `npm run fidelity` prints the table (scripts/fidelity.mjs); test/fidelity.test.ts checks that every
 * measured value is under its ceiling and that the README states the measured values as printed here; the tests of
 * each area (fk, ik, invariants, wire) take their ceilings from {@link CEILINGS}.
 *
 * Every measurement is deterministic: the fixtures are fixed, and the solves are float64 on one thread.
 */

import { readFileSync } from "node:fs";
import { arm, rot6ToMat, solveChunk, type Arm } from "../src/ik/index.js";
import { DreamClient, type PolicyConfig } from "../src/index.js";
import { matToRot6, mul3, norm3, rotvec, transpose3, type Mat3 } from "../src/linalg.js";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Json = any;

const fixture = (name: string): Json => JSON.parse(readFileSync(new URL(`../test/fixtures/${name}`, import.meta.url), "utf8"));
/** The angle of the turn from `a` to `b`, radians. */
const angle = (a: Mat3, b: Mat3) => norm3(rotvec(mul3(transpose3(a), b)));
const max = (values: Iterable<number>) => Math.max(-Infinity, ...values);

/**
 * The ceilings the tests hold each measurement to: each a margin above the measured value printed by
 * `npm run fidelity` (the README states both), round numbers so a re-measure need not move them.
 */
export const CEILINGS = {
  fk: { positionVsUrdf: 1e-12, rotationVsUrdf: 1e-12, positionVsDreamIk: 1e-6, jacobianVsFiniteDifferences: 1e-8 },
  ik: {
    /** Tool path vs dream_ik's, metres: tick; traj; {@link NULL_SPACE_ARM}'s capped traj. */
    toolTick: 5e-5,
    toolTraj: 3e-4,
    toolNullSpace: 1e-2,
    /** Joints vs dream_ik's, radians: tick; traj; {@link NULL_SPACE_ARM}'s traj, capped or not. */
    jointsTick: 5e-4,
    jointsTraj: 5e-3,
    jointsNullSpace: 0.1,
    /** The rotation reached vs dream_ik's, radians: up to six joints; seven; {@link NULL_SPACE_ARM}'s capped traj. */
    rotationSix: 2e-4,
    rotationSeven: 2e-3,
    rotationNullSpace: 0.1,
    /** Our orientation miss less dream_ik's, radians, row by row, every case. */
    orientationMissExcess: 1e-3,
    /** Our own orientation miss of the target, radians: six joints; seven (five-joint so101_faas cannot reach all). */
    orientationMissSix: 0.2,
    orientationMissSeven: 0.5,
    /** Joints vs dream_ik's traj solve run to convergence, at a joint limit (limit_stop.json), radians. */
    jointsAtLimit: 2e-3,
  },
  invariants: { tickPosition: 2e-4, tickAngle: 2e-3, trajPosition: 1.2e-2, trajAngle: 4e-2 },
  wire: { body: 1e-6, actions: 1e-12, ikJoints: 1e-3 },
} as const;

/**
 * The arm the two optimizers part on: ai_worker_faas has seven joints, a null space pyroki/jaxls (float32) and this
 * package's solve (float64) settle in differently, its joints up to 0.07 rad apart in traj; uncapped the tool paths
 * still agree, capped the different joints move the hand differently. (bracketbot_faas, seven joints too, agrees.)
 */
export const NULL_SPACE_ARM = "ai_worker_faas";

/** The ceilings one ik.json case is held to. */
export function ikCeilings(c: { arm: string; solve: string; speed_cap: number | null }, joints: number) {
  const k = CEILINGS.ik;
  const nullSpace = c.arm === NULL_SPACE_ARM && c.solve === "traj";
  const capped = nullSpace && c.speed_cap !== null;
  return {
    tool: c.solve === "tick" ? k.toolTick : capped ? k.toolNullSpace : k.toolTraj,
    joints: c.solve === "tick" ? k.jointsTick : nullSpace ? k.jointsNullSpace : k.jointsTraj,
    rotation: joints <= 6 ? k.rotationSix : capped ? k.rotationNullSpace : k.rotationSeven,
    orientationMissExcess: k.orientationMissExcess,
    // The five-joint so101_faas cannot reach every orientation: no ceiling of its own.
    orientationMiss: joints < 6 ? null : joints > 6 ? k.orientationMissSeven : k.orientationMissSix,
  };
}

export function measureFk() {
  const fk = fixture("fk.json");
  let positionVsUrdf = 0;
  let rotationVsUrdf = 0;
  let positionVsDreamIk = 0;
  for (const c of fk.cases) {
    const model = arm(c.arm, c.side);
    c.q.forEach((q: number[], k: number) => {
      const { position, rotation } = model.tool(q);
      const expected = c.poses[k];
      position.forEach((v, i) => (positionVsUrdf = Math.max(positionVsUrdf, Math.abs(v - expected[i][3]))));
      rotation.forEach((v, i) => (rotationVsUrdf = Math.max(rotationVsUrdf, Math.abs(v - expected[Math.floor(i / 3)][i % 3]))));
      position.forEach((v, i) => (positionVsDreamIk = Math.max(positionVsDreamIk, Math.abs(v - c.dream_ik_fk[k][i][3]))));
    });
  }
  let jacobianVsFiniteDifferences = 0;
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
      for (let r = 0; r < 3; r++) jacobianVsFiniteDifferences = Math.max(jacobianVsFiniteDifferences, Math.abs((a[r]! - b[r]!) / (2 * h) - jac[r]![j]!));
    }
  }
  return { cases: fk.cases.length as number, positionVsUrdf, rotationVsUrdf, positionVsDreamIk, jacobianVsFiniteDifferences };
}

/** One ik.json case solved: its tool path, joint and rotation distances to dream_ik's answer, worst over sides and rows. */
export interface IkCase {
  name: string;
  arm: string;
  solve: "tick" | "traj";
  capped: boolean;
  joints: number;
  ceilings: ReturnType<typeof ikCeilings>;
  tool: number;
  jointSpread: number;
  orientationMissExcess: number;
  rotation: number;
  orientationMiss: number;
  /** The solve's own answer per side, for the tests' step and limit checks. */
  sides: { side: string; model: Arm; q: number[][] }[];
}

export function measureIkCase(c: Json): IkCase {
  const cap = c.speed_cap ?? Infinity;
  const joints = arm(c.arm, c.sides[0]).free.length;
  const out: IkCase = {
    name: `${c.arm} ${c.variant} ${c.solve} cap ${c.speed_cap ?? "none"}`,
    arm: c.arm,
    solve: c.solve,
    capped: c.speed_cap !== null,
    joints,
    ceilings: ikCeilings(c, joints),
    tool: 0,
    jointSpread: 0,
    orientationMissExcess: -Infinity,
    rotation: 0,
    orientationMiss: 0,
    sides: [],
  };
  c.sides.forEach((side: string, i: number) => {
    const model = arm(c.arm, side);
    const { q } = solveChunk(model, c.xyz[i], c.rot6[i], c.q_from[i], { hz: c.hz, speedCap: cap, serve: { solve: c.solve }, qBefore: c.q_before[i] });
    out.sides.push({ side, model, q });
    q.forEach((row, k) => {
      const ours = model.tool(row);
      const theirs = model.tool(c.expected_q[i][k]).rotation;
      const target = rot6ToMat(c.rot6[i][k])!;
      out.tool = Math.max(out.tool, Math.hypot(...ours.position.map((v, j) => v - c.expected_xyz[i][k][j])));
      out.jointSpread = Math.max(out.jointSpread, ...row.map((v, j) => Math.abs(v - c.expected_q[i][k][j])));
      out.orientationMissExcess = Math.max(out.orientationMissExcess, angle(ours.rotation, target) - angle(theirs, target));
      out.rotation = Math.max(out.rotation, angle(ours.rotation, theirs));
      out.orientationMiss = Math.max(out.orientationMiss, angle(ours.rotation, target));
    });
  });
  return out;
}

export const ikCases = (): Json[] => fixture("ik.json").cases;

export function measureIk(cases: IkCase[] = ikCases().map(measureIkCase)) {
  const worst = (test: (c: IkCase) => boolean, key: "tool" | "jointSpread" | "rotation" | "orientationMiss" | "orientationMissExcess") => max(cases.filter(test).map((c) => c[key]));
  const tick = (c: IkCase) => c.solve === "tick";
  const traj = (c: IkCase) => c.solve === "traj";
  const nullSpace = (c: IkCase) => c.arm === NULL_SPACE_ARM && traj(c);
  const of = (name: string, capped: boolean) => (c: IkCase) => c.arm === name && traj(c) && c.capped === capped;
  const sevenJointArms = [...new Set(cases.filter((c) => c.joints > 6 && traj(c)).map((c) => c.arm))];
  return {
    cases: cases.length,
    arms: [...new Set(cases.map((c) => c.arm))],
    toolTick: worst(tick, "tool"),
    jointsTick: worst(tick, "jointSpread"),
    toolTraj: worst((c) => traj(c) && !(nullSpace(c) && c.capped), "tool"),
    jointsTraj: worst((c) => traj(c) && !nullSpace(c), "jointSpread"),
    /** Per seven-joint arm's traj, capped and not: tool path and joints vs dream_ik, and the ceilings they are held to. */
    sevenJointTraj: Object.fromEntries(
      sevenJointArms.map((name) => {
        const side = (capped: boolean) => {
          const ceilings = cases.find(of(name, capped))!.ceilings;
          return { tool: worst(of(name, capped), "tool"), joints: worst(of(name, capped), "jointSpread"), toolCeiling: ceilings.tool, jointsCeiling: ceilings.joints };
        };
        return [name, { capped: side(true), uncapped: side(false) }];
      }),
    ) as Record<string, Record<"capped" | "uncapped", { tool: number; joints: number; toolCeiling: number; jointsCeiling: number }>>,
    orientationMissExcess: worst(() => true, "orientationMissExcess"),
    orientationMissExcessSix: worst((c) => c.joints <= 6, "orientationMissExcess"),
    rotationSix: worst((c) => c.joints <= 6, "rotation"),
    rotationSeven: worst((c) => c.joints > 6 && !(nullSpace(c) && c.capped), "rotation"),
    rotationNullSpace: worst((c) => nullSpace(c) && c.capped, "rotation"),
    orientationMissSix: worst((c) => c.joints === 6, "orientationMiss"),
    orientationMissSeven: worst((c) => c.joints > 6, "orientationMiss"),
  };
}

/** One limit_stop.json side solved as the client solves a first chunk (its own filter off, the traj solve). */
export function solveLimitStop(f: Json, c: Json) {
  const model = arm(f.arm, c.side);
  const options = { hz: f.hz, speedCap: f.speed_cap ?? Infinity, serve: { smoothWindow: 0, seamBlendTicks: 0, solve: "traj" as const }, qBefore: c.q_before };
  return { model, q: solveChunk(model, c.xyz, c.rot6, c.q_from, options).q.map((row) => c.free.map((i: number) => row[i]!)) as number[][] };
}

/**
 * A two-arm SO-101 chunk at a joint limit, recorded on hardware (limit_stop.json): our traj solve against dream_ik's
 * run to convergence, and dream_ik's own answer against it, which stopped early (jaxls's relative cost change under
 * 1e-5 while the stiff limit cost kept each step short) and is 0.02 rad away. The worst side of each.
 */
export function measureLimitStop() {
  const f = fixture("limit_stop.json");
  const spread = (a: number[][], b: number[][]) => max(a.flatMap((row, k) => row.map((v, j) => Math.abs(v - b[k]![j]!))));
  return {
    sides: f.cases.length as number,
    joints: max(f.cases.map((c: Json) => spread(solveLimitStop(f, c).q, c.converged))),
    dream: max(f.cases.map((c: Json) => spread(c.dream, c.converged))),
  };
}

/** A joint path from `start` swinging every joint by `amplitude` rad (test/invariants.test.ts's). */
export function swing(start: readonly number[], amplitude: number, phase = 0, rows = 48, perRow = 0.4): number[][] {
  return Array.from({ length: rows }, (_, k) => start.map((v, j) => v + amplitude * (Math.sin(perRow * k + phase + j) - Math.sin(phase + j))));
}

/** Metal tracking a reachable path from rest (0.15 rad swing at 0.1 rad a row), 24 rows and 2: the worst row's miss. */
export function measureReach(solve: "tick" | "traj") {
  const metal = arm("metal");
  const q0 = [...metal.data.restPose];
  let position = 0;
  let turn = 0;
  for (const rows of [24, 2]) {
    const tools = swing(q0, 0.15, 0, rows, 0.1).map((q) => metal.tool(q));
    const xyz = tools.map((t) => [...t.position]);
    const rot6 = tools.map((t) => matToRot6(t.rotation));
    const res = solveChunk(metal, xyz, rot6, q0, { hz: 20, serve: { solve, smoothWindow: 0 } });
    res.q.forEach((row, k) => {
      const reached = metal.tool(row);
      position = Math.max(position, Math.hypot(...reached.position.map((v, i) => v - xyz[k]![i]!)));
      turn = Math.max(turn, angle(reached.rotation, tools[k]!.rotation));
    });
  }
  return { position, angle: turn };
}

export function measureInvariants() {
  const tick = measureReach("tick");
  const traj = measureReach("traj");
  return { tickPosition: tick.position, tickAngle: tick.angle, trajPosition: traj.position, trajAngle: traj.angle };
}

/** The largest difference between the numbers of two JSON values of one shape; Infinity when the shapes differ. */
export function numberDiff(a: unknown, b: unknown): number {
  if (typeof a === "number" && typeof b === "number") return Math.abs(a - b);
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length ? max([0, ...a.map((v, i) => numberDiff(v, b[i]))]) : Infinity;
  if (a !== null && b !== null && typeof a === "object" && typeof b === "object") {
    const keys = Object.keys(a).sort();
    if (JSON.stringify(keys) !== JSON.stringify(Object.keys(b).sort())) return Infinity;
    return max([0, ...keys.map((k) => numberDiff((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]))]);
  }
  return a === b ? 0 : Infinity;
}

const CONFIG_KEYS: Record<string, keyof PolicyConfig> = {
  checkpoint: "checkpoint",
  embodiment: "embodiment",
  strict: "strict",
  timeout_ms: "timeoutMs",
  obs_window: "obsWindow",
  obs_fps: "obsFps",
  execution_horizon: "executionHorizon",
  action_hz: "actionHz",
  sync_above_ms: "syncAboveMs",
  robot: "robot",
  ik: "ik",
};

/**
 * wire.json replayed in process (test/wire.test.ts replays it over HTTP and checks the headers and errors too): the
 * largest difference of a request body sent to Python's, of a returned action to Python's where no client IK runs,
 * and of a returned arm's joints to Python's (dream_ik's float32 solve) where it does, with the median over chunks.
 */
export async function measureWire() {
  const wire = fixture("wire.json");
  let body = 0;
  let actions = 0;
  const ikChunks: number[] = [];
  for (const scenario of wire.scenarios) {
    const clock = { ms: 0 };
    let advance = 0;
    let queue: Json[] = [];
    const fetch = async (url: string, init: RequestInit): Promise<Response> => {
      if (url.endsWith("/v1/models")) return new Response(JSON.stringify(scenario.models));
      const expected = queue.shift();
      if (!expected) throw new Error(`${scenario.name}: a request Python did not send: ${url}`);
      body = Math.max(body, numberDiff(JSON.parse(String(init.body)), expected.body));
      if (expected.path === "/v1/get_action") clock.ms += advance;
      return new Response(expected.reply.body, { status: expected.reply.status });
    };
    const client = new DreamClient({ apiKey: "dk_test_key", baseUrl: "http://wire.test", platformUrl: "http://wire.test", fetch, now: () => clock.ms, onWarning: () => {} });
    let ik = false;
    for (const step of scenario.steps) {
      queue = step.requests.filter((r: Json) => r.path !== "/v1/models");
      let value: unknown;
      try {
        if (step.op === "policy") {
          ik = "ik" in step.config;
          const config = Object.fromEntries(Object.entries(step.config).map(([key, v]) => [CONFIG_KEYS[key] ?? key, v])) as unknown as PolicyConfig;
          if (config.ik) {
            config.ikLeadMs = 0;
            const record = scenario.models?.find((m: { id: string }) => m.id === config.checkpoint);
            if (config.ik.head === "joint" && record) config.ik = { ...config.ik, embodimentId: record.embodiment };
          }
          await client.policy(config);
        } else if (step.op === "set_observation") client.setObservation(step.frames, step.observation, { timestamp: step.timestamp });
        else if (step.op === "get_action") {
          clock.ms = step.sent_s * 1000;
          advance = step.rtt_s * 1000;
          value = await client.getAction({ commanded: step.commanded });
        } else if (step.op === "reset_episode") await client.resetEpisode();
        else await client.close();
      } catch (error) {
        if (!step.error) throw error;
        continue;
      }
      if (step.op !== "get_action" || step.error) continue;
      const got = value as Record<string, unknown>;
      const want = step.result.action as Record<string, unknown>;
      for (const key of Object.keys(want)) {
        const diff = numberDiff(got[key], want[key]);
        if (ik && (key === "q_r_chunk" || key === "q_l_chunk")) ikChunks.push(diff);
        else actions = Math.max(actions, diff);
      }
    }
    await client.close();
  }
  const sorted = [...ikChunks].sort((a, b) => a - b);
  return { scenarios: wire.scenarios.length as number, body, actions, ikJoints: max(ikChunks), ikJointsMedian: sorted[Math.floor(sorted.length / 2)]!, ikChunks: ikChunks.length };
}

/** What `npm run fixtures:check` last recorded (a Python-only check: jaxls with vs without its sort patch). */
export function recordedUnpatched() {
  const manifest = fixture("manifest.json");
  return manifest.gen_fixtures.files["test/fixtures/ik.json"].checks.unpatched_jaxls as { bound: Record<string, number>; max: { dq_rad: number; tool_m: number; tool_m_redundant_capped: number } };
}

export async function measureFidelity() {
  return { fk: measureFk(), ik: measureIk(), limitStop: measureLimitStop(), invariants: measureInvariants(), wire: await measureWire(), unpatched: recordedUnpatched() };
}

export type Fidelity = Awaited<ReturnType<typeof measureFidelity>>;

/** A value as the README writes it: three significant figures; lengths in mm or µm, small numbers with an exponent. */
export function show(value: number, unit: "m" | "rad" | ""): string {
  if (unit === "m" && value >= 1e-4) return `${(value * 1000).toPrecision(3)} mm`;
  if (unit === "m" && value >= 1e-7) return `${(value * 1e6).toPrecision(3)} µm`;
  const number = value >= 1e-2 ? value.toPrecision(3) : value.toExponential(2).replace(/e([+-])0*(\d)/, "e$1$2").replace("e+", "e");
  return unit ? `${number}${unit === "m" ? " m" : " rad"}` : number;
}

/** One line of the table: what, under which conditions, the measured value and its ceiling (null: none, for context). */
export interface Row {
  quantity: string;
  conditions: string;
  value: number;
  ceiling: number | null;
  unit: "m" | "rad" | "";
}

/** The rows `npm run fidelity` prints. */
export function rows(f: Fidelity): Row[] {
  const c = CEILINGS;
  const row = (quantity: string, conditions: string, value: number, ceiling: number | null, unit: Row["unit"]): Row => ({ quantity, conditions, value, ceiling, unit });
  const out: Row[] = [
    row("FK position vs yourdfpy (float64)", `${f.fk.cases} arm-sides`, f.fk.positionVsUrdf, c.fk.positionVsUrdf, "m"),
    row("FK rotation entries vs yourdfpy", `${f.fk.cases} arm-sides`, f.fk.rotationVsUrdf, c.fk.rotationVsUrdf, ""),
    row("FK position vs dream_ik (float32)", `${f.fk.cases} arm-sides`, f.fk.positionVsDreamIk, c.fk.positionVsDreamIk, "m"),
    row("Jacobian vs finite differences", "metal, bracketbot_faas, ai_worker_faas", f.fk.jacobianVsFiniteDifferences, c.fk.jacobianVsFiniteDifferences, ""),
    row("IK tool path vs dream_ik", "tick, every case", f.ik.toolTick, c.ik.toolTick, "m"),
    row("IK joints vs dream_ik", "tick, every case", f.ik.jointsTick, c.ik.jointsTick, "rad"),
    row("IK tool path vs dream_ik", `traj, all but ${NULL_SPACE_ARM} capped`, f.ik.toolTraj, c.ik.toolTraj, "m"),
    row("IK joints vs dream_ik", `traj, all but ${NULL_SPACE_ARM}`, f.ik.jointsTraj, c.ik.jointsTraj, "rad"),
  ];
  for (const [name, m] of Object.entries(f.ik.sevenJointTraj)) {
    out.push(
      row("IK tool path vs dream_ik", `${name} traj, uncapped`, m.uncapped.tool, m.uncapped.toolCeiling, "m"),
      row("IK tool path vs dream_ik", `${name} traj, capped 0.5 rad/s`, m.capped.tool, m.capped.toolCeiling, "m"),
      row("IK joints vs dream_ik", `${name} traj, uncapped`, m.uncapped.joints, m.uncapped.jointsCeiling, "rad"),
      row("IK joints vs dream_ik", `${name} traj, capped 0.5 rad/s`, m.capped.joints, m.capped.jointsCeiling, "rad"),
    );
  }
  out.push(
    row("IK orientation miss, ours less dream_ik's", "every case, row by row", f.ik.orientationMissExcess, c.ik.orientationMissExcess, "rad"),
    row("IK orientation miss, ours less dream_ik's", "five and six joints", f.ik.orientationMissExcessSix, c.ik.orientationMissExcess, "rad"),
    row("IK rotation reached vs dream_ik's", "five and six joints", f.ik.rotationSix, c.ik.rotationSix, "rad"),
    row("IK rotation reached vs dream_ik's", `seven joints, all but ${NULL_SPACE_ARM} capped traj`, f.ik.rotationSeven, c.ik.rotationSeven, "rad"),
    row("IK rotation reached vs dream_ik's", `${NULL_SPACE_ARM} capped traj`, f.ik.rotationNullSpace, c.ik.rotationNullSpace, "rad"),
    row("IK orientation miss of the target", "six joints", f.ik.orientationMissSix, c.ik.orientationMissSix, "rad"),
    row("IK orientation miss of the target", "seven joints", f.ik.orientationMissSeven, c.ik.orientationMissSeven, "rad"),
    row("IK joints at a limit vs dream_ik converged", `so101_faas traj, recorded, ${f.limitStop.sides} arms`, f.limitStop.joints, c.ik.jointsAtLimit, "rad"),
    row("IK joints at a limit, dream_ik's vs converged", "same, dream_ik's own stop", f.limitStop.dream, null, "rad"),
    row("Metal tracking a reachable path", "tick, position", f.invariants.tickPosition, c.invariants.tickPosition, "m"),
    row("Metal tracking a reachable path", "tick, orientation", f.invariants.tickAngle, c.invariants.tickAngle, "rad"),
    row("Metal tracking a reachable path", "traj, position", f.invariants.trajPosition, c.invariants.trajPosition, "m"),
    row("Metal tracking a reachable path", "traj, orientation", f.invariants.trajAngle, c.invariants.trajAngle, "rad"),
    row("Wire: request bodies vs Python's", `${f.wire.scenarios} scenarios, every number`, f.wire.body, c.wire.body, ""),
    row("Wire: returned actions vs Python's", "no client IK", f.wire.actions, c.wire.actions, ""),
    row("Wire: returned joints vs Python's", `client IK, worst of ${f.wire.ikChunks} chunks`, f.wire.ikJoints, c.wire.ikJoints, "rad"),
    row("Wire: returned joints vs Python's", "client IK, median chunk", f.wire.ikJointsMedian, null, "rad"),
    row("fixtures:check (recorded), joints", "jaxls unpatched vs patched", f.unpatched.max.dq_rad, f.unpatched.bound.dq_rad!, "rad"),
    row("fixtures:check (recorded), tool", "same, all but seven-joint capped traj", f.unpatched.max.tool_m, f.unpatched.bound.tool_m!, "m"),
    row("fixtures:check (recorded), tool", "same, seven-joint capped traj", f.unpatched.max.tool_m_redundant_capped, f.unpatched.bound.tool_m_redundant_capped!, "m"),
  );
  return out;
}

/** The rows whose measured value reaches its ceiling. */
export const over = (f: Fidelity): Row[] => rows(f).filter((r) => r.ceiling !== null && !(r.value < r.ceiling));

/** The table as Markdown, as the README's Fidelity section holds it. */
export function table(f: Fidelity): string {
  const lines = rows(f).map((r) => [r.quantity, r.conditions, show(r.value, r.unit), r.ceiling === null ? "-" : show(r.ceiling, r.unit)]);
  return [["Quantity", "Conditions", "Measured", "Ceiling"], ["---", "---", "---:", "---:"], ...lines].map((cells) => `| ${cells.join(" | ")} |`).join("\n");
}
