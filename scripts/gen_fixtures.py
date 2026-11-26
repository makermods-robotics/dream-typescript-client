"""Golden fixtures for the TypeScript tests, from the upstream Python packages (npm run fixtures).

Run with the Python environment that has `dreamlabs` 1.0.42 and `dreamlabs-client` 1.1.7 installed (`npm run
fixtures` takes it from $DREAM_PYTHON, default ~/metal-arm-sdk/.venv/bin/python):

    $DREAM_PYTHON -I scripts/gen_fixtures.py [--only window,pacing,smoothing,fk,ik,client_ik,trained,wire]

Each fixture is the upstream code's own answer to inputs drawn here with fixed seeds. Each gen_X draws from its own
`np.random.Generator`, seeded with [BASE_SEED, crc32("X")], and runs in a process of its own (gen_wire patches dream's
modules and environment), single-threaded and with jaxls's cost order made independent of memory addresses
(#steady_cost_order): dream_ik's float32 answers otherwise change with thread scheduling and with whatever the process
allocated before. So `--only X` writes exactly what a full run writes for X:

- window.json: `dream._window.Window` over jittered 30 Hz ticks, pauses, repeated timestamps and refused ticks.
- pacing.json: `dream.pacing.Pacer` over round-trip sequences.
- smoothing.json: `dream.smoothing` and `dream.seam_join` helpers, and `SeamSmoother` over chunk sequences (joint
  space, and end-effector space with a float64 Metal arm from its URDF).
- fk.json: every built-in arm's tool pose, float64 from its URDF through yourdfpy, checked here against dream_ik's own
  (float32) forward kinematics.
- ik.json: `dream_ik.decode.solve_chunk`, tick and traj, on reachable paths of several arms.
- client_ik.json: `dream.client_ik`'s reach clamp and fence, camera slots and unit conversions (`_Wire`).
- trained.json: `dream.trained_inputs`: difflib's ratio and closest match on drawn prompts, `_plain`, the JPEG frame
  size, notes() on drawn frames and prompts, and the Python reprs, formats and quoting its lines and requests use.
- wire.json: `dream.DreamClient` end to end against a local HTTP server: every request it sent (path, headers, body),
  every answer it returned, each round trip it handed `on_round_trip()` and each line it warned about what a fine-tune
  was trained on, for the TypeScript client to replay against its own mock server.

Each run also records what it wrote in its section of test/fixtures/manifest.json (scripts/manifest.py), per fixture
so that `--only X` misdescribes none of the others: its sha256, seed and the dream_ik URDFs it was made from, and the
environment that wrote it (the upstream wheels' sha256 from $DREAM_WHEELS, the installed versions, the thread and
platform settings in force, since a caller's environment overrides the defaults set below, and whether jaxls's cost
order was patched: by gen_ik and gen_wire). A non-finite float raises rather than being written as null (JSON has no
NaN or infinity): null is Python's None.

The cost-order patch is a test harness choice, not upstream behavior, so

    $DREAM_PYTHON -I scripts/gen_fixtures.py --check-unpatched

generates nothing but re-solves every ik.json case twice, each in a fresh process (#check_unpatched): patched, which
must give ik.json back exactly, and with jaxls as shipped, whose answers must stay within UNPATCHED_BOUNDS of the
fixture's joints and tool path. It records the measured distances and the bounds in ik.json's manifest entry
(checks.unpatched_jaxls). Unpatched, the distances follow the heap, which follows the code that ran before: not the
same from one version of this script to the next, so it is a separate step, and a rerun of gen_ik that writes the
same ik.json keeps the record (a changed ik.json drops it). About five minutes on the Jetson.

Nothing here reaches the network or hardware.
"""

import argparse
import base64
import http.server
import itertools
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import threading
import types
import urllib.parse
import warnings
import zlib

os.environ.setdefault("DREAM_LOG", "off")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# One thread for XLA and BLAS: multi-threaded, the IK solves' float32 answers differ from run to run.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import manifest  # noqa: E402

OUT = manifest.ROOT / "test" / "fixtures"
BASE_SEED = 20261023
# How far jaxls unpatched may move ik.json's answers (--check-unpatched). Reordering float32 residual sums perturbs each
# step by ~1e-7, but the solves are iterative and stop on tolerances, so it lands far above that: over five address
# orders and six shuffled ones, up to 3.3e-3 rad on the joints (bracketbot_faas traj, seven joints: a null space) and
# 7.2e-5 m on the tool (so101_faas capped traj), 4.7e-4 m where a capped traj moves a seven-joint arm along its null
# space (ai_worker_faas). Each bound is ~3x the worst seen. The fixtures are always written patched, so the tests see
# the patched answers exactly; these bounds only show the patch reorders floating point. (How far the TypeScript solve
# is from these answers, and the ceilings ik.test.ts holds it to, is `npm run fidelity`: scripts/fidelity.ts.)
UNPATCHED_BOUNDS = {"dq_rad": 1e-2, "tool_m": 2.5e-4, "tool_m_redundant_capped": 2e-3}
WRITTEN: dict = {}  # the manifest's entry for each fixture written
URDFS: dict = {}  # file name to sha256, of every URDF the running generator loaded


def seed(name: str) -> list[int]:
    return [BASE_SEED, zlib.crc32(name.encode())]


def tolist(value):
    """JSON-ready: numpy arrays and scalars to lists and numbers."""
    if isinstance(value, np.ndarray):
        return tolist(value.tolist())
    if isinstance(value, np.generic):
        return tolist(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{value} in a fixture: JSON has no NaN or infinity")
    if isinstance(value, dict):
        return {str(k): tolist(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tolist(v) for v in value]
    return value


def write(name: str, data) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(tolist(data), separators=(",", ":"), allow_nan=False) + "\n")
    # each generator writes <its name>.json
    WRITTEN[str(path.relative_to(manifest.ROOT))] = {
        "sha256": manifest.sha256(path), "seed": seed(path.stem), "urdfs": dict(URDFS), "environment": manifest.environment(),
    }
    print(f"wrote {path} ({path.stat().st_size // 1024} KiB)", file=sys.stderr)


def steady_cost_order() -> None:
    """jaxls orders a problem's cost groups by str(treedef), which holds memory addresses: the residuals' order, and so
    dream_ik's float32 answers, change with the heap from process to process. Sorted here with the addresses left out."""
    from jaxls import _problem

    def steady_sorted(items, key=None, **kwargs):
        if getattr(key, "__name__", "") == "_sort_key":
            key = lambda x: re.sub(r"0x[0-9a-f]+", "", str(x))  # noqa: E731
        return sorted(items, key=key, **kwargs)

    _problem.sorted = steady_sorted


def frame(i: int) -> str:
    return base64.b64encode(f"jpeg-{i}".encode()).decode()


def jpeg_frame(i: int, width: int, height: int) -> str:
    """A frame that is only a JPEG's start of frame (what `dream.input_check._jpeg_size` reads), then a tag."""
    sof = b"\xff\xc0\x00\x11\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big") + b"\x03" + bytes(9)
    return base64.b64encode(b"\xff\xd8" + sof + b"\xff\xd9" + f"jpeg-{i}".encode()).decode()


# ---------------------------------------------------------------- window


def gen_window(rng: np.random.Generator) -> None:
    from dream import _window
    from dream.errors import DreamObservationError

    cases = []

    def run(name, size, period_s, ticks):
        window = _window.Window(size, period_s)
        steps = []
        for tick in ticks:
            if tick == "clear":
                window.clear()
                steps.append({"op": "clear"})
                continue
            frames, observation, ts = tick
            # the one non-finite input: null in the fixture, NaN again in the test
            step = {"op": "add", "frames": frames, "observation": observation, "timestamp": ts if math.isfinite(ts) else None}
            try:
                step["kept"] = window.add(frames, observation, ts)
            except DreamObservationError as error:
                step["error"] = str(error)
            try:
                step["built"] = window.build_observation()
            except DreamObservationError as error:
                step["built_error"] = str(error)
            steps.append(step)
        cases.append({"name": name, "size": size, "period_s": period_s, "steps": steps})

    keys = ("video.overhead", "video.wrist_l")

    def ticks_at(times, start=0):
        return [({k: frame(start + i) for k in keys}, {"state.right_arm": [i * 0.01] * 6, "prompt": "fold"}, t) for i, t in enumerate(times)]

    jitter = list(np.cumsum(1 / 30 + rng.normal(0, 0.002, 60)))
    run("30 Hz with jitter at 5 fps", 9, 0.2, ticks_at(jitter))
    paused = list(np.arange(0, 1, 1 / 30)) + list(np.arange(2.37, 3.4, 1 / 30))
    run("a pause restarts the schedule", 9, 0.2, ticks_at(paused))
    repeated = [0.0, 0.0, 0.033, 0.033, 0.066, 0.1, 0.1, 0.1, 0.133, 0.166, 0.2, 0.2, 0.233, 0.4, 0.41]
    run("repeated timestamps", 3, 0.2, ticks_at(repeated))
    fast = list(np.arange(0, 2, 1 / 120))
    run("120 Hz at 10 fps, window 4", 4, 0.1, ticks_at(fast))
    bad = ticks_at([0.0, 0.05, 0.3])
    bad.append(({keys[0]: frame(9)}, {}, 0.5))  # a frame key missing
    bad.append(({k: frame(10) for k in keys}, {}, 0.1))  # back in time
    bad.append(({k: frame(11) for k in keys}, {"video.overhead": 1}, 0.6))  # an observation key that is a frame key
    bad.append(({k: "" for k in keys}, {}, 0.7))  # an empty frame
    bad.append(({}, {}, 0.8))  # no frames
    bad.append(({k: frame(12) for k in keys}, {}, float("nan")))  # not finite
    bad.append("clear")
    bad.append(({"video.other": frame(13)}, {"x": 1}, 0.1))  # a new episode, new keys
    run("refused ticks and a clear", 9, 0.2, bad)
    write("window.json", {"cases": cases})


# ---------------------------------------------------------------- pacing


def gen_pacing(rng: np.random.Generator) -> None:
    from dream import pacing

    cases = []
    for name, (horizon, hz, sync_ms), rtts in (
        ("fast link", (30, 20.0, 700.0), [0.12, 0.15, 0.11, 0.2, 0.13, 0.12]),
        ("slows down and recovers", (30, 20.0, 700.0), [0.3, 0.8, 0.9, 0.95, 0.4, 0.3, 0.2, 0.25]),
        ("30 Hz, horizon 24", (24, 30.0, 500.0), [0.45, 0.55, 0.6, 0.62, 0.1, 0.1, 0.1]),
        ("one slow reply", (30, 20.0, 700.0), [0.2, 1.5, 0.2, 0.21, 0.19]),
    ):
        pacer = pacing.Pacer(horizon, hz, sync_ms)
        steps = []
        queries = [(0, 48), (10, 38), (20, 28), (24, 24), (29, 19), (30, 18), (40, 8), (47, 1), (48, 0), (5, 2)]
        steps.append({"mode": pacer.mode, "due": [pacer.refill_due(e, r) for e, r in queries]})
        for rtt in rtts:
            pacer.record(rtt)
            steps.append({"record": rtt, "mode": pacer.mode, "due": [pacer.refill_due(e, r) for e, r in queries]})
        cases.append({"name": name, "execution_horizon": horizon, "action_hz": hz, "sync_above_ms": sync_ms, "queries": queries, "steps": steps})
    write("pacing.json", {"cases": cases})


# ---------------------------------------------------------------- kinematics (float64, from the URDF)


class UrdfArm:
    """One dream_ik arm's tool pose in float64 through yourdfpy: the reference the TypeScript FK is held to."""

    def __init__(self, embodiment: str, side: str):
        import yourdfpy
        from dream_ik import specs

        self.spec = specs.EMBODIMENTS[embodiment].side_arm(side)
        self.urdf = yourdfpy.URDF.load(str(self.spec.urdf), load_meshes=False, build_scene_graph=True)
        URDFS[self.spec.urdf.name] = manifest.sha256(self.spec.urdf)
        tool = self.spec.tools["tcp"]
        self.tool = np.eye(4)
        self.tool[:3, :3] = np.array(tool.rotation)
        self.tool[:3, 3] = tool.offset
        self.root = self.urdf.base_link
        held = set(self.spec.held_joints)
        self.free = [i for i, name in enumerate(self.spec.joints) if name not in held]

    def pose(self, q):
        cfg = {name: float(v) for name, v in zip(self.spec.joints, q, strict=True)}
        self.urdf.update_cfg(cfg)
        return self.urdf.get_transform(frame_to=self.spec.flange, frame_from=self.root) @ self.tool

    def limits(self):
        return np.array([[self.urdf.joint_map[j].limit.lower, self.urdf.joint_map[j].limit.upper] for j in self.spec.joints])


class Float64Kinematics:
    """`dream.seam_join.Kinematics` over an arm's free joints: float64 FK and a central-difference Jacobian."""

    def __init__(self, arm: UrdfArm):
        self.arm = arm

    def _full(self, joints):
        q = np.zeros(len(self.arm.spec.joints))
        q[self.arm.free] = joints
        return q

    def pose(self, joints):
        T = self.arm.pose(self._full(joints))
        return T[:3, 3].copy(), T[:3, :3].copy()

    def jacobian(self, joints):
        from dream import seam_join

        joints = np.asarray(joints, dtype=np.float64)
        p0, r0 = self.pose(joints)
        out = np.zeros((6, len(joints)))
        h = 1e-7
        for j in range(len(joints)):
            up, down = joints.copy(), joints.copy()
            up[j] += h
            down[j] -= h
            pu, ru = self.pose(up)
            pd, rd = self.pose(down)
            out[:3, j] = (pu - pd) / (2 * h)
            out[3:, j] = (seam_join.rotvec(ru @ r0.T) - seam_join.rotvec(rd @ r0.T)) / (2 * h)
        return out


def smooth_path(rng: np.random.Generator, arm: UrdfArm, rows: int, scale: float = 0.25, seed_offset=None):
    """A smooth joint path inside the arm's margined limits around its rest pose: what a model's hand path follows."""
    spec = arm.spec
    lim = arm.limits()
    lo, hi = lim[:, 0] + spec.limit_margin + 0.05, lim[:, 1] - spec.limit_margin - 0.05
    rest = np.clip(np.asarray(spec.rest_pose, dtype=np.float64), lo, hi)
    t = np.arange(rows)[:, None] / rows
    phase = rng.uniform(0, 2 * np.pi, len(spec.joints))
    amp = rng.uniform(0.3, 1.0, len(spec.joints)) * scale * np.minimum(1.0, (hi - lo) / 2)
    q = rest + amp * np.sin(2 * np.pi * 0.6 * t + phase) - amp * np.sin(phase)
    if seed_offset is not None:
        q = q + seed_offset
    q = np.clip(q, lo, hi)
    held = [i for i, name in enumerate(spec.joints) if name in spec.held_joints]
    q[:, held] = 0.0
    return q


def eef_rows(arm: UrdfArm, q):
    poses = np.stack([arm.pose(row) for row in q])
    xyz = poses[:, :3, 3]
    rot6 = np.concatenate([poses[:, :3, 0], poses[:, :3, 1]], axis=-1)
    return poses, xyz, rot6


# ---------------------------------------------------------------- smoothing


def gen_smoothing(rng: np.random.Generator) -> None:
    from dream import seam_join, smoothing

    out: dict = {}
    out["savgol"] = []
    for count, cols, window, order in ((20, 3, 11, 3), (11, 2, 11, 3), (31, 4, 15, 3), (9, 1, 5, 2), (8, 2, 11, 3)):
        rows = rng.normal(size=(count, cols))
        out["savgol"].append({"rows": rows, "window": window, "polyorder": order, "expected": smoothing.savgol_filter(rows, window, order)})
    out["filter_from"] = []
    chunk = np.cumsum(rng.normal(0, 0.02, size=(48, 7)), axis=0)
    for clock in (0, 3, 10, 30, 42, 44, 47):
        out["filter_from"].append({"chunk": chunk, "clock": clock, "expected": smoothing.filter_from(chunk, clock)})
    out["fit_motion"] = []
    for times in ((-1.0, 0.0, 1.0, 2.0), (-1.0, 0.0), (0.0,), (-1.0, 0.0, 1.0)):
        rows = rng.normal(size=(len(times), 6))
        now, velocity = seam_join.fit_motion(rows, np.array(times))
        out["fit_motion"].append({"rows": rows, "times": times, "now": now, "velocity": velocity})
    out["velocity_blend"] = []
    for n in (1, 5, 16):
        plan = np.cumsum(rng.normal(0, 0.05, size=(n + 1, 6)), axis=0)
        now, velocity = rng.normal(size=6) * 0.1, rng.normal(size=6) * 0.02
        out["velocity_blend"].append({"now": now, "velocity": velocity, "plan": plan, "expected": seam_join.velocity_blend(now, velocity, plan)})
    out["rotations"] = []
    for _ in range(12):
        v = rng.normal(size=3) * rng.uniform(0, 3.1)
        R = seam_join.rotation_matrix(v)
        out["rotations"].append({"vector": v, "matrix": R, "rotvec": seam_join.rotvec(R)})
    for v in (np.zeros(3), np.array([np.pi, 0, 0]), np.array([0, 0, -np.pi + 1e-9])):
        R = seam_join.rotation_matrix(v)
        out["rotations"].append({"vector": v, "matrix": R, "rotvec": seam_join.rotvec(R)})
    out["nearest_row"] = []
    plan = np.cumsum(rng.normal(0, 0.03, size=(48, 6)), axis=0)
    held = np.concatenate([plan[:40], np.repeat(plan[40:41], 8, axis=0)])
    for pose, chunk in ((plan[17] + 0.001, plan), (plan[0] + 0.7, plan), (held[44], held), (plan[3], np.repeat(plan[3:4], 48, axis=0))):
        out["nearest_row"].append({"chunk": chunk, "pose": pose, "expected": smoothing.nearest_row(chunk, pose, smoothing.MAX_OFFSET_RAD)})
    out["anchor_row"] = []
    for pose, commanded in ((plan[10], None), (plan[10] + 0.01, plan[12]), (plan[10], plan[47]), (plan[10], plan[10] + 0.6)):
        out["anchor_row"].append({"previous": plan, "pose": pose, "commanded": commanded, "expected": smoothing.anchor_row(plan, pose, commanded)})
    out["old_motion"] = []
    for anchor, rows_in_flight in ((5, 7.4), (5, 7.5), (5, 8.5), (40, 9.0), (46, 0.2), (0, 0.0)):
        clock, rows, times = smoothing.old_motion(plan, anchor, rows_in_flight)
        out["old_motion"].append({"previous": plan, "anchor": anchor, "rows_in_flight": rows_in_flight, "clock": clock, "rows": rows, "times": times})

    def sequences(kinematics):
        """SeamSmoother over chunks: the first, joined ones, a held arm, an arm off its plan, a new shape, commanded."""
        smoother = smoothing.SeamSmoother(20.0, kinematics, smoothing.ARM_POSE_KEYS)
        arm = UrdfArm("metal", "right")
        steps = []
        base = smooth_path(rng, arm, 48, 0.3)
        previous = None
        for i, (rtt, offset, commanded_row, rows) in enumerate(
            ((0.3, 0.0, None, 48), (0.45, 0.002, None, 48), (0.4, 0.0, 9, 48), (2.6, 0.0, None, 48), (0.3, 0.9, None, 48), (0.3, 0.0, None, 32), (0.3, 0.0, 6, 32))
        ):
            q = smooth_path(rng, arm, rows, 0.3) + rng.normal(0, 0.004, size=(rows, 6))
            q = q - q[0] + (base[min(8 + i, 47)] if previous is None else previous[min(int(round(rtt * 20)), len(previous) - 1), :6])
            grip = np.linspace(0.2, 0.8, rows)[:, None]
            chunk_r = np.column_stack([q, grip])
            chunk_l = np.column_stack([q[::-1].copy(), grip])
            measured = (base[3] if previous is None else previous[2, :6]) + offset
            commanded = None if commanded_row is None or previous is None else {"q_r_chunk": previous[commanded_row]}
            observation = {"state.arm_r.joint_pos": measured, "state.arm_l.joint_pos": (chunk_l[0, :6] if previous is None else previous_l[2, :6])}
            action = {"q_r_chunk": chunk_r, "q_l_chunk": chunk_l, "other": "kept"}
            takeover = smoother.takeover_history("q_r_chunk", chunk_r.shape, measured, rtt, None if commanded is None else commanded["q_r_chunk"])
            result = smoother.smooth(observation, action, 0.0, rtt, commanded)
            steps.append({
                "observation": observation, "action": action, "rtt_s": rtt, "commanded": commanded,
                "takeover": None if takeover is None else {"clock": takeover[0], "joints": takeover[1], "before": [takeover[2], takeover[3]]},
                "expected": result,
            })
            previous = np.asarray(result["q_r_chunk"])
            previous_l = np.asarray(result["q_l_chunk"])
        return steps

    out["smoother_joint"] = sequences(None)
    metal = Float64Kinematics(UrdfArm("metal", "right"))
    out["smoother_eef"] = sequences({"q_r_chunk": metal})
    write("smoothing.json", out)


# ---------------------------------------------------------------- forward kinematics


def built_in_arms():
    from dream import client_ik
    from dream_ik import specs

    for name in client_ik.ARMS:
        if name == "yam":
            continue
        for side in specs.EMBODIMENTS[name].sides:
            yield name, side


def gen_fk(rng: np.random.Generator) -> None:
    from dream_ik.solver import ChunkIK

    cases = []
    for name, side in built_in_arms():
        arm = UrdfArm(name, side)
        lim = arm.limits()
        q = rng.uniform(lim[:, 0], lim[:, 1], size=(8, len(arm.spec.joints)))
        q[0] = arm.spec.rest_pose
        held = [i for i, j in enumerate(arm.spec.joints) if j in arm.spec.held_joints]
        q[:, held] = 0.0
        poses = np.stack([arm.pose(row) for row in q])
        ik = ChunkIK(name, "tcp", 20.0)
        dream_fk = ik.fk(q, side)
        assert np.abs(dream_fk[..., :3, 3] - poses[..., :3, 3]).max() < 2e-5, (name, side)
        assert np.abs(dream_fk[..., :3, :3] - poses[..., :3, :3]).max() < 2e-5, (name, side)
        cases.append({"arm": name, "side": side, "q": q, "poses": poses, "dream_ik_fk": dream_fk})
        print(f"fk {name} {side}", file=sys.stderr)
    write("fk.json", {"cases": cases})


# ---------------------------------------------------------------- the chunk solve


def gen_ik(rng: np.random.Generator) -> None:
    from dream_ik import decode, specs
    from dream_ik.solver import ChunkIK

    steady_cost_order()
    cases = []
    plans = (
        ("metal", 20.0, 48, ("tick", "traj")),
        ("makers", 20.0, 48, ("tick", "traj")),
        ("so101_faas", 30.0, 48, ("tick", "traj")),
        ("trossen_wb_faas", 20.0, 32, ("traj",)),
        ("bracketbot_faas", 20.0, 32, ("traj",)),
        ("ai_worker_faas", 20.0, 32, ("traj",)),
        ("seeed_faas", 20.0, 32, ("tick",)),
    )
    for name, hz, rows, solves in plans:
        emb = specs.EMBODIMENTS[name]
        arms = {side: UrdfArm(name, side) for side in emb.sides}
        ik = ChunkIK(name, "tcp", hz)
        for variant in ("exact", "noisy"):
            xyz, rot6, q_from, q_before, paths = [], [], [], [], []
            for side in emb.sides:
                path = smooth_path(rng, arms[side], rows, 0.3)
                _, x, r = eef_rows(arms[side], path)
                if variant == "noisy":
                    x = x + rng.normal(0, 0.002, x.shape)
                    r = r + rng.normal(0, 0.01, r.shape)
                xyz.append(x)
                rot6.append(r)
                start = path[0] + np.where(np.isin(arms[side].spec.joints, arms[side].spec.held_joints), 0.0, rng.normal(0, 0.02, path.shape[1]))
                q_from.append(start)
                q_before.append(np.stack([start - 0.004, start - 0.008]))
                paths.append(path)
            xyz, rot6, q_from, q_before = (np.stack(a) for a in (xyz, rot6, q_from, q_before))
            for solve in solves:
                for speed_cap in (np.inf, 0.5) if variant == "exact" else (np.inf,):
                    serve = specs.ServeSpec(solve=solve)
                    q, res = decode.solve_chunk(ik, xyz, rot6, q_from, speed_cap, serve, q_before=q_before)
                    reached = [np.stack([arms[side].pose(row) for row in q[i]]) for i, side in enumerate(emb.sides)]
                    cases.append({
                        "arm": name, "sides": list(emb.sides), "hz": hz, "solve": solve, "variant": variant,
                        "speed_cap": None if math.isinf(speed_cap) else speed_cap,
                        "xyz": xyz, "rot6": rot6, "q_from": q_from, "q_before": q_before,
                        "expected_q": q, "expected_xyz": [p[:, :3, 3] for p in reached],
                        "pos_miss": res.pos_miss,
                    })
                    print(f"ik {name} {variant} {solve} cap={speed_cap} miss p95 {np.percentile(res.pos_miss, 95) * 1e3:.2f} mm", file=sys.stderr)
    write("ik.json", {"cases": cases})


def resolve_ik(patched: bool) -> dict:
    """Re-solves every ik.json case from its inputs, with jaxls's cost order patched (#steady_cost_order) or as shipped,
    and returns the worst distance from the fixture's answers: joints, and the tool through the float64 URDF."""
    from dream_ik import decode, specs
    from dream_ik.solver import ChunkIK

    if patched:
        steady_cost_order()
    assert manifest.jaxls_sort_patched() == patched
    fixture = json.loads((OUT / "ik.json").read_text())
    solvers: dict = {}
    arms: dict = {}
    worst = dict.fromkeys(UNPATCHED_BOUNDS, 0.0)
    for c in fixture["cases"]:
        if c["arm"] not in solvers:  # one solver per arm, made in gen_ik's order: what was compiled before matters
            arms.update({(c["arm"], side): UrdfArm(c["arm"], side) for side in c["sides"]})
            solvers[c["arm"]] = ChunkIK(c["arm"], "tcp", c["hz"])
        cap = math.inf if c["speed_cap"] is None else c["speed_cap"]
        q, _ = decode.solve_chunk(
            solvers[c["arm"]], np.asarray(c["xyz"]), np.asarray(c["rot6"]), np.asarray(c["q_from"]), cap,
            specs.ServeSpec(solve=c["solve"]), q_before=np.asarray(c["q_before"]),
        )
        dq = float(np.abs(np.asarray(q) - np.asarray(c["expected_q"])).max())
        reached = np.stack([[arms[c["arm"], side].pose(row)[:3, 3] for row in q[i]] for i, side in enumerate(c["sides"])])
        dxyz = float(np.linalg.norm(reached - np.asarray(c["expected_xyz"]), axis=-1).max())
        # ik.test.ts's own split: a seven-joint arm's capped traj moves the hand along its null space
        redundant = len(arms[c["arm"], c["sides"][0]].free) > 6 and c["solve"] == "traj" and c["speed_cap"] is not None
        tool = "tool_m_redundant_capped" if redundant else "tool_m"
        worst["dq_rad"], worst[tool] = max(worst["dq_rad"], dq), max(worst[tool], dxyz)
        print(f"{'patched' if patched else 'unpatched'} {c['arm']} {c['variant']} {c['solve']} cap={cap}: |dq| {dq:.2e} rad, tool {dxyz:.2e} m", file=sys.stderr)
    return {"cases": len(fixture["cases"]), "max": worst}


def check_unpatched() -> None:
    """Shows the cost-order patch changes only floating-point order: re-solved patched, ik.json comes back exactly (so
    the re-solve is gen_ik's), and with jaxls as shipped it stays within UNPATCHED_BOUNDS. Each in a fresh process:
    unpatched, the order (and so the answer) changes with the heap, so the second is a bound, not equality. Recorded
    in ik.json's manifest entry, under checks.unpatched_jaxls."""
    results = {}
    for mode in ("patched", "unpatched"):
        run = subprocess.run([sys.executable, "-I", __file__, "--resolve-ik", mode], check=True, stdout=subprocess.PIPE, text=True)
        results[mode] = json.loads(run.stdout.splitlines()[-1])
    assert not any(results["patched"]["max"].values()), results["patched"]
    settings = {name: os.environ.get(name) for name in manifest.SETTINGS}  # the re-solves' own, as they inherit them
    result = {**results["unpatched"], "bound": UNPATCHED_BOUNDS, "patched_rerun_exact": True, "settings": settings}
    assert all(result["max"][k] < bound for k, bound in UNPATCHED_BOUNDS.items()), result
    manifest.check("gen_fixtures", "test/fixtures/ik.json", "unpatched_jaxls", result)


# ---------------------------------------------------------------- client IK helpers


def gen_client_ik(rng: np.random.Generator) -> None:
    from dream import client_ik
    from dream.errors import DreamConfigError
    from dream_ik import specs

    out: dict = {"clamp": [], "fence": [], "slots": [], "wires": []}
    points = rng.uniform(-0.3, 0.3, size=(6, 2))
    for _ in range(4):
        xy = rng.uniform(-0.5, 0.5, size=(20, 2))
        out["clamp"].append({"xy": xy, "points": points, "expected": client_ik.clamp_to_reach(xy, points)})
    for start in ((0.0, 0.0), (0.6, 0.6)):
        hands = np.asarray(start) + np.cumsum(rng.normal(0, 0.03, size=(30, 2)), axis=0)
        out["fence"].append({"hands": hands, "points": points, "expected": client_ik.reach_fence(hands, points)})
    for roles, model in ((("top", "wrist_l", "wrist_r"), ("overhead", "wrist_l", "wrist_r")), (("front", "wrist_r"), ("front", "wrist_r")), (("a", "b"), ("overhead", "wrist_l", "wrist_r"))):
        try:
            out["slots"].append({"roles": roles, "model": model, "expected": client_ik.camera_slots(roles, model, "ckpt")})
        except DreamConfigError as error:
            out["slots"].append({"roles": roles, "model": model, "error": str(error)})
    for name in ("so101_faas", "trossen_wb_faas", "galaxea_faas", "metal"):
        emb = specs.EMBODIMENTS[name]
        for side in emb.sides:
            spec = emb.side_arm(side)
            wire = client_ik._Wire(spec, emb.jaw_in_arm, None if spec.jaw is None or emb.jaw_in_arm else f"state.{side}_gripper")
            joints = rng.uniform(-1, 1, wire.joints)
            grip = np.array([0.3])
            rows = rng.uniform(-50, 50, size=(4, wire.width))
            entry = {"arm": name, "side": side, "joints": joints, "grip": 0.3, "rows": rows, "own": wire.own, "width": wire.width, "arm_units": wire.arm(joints, grip), "joints_of": wire.joints_of(rows)}
            if spec.jaw is not None:
                entry["drive"] = wire.drive(grip)
            out["wires"].append(entry)
    write("client_ik.json", out)


# ---------------------------------------------------------------- trained inputs


def gen_trained(rng: np.random.Generator) -> None:
    """`dream.trained_inputs`: the prompt comparison (difflib's ratio and closest match, `_plain`), the frame size read
    from a JPEG's header, and notes() on drawn frames and prompts, with the Python reprs and formats its lines use."""
    import difflib

    from dream import trained_inputs
    from dream.input_check import _jpeg_size

    words = ["pick", "up", "the", "red", "cube", "fold", "towels", "stack", "blocks", "Put", "it", "in", "box", "ß", "Ωmega", "naïve", "l'eau"]
    marks = ["", ".", "!", "?", " ", "  ", "\t", ",", ";:", "...", "\u3000"]

    def phrase():
        n = int(rng.integers(1, 6))
        text = " ".join(str(rng.choice(words)) for _ in range(n))
        if rng.random() < 0.3:
            text = text.upper() if rng.random() < 0.5 else text.title()
        return str(rng.choice(marks)) + text.replace(" ", str(rng.choice([" ", "  ", " \n "]))) + str(rng.choice(marks))

    ratios = []
    for _ in range(300):
        a, b = phrase(), phrase()
        if rng.random() < 0.2:
            b = a[: int(rng.integers(0, len(a) + 1))] + phrase()
        ratios.append({"a": a, "b": b, "ratio": difflib.SequenceMatcher(None, a, b).ratio()})
    long_b = "ab" * 150 + "c" * 30  # the popular-element heuristic: b of 200 or more
    for a in ("abc" * 40, "c" * 50 + "ab", "xyz"):
        ratios.append({"a": a, "b": long_b, "ratio": difflib.SequenceMatcher(None, a, long_b).ratio()})
    closest = []
    for _ in range(150):
        tasks = [trained_inputs._plain(phrase()) for _ in range(int(rng.integers(1, 6)))]
        word = trained_inputs._plain(phrase())
        closest.append({"word": word, "possibilities": tasks, "match": difflib.get_close_matches(word, tasks, n=1, cutoff=trained_inputs.CLOSE_PROMPT)})
    plain = [{"text": t, "plain": trained_inputs._plain(t)} for t in [phrase() for _ in range(100)] + ["  Stack!  ", "\u00a0fold\u2003the towel.", "ẞIG Σ", "\x1cx\x1f", "ﬂip the µ-cube", "flip the μ-cube", "ΣΑΣ ﬁ ŉ ǰ ᾳ"]]
    # Ties to the greater string by code point: an astral character against a BMP one above the surrogates.
    for word, tasks in (("abc", ["ab\U0001f600", "ab\uff21"]), ("abc", ["ab\uff21", "ab\U0001f600"]), ("pick it", ["pick ix", "pick iy"])):
        closest.append({"word": word, "possibilities": tasks, "match": difflib.get_close_matches(word, tasks, n=1, cutoff=trained_inputs.CLOSE_PROMPT)})
    folds = [[cp, chr(cp).casefold()] for cp in range(0x110000) if chr(cp).casefold() != chr(cp)]
    sizes = []
    for i in range(40):
        width, height = int(rng.integers(1, 4000)), int(rng.integers(1, 4000))
        good = jpeg_frame(i, width, height)
        sample = good if i % 4 else frame_variant(rng, good)
        sizes.append({"frame": sample, "size": _jpeg_size(sample)})
    vga = jpeg_frame(99, 640, 480)
    for sample in (vga + "==", vga + "====", vga.rstrip("=") + "=", vga + "*", "\u00e9" + vga, vga[:5] + "=" + vga[5:]):
        sizes.append({"frame": sample, "size": _jpeg_size(sample)})
    decodes = []
    for text in ["YQ==", "YQ=", "YQ", "YQ===", "YQ==YWI=", "Y=Q==", "YWJj", "YWJj====", "YW=Jj", "YWJ=", "YWJ", "Y", "YQ=*=", "=YQ==", "YWJjZA", "YWJjZA=x=", "Y W\nJ j", "", "====", "YQ==\u00e9", "\u00e9YQ=="]:
        try:
            decodes.append({"text": text, "bytes": list(base64.b64decode(text, validate=False))})
        except ValueError:
            decodes.append({"text": text, "bytes": None})
    notes = []
    for i in range(80):
        roles = ["front", "wrist_l", "wrist_r"][: int(rng.integers(1, 4))]
        cameras = {role: [[int(rng.integers(160, 1920)), int(rng.integers(120, 1080))] for _ in range(int(rng.integers(1, 3)))] for role in roles if rng.random() < 0.8}
        frames = {}
        for role in roles + (["depth"] if rng.random() < 0.2 else []):
            if rng.random() < 0.5 and role in cameras:
                w, h = cameras[role][0]
                w = max(1, int(w * (1 + rng.uniform(-0.07, 0.07))))
            else:
                w, h = int(rng.integers(100, 1920)), int(rng.integers(100, 1080))
            frames[f"video.{role}"] = [frame(i), jpeg_frame(i, w, h)]
        prompts = [phrase() for _ in range(int(rng.integers(0, 4)))]
        prompt = None if rng.random() < 0.1 else (prompts[0] if prompts and rng.random() < 0.3 else phrase())
        trained = {"cameras": cameras, "prompts": prompts}
        notes.append({"trained": trained, "frames": frames, "prompt": prompt, "notes": trained_inputs.notes(trained, "dream-1-ft:0a1b2c3d-x", frames, prompt)})
    reprs = [{"text": t, "repr": repr(t)} for t in ["it's", 'say "hi"', "both ' and \"", "tab\tnew\nline", "\x00\x7f\x85\xa0", "\u200b\u2028\ue000", "\U0001f600 ok", "back\\slash",
        # Unicode version boundaries: assigned after 15.0 (printed by a newer runtime, escaped by Python 3.12), and an
        # unassigned code point.
        "\U0001fae9 \U0001fa8f \u1c89 \U00031350", "\U000e0001 \U00100000"]]
    formats = [{"x": x, "text": f"{x:.2f}"} for x in [1.125, 1.375, 0.625, 1.7777777, 4 / 3, 16 / 9, 2.005, 1.0, 0.125, 3.875]]
    quotes = [{"text": t, "quoted": urllib.parse.quote(t, safe="")} for t in ["dream-1-ft:0a1b2c3d-towels", "a b/c?d#e", "x!'()*~._-", "naïve"]]
    # A camera role named __proto__ is a role like any other.
    proto = {"cameras": {"__proto__": [[640, 360]]}, "prompts": []}
    proto_frames = {"video.__proto__": [jpeg_frame(3, 640, 480)]}
    notes.append({"trained": proto, "frames": proto_frames, "prompt": None, "notes": trained_inputs.notes(proto, "dream-1-ft:0a1b2c3d-x", proto_frames, None)})
    # float() as dream.arm_file reads a URDF's numbers: Unicode digits and whitespace, underscores, specials.
    alphabet = list("0123456789._eE+-infaINF ") + ["\u0085", "\uff11", "\u0663", "\ufeff", "\u2003", "\x1c", "\u00b2", "\u2160"]
    spellings = ["\uff11\uff12.\uff15", "\u0085 1.5 \u0085", "\ufeff1", "\u0663.\u0661\u0664", "1 2", " ", "", "\u0661_\u0662", "1e5_0", ".5_0", "1_000.000_1", "Infinity", "-inF", "+nan", "1\u00b2", "\u2160", "1e", "e1", "-.e1", "\x1c2\x1f"]
    spellings += ["".join(rng.choice(alphabet, int(rng.integers(1, 9)))) for _ in range(400)]
    floats = []
    for spelling in spellings:
        try:
            value = float(spelling)
            floats.append({"text": spelling, "value": None if value != value else (str(value) if value in (float("inf"), float("-inf")) else value)})
        except ValueError:
            floats.append({"text": spelling, "error": True})
    bad = {"cameras": {"front": [None, [640, "x"], [0, 480], [640, 360, 1]], "wrist_l": "640x480", "wrist_r": [[640, 360]]}, "prompts": ["wave", None, 3]}
    try:
        bad_notes = trained_inputs.notes(bad, "dream-1-ft:0a1b2c3d-x", {"video.front": [jpeg_frame(1, 640, 480)], "video.wrist_r": [jpeg_frame(2, 640, 480)]}, "wave")
    except Exception as error:  # Python fails on it; the port leaves the bad sizes out
        bad_notes = f"{type(error).__name__}"
    write("trained.json", {"ratios": ratios, "closest": closest, "plain": plain, "folds": folds, "sizes": sizes, "decodes": decodes, "notes": notes, "malformed": {"trained": bad, "python": bad_notes}, "floats": floats, "reprs": reprs, "formats": formats, "quotes": quotes})


def frame_variant(rng: np.random.Generator, good: str) -> str:
    """A frame `_jpeg_size` may refuse: cut short, without padding, with characters outside base64, or not a JPEG."""
    kind = int(rng.integers(0, 4))
    if kind == 0:
        return good[: int(rng.integers(0, 16))]
    if kind == 1:
        return good.rstrip("=")
    if kind == 2:
        return good[:8] + "*\n" + good[8:]
    return base64.b64encode(bytes(rng.integers(0, 256, 24, dtype=np.uint8))).decode()


# ---------------------------------------------------------------- the wire, end to end


class Recorder(http.server.BaseHTTPRequestHandler):
    """Answers each request with the next scripted reply and records it: the gateway and the platform in one."""

    server_version = "fixture"
    script: list = []
    log: list = []
    models: list = []
    # The data service's GET /v1/fine_tuning/models/<id>/training_data, per checkpoint: (status, body); 404 not_found
    # for any other checkpoint, as for a model that is not one of the account's fine-tunes.
    training: dict = {}

    def log_message(self, *args):  # quiet
        pass

    def _record(self, body):
        headers = {k: v for k, v in self.headers.items() if k.lower() in ("content-type", "authorization", "dream-client-version", "user-agent", "accept")}
        entry = {"method": self.command, "path": self.path, "headers": headers, "body": body}
        training = re.fullmatch(r"/v1/fine_tuning/models/([^/]+)/training_data", self.path)
        if self.path == "/v1/models":
            reply = (200, json.dumps(Recorder.models))
        elif training:
            checkpoint = urllib.parse.unquote(training.group(1))
            reply = Recorder.training.get(checkpoint, (404, json.dumps({"error": {"code": "not_found", "message": f"no fine-tune {checkpoint}"}})))
        else:
            reply = Recorder.script.pop(0)
        entry["reply"] = {"status": reply[0], "body": reply[1]}
        Recorder.log.append(entry)
        return reply

    def do_GET(self):
        status, text = self._record(None)
        self._send(status, text)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        status, text = self._record(json.loads(raw))
        self._send(status, text)

    def _send(self, status, text):
        data = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def ok(actions, session="ses_1", code="DA-0000", reason=None, timing=None):
    status = {"code": code, "name": "ok" if code == "DA-0000" else "new_session"}
    if reason is not None:
        status["reason"] = reason
    return (200, json.dumps({"status": status, "session_id": session, "actions": tolist(actions), "timing": timing or {"serve_infer_ms": 180, "gateway_ms": 210}}))


def refusal(code, name, reason=None, retryable=False, retry_after_s=None, detail=None, http=200, session=None):
    status = {"code": code, "name": name, "reason": reason, "retryable": retryable, "retry_after_s": retry_after_s, "detail": detail, "trace_id": "trc_123"}
    body = {"status": status}
    if session:
        body["session_id"] = session
    return (http, json.dumps(body))


class Clock:
    """The scripted `time` the client measures round trips on: perf_counter returns the next scripted reading."""

    def __init__(self):
        self.readings: list[float] = []
        self.monotonic = __import__("time").monotonic
        self.sleep = __import__("time").sleep

    def perf_counter(self):
        return self.readings.pop(0)


def gen_wire(rng: np.random.Generator) -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    os.environ.update({"DREAM_GATEWAY_URL": url, "DREAM_API_URL": url, "DREAM_DATA_URL": url, "DREAM_API_KEY": "dk_test_key", "DREAM_CREDENTIALS": "/nonexistent/credentials"})

    import dream
    from dream import client as client_module
    from dream import client_ik, facade, transport

    steady_cost_order()
    # Two things a policy does on this computer, not on the wire, turned off in memory: the headers a client sends a
    # local check server on 127.0.0.1 (where this recorder listens), and the reminder of the input check, which reads
    # its record from the home directory.
    client_ik.CHECK_HOST = None
    facade._remind = lambda observation: None
    clock = Clock()
    client_module.time = types.SimpleNamespace(perf_counter=clock.perf_counter, monotonic=clock.monotonic, sleep=clock.sleep)
    # Each channel's User-Agent instance is random upstream, and the port is the OS's pick: both fixed here, so a
    # rerun writes the same bytes (the TypeScript test checks the instance's shape, and errors by type, not message).
    instances = itertools.count(1)
    transport.secrets = types.SimpleNamespace(token_hex=lambda n: f"{next(instances):0{2 * n}x}")
    # The client compiles its traj solve on a thread, racing its own fk compiles, and dream_ik's answers depend on
    # what was compiled before: run the warm-up where it starts.
    client_ik.threading = types.SimpleNamespace(Event=threading.Event, Thread=lambda target, name: types.SimpleNamespace(start=target))
    scenarios = []

    def scenario(name, models, body, training=None):
        Recorder.models = models
        Recorder.training = {checkpoint: (200, json.dumps(answer)) if isinstance(answer, dict) else answer for checkpoint, answer in (training or {}).items()}
        Recorder.log = []
        steps = []
        client = dream.DreamClient()
        trips = []
        client.on_round_trip(trips.append)

        def step(op, **fields):
            entry = {"op": op, **fields}
            start = len(Recorder.log)
            trips.clear()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    result = body_op(client, op, fields)
                    if result is not None:
                        entry["result"] = result
                except dream.DreamError as error:
                    entry["error"] = {"type": type(error).__name__, "message": str(error).replace(url, "http://127.0.0.1:PORT")}
                    for key in ("code", "reason", "retryable", "retry_after_s"):
                        if hasattr(error, key):
                            entry["error"][key] = getattr(error, key)
            entry["requests"] = Recorder.log[start:]
            # What policy() and get_action() warned about a fine-tune's training inputs (not the slow-network warning).
            entry["warnings"] = [str(w.message) for w in caught if not issubclass(w.category, dream.DreamSlowNetworkWarning)]
            returned = entry.get("result", {}).get("action") if op == "get_action" else None
            entry["round_trips"] = [
                {
                    "frames": trip.frames,
                    "observation": trip.observation,
                    # The action handed over is the very dict get_action() returned (compared in full in "result").
                    "action": None if trip.action is None else ("returned" if trip.action is returned else "other"),
                    "timing": None if trip.timing is None else {k: trip.timing.get(k) for k in ("serve_infer_ms", "gateway_ms")},
                    "error": None if trip.error is None else type(trip.error).__name__,
                    "trained": trip.trained,
                }
                for trip in trips
            ]
            steps.append(entry)
            return entry

        body(client, step)
        client.close()
        scenarios.append({"name": name, "models": models, "training": Recorder.training, "steps": steps})
        print(f"wire {name}: {len(steps)} steps", file=sys.stderr)

    def body_op(client, op, fields):
        if op == "policy":
            Recorder.script.extend(fields.pop("_replies", []))
            config = dict(fields["config"])
            config["_record"] = False
            client.policy(config)
            ik = getattr(client._policy, "_ik", None)
            if ik is not None:
                ik.traj_ready.wait()
            return None
        if op == "set_observation":
            client.set_observation(fields["frames"], fields["observation"], timestamp=fields["timestamp"])
            return None
        if op == "get_action":
            Recorder.script.extend(fields.pop("_replies"))
            clock.readings[:] = [fields["sent_s"], fields["sent_s"] + fields["rtt_s"]]
            try:
                action = client.get_action(commanded=fields.get("commanded"))
            finally:
                clock.readings.clear()  # a call that failed early read fewer
            return {"action": action, "mode": client.mode, "session_id": client.session_id, "timing": {k: v for k, v in (client.last_timing or {}).items() if k in ("serve_infer_ms", "gateway_ms", "request_bytes", "response_bytes")}}
        if op == "reset_episode":
            Recorder.script.extend(fields.pop("_replies", []))
            client.reset_episode()
            return {"session_id": client.session_id}
        if op == "close":
            Recorder.script.extend(fields.pop("_replies", []))
            client.close()
            return None
        raise AssertionError(op)

    def tick(step, frames, observation, t):
        step("set_observation", frames=frames, observation=observation, timestamp=t)

    yam_spec = {
        "state": {"arm_r.joint_pos": {"shape": [6]}, "arm_r.gripper": {"shape": [1]}, "arm_l.joint_pos": {"shape": [6]}, "arm_l.gripper": {"shape": [1]}},
        "image_window": {"top": {}, "left": {}, "right": {}},
        "action": {"q_r_chunk": {"shape": [48, 7]}, "q_l_chunk": {"shape": [48, 7]}},
    }

    def yam_obs(i):
        q = (0.1 + 0.01 * i) * np.array([1, -1, 1, 0.5, 0.2, -0.3])
        return {"state.arm_r.joint_pos": q, "state.arm_r.gripper": [0.5], "state.arm_l.joint_pos": -q, "state.arm_l.gripper": [0.4], "prompt": "fold the towel"}

    def yam_frames(i):
        return {"image_window.top": frame(i), "image_window.left": frame(100 + i), "image_window.right": frame(200 + i)}

    def yam_chunk(q0, slope):
        t = np.arange(48)[:, None]
        q = q0 + slope * t + 0.002 * np.sin(t)
        return np.column_stack([q, np.full(48, 0.5)])

    def yam_body(client, step):
        step("policy", config={"checkpoint": "dreamlabs/dream-1-beta", "embodiment": yam_spec, "timeout_ms": 5000, "action_hz": 20})
        t = 10.0
        for i in range(8):
            tick(step, yam_frames(i), yam_obs(i), t)
            t += 1 / 30
        q0 = yam_obs(7)["state.arm_r.joint_pos"]
        first = {"q_r_chunk": yam_chunk(q0, 0.004), "q_l_chunk": yam_chunk(-q0, -0.004), "q_r": q0.tolist(), "extra": {"note": 1}}
        step("get_action", sent_s=100.0, rtt_s=0.32, _replies=[ok(first, code="DA-0001", reason="new")])
        for i in range(8, 14):
            tick(step, yam_frames(i), yam_obs(i), t)
            t += 1 / 30
        second = {"q_r_chunk": yam_chunk(q0 + 0.03, 0.003), "q_l_chunk": yam_chunk(-q0 - 0.03, -0.003)}
        step("get_action", sent_s=101.0, rtt_s=0.41, commanded={"q_r_chunk": (q0 + 0.03).tolist()}, _replies=[ok(second)])
        step("get_action", sent_s=102.0, rtt_s=0.2, _replies=[refusal("DA-2000", "unavailable", "pool_warming", True, 3.5, "the model is starting")])
        step("get_action", sent_s=103.0, rtt_s=0.2, _replies=[refusal("DA-0001", "new_session", "idle_timeout", session="ses_2")])
        step("get_action", sent_s=104.0, rtt_s=0.25, _replies=[ok(first, session="ses_2")])
        step("reset_episode", _replies=[(200, json.dumps({"status": {"code": "DA-0000"}}))])
        step("get_action", sent_s=105.0, rtt_s=0.2, _replies=[])  # no observation since the reset
        tick(step, yam_frames(30), yam_obs(30), t + 5)
        step("get_action", sent_s=106.0, rtt_s=0.2, _replies=[refusal("DA-4000", "bad_request", "unknown_checkpoint")])
        step("get_action", sent_s=107.0, rtt_s=0.2, _replies=[refusal("DA-1000", "unauthorized", "unknown_key", http=401)])
        step("get_action", sent_s=108.0, rtt_s=0.2, _replies=[(502, "<html>bad gateway</html>")])
        step("get_action", sent_s=109.0, rtt_s=0.2, _replies=[(200, json.dumps({"actions": {}}))])
        step("get_action", sent_s=110.0, rtt_s=0.2, _replies=[refusal("DA-5000", "serve_failed", None, True, None, "inference failed")])
        bad = {"q_r_chunk": yam_chunk(q0, 0.0)[:40], "q_l_chunk": yam_chunk(q0, 0.0)}
        step("get_action", sent_s=111.0, rtt_s=0.2, _replies=[ok(bad, session="ses_9", code="DA-0001", reason="new")])
        bad_obs = dict(yam_obs(40))
        bad_obs["state.arm_r.joint_pos"] = [0.1, 0.2]
        tick(step, yam_frames(40), bad_obs, t + 6)
        step("get_action", sent_s=112.0, rtt_s=0.2, _replies=[])
        tick(step, yam_frames(41), {**yam_obs(41), "embodiment": "yam"}, t + 7)
        step("get_action", sent_s=113.0, rtt_s=0.2, _replies=[])
        step("close", _replies=[(200, json.dumps({"status": {"code": "DA-0000"}}))])

    scenario("yam station shape spec: sessions, refusals and checks", [], yam_body)

    # A fine-tune of a custom body: the embodiment config comes from GET /v1/models, the parts from per_key.
    body_config = {
        "state": {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]},
        "action": {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]},
        "video": {"front": "observation.images.front", "wrist_r": "observation.images.wrist_r"},
    }
    custom = {"id": "dream-1-ft:0a1b2c3d-towels", "owner": "mine", "base_model": "dreamlabs/dream-1-beta", "embodiment": "emb_0123", "servable": True, "created_at": "2026-10-01T12:00:00Z", "cameras": ["front", "wrist_r"], "embodiment_config": body_config}

    def parts_obs(i):
        q = 0.05 * i * np.ones(6)
        return {"state.right_arm": q, "state.right_gripper": [0.2], "state.left_arm": -q, "state.left_gripper": [0.3], "prompt": "towels"}

    def parts_frames(i):
        # The front camera at 4:3 where the fine-tune was trained at 16:9 (a line); the wrist within its aspect.
        return {"video.front": jpeg_frame(i, 640, 480), "video.wrist_r": jpeg_frame(50 + i, 320, 176)}

    def per_key_reply(q0, rows=48):
        t = np.arange(rows)[:, None]
        arm = q0 + 0.002 * t
        return {
            "per_key": {"action.right_arm": arm, "action.right_gripper": np.full(rows, 0.4), "action.left_arm": -arm, "action.left_gripper": np.full((rows, 1), 0.6), "action.extra": [[1]]},
            "q_r_chunk": np.column_stack([arm, np.full(rows, 0.4)]),
            "q_l_chunk": np.column_stack([-arm, np.full(rows, 0.6)]),
        }

    def custom_body(client, step):
        step("policy", config={"checkpoint": custom["id"], "robot": "other_embodiment"})
        for i in range(3):
            tick(step, parts_frames(i), parts_obs(i), 1.0 + i * 0.2)
        step("get_action", sent_s=50.0, rtt_s=0.3, _replies=[ok(per_key_reply(parts_obs(2)["state.right_arm"]), code="DA-0001", reason="new")])
        tick(step, parts_frames(3), parts_obs(3), 1.6)
        reply = per_key_reply(parts_obs(3)["state.right_arm"])
        reply["per_key"]["action.right_arm"] = None  # the serve answers no joint head on other_embodiment
        step("get_action", sent_s=51.0, rtt_s=0.35, commanded={"q_r_chunk": parts_obs(3)["state.right_arm"].tolist() + [0.4]}, _replies=[ok(reply)])
        missing = per_key_reply(parts_obs(3)["state.right_arm"])
        del missing["per_key"]["action.left_gripper"]
        step("get_action", sent_s=52.0, rtt_s=0.3, _replies=[ok(missing)])
        tick(step, {"video.front": frame(9), "video.wrist_r": frame(9)}, {k: v for k, v in parts_obs(4).items() if k != "state.left_gripper"}, 1.8)
        step("get_action", sent_s=53.0, rtt_s=0.3, _replies=[])
        step("reset_episode", _replies=[(200, json.dumps({"status": {"code": "DA-0000"}}))])
        tick(step, parts_frames(5), {**parts_obs(5), "prompt": "Fold  the towels!"}, 2.0)
        step("get_action", sent_s=54.0, rtt_s=0.3, _replies=[ok(per_key_reply(parts_obs(5)["state.right_arm"]), code="DA-0001", reason="new", session="ses_c2")])
        # A fine-tune the data service does not know (404 not_found: no lines), and one it cannot answer (a line).
        step("policy", config={"checkpoint": "dream-1-ft:0a1b2c3d-unknown", "embodiment": body_config}, _replies=[(200, json.dumps({"status": {"code": "DA-0000"}}))])
        step("policy", config={"checkpoint": "dream-1-ft:0a1b2c3d-flaky", "embodiment": body_config})
        step("policy", config={"checkpoint": "dream-1-ft:missing-model"})
        step("close", _replies=[])

    custom_training = {
        custom["id"]: {"cameras": {"front": [[640, 360], [1280, 720]], "wrist_r": [[640, 360]]}, "prompts": ["Fold the towels.", "stack the blocks"]},
        "dream-1-ft:0a1b2c3d-flaky": (503, json.dumps({"error": {"code": "unavailable", "message": "the data service is restarting"}})),
    }
    scenario("custom body fine-tune: config from the platform, per_key checks, other_embodiment", [custom], custom_body, custom_training)

    # Client IK on a Metal arm, for the YAM-trained base model: zero joints, grip as gripper, cameras renamed.
    metal = UrdfArm("metal", "right")
    beta = {"id": "dreamlabs/dream-1-beta", "owner": "public", "base_model": "dreamlabs/dream-1-beta", "embodiment": "yam", "servable": True, "created_at": None, "cameras": ["overhead", "wrist_l", "wrist_r"]}
    eef_parts = {"arm": 6, "gripper": 1, "eef_xyz": 3, "eef_rot6": 6, "grip": 1}

    def spans(sides, parts):
        out, at = {}, 0
        for side in sides:
            for part, width in parts.items():
                out[f"{side}_{part}"] = [at, at + width]
                at += width
        return out

    metal_body = {"state": spans(("right", "left"), eef_parts), "action": spans(("right", "left"), eef_parts), "video": {"top": "observation.images.top", "wrist_l": "observation.images.wrist_l", "wrist_r": "observation.images.wrist_r"}}
    path_r = smooth_path(rng, metal, 160, 0.35)
    path_l = smooth_path(rng, metal, 160, 0.35)

    def metal_obs(k):
        return {"state.right_arm": path_r[k], "state.right_gripper": [0.5], "state.right_grip": [0.5], "state.left_arm": path_l[k], "state.left_gripper": [0.3], "state.left_grip": [0.3], "prompt": "pick"}

    def metal_frames(i):
        return {"video.top": frame(i), "video.wrist_l": frame(10 + i), "video.wrist_r": frame(20 + i)}

    def eef_reply(k, noise=0.0005, rows=48, joint_head=False, sides=("right", "left"), paths=None, arms=None, drive=None):
        per_key = {}
        for side in sides:
            arm = arms[side]
            q = paths[side][k : k + rows]
            _, xyz, rot6 = eef_rows(arm, q)
            per_key[f"action.{side}_eef_xyz"] = xyz + rng.normal(0, noise, xyz.shape)
            per_key[f"action.{side}_eef_rot6"] = rot6 + rng.normal(0, noise * 5, rot6.shape)
            per_key[f"action.{side}_grip"] = np.linspace(0.3, 0.9, rows)
            per_key[f"action.{side}_arm"] = None if not joint_head else joint_head(side, q)
            if drive:
                per_key[f"action.{side}_gripper"] = np.linspace(30.0, 90.0, rows)[:, None]
        return {"per_key": per_key, "q_r": [0.0] * 7, "q_l": [0.0] * 7}

    def metal_body_fn(client, step):
        step("policy", config={"checkpoint": beta["id"], "embodiment": metal_body, "ik": {"arm": "metal", "head": "eef"}, "action_hz": 20})
        for i in range(3):
            tick(step, metal_frames(i), metal_obs(i), 5.0 + i / 30)
        # The body declares each arm's gripper part, so the reply carries it (in the gripper's own units).
        args = {"paths": {"right": path_r, "left": path_l}, "arms": {"right": metal, "left": metal}, "drive": True}
        step("get_action", sent_s=10.0, rtt_s=0.3, _replies=[ok(eef_reply(2, **args), code="DA-0001", reason="new")])
        tick(step, metal_frames(3), metal_obs(9), 5.4)
        step("get_action", sent_s=11.0, rtt_s=0.4, commanded={"q_r_chunk": path_r[9].tolist(), "q_l_chunk": path_l[9].tolist() + [0.3]}, _replies=[ok(eef_reply(9, **args))])
        tick(step, metal_frames(4), metal_obs(18), 5.8)
        step("get_action", sent_s=12.0, rtt_s=0.45, _replies=[ok(eef_reply(18, **args))])
        tick(step, metal_frames(5), {**metal_obs(19), "state.right_eef_xyz": [0, 0, 0]}, 6.0)
        step("get_action", sent_s=13.0, rtt_s=0.3, _replies=[])
        step("reset_episode", _replies=[(200, json.dumps({"status": {"code": "DA-0000"}}))])
        step("policy", config={"checkpoint": beta["id"], "embodiment": metal_body, "ik": {"arm": "metal", "head": "joint"}})
        step("policy", config={"checkpoint": beta["id"], "ik": {"arm": "metal", "head": "eef", "solve": "tick"}, "strict": False})
        step("policy", config={"checkpoint": beta["id"], "embodiment": metal_body, "ik": {"arm": "metal", "head": "eef", "solve": "tick"}, "action_hz": 30})
        tick(step, metal_frames(6), metal_obs(30), 7.0)
        step("get_action", sent_s=20.0, rtt_s=0.3, _replies=[ok(eef_reply(30, **args), code="DA-0001", reason="new", session="ses_t")])
        tick(step, metal_frames(7), metal_obs(40), 7.4)
        step("get_action", sent_s=21.0, rtt_s=0.36, _replies=[ok(eef_reply(40, **args), session="ses_t")])
        step("close", _replies=[(200, json.dumps({"status": {"code": "DA-0000"}}))])

    scenario("client IK on Metal for the YAM-trained base model", [beta], metal_body_fn)

    # Client IK on an SO-101 pair for a fine-tune of the robot (a FaaS beta robot): units and the gripper drive.
    so_r, so_l = UrdfArm("so101_faas", "right"), UrdfArm("so101_faas", "left")
    so_parts = {"arm": 6, "gripper": 1, "eef_xyz": 3, "eef_rot6": 6, "grip": 1}
    so_body = {"state": spans(("right", "left"), so_parts), "action": spans(("right", "left"), so_parts), "video": {"front": "observation.images.front", "wrist_r": "observation.images.wrist_r", "wrist_l": "observation.images.wrist_l"}}
    so_model = {"id": "dream-1-ft:5o101aaa-blocks", "owner": "mine", "base_model": "dreamlabs/dream-1-beta", "embodiment": "emb_so101", "servable": True, "created_at": "2026-10-10T08:00:00Z", "cameras": ["front", "wrist_l", "wrist_r"], "embodiment_config": so_body}
    so_paths = {"right": smooth_path(rng, so_r, 160, 0.3), "left": smooth_path(rng, so_l, 160, 0.3)}
    from dream_ik import specs as dream_specs

    def so_obs(k):
        return {
            "state.right_arm": so_paths["right"][k][:5], "state.right_grip": [0.6],
            "state.left_arm": so_paths["left"][k][:5], "state.left_grip": [0.2], "prompt": "stack",
        }

    def so_frames(i):
        return {"video.front": frame(i), "video.wrist_r": frame(10 + i), "video.wrist_l": frame(20 + i)}

    def so_joint_head(side, q):
        units = dream_specs.EMBODIMENTS["so101_faas"].side_arm(side).joint_units
        scale, offset = np.asarray(units[0]), np.asarray(units[1])
        return (q - offset) / scale

    def so_body_fn(client, step):
        step("policy", config={"checkpoint": so_model["id"], "ik": {"arm": "so101_faas", "head": "eef"}, "action_hz": 30})
        for i in range(2):
            tick(step, so_frames(i), so_obs(i), 3.0 + i / 30)
        args = {"paths": so_paths, "arms": {"right": so_r, "left": so_l}, "joint_head": so_joint_head, "drive": True}
        step("get_action", sent_s=30.0, rtt_s=0.25, _replies=[ok(eef_reply(1, **args), code="DA-0001", reason="new", session="ses_s")])
        tick(step, so_frames(2), so_obs(8), 3.4)
        step("get_action", sent_s=31.0, rtt_s=0.3, _replies=[ok(eef_reply(8, **args), session="ses_s")])
        tick(step, so_frames(3), {**so_obs(9), "state.right_gripper": [50.0]}, 3.5)
        step("get_action", sent_s=32.0, rtt_s=0.3, _replies=[])
        step("policy", config={"checkpoint": so_model["id"], "ik": {"arm": "so101_faas", "head": "joint"}, "action_hz": 30}, _replies=[(200, json.dumps({"status": {"code": "DA-0000"}}))])
        tick(step, so_frames(4), so_obs(20), 4.0)
        step("get_action", sent_s=40.0, rtt_s=0.3, _replies=[ok(eef_reply(20, **args), code="DA-0001", reason="new", session="ses_j")])
        step("close", _replies=[(200, json.dumps({"status": {"code": "DA-0000"}}))])

    so_training = {so_model["id"]: {"cameras": {"front": [[640, 480]], "wrist_l": [[640, 480]], "wrist_r": [[640, 480]]}, "prompts": ["stack them", "unstack"]}}
    scenario("client IK on SO-101 for a fine-tune of the robot: units and the gripper drive", [so_model], so_body_fn, so_training)
    server.shutdown()
    write("wire.json", {"client_version": dream.__version__, "scenarios": scenarios})


GENERATORS = {"window": gen_window, "pacing": gen_pacing, "smoothing": gen_smoothing, "fk": gen_fk, "ik": gen_ik, "client_ik": gen_client_ik, "trained": gen_trained, "wire": gen_wire}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default=",".join(GENERATORS), help="comma-separated: " + ",".join(GENERATORS))
    parser.add_argument("--check-unpatched", action="store_true", help="bound ik.json against jaxls unpatched, generating nothing")
    parser.add_argument("--resolve-ik", choices=("patched", "unpatched"), help=argparse.SUPPRESS)  # --check-unpatched's own
    args = parser.parse_args()
    if args.resolve_ik:
        print(json.dumps(resolve_ik(args.resolve_ik == "patched")))
        return
    if args.check_unpatched:
        check_unpatched()
        return
    names = args.only.split(",")
    if len(names) > 1:  # each in a fresh process, as `--only` runs it
        for name in names:
            subprocess.run([sys.executable, "-I", __file__, "--only", name], check=True)
        return
    GENERATORS[names[0]](np.random.default_rng(seed(names[0])))
    manifest.update("gen_fixtures", WRITTEN, base_seed=BASE_SEED)


if __name__ == "__main__":
    main()
