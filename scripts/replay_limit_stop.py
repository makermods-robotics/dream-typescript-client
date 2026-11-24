"""Replays test/fixtures/limit_stop.json's solve against the installed dreamlabs, offline (npm run limit-stop).

The fixture is one chunk's traj solve at a joint limit, recorded on hardware: its inputs, dream_ik's own answer (which
stopped early, on float rounding, so it moves from run to run and is kept as recorded) and the same solve run to
convergence in float64 (`converged`, what this package is held to). This script solves the recorded inputs again with
the installed dream_ik, in float64 with jaxls's stop tightened in memory (as the recording did), checks the answer
against `converged`, and writes the fixture back with `converged` replaced, `versions` naming the installed packages
and `replayed` saying how far the new answer moved. Nothing reaches the network or hardware.

    $DREAM_PYTHON -I scripts/replay_limit_stop.py
"""

import json
import os
import pathlib
import sys

os.environ["JAX_ENABLE_X64"] = "1"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

PATH = pathlib.Path(__file__).resolve().parent.parent / "test" / "fixtures" / "limit_stop.json"
# How far the replayed converged answer may be from the recorded one: the same solve to convergence, so only float order.
REPLAY_BOUND_RAD = 1e-6


def main() -> None:
    fixture = json.loads(PATH.read_text())
    try:
        from loguru import logger

        logger.remove()
    except ImportError:
        pass
    from importlib.metadata import version

    import jaxls
    import numpy as np
    from dream_ik.solver import ChunkIK
    from dream_ik.specs import SERVE, ServeSpec
    from jaxls._problem import AnalyzedLeastSquaresProblem as Problem

    tight = fixture["termination"]
    solve_once = Problem.solve
    Problem.solve = lambda self, *a, **k: solve_once(self, *a, **{**k, "termination": jaxls.TerminationConfig(**tight)})
    cases = fixture["cases"]
    solver = ChunkIK(fixture["arm"], SERVE.tool, fixture["hz"])
    assert list(solver.sides) == [c["side"] for c in cases], (solver.sides, [c["side"] for c in cases])
    targets = []
    for case in cases:
        xyz, rot6 = np.asarray(case["xyz"]), np.asarray(case["rot6"])
        c0, c1 = rot6[:, :3], rot6[:, 3:]
        pose = np.tile(np.eye(4), (len(xyz), 1, 1))
        pose[:, :3, 0], pose[:, :3, 1], pose[:, :3, 2], pose[:, :3, 3] = c0, c1, np.cross(c0, c1), xyz
        targets.append(pose)
    speed_cap = np.inf if fixture["speed_cap"] is None else fixture["speed_cap"]
    serve = ServeSpec(smooth_window=0, seam_blend_ticks=0, solve="traj")
    result = solver.solve_traj(
        np.asarray(targets), np.asarray([c["q_from"] for c in cases]), np.asarray([c["q_before"] for c in cases]), speed_cap, serve
    )
    moved = 0.0
    for i, case in enumerate(cases):
        converged = np.asarray(result.q[i])[:, case["free"]]
        moved = max(moved, float(np.max(np.abs(converged - np.asarray(case["converged"])))))
        case["converged"] = converged.tolist()
        case["dream_vs_converged"] = float(np.max(np.abs(np.asarray(case["dream"]) - converged)))
    print(f"replayed converged answer moved {moved:.3g} rad", file=sys.stderr)
    if moved > REPLAY_BOUND_RAD:
        sys.exit(f"the replayed solve moved {moved:.3g} rad (> {REPLAY_BOUND_RAD}): dream_ik changed; not written")
    recorded = fixture.get("recorded_versions", fixture["versions"])
    fixture["recorded_versions"] = recorded
    fixture["versions"] = {"dreamlabs": version("dreamlabs"), "dreamlabs-client": version("dreamlabs-client"), "jax": version("jax"), "jax_x64": True}
    fixture["replayed"] = {"converged_moved_rad": moved, "bound_rad": REPLAY_BOUND_RAD, "dream": "as recorded with recorded_versions: dream_ik's own early stop turns on float rounding"}
    PATH.write_text(json.dumps(fixture))
    print(f"wrote {PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
