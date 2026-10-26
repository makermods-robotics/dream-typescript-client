"""Writes src/ik/arms.data.ts: dream_ik's arm specs and URDF chains for every arm the client solves IK for.

Run with the Python environment that has `dreamlabs` installed (npm run arms). The arms are dream.client_ik.ARMS
less "yam", whose camera-pole obstacle and two-arm cost this package does not implement. Each side's spec keeps
dream_ik's own numbers: the URDF chain from its root link to the flange (each joint's origin, axis, type and limits),
the tool frame on the flange, the rest pose and weights, the speed caps, the limit margin, the held joints, the
gripper drive and the joint units. It also writes `ArmName`, the arms' names in order, and its section of
test/fixtures/manifest.json (see scripts/manifest.py).
"""

import importlib.metadata
import json
import math
import pathlib
import sys
import xml.etree.ElementTree as ET

from dream import client_ik
from dream_ik import specs

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import manifest  # noqa: E402

OUT = manifest.ROOT / "src" / "ik" / "arms.data.ts"
SKIPPED = {"yam": "obstacles (the camera pole) and the two-arm cost are not implemented"}
URDFS: dict = {}  # file name to sha256, of every URDF read


def floats(values):
    return [float(v) for v in values]


def chain(spec: specs.ArmSpec) -> dict:
    URDFS[spec.urdf.name] = manifest.sha256(spec.urdf)
    root = ET.parse(spec.urdf).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    parent_of = {j.find("child").get("link"): j for j in root.findall("joint")}
    links = {link.get("name") for link in root.findall("link")}
    roots = sorted(links - set(parent_of))
    assert len(roots) == 1, (spec.urdf, roots)
    path = []
    link = spec.flange
    while link != roots[0]:
        joint = parent_of[link]
        path.append(joint)
        link = joint.find("parent").get("link")
    path.reverse()
    out = []
    for joint in path:
        kind = joint.get("type")
        name = joint.get("name")
        if name in spec.fixed_joints:
            kind = "fixed"
        assert kind in ("fixed", "revolute", "prismatic"), (spec.urdf, name, kind)
        origin = joint.find("origin")
        xyz = floats((origin.get("xyz") or "0 0 0").split()) if origin is not None else [0.0, 0.0, 0.0]
        rpy = floats((origin.get("rpy") or "0 0 0").split()) if origin is not None else [0.0, 0.0, 0.0]
        entry = {"name": name, "type": kind, "xyz": xyz, "rpy": rpy}
        if kind != "fixed":
            axis = floats(joint.find("axis").get("xyz").split())
            assert abs(math.hypot(*axis) - 1.0) < 1e-9, (spec.urdf, name, axis)
            entry["axis"] = axis
        out.append(entry)
    on_path = [e["name"] for e in out if e["type"] != "fixed"]
    for name in on_path:
        assert name in spec.joints, (spec.urdf, name, spec.joints)
    limits = []
    for name in spec.joints:
        limit = joints[name].find("limit")
        limits.append([float(limit.get("lower")), float(limit.get("upper"))])
    return {"root": roots[0], "chain": out, "limits": limits}


def arm(spec: specs.ArmSpec) -> dict:
    tool = spec.tools["tcp"]
    return {
        "urdf": spec.urdf.name,
        "joints": list(spec.joints),
        "heldJoints": list(spec.held_joints),
        "flange": spec.flange,
        **chain(spec),
        "tool": {"offset": floats(tool.offset), "rotation": [floats(row) for row in tool.rotation]},
        "restPose": floats(spec.rest_pose),
        "restWeight": floats(spec.rest_weight),
        "maxSpeed": floats(spec.max_speed),
        "limitMargin": float(spec.limit_margin),
        "jaw": None if spec.jaw is None else floats(spec.jaw),
        "jointUnits": None if spec.joint_units is None else [floats(u) for u in spec.joint_units],
        "weights": {k: float(v) for k, v in spec.weights.model_dump().items()},
    }


def render(emb: dict) -> str:
    """One embodiment as TypeScript: one line per key of each arm, one line per chain joint."""
    def spec(value: dict, indent: str) -> list[str]:
        lines = ["{"]
        for key, item in value.items():
            if key == "chain":
                lines.append(f"{indent}  chain: [")
                lines += [f"{indent}    {json.dumps(joint)}," for joint in item]
                lines.append(f"{indent}  ],")
            else:
                lines.append(f"{indent}  {key}: {json.dumps(item)},")
        lines.append(f"{indent}}}")
        return lines

    head = f'{{ sides: {json.dumps(emb["sides"])}, jawInArm: {json.dumps(emb["jawInArm"])}, arm: '
    if "urdf" in emb["arm"]:
        lines = spec(emb["arm"], "  ")
        return head + "\n".join(lines) + " }"
    lines = ["{"]
    for side, value in emb["arm"].items():
        body = spec(value, "      ")
        lines.append(f"    {side}: " + "\n".join(body) + ",")
    lines.append("  } }")
    return head + "\n".join(lines)


def main() -> None:
    embodiments = {}
    for name in client_ik.ARMS:
        if name in SKIPPED:
            print(f"skipping {name}: {SKIPPED[name]}", file=sys.stderr)
            continue
        emb = specs.EMBODIMENTS[name]
        assert all(not emb.obstacles[s] for s in emb.sides), name
        assert emb.side_base is None, name
        embodiments[name] = {
            "sides": list(emb.sides),
            "jawInArm": emb.jaw_in_arm,
            # One spec for every side, or one per side (mirrored arms, per-side URDFs), as dream_ik's Embodiment.arm.
            "arm": arm(emb.arm) if not isinstance(emb.arm, dict) else {s: arm(emb.arm[s]) for s in emb.sides},
        }
    body = "{\n" + ",\n".join(f"  {json.dumps(name)}: {render(emb)}" for name, emb in embodiments.items()) + ",\n}"
    OUT.write_text(
        "// Generated by scripts/gen_arms.py from dreamlabs "
        f"{specs.__name__.split('.')[0]} specs.py and assets/*.urdf (dreamlabs {importlib.metadata.version('dreamlabs')}). Do not edit.\n"
        'import type { EmbodimentData } from "./arms.js";\n\n'
        f"export type ArmName = {' | '.join(json.dumps(name) for name in embodiments)};\n\n"
        f"export const EMBODIMENT_DATA: Readonly<Record<string, EmbodimentData>> = {body};\n"
    )
    print(f"wrote {OUT} ({len(embodiments)} arms)", file=sys.stderr)
    manifest.update("gen_arms", {str(OUT.relative_to(manifest.ROOT)): {"sha256": manifest.sha256(OUT), "urdfs": URDFS, "environment": manifest.environment()}})


if __name__ == "__main__":
    main()
