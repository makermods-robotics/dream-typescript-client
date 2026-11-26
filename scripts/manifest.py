"""test/fixtures/manifest.json: what the generated files came from, one section per generator script.

Each section holds its `files`, one entry per written file: its sha256, what the generator drew it with (seed, the
URDFs it read), and the `environment` of the process that wrote it (#environment): the upstream packages (the wheel's
sha256 when the wheel is found in $DREAM_WHEELS, default ../dreamlabs-versions/dl, and always the installed RECORD's
sha256), the installed versions, the thread and platform settings in force, and whether jaxls's cost order was patched.
Provenance is per file because `--only X` rewrites one file: a section-wide environment would describe the files the
run kept with the run's own. Only what is truly per section (gen_fixtures's base_seed) sits beside `files`.

An entry may also hold `checks`, results measured on that file's bytes (#check), each with the upstream packages and
versions it was measured with: kept over a rerun that writes the same bytes in the same environment, dropped when the
bytes or that environment change. Keys sorted and no timestamps: a rerun writes the same bytes.
"""

import functools
import hashlib
import importlib.metadata as metadata
import json
import os
import pathlib
import platform
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PATH = ROOT / "test" / "fixtures" / "manifest.json"
WHEELS = pathlib.Path(os.environ.get("DREAM_WHEELS", ROOT.parent / "dreamlabs-versions" / "dl"))
UPSTREAM = ("dreamlabs", "dreamlabs-client")
PACKAGES = (*UPSTREAM, "numpy", "scipy", "jax", "jaxlib", "jaxls", "pyroki", "yourdfpy")
# What decides dream_ik's float32 answers besides the code: XLA's and BLAS's threading and the JAX backend.
SETTINGS = ("JAX_PLATFORMS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XLA_FLAGS")


def sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


@functools.cache
def versions() -> dict:
    out = {"python": platform.python_version()}
    providers = metadata.packages_distributions()
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            if name in providers:  # jaxls and pyroki ship inside the dreamlabs wheel, with no metadata of their own
                out[name] = f"vendored in {providers[name][0]} {metadata.version(providers[name][0])}"
    return out


@functools.cache
def upstream() -> dict:
    out = {}
    for name in UPSTREAM:
        dist = metadata.distribution(name)
        record = next(f for f in dist.files if f.name == "RECORD" and f.parent.name.endswith(".dist-info"))
        entry = {"version": dist.version, "record_sha256": sha256(record.locate())}
        wheels = sorted(WHEELS.glob(f"{name.replace('-', '_')}-{dist.version}-*.whl"))
        if wheels:
            with zipfile.ZipFile(wheels[0]) as wheel:
                shipped = set(wheel.read(str(record)).decode().splitlines()) - {f"{record},,"}
            missing = shipped - set(dist.read_text("RECORD").splitlines())
            assert not missing, f"{wheels[0].name} is not what is installed: {sorted(missing)[:3]}"
            entry["wheel"] = wheels[0].name
            entry["wheel_sha256"] = sha256(wheels[0])
        else:
            print(f"manifest: no {name} {dist.version} wheel in {WHEELS}, recording the installed RECORD only", file=sys.stderr)
        out[name] = entry
    return out


def jaxls_sort_patched() -> bool:
    """Whether jaxls's cost order is patched in this process (gen_fixtures.steady_cost_order shadows the module's
    `sorted`): read off the module, not taken from the caller's word."""
    problem = sys.modules.get("jaxls._problem")
    return problem is not None and "sorted" in vars(problem)


def environment() -> dict:
    """The writing process's environment, as of now: the settings' values in force (null when unset), not the defaults
    a script would have set, since a caller's environment overrides them."""
    return {
        "jaxls_sort_patch": jaxls_sort_patched(),
        "settings": {name: os.environ.get(name) for name in SETTINGS},
        "upstream": upstream(),
        "versions": versions(),
    }


def _measured_with(env: dict) -> dict:
    """What a check's result depends on besides the bytes: the upstream packages and the installed versions."""
    return {"upstream": env["upstream"], "versions": env["versions"]}


def _read() -> dict:
    return json.loads(PATH.read_text()) if PATH.exists() else {}


def _write(manifest: dict) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {PATH}", file=sys.stderr)


def update(section: str, files: dict, **fields) -> None:
    """Rewrites `section`'s `fields` and its `files` (relative path to entry, each with its own environment) over the
    entries it had: the others are kept as they were, with the environment that wrote them."""
    manifest = _read()
    old = manifest.get(section, {}).get("files", {})
    merged = dict(old)
    for name, entry in files.items():
        checks = old.get(name, {}).get("checks")
        if checks and old[name].get("sha256") == entry["sha256"]:  # measured on these very bytes...
            # ...and in this environment: a check from another upstream or numpy/jax is measured again.
            kept = {key: c for key, c in checks.items() if c.get("environment") == _measured_with(entry["environment"])}
            if kept:
                entry = {**entry, "checks": kept}
        merged[name] = entry
    manifest[section] = {**fields, "files": merged}
    _write(manifest)


def check(section: str, name: str, key: str, result: dict) -> None:
    """Records `result` under `section`'s entry for `name`, in its `checks`, for the bytes the entry describes."""
    manifest = _read()
    entry = manifest[section]["files"][name]
    assert entry["sha256"] == sha256(ROOT / name), f"{name} changed since the manifest was written: regenerate it first"
    entry.setdefault("checks", {})[key] = {**result, "environment": _measured_with(environment())}
    _write(manifest)
