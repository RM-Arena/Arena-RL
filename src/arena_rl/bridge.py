"""Resolve a simulator checkout without vendoring its rules or physics."""
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys


def configure(root=None):
    candidate = root or os.environ.get("ARENA_ROOT")
    if not candidate:
        candidate = Path(__file__).resolve().parents[3] / "RMUC-Arena"
    root = Path(candidate).expanduser().resolve()
    for relative in ("rmuc_arena/env.py", "rmuc_arena/vector_env.py", "simulation/src/training_scenario.rs"):
        if not (root / relative).is_file():
            raise RuntimeError(f"Arena checkout needs the parallel environment and training scenarios: {root}. Set ARENA_ROOT or --arena-root.")
    loaded = sys.modules.get("rmuc_arena")
    if loaded and not Path(loaded.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Another Arena checkout is already imported; start a fresh Python process")
    os.environ["ARENA_ROOT"] = str(root)
    os.environ["RMUC_RESOURCE_ROOT"] = str(root)
    sys.path.insert(0, str(root)) if str(root) not in sys.path else None
    importlib.invalidate_caches()
    return root


def provenance(root):
    from rmuc_arena.engine import binary_path, Match
    files = ["config/rules.json", "config/battle_settings.json", "rmuc_arena/env.py",
             "rmuc_arena/vector_env.py", "rmuc_arena/engine.py", "rmuc_arena/action_protocol.py",
             "rmuc_arena/communication.py", "rmuc_arena/replays.py", "rmuc_arena/sdk.py"]
    files += [str(p.relative_to(root)) for p in sorted((root / "simulation/src").glob("*.rs"))]
    hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files}
    embedded = ["lib.rs", "rulebook.rs", "physics.rs", "penalties.rs", "structures.rs",
                "pickup.rs", "season2027.rs", "vision.rs", "training_scenario.rs", "../Cargo.lock"]
    engine_source = hashlib.sha256()
    for relative in embedded:engine_source.update((root/"simulation/src"/relative).read_bytes())
    with Match() as match:
        frame=match.snapshot()
    if frame["engine_hash"]!=engine_source.hexdigest():
        raise RuntimeError("Arena binary contains stale simulator sources; rebuild simulation/Cargo.toml")
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"root": str(root), "repository": "https://github.com/RM-Arena/Arena-1",
            "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "files": hashes, "engine_hash":frame["engine_hash"], "rules_hash":frame["rules_hash"], "binary_sha256": hashlib.sha256(binary_path().read_bytes()).hexdigest()}


def check_provenance(expected, actual, allow=False):
    same_build = (expected.get("engine_hash")==actual.get("engine_hash") and
                  expected.get("rules_hash")==actual.get("rules_hash")) if expected.get("engine_hash") else expected["binary_sha256"]==actual["binary_sha256"]
    if expected["files"] != actual["files"] or not same_build:
        if not allow:
            raise RuntimeError("Simulator differs from the checkpoint; use the recorded source/build or explicitly pass --allow-engine-change")
