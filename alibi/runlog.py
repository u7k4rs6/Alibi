"""Run directories, env.lock, and scrubbing artifacts before they are written.

An addition to the layout in docs/kickoff/02-technical-architecture.md section 2,
which names env.lock and the scrubbing requirement but gives them no module.
Everything that writes under artifacts/ goes through here, which is what makes
"nothing writes into artifacts/ without also writing env.lock" enforceable
rather than aspirational.

Two rules this file exists to hold:

  An unmeasured value is absent, not zero. Fields that could not be resolved on
  this host are written as null with a stated reason, never as a plausible
  looking default.

  Scrubbing happens at write time, not at publish time, per
  docs/kickoff/03-security-and-access.md section 4. An artifact that was
  scrubbed later was unscrubbed for a while.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts"

# Packages whose exact versions the architecture doc requires in env.lock.
# Absence is recorded as absence: on the day one CPU host the training stack is
# deliberately not installed, and writing 0 or "unknown" would read as a
# measurement.
TRACKED_PACKAGES = ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "numpy", "pyarrow")


def _scrub_patterns() -> list[tuple[re.Pattern, str]]:
    """Absolute paths, usernames and hostnames become stable placeholders."""
    patterns = [(re.compile(re.escape(str(REPO_ROOT))), "<repo>")]
    try:
        home = str(Path.home())
        if home and home != "/":
            patterns.append((re.compile(re.escape(home)), "<home>"))
    except (RuntimeError, OSError):
        pass
    for value in (_safe(getpass.getuser), _safe(socket.gethostname), _safe(platform.node)):
        if value and len(value) > 2:
            patterns.append((re.compile(re.escape(value)), "<host>" if "." in value else "<user>"))
    return patterns


def _safe(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001 - a missing identity is not an error here
        return None


def scrub(value):
    """Recursively replace machine specific strings.

    Applied to every artifact at write time. Git revision and the dirty flag
    are deliberately not scrubbed: section 4 of the security doc keeps them, and
    a dirty tree artifact stays labelled dirty.
    """
    patterns = _scrub_patterns()
    return _scrub_with(value, patterns)


def _scrub_with(value, patterns):
    if isinstance(value, str):
        for rx, replacement in patterns:
            value = rx.sub(replacement, value)
        return value
    if isinstance(value, dict):
        return {_scrub_with(k, patterns): _scrub_with(v, patterns) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub_with(v, patterns) for v in value]
    return value


def git_state() -> dict:
    """Revision and dirty flag. Kept deliberately, never cleaned."""

    def run(args):
        try:
            out = subprocess.run(
                ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    rev = run(["rev-parse", "HEAD"])
    status = run(["status", "--porcelain"])
    return {
        "revision": rev,
        "dirty": None if status is None else bool(status.strip()),
        "revision_absent_reason": None if rev else "no commits yet or not a git repository",
    }


def package_versions() -> dict:
    """Exact installed versions, or null plus a reason."""
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = {"version": version(name), "resolved": True}
        except PackageNotFoundError:
            out[name] = {
                "version": None,
                "resolved": False,
                "reason": "not installed on this host, see requirements-train.txt for the declared pin",
            }
    return out


def accelerator_state() -> dict:
    """CUDA, driver and GPU name, or absent with a reason. Never fabricated."""
    try:
        import torch
    except ImportError:
        return {"available": None, "reason": "torch is not installed on this host"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "torch reports no CUDA device on this host"}
    return {
        "available": True,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
    }


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def build_env_lock(
    config: dict,
    seeds: dict | None = None,
    sandbox: dict | None = None,
    dataset: dict | None = None,
    monitor_model_id: str | None = None,
) -> dict:
    """Everything architecture doc section 4 asks env.lock to carry."""
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
        },
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "packages": package_versions(),
        "accelerator": accelerator_state(),
        "sandbox_controls": sandbox if sandbox is not None else {},
        "datasets": dataset if dataset is not None else {},
        "monitor_model_id": monitor_model_id,
        "monitor_model_id_absent_reason": None if monitor_model_id else "no monitor is used on day 1",
        "seeds": seeds if seeds is not None else {},
        "config_hash": config_hash(config),
    }


def new_run_dir(kind: str, run_id: str | None = None) -> tuple[Path, str]:
    """A fresh run directory under artifacts/runs/."""
    if run_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{kind}-{stamp}"
    path = ARTIFACTS / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path, run_id


def write_json(path: Path, payload) -> None:
    """Scrub, then write. In that order, always."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scrub(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_run(run_dir: Path, config: dict, payload: dict, **env_lock_kwargs) -> None:
    """Write config.json, env.lock and the run's own payload, scrubbed.

    Nothing writes into artifacts/ without also writing env.lock, per
    docs/kickoff/04-cli-and-report-spec.md section 1, so the two happen in one
    call and there is no way to do one without the other.
    """
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "env.lock", build_env_lock(config, **env_lock_kwargs))
    write_json(run_dir / "result.json", payload)


def repo_relative(path: os.PathLike | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
