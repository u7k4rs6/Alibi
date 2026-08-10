"""Path scrubbing, verified on a real artifact.

The day one checklist in docs/kickoff/03-security-and-access.md section 8 asks
for exactly this: scrubbing in the logger, verified on one real artifact rather
than asserted on a synthetic string.
"""

from __future__ import annotations

import getpass
import json
from pathlib import Path

from alibi import runlog


def test_absolute_repo_paths_are_scrubbed() -> None:
    payload = {"path": str(runlog.REPO_ROOT / "alibi" / "env" / "executor.py")}
    assert runlog.scrub(payload) == {"path": "<repo>/alibi/env/executor.py"}


def test_home_directory_is_scrubbed() -> None:
    scrubbed = runlog.scrub({"cache": f"{Path.home()}/.cache/huggingface"})
    assert str(Path.home()) not in scrubbed["cache"]


def test_username_is_scrubbed() -> None:
    user = getpass.getuser()
    if len(user) <= 2:
        return
    assert user not in runlog.scrub({"note": f"run by {user} on this host"})["note"]


def test_scrubbing_recurses_through_lists_and_keys() -> None:
    payload = {str(runlog.REPO_ROOT): [{"nested": str(runlog.REPO_ROOT / "x")}]}
    assert runlog.scrub(payload) == {"<repo>": [{"nested": "<repo>/x"}]}


def test_env_lock_records_absence_rather_than_zero() -> None:
    """An unmeasured value is absent, not zero."""
    lock = runlog.build_env_lock({"command": "test"})
    for name, entry in lock["packages"].items():
        if not entry["resolved"]:
            assert entry["version"] is None, f"{name} invented a version"
            assert entry["reason"], f"{name} is absent with no stated reason"
    accelerator = lock["accelerator"]
    if not accelerator.get("available"):
        assert accelerator.get("reason"), "no accelerator, and no reason given"


def test_env_lock_keeps_the_dirty_flag() -> None:
    """Kept deliberately, per security doc section 4. Never quietly cleaned."""
    lock = runlog.build_env_lock({"command": "test"})
    assert "dirty" in lock["git"]
    assert "revision" in lock["git"]


def test_write_run_always_writes_env_lock(tmp_path: Path) -> None:
    """Nothing writes into artifacts/ without also writing env.lock."""
    runlog.write_run(tmp_path, {"command": "test"}, {"ok": True})
    assert (tmp_path / "env.lock").exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "result.json").exists()


def test_a_real_artifact_carries_no_absolute_paths(tmp_path: Path) -> None:
    """The verification the checklist asks for, on a genuinely written file."""
    from alibi.env.executor import Executor

    executor = Executor()
    runlog.write_run(
        tmp_path,
        {"command": "data check"},
        {"summary": {"n_problems": 0}},
        sandbox=executor.control_summary(),
    )
    for name in ("env.lock", "config.json", "result.json"):
        text = (tmp_path / name).read_text(encoding="utf-8")
        json.loads(text)
        assert str(runlog.REPO_ROOT) not in text, f"{name} leaked the repo path"
        assert str(Path.home()) not in text, f"{name} leaked the home directory"
