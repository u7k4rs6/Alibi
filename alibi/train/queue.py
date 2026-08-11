"""The breadth first run queue, resumable, with per run failure isolation.

Order is frozen in alibi/prereg.py: a0/s1, a1/s1, a2/s1, then seed 2, then seed
3. If the week dies on day 5 the deliverable is a complete three arm comparison
at fewer seeds, not two arms at three seeds. The tail is dropped, never the
front.

A halted run is marked FAILED with its reason and the queue moves on. The queue
stops only when three consecutive runs halt for the same reason, which is the
signal that the next run will fail the same way.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from alibi import halt as halt_module
from alibi import prereg, runlog

QUEUE_PATH = runlog.ARTIFACTS / "queue.json"


@dataclass
class QueueState:
    entries: list[dict] = field(default_factory=list)
    halt_reasons: list[str] = field(default_factory=list)
    stopped: bool = False
    stopped_reason: str = ""
    created_utc: str = ""
    updated_utc: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def build_queue(steps: int, dry_run: bool = False) -> QueueState:
    """The registered schedule, expanded into pending entries."""
    entries = [
        {
            "arm": arm,
            "seed": seed,
            "status": "pending",
            "run_id": None,
            "halt_reason": None,
            "steps": steps,
            "detail": "",
        }
        for arm, seed in prereg.PREREG.run_order.schedule()
    ]
    return QueueState(
        entries=entries,
        created_utc=datetime.now(timezone.utc).isoformat(),
    )


def load_queue(path: Path = QUEUE_PATH) -> QueueState | None:
    if not path.exists():
        return None
    return QueueState(**json.loads(path.read_text(encoding="utf-8")))


def save_queue(state: QueueState, path: Path = QUEUE_PATH) -> None:
    state.updated_utc = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.partial")
    tmp.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    tmp.rename(path)


def blocked_arms() -> set[str]:
    """Arms named by a `blocked-arms:` line in BLOCKED.md. Empty means global."""
    if not halt_module.BLOCKED_PATH.exists():
        return set()
    for line in halt_module.BLOCKED_PATH.read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("blocked-arms:"):
            return {a.strip() for a in line.split(":", 1)[1].split(",") if a.strip()}
    return set()


def next_pending(state: QueueState) -> dict | None:
    """The first entry that is not complete, skipping arms blocked by a collision.

    A blocked entry is marked failed with its reason rather than silently
    skipped, so the matrix records why it never ran.
    """
    blocked = blocked_arms()
    for entry in state.entries:
        if entry["status"] not in {"pending", "running"}:
            continue
        if entry["arm"] in blocked:
            entry["status"] = "failed"
            entry["halt_reason"] = "section_6_collision"
            entry["detail"] = (
                f"arm {entry['arm']} is blocked by BLOCKED.md: the view its monitor reads is empty, "
                "so the arm cannot measure what it was registered to measure"
            )
            continue
        return entry
    return None


def record_result(state: QueueState, entry: dict, result: dict) -> None:
    entry["run_id"] = result.get("run_id")
    if result.get("status") == "complete":
        entry["status"] = "complete"
        entry["detail"] = f"{result.get('steps', 0)} steps"
        state.halt_reasons.append(None)
    else:
        entry["status"] = "failed"
        entry["halt_reason"] = result.get("halt_reason")
        entry["detail"] = result.get("message", "")[:500]
        state.halt_reasons.append(result.get("halt_reason"))

    # Stop rule, set by the operator and superseding the earlier
    # three-consecutive-halts rule: the queue stops only when all nine runs are
    # complete, when more than half have failed, or when a section 6 collision
    # has written BLOCKED.md. Three consecutive halts for the same reason is
    # still recorded as a warning, because it is worth seeing, but it no longer
    # stops the queue on its own.
    failed = sum(1 for e in state.entries if e["status"] == "failed")
    complete = sum(1 for e in state.entries if e["status"] == "complete")
    total = len(state.entries)

    if failed * 2 > total:
        state.stopped = True
        state.stopped_reason = (
            f"{failed} of {total} runs have failed, which is more than half. "
            "Continuing would spend hours to produce a matrix that cannot support a comparison."
        )
    elif complete == total:
        state.stopped = True
        state.stopped_reason = f"all {total} runs are complete"
    elif halt_module.BLOCKED_PATH.exists():
        # A section 6 collision halts **the affected runs**, and stops the whole
        # queue only when nothing unaffected is left. The brief says to halt the
        # affected runs and continue with work that does not depend on the
        # collision, and a collision confined to one arm leaves the other arms
        # perfectly valid. BLOCKED.md declares its own scope on a line reading
        # `blocked-arms: a1` and a collision with no declared scope is global.
        blocked = blocked_arms()
        remaining = [e for e in state.entries if e["status"] == "pending" and e["arm"] not in blocked]
        if not blocked:
            state.stopped = True
            state.stopped_reason = (
                "BLOCKED.md exists with no declared arm scope, so the collision is treated as global. "
                "That is a finding, not a task, and the runs stop rather than being quietly corrected."
            )
        elif not remaining:
            state.stopped = True
            state.stopped_reason = (
                f"BLOCKED.md blocks arms {sorted(blocked)} and no unaffected run remains pending."
            )
    elif halt_module.should_stop_queue([r for r in state.halt_reasons]):
        state.stopped_reason = (
            f"WARN three consecutive runs halted for the same reason: {state.halt_reasons[-1]}. "
            "The queue continues per the operator's stop rule, but the next run will likely fail too."
        )


def progress(state: QueueState) -> dict:
    counts: dict[str, int] = {}
    for entry in state.entries:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    return {
        "total": len(state.entries),
        "by_status": dict(sorted(counts.items())),
        "stopped": state.stopped,
        "stopped_reason": state.stopped_reason,
        "complete_arms_at_seed_1": sorted(
            e["arm"] for e in state.entries if e["seed"] == 1 and e["status"] == "complete"
        ),
    }
