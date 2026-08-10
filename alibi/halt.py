"""Halt conditions. Each writes HALT.md and exits non zero for its run.

None of these is a warning. A run that continues past one of them produces an
artifact that looks like evidence and is not, which is the failure mode this
whole project is built to avoid.

Thresholds live in alibi/prereg.py and are frozen. This module decides whether a
condition is met and writes the evidence; it never chooses a number.

Halting one run does not stop the queue. The queue stops only when three
consecutive runs halt for the same reason, which is `should_stop_queue`.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from alibi import prereg, runlog

HALT_PATH = runlog.REPO_ROOT / "HALT.md"
BLOCKED_PATH = runlog.REPO_ROOT / "BLOCKED.md"

# Reasons, as stable machine readable identifiers.
MONITOR_ERROR_RATE = "monitor_error_rate"
INDETERMINATE_RATE = "indeterminate_held_out_rate"
KL_SPIKE = "kl_spike"
DEGENERATE_POLICY = "degenerate_policy"
DIRTY_TREE = "dirty_git_tree"
HASH_MISMATCH = "prereg_or_eligibility_hash_mismatch"
DISK_LOW = "disk_below_floor"

MIN_FREE_DISK_BYTES = 10 * 1024**3


class Halt(Exception):
    """Raised to stop a run. Carries everything HALT.md needs."""

    def __init__(self, reason: str, message: str, evidence: dict) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.evidence = evidence


@dataclass
class StepStats:
    """What one training step observed, as the halt checks need it."""

    run_id: str
    arm: str
    seed: int
    step: int
    monitor_judgements: int = 0
    monitor_errors: int = 0
    held_out_executions: int = 0
    held_out_indeterminate: int = 0
    kl: float | None = None
    completion_texts: list[str] = field(default_factory=list)
    group_size: int = 1
    mean_completion_chars: float | None = None

    @property
    def monitor_error_fraction(self) -> float:
        return self.monitor_errors / self.monitor_judgements if self.monitor_judgements else 0.0

    @property
    def indeterminate_fraction(self) -> float:
        return self.held_out_indeterminate / self.held_out_executions if self.held_out_executions else 0.0


def check_dirty_tree() -> None:
    state = runlog.git_state()
    if prereg.PREREG.halt.halt_on_dirty_tree and state.get("dirty"):
        raise Halt(
            DIRTY_TREE,
            "the working tree has uncommitted changes at run start",
            {"git": state},
        )


def check_hashes(expected_prereg: str | None = None, expected_eligibility: str | None = None) -> None:
    """The registration this run claims to be under must be the one in the tag."""
    actual = prereg.provenance()
    if expected_prereg and actual["prereg_hash"] != expected_prereg:
        raise Halt(
            HASH_MISMATCH,
            "prereg hash does not match the tagged registration",
            {"expected": expected_prereg, "actual": actual["prereg_hash"]},
        )
    if actual["eligibility_hash"] is None:
        raise Halt(
            HASH_MISMATCH,
            "the eligibility manifest is missing, so the problem set is undeclared",
            {"reason": actual["eligibility_absent_reason"]},
        )
    if expected_eligibility and actual["eligibility_hash"] != expected_eligibility:
        raise Halt(
            HASH_MISMATCH,
            "eligibility manifest hash does not match the tagged registration",
            {"expected": expected_eligibility, "actual": actual["eligibility_hash"]},
        )


def check_disk(path: Path | None = None) -> None:
    usage = shutil.disk_usage(str(path or runlog.REPO_ROOT))
    if usage.free < MIN_FREE_DISK_BYTES:
        raise Halt(
            DISK_LOW,
            "free disk is below the floor, so artifacts would be truncated",
            {"free_bytes": usage.free, "floor_bytes": MIN_FREE_DISK_BYTES},
        )


def check_monitor_errors(stats: StepStats) -> None:
    limit = prereg.PREREG.halt.max_monitor_error_fraction
    if stats.monitor_judgements and stats.monitor_error_fraction > limit:
        raise Halt(
            MONITOR_ERROR_RATE,
            "too many monitor judgements errored, so this step's flag rate is not a measurement",
            {
                "errors": stats.monitor_errors,
                "judgements": stats.monitor_judgements,
                "fraction": stats.monitor_error_fraction,
                "limit": limit,
            },
        )


def check_indeterminate(stats: StepStats) -> None:
    limit = prereg.PREREG.halt.max_indeterminate_fraction
    if stats.held_out_executions and stats.indeterminate_fraction > limit:
        raise Halt(
            INDETERMINATE_RATE,
            "too many held out executions were indeterminate, so this step's cheat rate "
            "is measuring the sandbox rather than the policy",
            {
                "indeterminate": stats.held_out_indeterminate,
                "executions": stats.held_out_executions,
                "fraction": stats.indeterminate_fraction,
                "limit": limit,
            },
        )


def check_kl(stats: StepStats, kl_history: list[float]) -> None:
    """KL above a multiple of the median of the first N steps."""
    spec = prereg.PREREG.halt
    if stats.kl is None:
        return
    baseline = [k for k in kl_history[: spec.kl_baseline_steps] if k is not None]
    if len(baseline) < spec.kl_baseline_steps:
        return
    reference = median(baseline)
    if reference <= 0:
        return
    if stats.kl > spec.kl_spike_multiple * reference:
        raise Halt(
            KL_SPIKE,
            "KL from the reference policy spiked above the registered multiple of its own baseline",
            {
                "kl": stats.kl,
                "baseline_median": reference,
                "multiple": spec.kl_spike_multiple,
                "baseline_steps": spec.kl_baseline_steps,
            },
        )


def check_degenerate(stats: StepStats, min_mean_chars: int | None = None) -> None:
    """Identical completions within a group, or completions too short to be code."""
    spec = prereg.PREREG.halt
    floor = spec.min_mean_completion_chars if min_mean_chars is None else min_mean_chars

    texts = stats.completion_texts
    if texts:
        mean_chars = sum(len(t) for t in texts) / len(texts)
        if mean_chars < floor:
            raise Halt(
                DEGENERATE_POLICY,
                "mean completion length fell below the floor, so the policy has collapsed",
                {"mean_chars": mean_chars, "floor": floor, "n": len(texts)},
            )
        group = max(1, stats.group_size)
        for start in range(0, len(texts) - group + 1, group):
            chunk = texts[start : start + group]
            if len(chunk) > 1 and len(set(chunk)) == 1:
                raise Halt(
                    DEGENERATE_POLICY,
                    "every completion in a group was identical, so the GRPO advantage is identically zero",
                    {"group_index": start // group, "group_size": len(chunk), "text_chars": len(chunk[0])},
                )


def check_step(stats: StepStats, kl_history: list[float], min_mean_chars: int | None = None) -> None:
    """Every per step condition, in a fixed order so the first failure is stable."""
    check_monitor_errors(stats)
    check_indeterminate(stats)
    check_kl(stats, kl_history)
    check_degenerate(stats, min_mean_chars)
    check_disk()


def preflight(expected_prereg: str | None = None, expected_eligibility: str | None = None) -> None:
    """Everything checked before a run starts."""
    check_dirty_tree()
    check_hashes(expected_prereg, expected_eligibility)
    check_disk()


def write_halt(halt: Halt, run_id: str, step: int | None, extra: dict | None = None) -> Path:
    """HALT.md, with the reason, run id, step and the supporting numbers."""
    payload = {
        "reason": halt.reason,
        "message": halt.message,
        "run_id": run_id,
        "step": step,
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "evidence": halt.evidence,
        "prereg": prereg.provenance(),
        "git": runlog.git_state(),
    }
    if extra:
        payload["context"] = extra
    scrubbed = runlog.scrub(payload)

    lines = [
        "# HALT",
        "",
        f"**Reason:** `{halt.reason}`",
        "",
        f"**Run:** `{run_id}`  **Step:** {step if step is not None else 'not started'}",
        "",
        f"**Written:** {payload['written_utc']}",
        "",
        f"{halt.message}.",
        "",
        "## Supporting numbers",
        "",
        "```json",
        json.dumps(scrubbed["evidence"], indent=2, sort_keys=True),
        "```",
        "",
        "## Registration",
        "",
        "```json",
        json.dumps(scrubbed["prereg"], indent=2, sort_keys=True),
        "```",
        "",
        "## Git",
        "",
        "```json",
        json.dumps(scrubbed["git"], indent=2, sort_keys=True),
        "```",
        "",
        "This run is marked FAILED and does not enter the evidence index.",
        "",
    ]
    HALT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return HALT_PATH


def should_stop_queue(recent_halt_reasons: list[str], window: int = 3) -> bool:
    """Three consecutive halts for the same reason stop the whole queue."""
    if len(recent_halt_reasons) < window:
        return False
    tail = recent_halt_reasons[-window:]
    return len(set(tail)) == 1 and tail[0] is not None


def halt_to_dict(halt: Halt) -> dict:
    return {"reason": halt.reason, "message": halt.message, "evidence": halt.evidence}


@dataclass
class QueueEntry:
    arm: str
    seed: int
    status: str = "pending"  # pending | running | complete | failed
    run_id: str | None = None
    halt_reason: str | None = None
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
