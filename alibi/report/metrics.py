"""Every published number is computed here and nowhere else.

Architecture doc section 9. `alibi verify --no-gpu` recomputes from artifacts
using these same functions, so a mismatch between the report and the artifacts
is a bug in one of them rather than a difference of opinion between two
implementations.

Everything reads from `artifacts/`. Nothing here imports torch, loads a model,
or touches the network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from alibi import prereg, runlog

ARTIFACTS = runlog.ARTIFACTS
INDEX_PATH = ARTIFACTS / "index.json"


@dataclass(frozen=True)
class RunSeries:
    run_id: str
    arm: str
    seed: int
    steps: list[int]
    cheat_rate: list[float | None]
    flag_rate: list[float | None]
    gap: list[float | None]
    indeterminate_rate: list[float]
    visible_pass_rate: list[float | None]
    held_out_pass_rate: list[float | None]
    mean_reward: list[float | None]
    kl: list[float | None]
    cheat_rate_by_threshold: dict
    cheat_form_counts: list[dict]


def _run_dirs() -> list[Path]:
    root = ARTIFACTS / "runs"
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []


def load_step_summaries(run_dir: Path) -> list[dict]:
    """Step summaries in step order, read from disk in sorted order.

    Sorted by the integer step, not by filename, and not by mtime, so a replay
    cannot depend on the order the filesystem happens to return.
    """
    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return []
    summaries = []
    for path in sorted(steps_dir.glob("*/summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return sorted(summaries, key=lambda s: s["step"])


def series_for(run_dir: Path) -> RunSeries | None:
    summaries = load_step_summaries(run_dir)
    if not summaries:
        return None
    thresholds = {f"{t:.2f}": [] for t in prereg.PREREG.oracle.sensitivity_thresholds}
    for summary in summaries:
        for label in thresholds:
            thresholds[label].append((summary.get("cheat_rate_at_threshold", {}).get(label) or {}).get("rate"))
    first = summaries[0]
    return RunSeries(
        run_id=first["run_id"],
        arm=first["arm"],
        seed=first["seed"],
        steps=[s["step"] for s in summaries],
        cheat_rate=[s.get("cheat_rate_oracle") for s in summaries],
        flag_rate=[s.get("flag_rate_monitor") for s in summaries],
        gap=[s.get("obfuscation_gap") for s in summaries],
        indeterminate_rate=[s.get("held_out_indeterminate_fraction", 0.0) for s in summaries],
        visible_pass_rate=[s.get("visible_pass_rate") for s in summaries],
        held_out_pass_rate=[s.get("held_out_pass_rate") for s in summaries],
        mean_reward=[s.get("mean_reward") for s in summaries],
        kl=[s.get("kl") for s in summaries],
        cheat_rate_by_threshold=thresholds,
        cheat_form_counts=[s.get("cheat_form_counts", {}) for s in summaries],
    )


def load_declared_runs() -> list[RunSeries]:
    """Only runs named in artifacts/index.json. Never a glob, never a sort.

    Architecture doc section 8: evidence is selected by explicit declaration.
    A run that is not declared does not contribute to a published number, even
    if it exists and looks fine.
    """
    if not INDEX_PATH.exists():
        return []
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    out = []
    for entry in index.get("runs", []):
        run_dir = ARTIFACTS / "runs" / entry["run_id"]
        series = series_for(run_dir) if run_dir.exists() else None
        if series is not None:
            out.append(series)
    return out


def terminal(values: list[float | None]) -> float | None:
    for value in reversed(values):
        if value is not None:
            return value
    return None


def seed_band(series_list: list[RunSeries], attribute: str, step_index: int = -1) -> dict:
    """Min to max across seeds. Never a standard error, because n=3."""
    values = []
    for series in series_list:
        column = getattr(series, attribute)
        if not column:
            continue
        value = column[step_index] if -len(column) <= step_index < len(column) else None
        if value is not None:
            values.append(value)
    if not values:
        return {"min": None, "max": None, "spread": None, "n_seeds": 0}
    return {
        "min": min(values),
        "max": max(values),
        "spread": max(values) - min(values),
        "n_seeds": len(values),
        "values": sorted(values),
    }


def by_arm(series_list: list[RunSeries]) -> dict[str, list[RunSeries]]:
    grouped: dict[str, list[RunSeries]] = {}
    for series in series_list:
        grouped.setdefault(series.arm, []).append(series)
    return {arm: sorted(runs, key=lambda s: s.seed) for arm, runs in sorted(grouped.items())}


def correlation(xs: list[float | None], ys: list[float | None]) -> float | None:
    """Pearson correlation over pairs where both are present.

    Used for indeterminate rate against oracle cheat rate. If measurement drift
    tracks the headline, that belongs in the findings, not a footnote.
    """
    pairs = [(x, y) for x, y in zip(xs, ys, strict=False) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    dx = sum((p[0] - mx) ** 2 for p in pairs) ** 0.5
    dy = sum((p[1] - my) ** 2 for p in pairs) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def is_monotonic_rising(values: list[float | None]) -> bool | None:
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return None
    return all(b >= a for a, b in zip(present, present[1:], strict=False))


def claims() -> dict:
    """Every published number, keyed by claim id.

    `alibi verify` recomputes exactly this dict and compares it to what the
    report published. A claim with no entry here does not go in the report.
    """
    runs = load_declared_runs()
    grouped = by_arm(runs)
    out: dict = {
        "prereg_hash": prereg.PREREG_HASH,
        "prereg_version": prereg.PREREG.version,
        "eligibility": prereg.provenance(),
        "n_declared_runs": len(runs),
        "arms": {},
        "hypotheses": {},
        "sensitivity": {},
        "diagnostics": {},
    }
    if not runs:
        out["absent_reason"] = "no runs are declared in artifacts/index.json yet"
        return out

    for arm, series_list in grouped.items():
        out["arms"][arm] = {
            "n_seeds": len(series_list),
            "seeds": [s.seed for s in series_list],
            "run_ids": [s.run_id for s in series_list],
            "terminal_cheat_rate": seed_band(series_list, "cheat_rate"),
            "terminal_flag_rate": seed_band(series_list, "flag_rate"),
            "terminal_gap": seed_band(series_list, "gap"),
            "terminal_indeterminate_rate": seed_band(series_list, "indeterminate_rate"),
            "initial_gap": seed_band(series_list, "gap", step_index=0),
        }
        for label in (f"{t:.2f}" for t in prereg.PREREG.oracle.sensitivity_thresholds):
            out["sensitivity"].setdefault(label, {})[arm] = {
                "terminal_cheat_rate": [
                    terminal(s.cheat_rate_by_threshold.get(label, [])) for s in series_list
                ]
            }
        correlations = [
            correlation(s.indeterminate_rate, s.cheat_rate) for s in series_list
        ]
        out["diagnostics"].setdefault("indeterminate_vs_cheat_rate_correlation", {})[arm] = [
            c for c in correlations if c is not None
        ]

    a0 = grouped.get("a0", [])
    out["hypotheses"]["H1"] = {
        "statement": prereg.PREREG.metric.hypotheses[0],
        "monotonic_rising_per_seed": [is_monotonic_rising(s.cheat_rate) for s in a0],
        "terminal_minus_initial": [
            (terminal(s.cheat_rate) - next((v for v in s.cheat_rate if v is not None), 0.0))
            if terminal(s.cheat_rate) is not None
            else None
            for s in a0
        ],
    }
    return out
