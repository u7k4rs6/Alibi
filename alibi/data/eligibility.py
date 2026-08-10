"""Build the frozen eligible problem set from a data check run.

The rules are in alibi/prereg.py and are not chosen here. This module applies
them and records, per excluded problem, which rule excluded it and the number
that triggered it. An auditor can therefore reconstruct the denominator without
rerunning anything, which is the point.

Written once, before the tag. A run whose eligibility hash differs from the
tagged one is halted by alibi.halt rather than quietly compared.
"""

from __future__ import annotations

import json
from pathlib import Path

from alibi import prereg
from alibi.prereg import ELIGIBILITY_PATH

RULE_REFERENCE_HELD_OUT = "reference_solution_fails_held_out"
RULE_REFERENCE_VISIBLE = "reference_solution_fails_visible"
RULE_NO_CHEAT = "no_constructible_synthetic_cheat"
RULE_TOO_FEW_HELD_OUT = "fewer_than_min_held_out_tests"


def decide(record: dict) -> tuple[bool, list[dict]]:
    """Apply the frozen rules to one problem's data check record."""
    spec = prereg.PREREG.eligibility
    reasons: list[dict] = []

    reference = record["reference"]
    cheat = record["cheat"]

    if spec.require_reference_passes_visible and (reference.get("visible_pass_fraction") or 0.0) < 1.0:
        reasons.append(
            {
                "rule": RULE_REFERENCE_VISIBLE,
                "value": reference.get("visible_pass_fraction"),
                "note": "the visible reward signal is meaningless where the reference cannot pass it",
            }
        )
    if spec.require_reference_passes_held_out and (reference.get("held_out_pass_fraction") or 0.0) < 1.0:
        reasons.append(
            {
                "rule": RULE_REFERENCE_HELD_OUT,
                "value": reference.get("held_out_pass_fraction"),
                "note": "a held out failure here says something about EvalPlus or our harness, not the policy",
            }
        )
    if spec.require_constructible_cheat and not cheat.get("constructed"):
        reasons.append(
            {
                "rule": RULE_NO_CHEAT,
                "value": cheat.get("reason", ""),
                "note": "without a synthetic cheat the problem cannot contribute to the gate",
            }
        )
    if record["n_held_out"] < spec.min_held_out_tests:
        reasons.append(
            {
                "rule": RULE_TOO_FEW_HELD_OUT,
                "value": record["n_held_out"],
                "note": "a held out set this small is noise rather than a generalisation measurement",
            }
        )
    return (not reasons), reasons


def build(records: list[dict], split_sha256: str) -> dict:
    """The eligibility manifest, ready to hash."""
    eligible: list[int] = []
    excluded: list[dict] = []
    for record in sorted(records, key=lambda r: r["task_id"]):
        ok, reasons = decide(record)
        if ok:
            eligible.append(record["task_id"])
        else:
            excluded.append(
                {
                    "task_id": record["task_id"],
                    "rules": [r["rule"] for r in reasons],
                    "detail": reasons,
                }
            )

    document = {
        "prereg_version": prereg.PREREG.version,
        "prereg_hash": prereg.PREREG_HASH,
        "split_sha256": split_sha256,
        "rules": {
            "require_reference_passes_visible": prereg.PREREG.eligibility.require_reference_passes_visible,
            "require_reference_passes_held_out": prereg.PREREG.eligibility.require_reference_passes_held_out,
            "require_constructible_cheat": prereg.PREREG.eligibility.require_constructible_cheat,
            "min_held_out_tests": prereg.PREREG.eligibility.min_held_out_tests,
        },
        "task_ids": eligible,
        "n_eligible": len(eligible),
        "n_excluded": len(excluded),
        "excluded": excluded,
        "excluded_by_rule": _count_by_rule(excluded),
    }
    document["eligibility_hash"] = prereg.eligibility_hash(document)
    return document


def _count_by_rule(excluded: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for entry in excluded:
        for rule in entry["rules"]:
            counts[rule] = counts.get(rule, 0) + 1
    return dict(sorted(counts.items()))


def write(document: dict, path: Path = ELIGIBILITY_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
