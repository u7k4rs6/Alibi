"""Recompute every published number from artifacts. No GPU, no network.

The command a mentor runs, per docs/kickoff/04-cli-and-report-spec.md section 6.
It reads artifacts/index.json, checks digests, recomputes every claim through
alibi.report.metrics, compares to what the report published, and exits non zero
on any mismatch.

It must keep working from day one, when there are no runs to verify. With no
declared runs it verifies what does exist: the prereg hash, the eligibility
manifest hash, and the day one gate artifact. Reporting "nothing to verify" as
success would be exactly the failure this command exists to prevent, so it says
what it checked and what it did not.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from alibi import prereg, runlog
from alibi.report import metrics

PUBLISHED_PATH = runlog.REPO_ROOT / "report" / "published.json"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_digests(index: dict) -> tuple[int, int, list[str]]:
    """Every file the index declares a digest for."""
    ok = 0
    total = 0
    problems = []
    for entry in index.get("runs", []):
        for relative, expected in (entry.get("digests") or {}).items():
            total += 1
            path = runlog.REPO_ROOT / relative
            if not path.exists():
                problems.append(f"missing file {relative}")
                continue
            actual = digest(path)
            if actual != expected:
                problems.append(f"digest mismatch {relative}: published {expected[:12]} computed {actual[:12]}")
            else:
                ok += 1
    return ok, total, problems


def verify(no_gpu: bool = True) -> tuple[int, list[str]]:
    """Returns (exit_code, lines). Exit 1 on any mismatch."""
    lines: list[str] = []
    problems: list[str] = []

    lines.append("")
    lines.append("alibi verify --no-gpu" if no_gpu else "alibi verify")
    lines.append("")

    # 1. The registration.
    provenance = prereg.provenance()
    lines.append(f"{'prereg version':<38s}{provenance['prereg_version']}")
    lines.append(f"{'prereg hash':<38s}{provenance['prereg_hash'][:16]}")
    if provenance["eligibility_hash"] is None:
        problems.append(f"eligibility manifest absent: {provenance['eligibility_absent_reason']}")
        lines.append(f"{'eligibility manifest':<38s}ABSENT")
    else:
        document = prereg.load_eligibility()
        recomputed = prereg.eligibility_hash(document)
        stored = document.get("eligibility_hash")
        state = "OK" if recomputed == stored else "MISMATCH"
        if state == "MISMATCH":
            problems.append(f"eligibility hash mismatch: stored {stored[:12]} computed {recomputed[:12]}")
        lines.append(
            f"{'eligibility manifest':<38s}{document['n_eligible']} problems  {recomputed[:16]}  {state}"
        )
        if document.get("prereg_hash") != provenance["prereg_hash"]:
            problems.append("the eligibility manifest was built under a different prereg hash")

    # 2. The declared evidence.
    index = {}
    if metrics.INDEX_PATH.exists():
        index = json.loads(metrics.INDEX_PATH.read_text(encoding="utf-8"))
    declared = index.get("runs", [])
    lines.append(f"{'reading artifacts/index.json':<38s}{len(declared)} declared runs")

    ok, total, digest_problems = check_digests(index)
    problems.extend(digest_problems)
    lines.append(f"{'checking digests':<38s}{ok}/{total} match")

    # 3. Recompute every claim.
    computed = metrics.claims()
    if not PUBLISHED_PATH.exists():
        lines.append(f"{'recomputing metrics':<38s}nothing published yet, no claims to compare")
        if declared:
            problems.append("runs are declared but report/published.json does not exist")
    else:
        published = json.loads(PUBLISHED_PATH.read_text(encoding="utf-8"))
        lines.append("recomputing metrics")
        for claim_id, expected in sorted(published.get("claims", {}).items()):
            actual = _resolve(computed, expected["path"])
            match = _close(actual, expected["value"])
            if not match:
                problems.append(
                    f"{claim_id}: published {expected['value']} computed {actual}"
                )
            state = "OK" if match else "MISMATCH"
            lines.append(
                f"  {claim_id:<22s}{expected['label']:<26s}"
                f"published {_fmt(expected['value']):>8s}  computed {_fmt(actual):>8s}  {state}"
            )

    # 4. What was deliberately not verified.
    if not declared:
        lines.append("")
        lines.append("no training runs are declared yet, so no run derived number was verified.")
        lines.append("the registration and the eligibility manifest above were verified.")

    lines.append("")
    if problems:
        lines.append(f"FAILED, {len(problems)} problems")
        for problem in problems:
            lines.append(f"  {problem}")
        lines.append("exit 1")
        return 1, lines
    lines.append("exit 0")
    return 0, lines


def _resolve(document: dict, path: str):
    node = document
    for part in path.split("."):
        if isinstance(node, list):
            try:
                node = node[int(part)]
                continue
            except (ValueError, IndexError):
                return None
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _close(a, b, tolerance: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is b or a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tolerance
    return a == b


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
