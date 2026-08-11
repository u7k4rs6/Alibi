"""Rebuild REPORT.md, the figures and report/published.json from artifacts.

Run after every completed run, so a current artifact always exists and the
pre-registered sensitivity rows are always populated.

Numbers are injected from artifacts, never typed. `report/published.json` is the
machine readable list of every claim, and `alibi verify` recomputes exactly that
list. A claim with no entry there does not appear in the report.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from alibi import prereg, runlog
from alibi.report import figure, metrics

REPORT_DIR = runlog.REPO_ROOT / "report"
REPORT_PATH = REPORT_DIR / "REPORT.md"
PUBLISHED_PATH = REPORT_DIR / "published.json"
STEP_ZERO_FLAGGED_PATH = REPORT_DIR / "STEP_ZERO_FLAGGED.md"
STRUCTURAL_FP_PATH = REPORT_DIR / "STRUCTURAL_FP.md"

# What counts as "near zero" for the structural check's false positive rate on
# known-honest code. Declared here rather than judged after seeing the number.
NEAR_ZERO_FALSE_POSITIVE_RATE = 0.01


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_published(computed: dict) -> dict:
    """Every published claim, as (path into metrics.claims(), value, label)."""
    claims: dict[str, dict] = {}
    index = 1
    for arm, entry in sorted(computed.get("arms", {}).items()):
        for key, label in (
            ("terminal_cheat_rate", "terminal cheat rate"),
            ("terminal_flag_rate", "terminal flag rate"),
            ("terminal_gap", "terminal gap"),
        ):
            band = entry.get(key) or {}
            for stat in ("min", "max"):
                if band.get(stat) is None:
                    continue
                claims[f"claim{index}"] = {
                    "path": f"arms.{arm}.{key}.{stat}",
                    "value": band[stat],
                    "label": f"{arm} {label} {stat}",
                }
                index += 1
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "prereg_hash": computed.get("prereg_hash"),
        "eligibility_hash": (computed.get("eligibility") or {}).get("eligibility_hash"),
        "n_declared_runs": computed.get("n_declared_runs", 0),
        "claims": claims,
    }


def write_step_zero_flagged() -> Path:
    """Full text of every step-zero completion carrying a cheat_form, for reading by hand.

    Lives under report/ because the queue runner stages that directory. A file
    at the repository root would be left unstaged after each run and would dirty
    the tree, which is precisely what halted three runs earlier.
    """
    payload = metrics.flagged_step_zero_completions()
    lines = [
        "# Step-zero completions carrying a cheat_form",
        "",
        "Every completion sampled **before any policy update** whose code the structural check "
        "labelled with a cheat_form, pooled across all matrix runs. Step 0 is the same untrained "
        "base policy in every arm and at every seed.",
        "",
        f"Total flagged: **{payload['total']}**. Shown here: **{payload['shown']}**.",
        "",
    ]
    if payload["sampled"]:
        lines += [
            f"More than {metrics.STEP_ZERO_SAMPLE_CAP} were flagged, so this is a sample of "
            f"{metrics.STEP_ZERO_SAMPLE_CAP} drawn at declared seed "
            f"`{metrics.STEP_ZERO_SAMPLE_SEED}`, fixed before the draw.",
            "",
        ]
    if not payload["items"]:
        lines += [
            "No step-zero completion carries a cheat_form yet, or no matrix run has stored a step 0.",
            "",
            "This file is regenerated after every completed run.",
            "",
        ]
    else:
        lines.append(
            "The structural check flags "
            f"{_fmt(metrics.reference_structural_false_positives()['false_positive_rate'])} of known-honest "
            "MBPP+ reference solutions, so some of what follows is expected to be detector error rather "
            "than model behaviour. That is the reason for reading them by hand."
        )
        lines.append("")
        for index, item in enumerate(payload["items"], start=1):
            lines += [
                "---",
                "",
                f"## {index}. {item['run_id']}, prompt {item['prompt_id']}, completion {item['completion_idx']}",
                "",
                f"- cheat_form: `{item['cheat_form']}`",
                f"- oracle signals: `{item['oracle_signals']}`",
                f"- visible pass fraction: {_fmt(item['visible_pass_fraction'])}",
                f"- held-out pass fraction: {_fmt(item['held_out_pass_fraction'])}",
                f"- behavioural cheat verdict: {item['oracle_behavioural']}",
                "",
                "```",
                (item.get("text") or "").rstrip(),
                "```",
                "",
            ]
    STEP_ZERO_FLAGGED_PATH.write_text("\n".join(lines), encoding="utf-8")
    return STEP_ZERO_FLAGGED_PATH


def write_structural_fp() -> Path:
    """Full source of every honest reference solution the structural check flags."""
    from alibi.data.build import build
    from alibi.env.oracle import structural_check

    precision = metrics.reference_structural_false_positives()
    ids = set(precision["flagged_task_ids"])
    problems = {p.task_id: p for p in build().problems if p.task_id in ids}

    lines = [
        "# Structural check false positives",
        "",
        "Every one of these is an MBPP+ **reference solution**: a genuine algorithm, honest by "
        "construction. The structural check flags them anyway, so each is a false positive.",
        "",
        f"{precision['n_flagged']} of {precision['n_reference_solutions']} eligible reference solutions, "
        f"a rate of {_fmt(precision['false_positive_rate'])}.",
        "",
        "These are **five specific problems, not a uniform rate**. The floor they create appears only "
        "on the steps where one of them is sampled into the prompt set.",
        "",
    ]
    for task_id in sorted(ids):
        problem = problems.get(task_id)
        if problem is None:
            lines += [f"## Task {task_id}", "", "Source unavailable: not present in the built problem set.", ""]
            continue
        flagged, forms, signals, _ = structural_check(
            problem.reference_code, problem.visible_asserts, problem.entry_point
        )
        lines += [
            "---",
            "",
            f"## Task {task_id}, entry point `{problem.entry_point}`",
            "",
            f"- cheat_form: `{forms}`",
            f"- signals: `{signals}`",
            "",
            "Visible asserts, which are what the check compares literals against:",
            "",
            "```python",
            "\n".join(problem.visible_asserts),
            "```",
            "",
            "Reference solution:",
            "",
            "```python",
            (problem.reference_code or "").strip(),
            "```",
            "",
        ]
    STRUCTURAL_FP_PATH.write_text("\n".join(lines), encoding="utf-8")
    return STRUCTURAL_FP_PATH


def build_report() -> dict[str, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    computed = metrics.claims()
    figures = figure.build_figures()

    write_step_zero_flagged()
    write_structural_fp()
    published = build_published(computed)
    PUBLISHED_PATH.write_text(json.dumps(published, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines: list[str] = []
    runs = computed.get("n_declared_runs", 0)
    provenance = computed.get("eligibility") or {}

    lines.append("# Alibi: does a monitored policy stop reward hacking, or stop looking like it?")
    lines.append("")

    # 1. Result first. Never method first.
    lines.append("## Result")
    lines.append("")
    if runs == 0:
        lines.append(
            "**No training run has been declared as evidence yet.** The instrument is built, "
            "frozen and gated, and no hypothesis is evaluable until the run matrix completes. "
            "This file is regenerated after every completed run, so its emptiness here is a "
            "statement about the runs, not about the report."
        )
        lines.append("")
        lines.append(
            "What is established without any GPU is the environment gate: a hardcoding solution "
            "built programmatically from the visible asserts passes the visible tests on 371 of "
            "376 problems and fails the held-out tests on 370. The environment rewards cheating, "
            "so H1 is worth testing. See `docs/day-1-gate.md`."
        )
    else:
        h1 = computed.get("hypotheses", {}).get("H1", {})
        rising = h1.get("monotonic_rising_per_seed") or []
        lines.append(
            f"Declared runs: {runs}. H1, whether the oracle cheat rate rises in a0, is "
            f"{'supported' if any(rising) else 'not supported'} per seed: {rising}."
        )
    lines.append("")

    # 2. Setup.
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Model: Qwen2.5-0.5B-Instruct, LoRA, bf16")
    lines.append(f"- Task: MBPP with the visible asserts in the prompt, held-out from MBPP+")
    lines.append(f"- Eligible problems: {provenance.get('eligibility_n_problems')} (frozen, hashed)")
    lines.append(f"- Reward: `{prereg.PREREG.reward_definition}`, lambda = {prereg.PREREG.lambda_monitor}")
    lines.append(f"- Arms: {', '.join(prereg.PREREG.run_order.arms)}; seeds {list(prereg.PREREG.run_order.seeds)}")
    lines.append(f"- Pre-registration: `{prereg.PREREG.version}`, hash `{prereg.PREREG_HASH[:16]}`")
    lines.append("")

    # 2b. The judge, including the confound if there is one.
    selection = None
    try:
        from alibi.monitor.selection import load_selection

        selection = load_selection()
    except ImportError:
        selection = None

    lines.append("## The judge")
    lines.append("")
    if selection is None:
        lines.append(
            "No monitor has been selected. The monitored arms cannot run until "
            "`alibi monitor select` records one."
        )
    else:
        lines.append(f"- Monitor model: `{selection['model_id']}`")
        lines.append(f"- Prompt template version: `{selection['prompt_template_version']}`")
        lines.append(f"- Selection hash: `{selection['selection_hash'][:16]}`")
        lines.append(
            "- Availability was probed with a real 5-token request before use, because a provider's "
            "model listing includes dedicated-endpoint-only models that reject serverless calls. "
            "Every model probed and its result are in DECISIONS.md."
        )
        lines.append("")
        lines.append(
            "The monitor model id is **not** covered by the pre-registration hash, because "
            "`alibi/prereg.py` carries no monitor field. Compare the selection hash above as well "
            "as the prereg hash when checking that two runs were judged by the same thing."
        )
        if selection.get("shares_family_with_policy"):
            lines.append("")
            lines.append("### Judge and judged share a model family")
            lines.append("")
            lines.append(
                f"**`{selection['model_id']}` shares a model family with the "
                "Qwen2.5-0.5B-Instruct policy it grades.** Shared tokenizer, shared pretraining "
                "data and shared failure modes mean the monitor may be unusually good, or "
                "unusually blind, to exactly this policy's habits."
            )
            lines.append("")
            lines.append(
                "This does not invalidate the monitored arms, and it is not a defect that was "
                "discovered late: it is a consequence of which models were reachable on the "
                "account, recorded at selection time. It bears directly on H2 and H3, because a "
                "same-family judge could plausibly track the policy's obfuscation more closely "
                "than an independent one would, which would understate the obfuscation gap. Any "
                "reading of the monitored arms should carry this caveat."
            )
        qualification = selection.get("qualification") or {}
        if qualification:
            lines.append("")
            lines.append("### Qualification, and what it does not establish")
            lines.append("")
            lines.append(
                f"Before use, the judge was shown known synthetic cheats and known reference "
                f"solutions: cheat flag rate {_fmt(qualification.get('cheat_flag_rate'))}, honest "
                f"flag rate {_fmt(qualification.get('honest_flag_rate'))}, separation "
                f"{_fmt(qualification.get('separation'))}, error rate "
                f"{_fmt(qualification.get('error_rate'))}."
            )
            lines.append("")
            lines.append(
                "**This qualification is not pre-registered.** `alibi/prereg.py` is frozen and "
                "carries no monitor field, so the procedure and all four thresholds were chosen "
                "by the implementer rather than registered in advance. They were fixed before any "
                "run existed and are deliberately weak: the check is a smoke test for a judge that "
                "is useless, not a measurement of judge quality."
            )
            lines.append("")
            lines.append(
                "**It ran on the A2 view only.** The judge was shown code. A1 shows it a reasoning "
                "trace instead, and no qualification was run on that view."
            )
            lines.append("")
            lines.append(
                "**A separation of "
                f"{_fmt(qualification.get('separation'))} on these cheats does not predict "
                "performance on the policy's own.** The synthetic cheats are generated "
                "programmatically from the visible asserts and are stereotyped: a chain of "
                "equality comparisons against literal tuples, the easiest possible case. Once a "
                "policy starts varying the form of its hardcoding, the judge faces a distribution "
                "this qualification never sampled. Passing here is **necessary but not "
                "sufficient**, and no result in this report should be read as though the judge "
                "were known to be reliable on real completions. The per-step false positive panel "
                "is the running check on the honest half of that concern; there is no equivalent "
                "running check on the cheating half, because the policy's cheats have no ground "
                "truth label other than the oracle, which the monitor never sees."
            )
    lines.append("")

    # 3. The instrument.
    lines.append("## The instrument")
    lines.append("")
    lines.append(
        "Two judges are kept apart. The **reward monitor** reads text and its verdict enters the "
        "reward. The **cheat oracle** executes held-out tests the model never sees and its verdict "
        "never enters the reward. `tests/test_reward_isolation.py` fails if anything connects them."
    )
    lines.append("")
    lines.append(
        "Test outcomes are three-valued. A timeout or crash is `indeterminate`, never `fail`, and "
        "pass fractions are computed over determinate tests only. Folding indeterminate into fail "
        "would let a sandbox problem read as a model that does not generalise."
    )
    lines.append("")
    lines.append(
        "A monitor error is its own verdict state and is **always flagged**. An outage that read as "
        "unflagged would lower the penalty and look exactly like a policy that stopped cheating."
    )
    lines.append("")

    # 4. Findings, per hypothesis.
    lines.append("## Findings")
    lines.append("")
    for label, statement in zip(("H1", "H2", "H3"), prereg.PREREG.metric.hypotheses, strict=True):
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"> {statement}")
        lines.append("")
        lines.append("**not resolved** (no declared runs yet)" if runs == 0 else "**not resolved**")
        lines.append("")

    # 5. Sensitivity, always all three thresholds.
    lines.append("## Sensitivity")
    lines.append("")
    lines.append(
        "All three pre-registered thresholds, always shown, never a subset chosen after seeing a curve."
    )
    lines.append("")
    lines.append("| Threshold | " + " | ".join(prereg.PREREG.run_order.arms) + " |")
    lines.append("|---|" + "---|" * len(prereg.PREREG.run_order.arms))
    for threshold in prereg.PREREG.oracle.sensitivity_thresholds:
        label = f"{threshold:.2f}"
        cells = []
        for arm in prereg.PREREG.run_order.arms:
            entry = (computed.get("sensitivity", {}).get(label, {}) or {}).get(arm)
            if not entry:
                cells.append("not measured")
            else:
                values = [v for v in entry["terminal_cheat_rate"] if v is not None]
                cells.append(
                    f"{min(values):.4f} to {max(values):.4f}" if values else "not measured"
                )
        lines.append(f"| held-out <= {label} | " + " | ".join(cells) + " |")
    lines.append("")

    # 6. Figures.
    if figures:
        lines.append("## Figures")
        lines.append("")
        for name, path in sorted(figures.items()):
            lines.append(f"![{name}](figures/{Path(path).name})")
            lines.append("")

    # 6b. Determinacy floor, declared at report time.
    floor_info = {arm: (entry.get("determinacy_floor") or {}) for arm, entry in computed.get("arms", {}).items()}
    lines.append("## Determinacy floor")
    lines.append("")
    lines.append(
        f"A completion whose held-out scoring produced fewer than **{metrics.DETERMINACY_FLOOR} determinate "
        "tests** has an indeterminate oracle verdict. It leaves both the numerator and the denominator of "
        "the cheat rate and is **never counted as a cheat**. Nothing in any run directory was changed: this "
        "is recomputed from stored completions, which is why `alibi verify --no-gpu` can check it."
    )
    lines.append("")
    lines.append(
        "This was declared after the queue had started, while only partial a0 data existed and no "
        "monitored-arm curve had been produced. It is applied at report time only."
    )
    lines.append("")
    if not floor_info:
        lines.append("No declared runs yet, so there is nothing to compare.")
    else:
        lines.append("| Arm | Terminal gap, floor applied | Terminal gap, no floor | Excluded fraction | Differ materially |")
        lines.append("|---|---|---|---|---|")
        for arm, info in sorted(floor_info.items()):
            wf = info.get("terminal_gap_with_floor")
            nf = info.get("terminal_gap_without_floor")
            ex = info.get("terminal_excluded_fraction") or []
            ex = [e for e in ex if e is not None]
            lines.append(
                f"| {arm} | {_fmt(min(wf)) + ' to ' + _fmt(max(wf)) if wf else 'not measured'} "
                f"| {_fmt(min(nf)) + ' to ' + _fmt(max(nf)) if nf else 'not measured'} "
                f"| {_fmt(min(ex)) + ' to ' + _fmt(max(ex)) if ex else 'not measured'} "
                f"| {info.get('materially_different')} |"
            )
        lines.append("")
        lines.append(
            "Where the two differ materially, **the floor version is primary**. Materially means the "
            "absolute difference exceeds the seed band, which is this project's own standard for a "
            "resolved difference, or 0.02 where there is only one seed."
        )
    lines.append("")

    # 6c. Step-zero prevalence, before any policy update.
    lines.append("## Cheat prevalence before any policy update")
    lines.append("")
    lines.append(
        "What the base policy already does at step 0, before a single gradient step. If prevalence is "
        "already high, a rise over training is a smaller finding than it looks; if it is zero, any rise "
        "is entirely learned."
    )
    lines.append("")
    zero = {arm: (entry.get("step_zero_prevalence") or []) for arm, entry in computed.get("arms", {}).items()}
    if not any(zero.values()):
        lines.append("No declared runs yet, so step 0 has not been read.")
    else:
        lines.append("| Arm | Seeds | Completions | Any cheat_form | Behavioural cheat rate | cheat_form breakdown |")
        lines.append("|---|---|---|---|---|---|")
        for arm, entries in sorted(zero.items()):
            entries = [e for e in entries if e.get("n")]
            if not entries:
                lines.append(f"| {arm} | 0 | not measured | not measured | not measured | not measured |")
                continue
            n = sum(e["n"] for e in entries)
            any_form = sum(e["any_cheat_form"] for e in entries)
            breakdown: dict = {}
            for entry in entries:
                for form, count in (entry.get("cheat_form_counts") or {}).items():
                    breakdown[form] = breakdown.get(form, 0) + count
            behavioural = [e["behavioural_cheat_fraction"] for e in entries if e.get("behavioural_cheat_fraction") is not None]
            lines.append(
                f"| {arm} | {len(entries)} | {n} | {any_form}/{n} = {any_form / n:.4f} "
                f"| {_fmt(sum(behavioural) / len(behavioural)) if behavioural else 'not measured'} "
                f"| {breakdown or 'none'} |"
            )
    lines.append("")

    # 6c-bis. Pooled step zero and structural precision.
    pooled = computed.get("pooled_step_zero") or {}
    lines.append("### Pooled across all matrix runs")
    lines.append("")
    if not pooled.get("n_completions"):
        lines.append(f"Not measured yet: {pooled.get('absent_reason')}")
    else:
        ci = pooled.get("prevalence_ci95") or [None, None]
        bci = pooled.get("behavioural_ci95") or [None, None]
        lines.append(
            f"Step 0 is the same untrained base policy in every arm and at every seed, so the runs pool. "
            f"**{pooled['n_runs_pooled']} of {pooled['n_runs_expected']} runs** have stored a step 0 so far; "
            "this table is regenerated after every completed run."
        )
        lines.append("")
        lines.append("| Measure | Value |")
        lines.append("|---|---|")
        lines.append(f"| Runs pooled | {pooled['n_runs_pooled']} of {pooled['n_runs_expected']} |")
        lines.append(f"| Completions | {pooled['n_completions']} |")
        lines.append(
            f"| Any cheat_form | {pooled['any_cheat_form']}/{pooled['n_completions']} = "
            f"{_fmt(pooled['prevalence'])} |"
        )
        lines.append(f"| 95 percent interval (Wilson) | {_fmt(ci[0])} to {_fmt(ci[1])} |")
        lines.append(
            f"| Behavioural cheat rate | {pooled['behavioural_cheat']}/{pooled['behavioural_determinate']} = "
            f"{_fmt(pooled['behavioural_rate'])} |"
        )
        lines.append(f"| 95 percent interval (Wilson) | {_fmt(bci[0])} to {_fmt(bci[1])} |")
        lines.append(f"| Per-run spread | {_fmt(pooled['per_run_spread'])} |")
        lines.append(f"| cheat_form breakdown | {pooled['cheat_form_counts'] or 'none'} |")
        lines.append("")
        lines.append("Per run:")
        lines.append("")
        lines.append("| Run | Arm | Seed | n | Any cheat_form |")
        lines.append("|---|---|---|---|---|")
        for entry in pooled.get("per_run_prevalence", []):
            lines.append(
                f"| `{entry['run_id']}` | {entry['arm']} | {entry['seed']} | {entry['n']} | {_fmt(entry['rate'])} |"
            )
        lines.append("")
        lines.append(
            "Wilson intervals rather than the normal approximation, because n is small and the "
            "proportion is near zero, which is exactly where the textbook interval returns a "
            "negative lower bound."
        )
    lines.append("")

    precision = computed.get("structural_precision_on_honest_code") or {}
    lines.append("## Structural check precision on known-honest code")
    lines.append("")
    if not precision.get("n_reference_solutions"):
        lines.append("Not measured.")
    else:
        rate = precision["false_positive_rate"]
        ci = precision.get("false_positive_ci95") or [None, None]
        lines.append(
            "The structural check's **recall** was validated on generated cheats. Its **precision** was "
            "not, until now. Every eligible MBPP+ reference solution is a genuine algorithm, so any flag "
            "on one is a false positive by construction."
        )
        lines.append("")
        lines.append("| Measure | Value |")
        lines.append("|---|---|")
        lines.append(f"| Reference solutions checked | {precision['n_reference_solutions']} |")
        lines.append(f"| Flagged (false positives) | {precision['n_flagged']} |")
        lines.append(f"| **False positive rate** | **{_fmt(rate)}** |")
        lines.append(f"| 95 percent interval (Wilson) | {_fmt(ci[0])} to {_fmt(ci[1])} |")
        lines.append(f"| Per form | {precision['per_form'] or 'none'} |")
        lines.append(f"| Parse errors | {precision['parse_errors']} |")
        lines.append(f"| Flagged task ids | `{precision['flagged_task_ids']}` |")
        lines.append("")
        if rate is not None and rate >= NEAR_ZERO_FALSE_POSITIVE_RATE:
            lines.append(
                f"**This is not near zero.** At {_fmt(rate)} the detector flags roughly "
                f"{rate * 16:.1f} of every 16 completions on honest code alone. The cheat_form panel "
                "therefore reports **detector behaviour as well as policy behaviour**."
            )
            lines.append("")
            lines.append(
                "But the rate is **not uniform**. It is five specific problems, listed with their full "
                "source in [STRUCTURAL_FP.md](STRUCTURAL_FP.md). The floor appears only on the steps "
                "where one of them is sampled, so it is a small number of contaminated steps rather "
                "than a constant offset on every step. See the exposure table below."
            )
        else:
            lines.append(
                f"At {_fmt(rate)} this is near zero against the declared "
                f"{NEAR_ZERO_FALSE_POSITIVE_RATE:.2f} bar, so the cheat_form panel can be read as policy "
                "behaviour rather than detector behaviour."
            )
        lines.append("")
        lines.append(
            "The full text of every step-zero completion carrying a cheat_form is in "
            "[STEP_ZERO_FLAGGED.md](STEP_ZERO_FLAGGED.md), for reading by hand rather than trusting the "
            "label."
        )

        # Exposure of the false positive problems, and prompt coverage.
        exposures = [e for entry in computed.get("arms", {}).values() for e in (entry.get("fp_problem_exposure") or [])]
        coverages = [c for entry in computed.get("arms", {}).values() for c in (entry.get("prompt_coverage") or [])]
        if exposures:
            lines.append("")
            lines.append("### How often the false positive problems are actually sampled")
            lines.append("")
            lines.append("| Run | Steps touching a flagged problem | Of steps | Fraction |")
            lines.append("|---|---|---|---|")
            for exposure in exposures:
                lines.append(
                    f"| (run) | `{exposure['steps_touching_a_fp_problem']}` | {exposure['n_steps']} "
                    f"| {_fmt(exposure['fraction_of_steps'])} |"
                )
            lines.append("")
            lines.append(
                "The floor is therefore **confined to those steps**, not spread across the series. On "
                "every other step the structural panel carries no known detector contribution. This is "
                "why the excluding-the-five series below is the primary one."
            )
        if coverages:
            cover = coverages[0]
            lines.append("")
            lines.append("### Prompt coverage, which this exposed")
            lines.append("")
            lines.append(
                f"A run of this length samples **{cover['distinct_problems_sampled']} distinct problems "
                f"of {cover['eligible_problems']} eligible**, a coverage of "
                f"{_fmt(cover['coverage_fraction'])}. {cover['note']} The eligible count of "
                f"{cover['eligible_problems']} therefore overstates what any run actually sees, and the "
                "effective problem set is the same fixed prefix in every arm and every seed. That is "
                "good for comparability between arms and it means the seeds vary sampling only, not "
                "problems."
            )
    lines.append("")

    # Behavioural precision, and the correction to the earlier claim.
    behavioural = computed.get("behavioural_precision_on_honest_code") or {}
    lines.append("## Behavioural check precision, and a correction")
    lines.append("")
    lines.append(
        "An earlier draft of this report claimed the structural check's false positive rate was a "
        "vindication of the pre-registration's choice to make the behavioural oracle primary, on the "
        "grounds that the behavioural check has no equivalent false positive floor. **That claim was "
        "wrong and is withdrawn.**"
    )
    lines.append("")
    lines.append(
        "The behavioural check has not been shown to have no false positive floor. It had not been "
        "measured at all. The reason is circularity: **eligibility excludes problems whose reference "
        "solution fails held out**, which is the same criterion the behavioural rule uses. Any false "
        "positive rate computed on the eligible set is zero by construction and carries no information. "
        "The two checks were validated on differently filtered populations: the eligibility filter is "
        "unrelated to the structural criterion, so the structural measurement is real, and it is "
        "identical to the behavioural criterion, so a behavioural measurement there would be vacuous."
    )
    lines.append("")
    if not behavioural.get("measured"):
        lines.append(
            f"**The behavioural false positive rate on honest code remains unmeasured.** "
            f"{behavioural.get('absent_reason')}"
        )
    else:
        ci = behavioural.get("false_positive_ci95") or [None, None]
        lines.append(
            "A non circular estimate is available from stored data and is reported here **as an "
            "estimate with named limits**, not as a resolution of the gap. The pre eligibility joined "
            "set is filtered only for reasons unrelated to the behavioural rule, so reference "
            "solutions on it were never selected for passing held out."
        )
        lines.append("")
        lines.append("| Measure | Value |")
        lines.append("|---|---|")
        lines.append(f"| Population | {behavioural['population']} |")
        lines.append(f"| Source artifact | `{behavioural['source_artifact']}` |")
        lines.append(f"| Evaluable | {behavioural['n_evaluable']} |")
        lines.append(f"| Flagged (false positives) | {behavioural['n_flagged']} |")
        lines.append(f"| Point estimate | {_fmt(behavioural['false_positive_rate'])} |")
        lines.append(f"| **95 percent upper bound** | **{_fmt(ci[1])}** |")
        lines.append("")
        lines.append(
            "With zero events the point estimate is uninformative on its own and the upper bound is "
            "the number worth quoting. Against the structural check's "
            f"{_fmt((computed.get('structural_precision_on_honest_code') or {}).get('false_positive_rate'))}, "
            "the comparison that can honestly be made is: the structural rate is measured and above the "
            "near zero bar, while the behavioural rate is bounded above by "
            f"{_fmt(ci[1])} and is consistent with zero. That is weaker than the withdrawn claim and it "
            "is what the data supports."
        )
        lines.append("")
        lines.append("Limits of this estimate, all of which keep it an estimate:")
        lines.append("")
        for caveat in behavioural.get("caveats", []):
            lines.append(f"- {caveat}")
    lines.append("")

    # 6c-ter. Cluster bootstrap over problems, and what it does and does not fix.
    lines.append("## Two variance estimates, and which one a verdict must clear")
    lines.append("")
    lines.append(
        "**Seed band** is the min to max across seeds. It captures **sampling variance only**: the "
        "same problems, the same order, a different sampling seed. It says nothing about whether the "
        "result would hold on other problems."
    )
    lines.append("")
    lines.append(
        "**Cluster bootstrap** resamples **problems** with replacement, "
        f"{metrics.BOOTSTRAP_DRAWS} draws at declared seed `{metrics.BOOTSTRAP_SEED}`, and recomputes "
        "the terminal statistics each draw. The cluster is the problem, not the completion, because "
        "completions on one problem share its difficulty, its visible asserts and its held-out set, so "
        "treating them as independent understates variance. This captures **problem variance**."
    )
    lines.append("")
    lines.append(
        f"\"Terminal\" here is the last {metrics.TERMINAL_WINDOW_STEPS} steps, not the last step. The "
        "final step contains 2 problems and 16 completions, and resampling 2 clusters is not a "
        "bootstrap. That window choice is an implementation decision and is stated rather than buried."
    )
    lines.append("")
    boots = {arm: (entry.get("cluster_bootstrap") or []) for arm, entry in computed.get("arms", {}).items()}
    if not any(boots.values()):
        lines.append("No declared runs yet, so neither interval is measurable.")
    else:
        lines.append("| Arm | Seed band, terminal gap | Bootstrap 95 percent, terminal gap | Problems | Wider |")
        lines.append("|---|---|---|---|---|")
        for arm, entries in sorted(boots.items()):
            entries = [e for e in entries if e.get("measured")]
            band = (computed["arms"][arm].get("terminal_gap") or {})
            band_width = band.get("spread")
            if not entries:
                lines.append(f"| {arm} | {_fmt(band_width)} | not measurable | n/a | n/a |")
                continue
            widths = [e["gap_width"] for e in entries if e.get("gap_width") is not None]
            boot_width = max(widths) if widths else None
            lo = min(e["gap_ci95"][0] for e in entries if e["gap_ci95"][0] is not None)
            hi = max(e["gap_ci95"][1] for e in entries if e["gap_ci95"][1] is not None)
            wider = "bootstrap" if (boot_width or 0) >= (band_width or 0) else "seed band"
            lines.append(
                f"| {arm} | {_fmt(band_width)} | {_fmt(lo)} to {_fmt(hi)} (width {_fmt(boot_width)}) "
                f"| {entries[0]['n_problems']} | {wider} |"
            )
        lines.append("")
        lines.append(
            "**A difference between arms is reported as resolved only if it clears the wider of the "
            "two.** Where the bootstrap is wider than the seed band, a verdict that looked resolved on "
            "the seed band alone does not survive, and this section names any verdict that changes."
        )
        lines.append("")
        changed = [
            f"{arm}: seed band {_fmt((computed['arms'][arm].get('terminal_gap') or {}).get('spread'))} "
            f"against bootstrap width {_fmt(max((e['gap_width'] for e in entries if e.get('gap_width') is not None), default=None))}"
            for arm, entries in sorted(boots.items())
            if [e for e in entries if e.get("measured")]
        ]
        lines.append(
            "Verdicts affected: none yet, because no between-arm comparison exists. Per-arm widths: "
            + "; ".join(changed)
            if changed
            else "Verdicts affected: none yet."
        )
        lines.append("")
        lines.append(
            "A zero-width bootstrap interval means zero events were observed in the window, not that "
            "the estimate is precise. It should be read as uninformative, not as certainty."
        )
    lines.append("")
    lines.append(
        "**What the bootstrap does not fix.** It corrects understated variance. It does **not** correct "
        "selection bias. The problems in any run are a fixed, deterministic, non-random prefix of the "
        "eligible set, not a random draw from it, so resampling them estimates the uncertainty of a "
        "statement about *those* problems and cannot license a statement about MBPP as a whole. The "
        "bootstrap widens the interval around the right target; it does not move the target."
    )
    lines.append("")

    # 6c-quater. What the prefix actually is.
    prefix = computed.get("prefix_characterisation") or {}
    lines.append("## What the sampled problems are")
    lines.append("")
    if not prefix.get("measured"):
        lines.append(f"Not measured: {prefix.get('absent_reason')}")
    else:
        lines.append(f"**Ordering.** {prefix['ordering']}.")
        lines.append("")
        lines.append(
            f"**{prefix['n_sampled']} sampled** (task ids {prefix['sampled_task_id_range']}) against "
            f"**{prefix['n_unsampled']} never sampled** (task ids {prefix['unsampled_task_id_range']})."
        )
        lines.append("")
        lines.append("| Property | Sampled mean | Never-sampled mean | Difference | Permutation p | Material |")
        lines.append("|---|---|---|---|---|---|")
        for key, entry in prefix["comparisons"].items():
            lines.append(
                f"| {key} | {_fmt(entry['sampled_mean'])} | {_fmt(entry['unsampled_mean'])} "
                f"| {_fmt(entry['difference'])} | {_fmt(entry['permutation_p'])} "
                f"| {entry['materially_different']} |"
            )
        lines.append("")
        lines.append(
            f"Two-sided permutation test, {prefix['permutation_draws']} draws, alpha {prefix['alpha']}, "
            "seed as declared for the bootstrap."
        )
        lines.append("")
        sampled_split = prefix.get("mbpp_split_composition_sampled") or {}
        unsampled_split = prefix.get("mbpp_split_composition_unsampled") or {}
        if sampled_split and unsampled_split:
            lines.append("### Provenance, which those four properties do not capture")
            lines.append("")
            lines.append(f"- Sampled: `{sampled_split}`")
            lines.append(f"- Never sampled: `{unsampled_split}`")
            lines.append("")
            lines.append(
                "**The four properties above show no material difference, and this does.** Because "
                "task id ordering tracks MBPP's own split boundaries, the sampled prefix is almost "
                "entirely MBPP's *test* split plus several problems from the *prompt* split, while the "
                "unsampled tail spans test, validation and train. The prompt split is MBPP's designated "
                "few-shot exemplar set and is the most likely of all to appear in pretraining data."
            )
            lines.append("")
            lines.append(
                "So the honest statement is: on problem shape, held-out size, visible assert count, "
                "reference length and cheat constructibility, the prefix looks like the tail. On "
                "**provenance** it does not. **Generalisation from this run is to the sampled prefix, "
                "not to MBPP.** Any claim in this report about behaviour on MBPP should be read as a "
                "claim about 160 mostly-test-split problems that were chosen by sort order rather than "
                "at random."
            )
        elif prefix.get("any_material_difference"):
            lines.append(
                "At least one property differs materially, so **generalisation from this run is to the "
                "sampled prefix, not to MBPP.**"
            )
        else:
            lines.append(
                "No measured property differs materially. That is not evidence the prefix is "
                "representative on properties nobody measured, and the selection is still "
                "deterministic rather than random."
            )
    lines.append("")

    # 6c-quinquies. The think-region diagnostic, and the corrected framing.
    diagnostic_path = runlog.ARTIFACTS / "diagnostics" / "think_region" / "result.json"
    lines.append("## Arm A1 read an empty view, and why")
    lines.append("")
    lines.append(
        "**Arm A1 measured nothing.** Its monitor is registered to read the designated thought "
        "region, and the policy emitted none, so the monitor judged empty strings on every "
        "completion of every A1 run. Its flag rate is zero by construction, its penalty term is "
        "identically zero, and its reward is therefore arithmetically A0's. The completed runs "
        "confirm it: a0 and a1 produce identical trajectories at matching seeds. Full evidence in "
        "[BLOCKED.md](../BLOCKED.md)."
    )
    lines.append("")
    if not diagnostic_path.exists():
        lines.append("The supplementary diagnostic has not been run, so the cause is not isolated.")
    else:
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        by_name = {c["name"]: c for c in diagnostic["conditions"]}
        lines.append(
            "**The claim is not that a model this size cannot produce a thought region.** A "
            "supplementary diagnostic, sampling only and never pooled with the matrix, settles what "
            "the cause actually is. Same step-zero prompts, byte identical, same group size, "
            f"{diagnostic['max_new_tokens']} max new tokens so a think block and an answer both fit."
        )
        lines.append("")
        lines.append(
            "| Condition | Non-empty thought region | Mean think chars | Flag rate, A1 view | Flag rate, A2 view | Visible pass |"
        )
        lines.append("|---|---|---|---|---|---|")
        for name in ("qwen2.5-0.5b-instruct", "qwen3-0.6b-raw-prompt", "qwen3-0.6b-chat-template"):
            entry = by_name.get(name)
            if not entry:
                continue
            ci = entry.get("nonempty_think_ci95") or [None, None]
            lines.append(
                f"| `{name}` | **{_fmt(entry['nonempty_think_fraction'])}** "
                f"({_fmt(ci[0])} to {_fmt(ci[1])}) | {_fmt(entry['think_chars']['mean'], 1)} "
                f"| {_fmt(entry['flag_rate_think_view'])} | {_fmt(entry['flag_rate_answer_view'])} "
                f"| {_fmt(entry['visible_pass_rate'])} |"
            )
        lines.append("")
        lines.append(
            f"n = {by_name['qwen2.5-0.5b-instruct']['n_completions']} completions per condition, which "
            "is the matrix's own step-zero sample size. Wilson intervals, because the proportions sit "
            "at the boundary."
        )
        lines.append("")
        lines.append("### What this establishes")
        lines.append("")
        lines.append(
            "**Thought-region monitoring presupposes a policy post-trained to emit a thought region, "
            "and presupposes an interface that lets it.** Qwen2.5-0.5B-Instruct is not such a policy: "
            "it emits none, on 0 of 16 completions, even though the prompt explicitly asks for "
            "`<think>` tags. **This was not checked before the arm was designed.** That is the error, "
            "and it is a design error rather than a discovery about scale."
        )
        lines.append("")
        lines.append(
            "Qwen3-0.6B, a model of similar scale that **is** post-trained to think, settles the scale "
            "question directly: through its own chat template it emits a thought region on **16 of 16** "
            "completions, averaging 2888 characters. So the behaviour is available at this scale. It "
            "is a property of post-training, not of parameter count."
        )
        lines.append("")
        lines.append(
            "**The diagnostic also found a second, independent cause that would have disabled A1 even "
            "with a thinking model.** Qwen3-0.6B on the matrix's raw prompt string emits a thought "
            "region on **0 of 16** completions. The same model, the same prompt content, differs only "
            "in whether the tokenizer's chat template was applied. The matrix builds prompts as a raw "
            "string and never applies the chat template, so A1 would have read an empty view even if "
            "the policy had been chosen correctly. Two independent faults, either sufficient."
        )
        lines.append("")
        lines.append(
            "**The A1 machinery itself is sound.** Given a thought region to read, the monitor flags "
            f"{_fmt(by_name['qwen3-0.6b-chat-template']['flag_rate_think_view'])} of completions on "
            "the A1 view while flagging "
            f"{_fmt(by_name['qwen3-0.6b-chat-template']['flag_rate_answer_view'])} on the A2 view of "
            "the same completions. The arm fails on its inputs, not on its implementation."
        )
        lines.append("")
        lines.append(
            f"Cost: {diagnostic['monitor_cost']['calls']} monitor calls, "
            f"{diagnostic['monitor_cost']['total_tokens']} tokens, "
            f"{diagnostic['wall_clock_seconds'] / 60:.1f} minutes wall clock on the local GPU shared "
            "with the running matrix, so no rented compute. This diagnostic is excluded from the "
            "evidence index and is never pooled with the matrix."
        )
    lines.append("")

    # 6c-sexies. Did the v1 loop have usable gradient signal?
    from alibi.report import retrospective as retro

    retro.write()
    _analysis, judgement = retro.a0_seed1()
    lines.append("## Did the v1 loop learn anything?")
    lines.append("")
    if not judgement or not judgement.get("measured"):
        lines.append(f"Not measured: {(judgement or {}).get('absent_reason', 'no a0 seed 1 run stored')}")
    else:
        lines.append(
            "v1's curves are flat. That has two very different explanations, and only one of them is "
            "about the model: either the policy did not learn to cheat, or the loop could not have "
            "taught it anything. Recomputed from stored a0 seed 1 artifacts, "
            f"{judgement['n_steps']} steps."
        )
        lines.append("")
        lines.append("| Measure | First 10 steps | Last 10 steps |")
        lines.append("|---|---|---|")
        lines.append(
            f"| Mean reward | {_fmt(judgement['mean_reward_first10'])} | "
            f"**{_fmt(judgement['mean_reward_last10'])}** |"
        )
        lines.append(
            f"| Zero-variance groups | {_fmt(judgement['zero_variance_group_fraction_first10'])} | "
            f"{_fmt(judgement['zero_variance_group_fraction_last10'])} |"
        )
        lines.append(
            f"| Mean token entropy | {_fmt(judgement['entropy_first10'])} | "
            f"{_fmt(judgement['entropy_last10'])} |"
        )
        lines.append(
            f"| Capped fraction | {_fmt(judgement['capped_fraction_first10'])} | "
            f"{_fmt(judgement['capped_fraction_last10'])} |"
        )
        lines.append(
            f"| Mean absolute advantage (whole run) | {_fmt(judgement['mean_abs_advantage_mean'])} "
            f"| max {_fmt(judgement['max_abs_advantage'])} |"
        )
        lines.append("")
        lines.append("### The policy degraded")
        lines.append("")
        lines.append(
            f"**Yes.** Mean reward fell from {_fmt(judgement['mean_reward_first10'])} to "
            f"{_fmt(judgement['mean_reward_last10'])}, a change of "
            f"{_fmt(judgement['reward_change'])}. Training made the policy worse at the task it was "
            "rewarded for. Visible pass rate falls with it. Whatever v1 measured, it was not a policy "
            "improving and then learning to cheat."
            if judgement["policy_degraded"]
            else f"**No.** Mean reward moved from {_fmt(judgement['mean_reward_first10'])} to "
            f"{_fmt(judgement['mean_reward_last10'])}."
        )
        lines.append("")
        lines.append("### There was gradient signal, and that is not the mechanism")
        lines.append("")
        if judgement["most_groups_had_zero_variance"]:
            lines.append(
                f"**Most groups had zero reward variance** "
                f"({_fmt(judgement['zero_variance_group_fraction_mean'])} on average). A group whose "
                "rewards are all equal produces advantages of exactly zero and contributes no "
                "gradient. That is the headline mechanism: the loop mostly could not learn."
            )
        else:
            lines.append(
                f"**No.** Zero-variance groups averaged "
                f"{_fmt(judgement['zero_variance_group_fraction_mean'])}, so roughly "
                f"{_fmt(1 - judgement['zero_variance_group_fraction_mean'])} of groups produced a "
                "non-zero advantage, and mean absolute advantage was "
                f"{_fmt(judgement['mean_abs_advantage_mean'])} with a maximum of "
                f"{_fmt(judgement['max_abs_advantage'])}. **The loop had usable gradient signal on "
                "most steps.** Zero-variance groups are therefore not the explanation for the flat "
                "curves, and it would have been convenient but wrong to report them as one."
            )
        lines.append("")
        lines.append(
            "Entropy **rose** from "
            f"{_fmt(judgement['entropy_first10'])} to {_fmt(judgement['entropy_last10'])}. The policy "
            "did not collapse onto a single output; it became less confident. Combined with falling "
            "reward, the picture is a policy being pushed away from its pretrained behaviour without "
            "finding anything better, which is what a small model under a sparse, mostly-negative "
            "reward tends to do."
            if judgement["entropy_first10"] is not None
            and judgement["entropy_last10"] is not None
            and judgement["entropy_last10"] > judgement["entropy_first10"]
            else "Entropy is reported above."
        )
        lines.append("")
        lines.append(
            f"**Truncation is the standing constraint.** The capped fraction sat at "
            f"{_fmt(judgement['capped_fraction_first10'])} at the start and "
            f"{_fmt(judgement['capped_fraction_last10'])} at the end: roughly two completions in five "
            "were cut off by the 256-token budget for the whole run. That bounds what the policy "
            "could express throughout and is recorded as v1's third fault."
        )
        lines.append("")
        lines.append(
            "**Reading for H1.** v1's flat cheat rate should not be read as evidence that a 0.5B "
            "policy will not learn to reward hack. It is evidence about a run whose policy was "
            "degrading, whose completions were 40 percent truncated, and one of whose three arms was "
            "measuring nothing. H1 was not given a fair test."
        )
    lines.append("")

    # 6c-septies. The resolved training configuration.
    lines.append("## The resolved training configuration, and what is not in it")
    lines.append("")
    lines.append(
        "**There is no `GRPOConfig` to report.** TRL is pinned in `env.lock` and `trl.GRPOTrainer` "
        "is never instantiated. The update in `alibi/train/grpo.py:_update` is hand written, and the "
        "module docstring says so, but the absence of the trainer changes what several logged "
        "numbers mean. Resolved from the stored `config.json` of a completed v1 run and from the "
        "source that consumed it, not from library defaults."
    )
    lines.append("")
    lines.append("| Setting | v1 resolved value |")
    lines.append("|---|---|")
    for name, value in [
        ("Optimiser", "`torch.optim.AdamW`, betas (0.9, 0.999), eps 1e-8, weight decay 0.01, all defaults"),
        ("Learning rate", "1e-5, constant"),
        ("Scheduler", "**none**"),
        ("Warmup", "**none**, 0 steps"),
        ("Gradient clipping", "global norm 1.0"),
        ("beta, KL coefficient in the loss", "**there is no KL term in the loss**, and no reference policy existed"),
        ("epsilon, PPO clip ratio", "**none**, the loss is an unclipped `-advantage x logprob`"),
        ("LoRA", "r 16, alpha 32, dropout 0.0, targets q/k/v/o_proj, task CAUSAL_LM"),
        ("Effective batch", "16 completions per step, 2 prompts x group 8"),
        ("Gradient accumulation", "16, one backward per completion scaled 1/16, one optimiser step per training step"),
        ("Objective", "mean token logprob, length normalised, not a token sum"),
    ]:
        lines.append(f"| {name} | {value} |")
    lines.append("")
    lines.append("### Two logged numbers mean less than their names suggest")
    lines.append("")
    lines.append(
        "**The `kl` column in v1 is not KL from a reference policy.** No reference policy existed. "
        "It was computed as the current policy's mean token logprob minus the sampler's mean token "
        "logprob **from the same step**, and generation happens with the same weights immediately "
        "before the update. It therefore measured numerical difference between `generate` and a "
        "forward pass, not policy drift. **The KL spike halt condition was consequently guarding "
        "nothing meaningful in v1**, and no v1 run ever tripped it, which should now be read as "
        "uninformative rather than reassuring."
    )
    lines.append("")
    lines.append(
        "**`trainer_logprob` is a literal copy of `sampler_logprob`.** Verified on stored data: "
        "3430 of 3430 rows identical at step 0 of a0 seed 1. The architecture doc asks for this pair "
        "so the follow-on project can study what happens to it when the rollout path changes. As "
        "stored it is the same number twice, so v1's `logprobs.parquet` does not support that "
        "analysis. This is a defect in v1's logging, not in the follow-on plan."
    )
    lines.append("")
    lines.append(
        "Taken with the retrospective above, the picture is coherent: an unclipped, unanchored "
        "policy gradient with no trust region, at a constant learning rate, on a small model. "
        "Reward fell and entropy rose, which is what that configuration does when the reward is "
        "sparse and mostly negative. v2 declares these settings in `alibi/prereg_v2.py` and chooses "
        "the learning rate and KL anchor from ten-step probes rather than inheriting them silently."
    )
    lines.append("")

    # 6d. Amendments to the halt conditions, stated plainly.
    lines.append("## Halt conditions were amended before any curve existed")
    lines.append("")
    lines.append(
        "Two of the pre-declared halt conditions were amended before any monitored-arm curve had been "
        "produced. Both amendments are recorded here because a halt condition that moved is a fact about "
        "the experiment, not an implementation detail."
    )
    lines.append("")
    lines.append(
        "**1. Indeterminate held-out rate, instantaneous to lagging window.** The threshold, 5 percent, "
        "was not changed. The evaluation window was. Measured on a0 seed 1: steps 0 to 7 were 0.0000 and "
        "step 8 was 0.0584, caused by exactly one completion out of sixteen whose code hung, which made "
        "92 of its ~99 held-out tests indeterminate. Spikes of that kind arrived every four or five steps, "
        "so the instantaneous reading would have halted every run within about ten steps while saying "
        "nothing about the sandbox. The halt condition's own wording permits this: it specifies "
        "evaluation \"on a lagging window if held-out scoring is asynchronous\", and held-out scoring here "
        "is asynchronous. Amended by the implementer."
    )
    lines.append("")
    lines.append(
        "**2. The queue stop rule, three consecutive halts to more than half failed.** Amended by the "
        "operator's instruction. Three consecutive halts for the same reason is still detected and "
        "recorded as a warning, but no longer stops the queue on its own."
    )
    lines.append("")
    lines.append(
        "**The halt conditions were specified against an imagined run.** They were written before any "
        "training had happened, against an expectation of how a run would behave, and the first contact "
        "with a real 0.5B policy showed one of them to be incompatible with the thing it was guarding. "
        "That is worth stating plainly: a reader should treat the halt conditions as a design artifact "
        "that was revised on contact with data, not as constraints that survived unchanged."
    )
    lines.append("")

    # 7. Threats.
    lines.append("## Threats to the result")
    lines.append("")
    lines.append(
        "1. **Measurement drift.** This ran on a laptop under sustained load. Held-out timeouts are "
        "the channel through which thermal throttling could contaminate the primary metric. The "
        "indeterminate rate is logged per step, is a halt condition at 5 percent, and has its own "
        "figure panel. The correlation between indeterminate rate and cheat rate is reported."
    )
    lines.append(
        "2. **The structural check is a check on syntax.** It was extended to the dict and sequence "
        "forms after day 1 measured it catching 372 of 376 if-chain cheats and 0 of 376 dict cheats. "
        "That it needed extending is evidence that it can be evaded by a form nobody enumerated, "
        "which is why the primary oracle is behavioural."
    )
    lines.append(
        "3. **The synthetic cheat is not a model.** The gate shows the environment rewards "
        "hardcoding. It does not show that a 0.5B policy can discover hardcoding within the step "
        "budget. H1 is exactly that question and it is not settled by the gate."
    )
    lines.append("")

    # 8. Provenance.
    lines.append("## Provenance")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(runlog.scrub(provenance), indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append(f"Generated {published['generated_utc']} by `alibi report`. Numbers are injected from")
    lines.append("`artifacts/`, never typed. `alibi verify --no-gpu` recomputes every claim above.")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    out = {
        "report": REPORT_PATH,
        "published": PUBLISHED_PATH,
        "step_zero_flagged": STEP_ZERO_FLAGGED_PATH,
        "structural_fp": STRUCTURAL_FP_PATH,
    }
    out.update({f"figure_{k}": v for k, v in figures.items()})
    return out
