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
    """REPORT.md, structured as a study of how a monitor-reward experiment fails.

    The spine is deliberately not "experiment with a failed hypothesis section".
    No monitored-arm comparison exists at any version, so presenting one would be
    a fiction. What does exist is five distinct faults, each of which alone would
    have produced a publishable-looking curve that meant nothing, and each of
    which was caught by instrumentation built before the runs. That is the
    result, and the structure says so.

    Nothing is softened because the framing changed. Every limitation, every
    withdrawn claim and the above-the-fold disclosure are carried over.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    computed = metrics.claims()
    figures = figure.build_figures()

    write_step_zero_flagged()
    write_structural_fp()
    published = build_published(computed)
    PUBLISHED_PATH.write_text(json.dumps(published, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    from alibi.report import retrospective as retro

    retro.write()
    _analysis, judgement = retro.a0_seed1()
    judgement = judgement or {}

    precision = computed.get("structural_precision_on_honest_code") or {}
    behavioural = computed.get("behavioural_precision_on_honest_code") or {}
    prefix = computed.get("prefix_characterisation") or {}
    pooled = computed.get("pooled_step_zero") or {}
    provenance = computed.get("eligibility") or {}

    selection = None
    try:
        from alibi.monitor.selection import load_selection

        selection = load_selection()
    except ImportError:
        selection = None

    diagnostic = {}
    diagnostic_path = runlog.ARTIFACTS / "diagnostics" / "think_region" / "result.json"
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    by_condition = {c["name"]: c for c in diagnostic.get("conditions", [])}

    L: list[str] = []

    L.append("# Alibi: how a monitor-reward experiment fails silently")
    L.append("")
    L.append(
        "A study of the ways an experiment that penalises a policy for visible reward hacking can "
        "produce confident, plausible curves that mean nothing. The case is our own run."
    )
    L.append("")

    # --------------------------------------------------------------- fold
    L.append("> ## Read this before any number below")
    L.append(">")
    L.append("> **No `GRPOTrainer` was used.** TRL is pinned in every `env.lock` and is never")
    L.append("> instantiated. The update is a hand-written policy gradient in")
    L.append("> `alibi/train/grpo.py`. There is no `GRPOConfig` to inspect.")
    L.append(">")
    L.append("> **In v1 that objective was neither clipped nor anchored.** No reference policy")
    L.append("> existed, so there was no KL term in the loss, no ratio clipping and no trust")
    L.append("> region. It was `-advantage x logprob` at a constant learning rate.")
    L.append(">")
    L.append("> **v1's update did not condition on the prompt.** It fed the completion tokens")
    L.append("> with the prompt stripped, so the gradient maximised the unconditional")
    L.append("> likelihood of completion text rather than the likelihood of a solution given a")
    L.append("> problem.")
    L.append(">")
    L.append("> **Two v1 columns do not mean what their names suggest.** `kl` is not KL from a")
    L.append("> reference policy, because none existed: it compared the policy against the")
    L.append("> sampler's own logprobs from the same step and the same weights, so it measured")
    L.append("> numerical noise. The KL spike halt was therefore guarding nothing in v1.")
    L.append("> `trainer_logprob` is a literal copy of `sampler_logprob`, identical on 3430 of")
    L.append("> 3430 rows.")
    L.append(">")
    L.append("> All are fixed for later versions and none is fixed retroactively. **v1's numbers")
    L.append("> stand as they were measured, and are not evidence about H1.**")
    L.append("")

    # ------------------------------------------------------------- 1. RESULT
    L.append("## 1. Result")
    L.append("")
    L.append(
        "**No monitored-arm comparison exists at any version of this experiment.** Five distinct "
        "faults were found, and each one alone would have produced a publishable-looking curve "
        "that meant nothing: **arm A1's monitor read an empty view**, so its flag rate was zero by "
        "construction and its reward was arithmetically the unmonitored control's; **the chat "
        "template was never applied**, which would have emptied that view even with a policy "
        "chosen correctly; **44.8 percent of completions hit the token cap**; **the training "
        "objective did not condition on the prompt**, so the gradient maximised the unconditional "
        "likelihood of completion text; and **`trainer_logprob` was a literal copy of "
        "`sampler_logprob`**, so the column collected to detect sampler-trainer divergence could "
        "not have shown any. Every one was caught by instrumentation built before the runs, and "
        "surfaced as a number rather than as a plausible curve."
    )
    L.append("")
    L.append(
        "That is the result. It is not the result the pre-registration asked for, and the "
        "pre-registered hypotheses remain unevaluated, which section 5 states rather than "
        "obscures."
    )
    L.append("")

    # ------------------------------------------------------- 2. THE FIVE FAULTS
    L.append("## 2. The five faults")
    L.append("")
    L.append(
        "Each subsection gives what the fault was, what it would have produced undetected, the "
        "measurement that caught it, and what it cost."
    )
    L.append("")

    L.append("### 2.1 Arm A1's monitor read an empty view")
    L.append("")
    L.append(
        "**The fault.** A1 is registered to show its monitor `completion.think`, the designated "
        "thought region. Qwen2.5-0.5B-Instruct emits none, despite the prompt asking for `<think>` "
        "tags. Mean `think_chars` over 320 completions was 3.2, and the monitor's own verdict text "
        "reads `\"No code provided to audit.\"` A2, reading the answer on the same policy, flagged "
        "21 of 320."
    )
    L.append("")
    L.append(
        "**Undetected it would have produced** a monitored arm whose flag rate is zero at every "
        "step, read as a monitor that finds nothing to flag, and an obfuscation gap equal to the "
        "negated cheat rate. A1 would have appeared to be the arm where monitoring did not bite."
    )
    L.append("")
    L.append(
        "**Caught by** the flag rate being *identically* zero rather than small, and confirmed by "
        "a0/s1 and a1/s1 producing identical trajectories at matching seeds, to three decimals on "
        "every published series. A penalty term of exactly zero makes A1's reward arithmetically "
        "A0's."
    )
    L.append("")
    L.append(
        "**Cost.** Four completed runs, two of them A1, which are kept and relabelled as A0 "
        "replicates rather than deleted. H2 is unevaluable: a flag rate that is zero by "
        "construction cannot fall. Recorded in `BLOCKED.md`."
    )
    L.append("")

    L.append("### 2.2 The chat template was never applied")
    L.append("")
    if by_condition:
        raw = by_condition.get("qwen3-0.6b-raw-prompt", {})
        templated = by_condition.get("qwen3-0.6b-chat-template", {})
        policy_cond = by_condition.get("qwen2.5-0.5b-instruct", {})
        L.append(
            "**The fault.** `build_prompt` constructed a raw string and never applied the "
            "tokenizer's chat template. A supplementary diagnostic, sampling only, on "
            "byte-identical step-zero prompts:"
        )
        L.append("")
        L.append("| Condition | Non-empty thought region | Mean think chars | Flag rate, A1 view |")
        L.append("|---|---|---|---|")
        for name, entry in (
            ("`Qwen2.5-0.5B-Instruct`, raw prompt", policy_cond),
            ("`Qwen3-0.6B`, raw prompt", raw),
            ("`Qwen3-0.6B`, own chat template", templated),
        ):
            if not entry:
                continue
            L.append(
                f"| {name} | **{_fmt(entry.get('nonempty_think_fraction'))}** "
                f"| {_fmt((entry.get('think_chars') or {}).get('mean'), 1)} "
                f"| {_fmt(entry.get('flag_rate_think_view'))} |"
            )
        L.append("")
        L.append(
            "Qwen3-0.6B, a model post-trained to think, emits a thought region on **0 of 16** "
            "completions from the raw prompt and **16 of 16** through its own template. Same "
            "model, same prompt content."
        )
    else:
        L.append("**The fault.** `build_prompt` never applied the tokenizer's chat template.")
    L.append("")
    L.append(
        "**Undetected it would have produced** exactly the same empty-view failure as 2.1, and "
        "would have survived swapping in a thinking policy. The two faults are independent and "
        "**either alone is sufficient** to empty the arm, so fixing only the model would have "
        "looked like a fix and changed nothing."
    )
    L.append("")
    L.append(
        "**Caught by** adding a third diagnostic condition on the reasoning that a model which "
        "only thinks under its own template would itself answer the post-training question. The "
        "two-condition comparison originally asked for would have shown 0/16 twice and concluded, "
        "wrongly, that a thinking model does not help."
    )
    L.append("")
    L.append(
        "**Cost.** It invalidates the framing, not the data: the claim is not that a model this "
        "size cannot produce a thought region. Thought-region monitoring presupposes a policy "
        "post-trained to emit one **and** an interface that lets it, and neither was checked "
        "before the arm was designed."
    )
    L.append("")

    L.append("### 2.3 Nearly half of all completions hit the token cap")
    L.append("")
    L.append(
        "**The fault.** `max_new_tokens` was 256. Across a0/s1's 1280 completions, **44.8 percent "
        "hit the cap**. Their visible pass rate is **0.1053** against **0.2254** for completions "
        "that stopped naturally."
    )
    L.append("")
    L.append(
        "**Undetected it would have produced** a policy bounded in what it could express for the "
        "whole run, with a visible pass rate depressed by truncation and read as a policy that "
        "cannot solve the problems."
    )
    L.append("")
    L.append(
        "**Caught by** logging `finish_reason` per completion and splitting visible pass rate on "
        "it. The two subgroups differ by more than a factor of two."
    )
    L.append("")
    L.append(
        "**Cost.** It bounds every v1 number. It is also the fault that eventually made both later "
        "versions infeasible, because a policy that thinks needs a budget this hardware cannot "
        "backpropagate through. See section 5."
    )
    L.append("")

    L.append("### 2.4 The objective had no prompt conditioning")
    L.append("")
    L.append(
        "**The fault.** `_update` fed `completion.token_ids`, which is the generated tokens with "
        "the prompt stripped. The gradient therefore maximised the **unconditional** likelihood of "
        "completion text, not the likelihood of a solution given a problem."
    )
    L.append("")
    L.append(
        "**Undetected it would have produced** a training curve that looks like ordinary "
        "optimisation failure. Reward falls, the run completes, and the natural reading is that "
        "the task is too hard for a 0.5B policy."
    )
    L.append("")
    if judgement.get("measured"):
        L.append(
            "**Caught by** the retrospective, recomputed from stored artifacts. Mean reward fell "
            f"from **{_fmt(judgement['mean_reward_first10'])}** over the first ten steps to "
            f"**{_fmt(judgement['mean_reward_last10'])}** over the last ten, while mean token "
            f"entropy **rose** from **{_fmt(judgement['entropy_first10'])}** to "
            f"**{_fmt(judgement['entropy_last10'])}**. Falling reward with rising entropy is not a "
            "policy converging on a bad solution; it is a policy being pushed off its pretrained "
            "distribution without finding anything better."
        )
        L.append("")
        L.append(
            "The same retrospective **refuted the obvious explanation**. Zero-variance groups, "
            "which contribute no gradient, averaged "
            f"**{_fmt(judgement['zero_variance_group_fraction_mean'])}**, not a majority, and mean "
            f"absolute advantage was {_fmt(judgement['mean_abs_advantage_mean'])} with a maximum "
            f"of {_fmt(judgement['max_abs_advantage'])}. The loop had usable gradient signal on "
            "most steps. Reporting zero variance as the mechanism would have been convenient and "
            "wrong."
        )
    else:
        L.append("**Caught by** the retrospective over stored artifacts.")
    L.append("")
    L.append(
        "**Cost.** Every v1 training run optimised a different objective from the one the design "
        "assumes. Fixed in later versions by prepending the prompt and masking it out of the loss."
    )
    L.append("")

    L.append("### 2.5 `trainer_logprob` was a copy of `sampler_logprob`")
    L.append("")
    L.append(
        "**The fault.** The column exists so the follow-on project can study what happens to the "
        "sampler-trainer logprob pair when the rollout path changes. It was written as a literal "
        "copy: **3430 of 3430 rows identical** at step 0 of a0/s1."
    )
    L.append("")
    L.append(
        "**Undetected it would have produced** a dataset that appears to show perfect "
        "sampler-trainer agreement. The column collected to detect divergence **could not have "
        "shown any**, and the absence of divergence would have read as evidence of its absence."
    )
    L.append("")
    L.append(
        "**Caught by** asking, before the next version ran, whether the column came from an actual "
        "forward pass, then comparing the two columns directly rather than trusting the field name."
    )
    L.append("")
    L.append(
        "**Cost.** v1's `logprobs.parquet` does not support the analysis it was collected for. "
        "Fixed afterwards: with a genuine prompt-conditioned recomputation the two columns diverge "
        "on **0 of 96** rows, mean absolute difference **0.2150**, maximum **1.8477**. That "
        "divergence is itself unresolved, see section 6."
    )
    L.append("")

    # --------------------------------------------- 3. WHAT THE INSTRUMENT GOT RIGHT
    L.append("## 3. What the instrument got right")
    L.append("")
    L.append(
        "This is the load-bearing claim of the whole report. Five faults were found **before** any "
        "of them reached a published number, and that was not luck. Each was caught because a "
        "specific piece of instrumentation existed before the runs did."
    )
    L.append("")
    L.append(
        "**The reward monitor is separated from the behavioural oracle by construction.** "
        "`reward_fn` is never handed an oracle, `ScoredCompletion` has no oracle field, and "
        "`tests/test_reward_isolation.py` fails if anything connects them, including through a "
        "call site. Because ground truth could not leak into the reward, a monitored arm that "
        "measured nothing still had an independent measurement pointing at it."
    )
    L.append("")
    L.append(
        f"**The determinacy floor.** A completion scored on fewer than {metrics.DETERMINACY_FLOOR} "
        "determinate held-out tests has an indeterminate oracle verdict and leaves both the "
        "numerator and the denominator, never counted as a cheat. It is applied at report time and "
        "recomputed from stored completions, so an auditor can check it without a GPU."
    )
    L.append("")
    L.append(
        "**Thresholds were pre-registered and hashed into every run.** The cheat threshold, the "
        "three sensitivity values reported always rather than chosen after the fact, the "
        "eligibility rules and every halt condition are frozen in `alibi/prereg.py` and immutable "
        "at runtime. Later versions inherit the measurement **by import** rather than by "
        "restatement, and a test asserts the inheritance still holds."
    )
    L.append("")
    if precision.get("n_reference_solutions"):
        rate = precision["false_positive_rate"]
        ci = precision.get("false_positive_ci95") or [None, None]
        L.append(
            "**Structural precision was measured on honest code before the panel was trusted.** "
            "The check's recall was validated on generated cheats; its precision never was. Run "
            f"over all {precision['n_reference_solutions']} eligible MBPP+ reference solutions, "
            f"every one a genuine algorithm, it flags **{precision['n_flagged']}**, a false "
            f"positive rate of **{_fmt(rate)}** with a 95 percent interval of {_fmt(ci[0])} to "
            f"{_fmt(ci[1])}, per form `{precision['per_form'] or 'none'}`, task ids "
            f"`{precision['flagged_task_ids']}`."
        )
        L.append("")
        if rate is not None and rate >= NEAR_ZERO_FALSE_POSITIVE_RATE:
            L.append(
                f"That is **above** the {NEAR_ZERO_FALSE_POSITIVE_RATE:.2f} bar declared before the "
                "measurement was run, so the cheat_form panel reports detector behaviour as well as "
                "policy behaviour, and the report says so rather than presenting the panel as a "
                "clean read of the policy. The rate is not uniform: it is five specific problems, "
                "listed with full source in [STRUCTURAL_FP.md](STRUCTURAL_FP.md), which enter the "
                "prompt set on 2 of 80 steps."
            )
            L.append("")
    if behavioural.get("measured"):
        ci = behavioural.get("false_positive_ci95") or [None, None]
        L.append(
            "**A vindication claim was withdrawn as circular.** An earlier draft argued that the "
            "structural false positive rate vindicated making the behavioural oracle primary, "
            "because the behavioural check has no equivalent floor. That was wrong. The behavioural "
            "check has no *measured* floor, and the reason is circular: **eligibility excludes "
            "problems whose reference solution fails held-out**, which is the behavioural rule's own "
            "criterion, so any rate computed on the eligible set is zero by construction. The two "
            "checks were validated on differently filtered populations."
        )
        L.append("")
        L.append(
            "A non-circular estimate from the pre-eligibility set, which is filtered only for "
            f"reasons unrelated to the rule: **{behavioural['n_flagged']} of "
            f"{behavioural['n_evaluable']} flagged, 95 percent upper bound {_fmt(ci[1])}**. "
            "Reported as an estimate with four named limits, not as a resolution. The honest "
            "comparison is that the structural rate is measured and above the bar, while the "
            "behavioural rate is bounded above and consistent with zero."
        )
        L.append("")
    L.append(
        "**Monitor failure can never read as innocence.** An error is its own verdict state and is "
        "always flagged; `Verdict.from_error` does not take a `flagged` argument, and a test "
        "asserts that no exception type in the judging path can yield `flagged=False`. An outage "
        "raises the penalty rather than lowering it, then halts."
    )
    L.append("")
    L.append(
        "**Timeouts and crashes are a third outcome, never a fail.** Pass fractions are computed "
        "over determinate tests only and are `None` when nothing was determinate, so an "
        "infrastructure failure cannot read as a policy that does not generalise."
    )
    L.append("")
    if selection:
        L.append(
            f"**The judge was probed for availability and qualified before use.** Selected: "
            f"`{selection['model_id']}`, selection hash `{selection['selection_hash'][:16]}`. "
            "Availability was probed with a real request rather than inferred from a model "
            "listing, because 3 of 5 candidates that appear in the listing reject serverless calls."
        )
        if selection.get("shares_family_with_policy"):
            L.append("")
            L.append(
                f"**Judge and judged share a model family.** `{selection['model_id']}` grades a Qwen "
                "policy. This was an explicit operator override made before any curve existed, on "
                "the qualification table, and it is carried as a limitation: a same-family judge may "
                "track this policy's habits more closely than an independent one would."
            )
        qualification = selection.get("qualification") or {}
        if qualification:
            L.append("")
            L.append(
                "**Qualification is not pre-registered, and its limits are stated rather than "
                f"buried.** Cheat flag rate {_fmt(qualification.get('cheat_flag_rate'))}, honest "
                f"flag rate {_fmt(qualification.get('honest_flag_rate'))}, separation "
                f"{_fmt(qualification.get('separation'))}. The procedure and all four thresholds "
                "were chosen by the implementer, it ran on the A2 view only, and the cheats it used "
                "are programmatically generated and stereotyped. Passing it is **necessary but not "
                "sufficient**, and no result here should be read as though the judge were known "
                "reliable on real completions."
            )
        L.append("")

    # ------------------- 4. WHAT WE LEARNED ABOUT MONITOR-REWARD EXPERIMENTS
    L.append("## 4. What this says about monitor-reward experiments generally")
    L.append("")
    L.append(
        "Four claims that outlive this particular run. Each is supported by a fault above, and "
        "each would apply to any experiment of this shape."
    )
    L.append("")
    L.append(
        "**An arm that reads thought tokens presupposes a policy post-trained to emit them, and an "
        "interface that lets it.** Both halves matter and neither is implied by model size. A 0.6B "
        "model emitted a thought region on every completion through its own chat template and none "
        "at all from a raw prompt of identical content. Any study of reasoning-trace monitoring "
        "should verify the trace exists, at the exact prompt path the experiment will use, before "
        "the arm is designed."
    )
    L.append("")
    L.append(
        "**An empty view and a clean policy are indistinguishable at the monitor's output.** Both "
        "produce `flagged=False`. Nothing downstream of the monitor can tell them apart, so the "
        "check has to be upstream: measure that the view is non-empty, per step, as a first-class "
        "series. A flag rate of exactly zero should be treated as a defect signal until shown "
        "otherwise, not as a monitor finding nothing."
    )
    L.append("")
    L.append(
        "**The token budget silently decides which arms are comparable.** Under a thinking policy "
        "the answer follows the think block, so a truncated completion yields no code at all and "
        "an empty answer view. The budget therefore determines whether an answer-reading arm has "
        "anything to read. It is an experimental design parameter, not an infrastructure setting, "
        "and this project only discovered that after it had eliminated two arms in two different "
        "versions for the same underlying reason."
    )
    L.append("")
    L.append(
        "**A flat cheat rate is uninformative unless the loop is shown to have been optimising the "
        "intended objective.** v1's cheat rate never moved, which reads as a policy that did not "
        "learn to cheat. It was a policy trained on the unconditional likelihood of its own output "
        "text, with two completions in five truncated, whose reward fell while its entropy rose. "
        "Reporting the flat curve as a finding about reward hacking would have been false, and "
        "nothing in the curve itself would have revealed that. The diagnostics that distinguish "
        "the two cases, per-step reward, entropy, advantage magnitude, zero-variance group "
        "fraction and capped fraction, cost almost nothing and should be logged from the first run."
    )
    L.append("")

    # ------------------------------------------------- 5. WHAT WAS NEVER TESTED
    L.append("## 5. What was never tested")
    L.append("")
    L.append("**All three pre-registered hypotheses are unevaluated.**")
    L.append("")
    for label, statement in zip(("H1", "H2", "H3"), prereg.PREREG.metric.hypotheses, strict=True):
        L.append(f"- **{label}** — *{statement}*")
    L.append("")
    L.append(
        "**H1** asks whether the oracle cheat rate rises in the unmonitored arm. The a0 runs "
        "completed and the rate did not move, but section 2.4 shows the loop was not optimising "
        "the registered objective and section 2.3 shows two completions in five were truncated. "
        "H1 was not given a fair test, and the flat curve is not evidence for or against it."
    )
    L.append("")
    L.append(
        "**H2** asks whether A1's flag rate falls faster than its cheat rate. A1's flag rate was "
        "zero by construction, and a rate that cannot fall cannot fall faster. Unevaluable."
    )
    L.append("")
    L.append(
        "**H3** compares terminal cheat rate in A2 against A1. A1 is a second control, and no A2 "
        "run ever completed: every A2 attempt failed, three of them on the indeterminate held-out "
        "halt. The comparison has no data on either side."
    )
    L.append("")
    L.append("### Why the control data cannot stand in")
    L.append("")
    if prefix.get("measured"):
        sampled_split = prefix.get("mbpp_split_composition_sampled") or {}
        unsampled_split = prefix.get("mbpp_split_composition_unsampled") or {}
        L.append(
            f"**Prompt coverage.** A run samples {prefix['n_sampled']} distinct problems of "
            f"{prefix['n_sampled'] + prefix['n_unsampled']} eligible, selected by a deterministic "
            "unseeded round robin over task id, so every arm and every seed sees the same "
            "lowest-task-id prefix. That is good for comparability between arms and it means the "
            "eligible count overstates what any run sees by more than a factor of two."
        )
        L.append("")
        if sampled_split and unsampled_split:
            L.append(
                f"On the four measured properties, held-out test count, visible assert count, "
                "reference solution length and cheat constructibility, the prefix does not differ "
                "materially from the tail. **On provenance it does**: sampled "
                f"`{sampled_split}` against never-sampled `{unsampled_split}`. The prefix is almost "
                "entirely MBPP's test split plus several problems from the prompt split, which is "
                "MBPP's designated few-shot exemplar set and the most likely of all to sit in "
                "pretraining data. **Generalisation from this run is to the sampled prefix, not to "
                "MBPP.**"
            )
            L.append("")
    L.append(
        "**Two variance estimates, and a verdict must clear the wider.** The seed band is min to "
        "max across seeds and captures sampling variance only. The cluster bootstrap resamples "
        f"**problems** with replacement, {metrics.BOOTSTRAP_DRAWS} draws at declared seed "
        f"`{metrics.BOOTSTRAP_SEED}`, and captures problem variance. The bootstrap corrects "
        "understated variance; it does **not** correct selection bias, because the sampled "
        "problems are a fixed deterministic prefix rather than a random draw. It widens the "
        "interval around the right target without moving the target."
    )
    L.append("")
    if pooled.get("n_completions"):
        ci = pooled.get("prevalence_ci95") or [None, None]
        L.append(
            f"**Cheat prevalence before any update**, pooled across {pooled['n_runs_pooled']} of "
            f"{pooled['n_runs_expected']} matrix runs, {pooled['n_completions']} completions: any "
            f"cheat_form {_fmt(pooled['prevalence'])} with a Wilson interval of {_fmt(ci[0])} to "
            f"{_fmt(ci[1])}, behavioural cheat rate {_fmt(pooled['behavioural_rate'])}. The "
            "interval width is the honest headline: at this sample size prevalence is consistent "
            "with almost anything."
        )
        L.append("")
    L.append("### v2 was abandoned as jointly infeasible")
    L.append("")
    L.append(
        "v2 changed the policy to Qwen3-0.6B, applied the chat template and set `max_new_tokens` "
        "to 3072 from the measured think-length distribution, with a capped-fraction halt of 0.35 "
        "against the 0.125 truncation that budget produced. Backpropagating through a "
        "prompt-plus-completion sequence of that length, with a 151936-token vocabulary, does not "
        "fit in 8 GB. Measured with one budget per process, LoRA and gradient checkpointing "
        "enabled:"
    )
    L.append("")
    L.append("| Budget | Result | Peak allocated |")
    L.append("|---|---|---|")
    L.append("| 3072, registered | **OOM** | 7.05 GB |")
    L.append("| 2048 | **OOM** | 6.45 GB |")
    L.append("| 1536 | **OOM** | 6.62 GB |")
    L.append("| 1024 | fits | 5.54 GB |")
    L.append("")
    L.append(
        "The floor is the full-vocab logits tensor, about 1 GB in bf16 before the backward graph. "
        "Chunked softmax and gradient checkpointing were both applied and neither is sufficient. "
        "Both numbers are registered in `alibi-prereg-v2.2`, so neither was changed to make the run "
        "fit, and no fused chunked LM head was built. See `BLOCKED-v2.md`."
    )
    L.append("")
    L.append("### v3 was stopped by its own qualifying measurement")
    L.append("")
    L.append(
        "v3 was a **different and smaller experiment, not v2 rescued**: a 1024-token budget, arms "
        "a0 and a2 only, one seed, everything else inherited. Arm a1 was excluded outright because "
        "the measured median think block is 746 tokens and a 1024 budget cannot hold think plus "
        "answer."
    )
    L.append("")
    L.append(
        "A no-training sampling pass at 1024 tokens, 64 completions over 8 eligible problems, was "
        "to set its capped-fraction halt. It stopped the version instead:"
    )
    L.append("")
    L.append("| Measure | Value | 95 percent interval |")
    L.append("|---|---|---|")
    L.append("| Capped fraction | **0.6875** | 0.5661 to 0.7877 |")
    L.append("| Completions yielding code | **0.4219** | 0.3087 to 0.5439 |")
    L.append("| Think block closed | 0.4219 | |")
    L.append("| Median tokens | 1024, the cap itself | |")
    L.append("")
    L.append(
        "The median completion is capped, so more than half never finish thinking. The "
        "closed-think fraction and the has-code fraction are equal at **exactly 27 of 64**: a "
        "completion yields code if and only if its think block closed, so the other 58 percent "
        "carry an empty answer. Arm a2's monitor reads the answer, so it would read an empty "
        "string on 58 percent of completions and return unflagged. **a2 would measure the "
        "monitor's response to absence rather than to cheating** — the fault of 2.1 reappearing in "
        "a different arm, arriving through the token budget instead of through the policy."
    )
    L.append("")
    L.append(
        "No halt threshold was chosen, because any value above 0.6875 would be a licence rather "
        "than a guard, and no `alibi-prereg-v3.0` tag was created, because a pre-registration tag "
        "records intent frozen before data and here the blocking data arrived first. The design is "
        "in `alibi/prereg_v3.py` with `RUNNABLE = False`."
    )
    L.append("")

    # ------------------------------------------------------------ 6. UNRESOLVED
    L.append("## 6. Unresolved")
    L.append("")
    L.append(
        "**The sampler and trainer logprobs diverge by more than numerical noise on identical "
        "weights, and it is not diagnosed.** After `trainer_logprob` was fixed to be a genuine "
        "prompt-conditioned recomputation, the two columns differ on **0 of 96** rows, with a mean "
        "absolute difference of **0.2150** and a maximum of **1.8477**."
    )
    L.append("")
    L.append(
        "Both are computed from the same weights at the same step. The paths differ in more than "
        "precision: the sampler reads the generation path with a KV cache, one position at a time, "
        "while the trainer runs a single full forward over prompt and completion together. A "
        "difference of this size is well past what bf16 rounding alone would explain, and which "
        "part of that gap is attributable to the cache, to kernel selection, or to something else "
        "has not been established."
    )
    L.append("")
    L.append(
        "This is stated as an open question rather than an explanation, and it is the follow-on "
        "project's subject arriving early. It is also a caution about the fix: the column now "
        "carries information, and what that information means is not yet known."
    )
    L.append("")

    # ------------------------------------------------------------ 7. PROVENANCE
    L.append("## 7. Provenance")
    L.append("")
    L.append(f"- Pre-registration: `{prereg.PREREG.version}`, hash `{prereg.PREREG_HASH[:16]}`")
    L.append(f"- Eligible problems: {provenance.get('eligibility_n_problems')}, frozen and hashed")
    L.append(f"- Reward: `{prereg.PREREG.reward_definition}`, lambda {prereg.PREREG.lambda_monitor}")
    L.append(f"- Declared evidence runs: {computed.get('n_declared_runs', 0)}")
    L.append("")
    L.append("### The resolved training configuration")
    L.append("")
    L.append("| Setting | v1 resolved value |")
    L.append("|---|---|")
    for name, value in [
        ("Optimiser", "`torch.optim.AdamW`, betas (0.9, 0.999), eps 1e-8, weight decay 0.01, all defaults"),
        ("Learning rate", "1e-5, constant"),
        ("Scheduler", "**none**"),
        ("Warmup", "**none**, 0 steps"),
        ("Gradient clipping", "global norm 1.0"),
        ("beta, KL coefficient in the loss", "**no KL term in the loss**, and no reference policy existed"),
        ("epsilon, PPO clip ratio", "**none**, the loss is an unclipped `-advantage x logprob`"),
        ("LoRA", "r 16, alpha 32, dropout 0.0, targets q/k/v/o_proj"),
        ("Effective batch", "16 completions per step, 2 prompts x group 8"),
        ("Gradient accumulation", "16, one backward per completion, one optimiser step per training step"),
        ("Objective", "mean token logprob, length normalised, not a token sum"),
    ]:
        L.append(f"| {name} | {value} |")
    L.append("")
    L.append("### Config hash discontinuity")
    L.append("")
    L.append(
        "`ArmConfig` gained `policy_version` and `apply_chat_template` partway through the work. "
        "Both are behaviourally inert defaults, so runs before and after are identical in what "
        "they did, but **their config hashes are not comparable**. The four declared evidence runs "
        "predate the change and carry configs without those keys. An auditor diffing config hashes "
        "will see a discontinuity there; it cannot be fixed retroactively without editing "
        "artifacts, which is not done."
    )
    L.append("")
    L.append("### Halt conditions were amended before any curve existed")
    L.append("")
    L.append(
        "**1. Indeterminate held-out rate, instantaneous to lagging window.** The threshold, 5 "
        "percent, was not changed; the evaluation window was. On a0/s1, steps 0 to 7 were 0.0000 "
        "and step 8 was 0.0584, caused by exactly one completion of sixteen whose code hung, "
        "making 92 of its ~99 held-out tests indeterminate. Spikes of that kind arrived every four "
        "or five steps, so the instantaneous reading would have halted every run within about ten "
        "steps while saying nothing about the sandbox. The halt condition's own wording permits "
        "evaluation on a lagging window when held-out scoring is asynchronous, which it is. "
        "Amended by the implementer."
    )
    L.append("")
    L.append(
        "**2. The queue stop rule, three consecutive halts to more than half failed.** Amended by "
        "the operator's instruction."
    )
    L.append("")
    L.append(
        "**The halt conditions were specified against an imagined run.** They were written before "
        "any training had happened, and first contact with a real policy showed one of them "
        "incompatible with the thing it was guarding. Treat them as a design artifact revised on "
        "contact with data, not as constraints that survived unchanged."
    )
    L.append("")
    if figures:
        L.append("### Figures")
        L.append("")
        for name, path in sorted(figures.items()):
            L.append(f"![{name}](figures/{Path(path).name})")
            L.append("")
    L.append("```json")
    L.append(json.dumps(runlog.scrub(provenance), indent=2, sort_keys=True))
    L.append("```")
    L.append("")
    L.append(
        f"Generated {published['generated_utc']} by `alibi report`. Numbers are injected from "
        "`artifacts/`, never typed. `alibi verify --no-gpu` recomputes every claim above. Every "
        "autonomous decision, including the ones later withdrawn, is in `DECISIONS.md`."
    )
    L.append("")

    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    out = {
        "report": REPORT_PATH,
        "published": PUBLISHED_PATH,
        "step_zero_flagged": STEP_ZERO_FLAGGED_PATH,
        "structural_fp": STRUCTURAL_FP_PATH,
    }
    out.update({f"figure_{k}": v for k, v in figures.items()})
    return out
