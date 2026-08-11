"""Ten-step hyperparameter probes on Qwen3-0.6B. Probes, never evidence.

Run ids are prefixed `probe-`, which matches neither the v1 nor the v2 matrix
pattern, so a probe cannot enter the evidence index or pool with either matrix.
`artifacts/index.json` also excludes the prefix by policy.

Ten steps suffices because v1's degradation was visible inside ten: mean reward
over its first ten steps was already below its step-zero value.

Three conditions:

  A  current hyperparameters, learning rate 1e-5, beta 0.0
  B  learning rate reduced tenfold, 1e-6, beta 0.0
  C  a non-zero KL anchor, beta 0.02, learning rate unchanged. v1 ran at beta 0
     with no reference policy at all, so this is the condition the operator
     specified rather than the hundredfold learning rate cut.

Also checks, directly, the thing that cannot be recovered from v1's stored data:
whether a completion with positive advantage had its logprob increased by the
update that followed it.
"""

from __future__ import annotations

import json
import math
import statistics
import time

from alibi import prereg_v2, runlog
from alibi.train.loop import ArmConfig

OUT_DIR = runlog.ARTIFACTS / "diagnostics" / "probes"
PROBE_STEPS = 10

# Ordered by algorithmic completeness, most complete first, because that is the
# order in which a passing probe is preferred. An unclipped, unanchored
# objective is not GRPO, so a probe that merely holds reward flat by lowering
# the learning rate is a quieter version of the wrong algorithm rather than the
# right one.
CONDITIONS = (
    {"label": "D-anchor-and-clip", "learning_rate": 1e-5, "beta": 0.02,
     "clip_epsilon": 0.2, "inner_epochs": 2,
     "note": "KL anchor plus ratio clipping over two inner epochs. This is the "
             "GRPO objective. Clipping cannot bind at one inner epoch, because "
             "the ratio is then identically one."},
    {"label": "C-kl-anchor", "learning_rate": 1e-5, "beta": 0.02,
     "clip_epsilon": 0.0, "inner_epochs": 1,
     "note": "non-zero KL anchor against the base policy, which v1 never had"},
    {"label": "B-lr-10x-lower", "learning_rate": 1e-6, "beta": 0.0,
     "clip_epsilon": 0.0, "inner_epochs": 1,
     "note": "learning rate reduced tenfold, still the unanchored objective"},
    {"label": "A-current", "learning_rate": 1e-5, "beta": 0.0,
     "clip_epsilon": 0.0, "inner_epochs": 1,
     "note": "current hyperparameters, unchanged from v1"},
)

# A probe is preferred in this order when more than one holds reward flat.
PREFERENCE = ("D-anchor-and-clip", "C-kl-anchor", "B-lr-10x-lower", "A-current")


def build_config(condition: dict) -> ArmConfig:
    policy = prereg_v2.PREREG_V2.policy
    return ArmConfig(
        arm="a0",
        seed=1,
        steps=PROBE_STEPS,
        group_size=policy.group_size,
        prompts_per_step=policy.prompts_per_step,
        max_new_tokens=policy.max_new_tokens,
        model_id=policy.model_id,
        monitor_name="rules",
        policy_version=prereg_v2.PREREG_VERSION,
        apply_chat_template=policy.apply_chat_template,
        learning_rate=condition["learning_rate"],
        beta=condition["beta"],
        clip_epsilon=condition.get("clip_epsilon", 0.0),
        inner_epochs=condition.get("inner_epochs", 1),
        label=condition["label"],
    )


def summarise(run_id: str) -> dict:
    """Per step reward, entropy, capped fraction and KL for one probe."""
    import glob

    rows = []
    for path in sorted(glob.glob(str(runlog.ARTIFACTS / "runs" / run_id / "steps/*/summary.json"))):
        summary = json.loads(open(path).read())
        training = summary.get("training") or {}
        rows.append(
            {
                "step": summary["step"],
                "mean_reward": summary.get("mean_reward"),
                "entropy": training.get("mean_token_entropy"),
                "capped_fraction": summary.get("capped_fraction"),
                "kl": summary.get("kl"),
                "zero_variance_group_fraction": training.get("zero_variance_group_fraction"),
                "visible_pass_rate": summary.get("visible_pass_rate"),
            }
        )
    if not rows:
        return {"run_id": run_id, "measured": False, "absent_reason": "no steps were written"}

    rewards = [r["mean_reward"] for r in rows if r["mean_reward"] is not None]
    first, last = rewards[0], statistics.fmean(rewards[-3:]) if len(rewards) >= 3 else rewards[-1]

    def col(key):
        return [r[key] for r in rows if r.get(key) is not None]

    entropy, capped, kl = col("entropy"), col("capped_fraction"), col("kl")
    return {
        "run_id": run_id,
        "measured": True,
        "steps": rows,
        "reward_step0": first,
        "reward_last3": last,
        "reward_change": last - first,
        # Flat or rising. A probe that falls is a loop degrading the policy.
        "holds_or_rises": last >= first,
        "entropy_first3": statistics.fmean(entropy[:3]) if entropy else None,
        "entropy_last3": statistics.fmean(entropy[-3:]) if entropy else None,
        "capped_mean": statistics.fmean(capped) if capped else None,
        "kl_mean": statistics.fmean(kl) if kl else None,
        "kl_max": max(kl) if kl else None,
    }


def advantage_direction_check(run_id: str) -> dict:
    """Did a positive advantage actually raise that completion's logprob?

    **This cannot be recovered from v1's stored data**, for two reasons that are
    worth stating rather than working around: `trainer_logprob` was written as a
    literal copy of `sampler_logprob`, so it is not a recomputation under the
    updated policy, and no prompt repeats within a run, so there is no natural
    same-prompt comparison across steps.

    Here it is measured directly instead. For each step the completions are
    re-scored under the policy **after** that step's update, and the sign of the
    logprob change is compared against the sign of the advantage. Under a
    correct policy gradient the two should agree more often than not.
    """
    import glob

    from alibi.report.metrics import load_step_completions

    path_pattern = str(runlog.ARTIFACTS / "runs" / run_id / "steps/*/advantage_check.json")
    checks = [json.loads(open(p).read()) for p in sorted(glob.glob(path_pattern))]
    if not checks:
        return {"measured": False, "absent_reason": "no advantage checks were recorded"}
    agree = sum(c["n_agree"] for c in checks)
    total = sum(c["n_compared"] for c in checks)
    return {
        "measured": True,
        "n_steps": len(checks),
        "n_compared": total,
        "n_agree": agree,
        "agreement_rate": (agree / total) if total else None,
        "per_step": checks,
    }


def main() -> dict:
    from alibi.train import runner as v1_runner

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results = []
    for condition in CONDITIONS:
        config = build_config(condition)
        run_id = f"probe-{condition['label']}-{config.hash()[:8]}"
        v1_runner.log(f"probe {condition['label']}: lr={condition['learning_rate']} beta={condition['beta']}")
        outcome = v1_runner.run_subprocess(config, run_id)
        summary = summarise(run_id)
        summary.update(
            {
                "label": condition["label"],
                "note": condition["note"],
                "learning_rate": condition["learning_rate"],
                "beta": condition["beta"],
                "clip_epsilon": condition.get("clip_epsilon", 0.0),
                "inner_epochs": condition.get("inner_epochs", 1),
                "status": outcome.get("status"),
                "halt_reason": outcome.get("halt_reason"),
                "advantage_direction": advantage_direction_check(run_id),
            }
        )
        results.append(summary)
        v1_runner.log(
            f"probe {condition['label']} -> {outcome.get('status')} "
            f"reward {summary.get('reward_step0')} to {summary.get('reward_last3')} "
            f"holds={summary.get('holds_or_rises')}"
        )

    passing = {r["label"]: r for r in results if r.get("measured") and r.get("holds_or_rises")}
    # Preference, not order of completion: prefer the algorithmically complete
    # objective over a quieter version of the wrong one.
    chosen = next((passing[label] for label in PREFERENCE if label in passing), None)
    document = {
        "diagnostic": "hyperparameter_probes",
        "declared": "Probes, never evidence. Excluded from the evidence index by run-id prefix and by policy.",
        "steps_per_probe": PROBE_STEPS,
        "policy": prereg_v2.POLICY_MODEL,
        "conditions": results,
        "chosen": chosen["label"] if chosen else None,
        "preference_order": list(PREFERENCE),
        "all_passing": sorted(passing),
        "chosen_reason": (
            f"{chosen['label']} held reward flat or rising, {chosen['reward_step0']:.4f} to "
            f"{chosen['reward_last3']:.4f}, and is the most algorithmically complete of the "
            f"{len(passing)} passing condition(s) {sorted(passing)}. An unclipped, unanchored "
            "objective is not GRPO, so a low learning rate that merely holds reward flat is "
            "preferred only when nothing more complete passes."
            if chosen
            else "no condition held reward flat or rising, so the stage was not launched"
        ),
        "wall_clock_seconds": round(time.monotonic() - started, 1),
    }
    (OUT_DIR / "result.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(document)
    return document


def write_markdown(document: dict) -> None:
    lines = [
        "# Hyperparameter probes",
        "",
        f"Ten steps each on `{document['policy']}` with the chat template, arm a0 only. "
        "**Probes, never evidence.** Excluded from the evidence index by run-id prefix and by policy.",
        "",
        "| Probe | lr | beta | clip | inner epochs | Reward step 0 to last 3 | Holds or rises | Entropy first3 to last3 | Capped | KL mean |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(value, digits=4):
        return "not measured" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))

    for entry in document["conditions"]:
        if not entry.get("measured"):
            lines.append(
                f"| {entry['label']} | {entry['learning_rate']} | {entry['beta']} | run produced no steps "
                f"| n/a | n/a | n/a | n/a | {entry.get('halt_reason') or entry.get('status')} |"
            )
            continue
        lines.append(
            f"| {entry['label']} | {entry['learning_rate']} | {entry['beta']} "
            f"| {entry.get('clip_epsilon', 0.0)} | {entry.get('inner_epochs', 1)} "
            f"| {fmt(entry['reward_step0'])} to {fmt(entry['reward_last3'])} "
            f"| **{entry['holds_or_rises']}** "
            f"| {fmt(entry['entropy_first3'])} to {fmt(entry['entropy_last3'])} "
            f"| {fmt(entry['capped_mean'])} | {fmt(entry['kl_mean'])} |"
        )
    lines += ["", f"**Chosen: {document['chosen'] or 'none'}.** {document['chosen_reason']}", ""]

    lines += [
        "## Did a positive advantage raise the logprob?",
        "",
        "This could not be recovered from v1's stored data. `trainer_logprob` was written as a "
        "literal copy of `sampler_logprob` rather than a recomputation under the updated policy, and "
        "no prompt repeats within a run, so there was no same-prompt comparison to make. Measured "
        "directly in the probes instead: each step's completions are re-scored under the policy "
        "**after** that step's update, and the sign of the logprob change is compared against the "
        "sign of the advantage.",
        "",
        "| Probe | Completions compared | Sign agreement |",
        "|---|---|---|",
    ]
    for entry in document["conditions"]:
        check = entry.get("advantage_direction") or {}
        if not check.get("measured"):
            lines.append(f"| {entry['label']} | not measured | {check.get('absent_reason', '')} |")
            continue
        lines.append(
            f"| {entry['label']} | {check['n_compared']} | {fmt(check['agreement_rate'])} |"
        )
    lines += [
        "",
        "Agreement near 0.5 would mean the update is not moving logprobs in the direction the "
        "advantage asks for, which would point at the optimiser or the loss rather than at the "
        "reward. Agreement well above 0.5 means the gradient is being applied as intended and the "
        "problem, if any, is upstream in what the reward is rewarding.",
        "",
    ]
    (runlog.REPO_ROOT / "report" / "PROBES.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
