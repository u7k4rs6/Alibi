"""Did the v1 loop have usable gradient signal, and did the policy degrade?

Computed entirely from stored v1 artifacts. v1 did not log advantage statistics
or entropy per step, so both are **recomputed** here from what was stored:

  advantage  the GRPO formula is deterministic given the group's rewards, and
             every reward is in completions.jsonl, so the advantages are
             recoverable exactly rather than approximated
  entropy    logprobs.parquet holds the sampler's logprob for every generated
             token. Tokens were drawn from the policy at temperature 1.0, so the
             mean negative logprob is an unbiased single-sample estimator of the
             sampling entropy
  capped     finish_reason is per completion, and "length" means the completion
             hit the token budget rather than stopping

The question this answers is the one that decides whether v1's flat curves mean
"the policy did not learn to cheat" or "the loop could not have taught it
anything". Those are different findings and only one of them is about the model.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from alibi import runlog

ZERO_VARIANCE_TOLERANCE = 1e-8


def _steps(run_dir: Path) -> list[int]:
    return sorted(int(p.parent.name) for p in run_dir.glob("steps/*/summary.json"))


def _advantages(rewards: list[float], group_size: int) -> tuple[list[float], int, int]:
    """The GRPO advantages for one step, and how many groups had zero variance."""
    magnitudes: list[float] = []
    zero_variance = 0
    groups = 0
    for start in range(0, len(rewards), group_size):
        chunk = rewards[start : start + group_size]
        if len(chunk) < 2:
            continue
        groups += 1
        mean = statistics.fmean(chunk)
        spread = statistics.pstdev(chunk)
        if spread < ZERO_VARIANCE_TOLERANCE:
            zero_variance += 1
        magnitudes.extend(abs((value - mean) / (spread + 1e-6)) for value in chunk)
    return magnitudes, zero_variance, groups


def _entropy_for_step(run_dir: Path, step: int) -> float | None:
    """Mean negative sampler logprob per token, over the step's completions."""
    path = run_dir / "steps" / f"{step:05d}" / "logprobs.parquet"
    if not path.exists():
        return None
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=["completion_idx", "sampler_logprob"])
    except Exception:  # noqa: BLE001 - a missing diagnostic is absent, not fatal
        return None
    import math

    per_completion: dict[int, list[float]] = {}
    for row in table.to_pylist():
        value = row.get("sampler_logprob")
        # Positions past the end-of-sequence token are padding, and their logprob
        # is -inf. Including them made the entropy estimate inf rather than a
        # number. They are dropped rather than clamped: a padding position is not
        # a sampled token and does not belong in an entropy over sampled tokens.
        if value is not None and math.isfinite(value):
            per_completion.setdefault(row["completion_idx"], []).append(value)
    means = [-statistics.fmean(v) for v in per_completion.values() if v]
    return statistics.fmean(means) if means else None


def analyse(run_dir: Path, group_size: int = 8) -> dict:
    """Per-step training diagnostics for one stored run."""
    from alibi.report.metrics import load_step_completions

    rows = []
    for step in _steps(run_dir):
        records = load_step_completions(run_dir, step)
        if not records:
            continue
        rewards = [r["reward"] for r in records if r.get("reward") is not None]
        magnitudes, zero_variance, groups = _advantages(rewards, group_size)
        capped = sum(1 for r in records if r.get("finish_reason") == "length")
        rows.append(
            {
                "step": step,
                "n": len(records),
                "mean_reward": statistics.fmean(rewards) if rewards else None,
                "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else None,
                "n_groups": groups,
                "zero_variance_groups": zero_variance,
                "zero_variance_group_fraction": (zero_variance / groups) if groups else None,
                "mean_abs_advantage": statistics.fmean(magnitudes) if magnitudes else None,
                "max_abs_advantage": max(magnitudes) if magnitudes else None,
                "mean_token_entropy": _entropy_for_step(run_dir, step),
                "capped_fraction": capped / len(records),
                "n_capped": capped,
            }
        )
    return {"run_id": run_dir.name, "steps": rows}


def verdict(analysis: dict) -> dict:
    """Two plain statements: did the policy degrade, was there gradient signal."""
    rows = analysis["steps"]
    if len(rows) < 20:
        return {"measured": False, "absent_reason": f"only {len(rows)} steps stored, too few to judge"}

    def col(key):
        return [r[key] for r in rows if r.get(key) is not None]

    rewards = col("mean_reward")
    first, last = statistics.fmean(rewards[:10]), statistics.fmean(rewards[-10:])
    zero_variance = col("zero_variance_group_fraction")
    entropy = col("mean_token_entropy")
    capped = col("capped_fraction")
    advantage = col("mean_abs_advantage")

    mean_zero_variance = statistics.fmean(zero_variance) if zero_variance else None
    degraded = last < first
    # "Most groups" is the operator's phrase and is taken at its plain meaning.
    no_signal = mean_zero_variance is not None and mean_zero_variance > 0.5

    return {
        "measured": True,
        "n_steps": len(rows),
        "mean_reward_first10": first,
        "mean_reward_last10": last,
        "reward_change": last - first,
        "policy_degraded": degraded,
        "zero_variance_group_fraction_mean": mean_zero_variance,
        "zero_variance_group_fraction_first10": statistics.fmean(zero_variance[:10]) if zero_variance else None,
        "zero_variance_group_fraction_last10": statistics.fmean(zero_variance[-10:]) if zero_variance else None,
        "most_groups_had_zero_variance": no_signal,
        "mean_abs_advantage_mean": statistics.fmean(advantage) if advantage else None,
        "max_abs_advantage": max(col("max_abs_advantage")) if col("max_abs_advantage") else None,
        "entropy_first10": statistics.fmean(entropy[:10]) if entropy else None,
        "entropy_last10": statistics.fmean(entropy[-10:]) if entropy else None,
        "capped_fraction_mean": statistics.fmean(capped) if capped else None,
        "capped_fraction_first10": statistics.fmean(capped[:10]) if capped else None,
        "capped_fraction_last10": statistics.fmean(capped[-10:]) if capped else None,
    }


def a0_seed1() -> tuple[dict, dict] | tuple[None, None]:
    """The retrospective the operator asked for, on a0 seed 1."""
    candidates = sorted((runlog.ARTIFACTS / "runs").glob("a0-seed1-*"))
    if not candidates:
        return None, None
    analysis = analyse(candidates[0])
    return analysis, verdict(analysis)


def write(path: Path | None = None) -> Path | None:
    analysis, judgement = a0_seed1()
    if analysis is None:
        return None
    path = path or runlog.ARTIFACTS / "diagnostics" / "v1_retrospective.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"analysis": analysis, "verdict": judgement}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
