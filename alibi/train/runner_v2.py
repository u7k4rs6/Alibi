"""The v2 staged runner. Runs a0/s1, a1/s1, a2/s1 and stops.

Staged on purpose. If H1 does not move at seed 1, adding seeds will not save it,
and the operator wants the decision rather than a night of compute.

v2 run directories are prefixed `v2-`, which keeps them out of
`alibi.report.metrics.matrix_run_dirs`, whose pattern matches v1 ids only. So v2
cannot be pooled with v1 by accident: the separation is enforced by the id, not
by remembering to filter.

The v1 queue is never read or written by this module. It has its own queue file.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from alibi import prereg_v2, runlog
from alibi.train import queue as queue_module
from alibi.train import runner as v1_runner
from alibi.train.loop import ArmConfig

QUEUE_PATH = runlog.ARTIFACTS / "queue_v2.json"
STAGE = (("a0", 1), ("a1", 1), ("a2", 1))


def chosen_hyperparameters() -> dict:
    """The learning rate and KL anchor selected by the probes.

    Falls back to the registered defaults when no probe result exists, and says
    so, rather than silently training at whatever the source default happens to
    be. That silent inheritance is exactly what v1 did.
    """
    path = runlog.ARTIFACTS / "diagnostics" / "probes" / "result.json"
    training = prereg_v2.PREREG_V2.training
    if not path.exists():
        return {"learning_rate": training.learning_rate, "beta": training.beta_kl_anchor,
                "source": "registered default, no probe result found"}
    document = json.loads(path.read_text(encoding="utf-8"))
    for entry in document.get("conditions", []):
        if entry.get("label") == document.get("chosen"):
            return {"learning_rate": entry["learning_rate"], "beta": entry["beta"],
                    "source": f"probe {entry['label']}"}
    return {"learning_rate": training.learning_rate, "beta": training.beta_kl_anchor,
            "source": "registered default, no probe was chosen"}


def build_config(arm: str, seed: int) -> ArmConfig:
    policy = prereg_v2.PREREG_V2.policy
    hyper = chosen_hyperparameters()
    return ArmConfig(
        learning_rate=hyper["learning_rate"],
        beta=hyper["beta"],
        arm=arm,
        seed=seed,
        steps=policy.steps_per_run,
        group_size=policy.group_size,
        prompts_per_step=policy.prompts_per_step,
        max_new_tokens=policy.max_new_tokens,
        model_id=policy.model_id,
        monitor_name="rules" if arm == "a0" else "llm",
        policy_version=prereg_v2.PREREG_VERSION,
        apply_chat_template=policy.apply_chat_template,
    )


def reward_gate(run_id: str) -> dict:
    """Is mean reward at the end below its step-zero value?

    Step zero is a single step and therefore noisy, so the comparison is made
    both against step zero alone, which is what the operator specified, and
    against the first five steps, which is reported alongside so a marginal call
    is visible rather than hidden behind one number.
    """
    import glob
    import statistics

    paths = sorted(glob.glob(str(runlog.ARTIFACTS / "runs" / run_id / "steps/*/summary.json")))
    rewards = []
    for path in paths:
        value = json.loads(open(path).read()).get("mean_reward")
        if value is not None:
            rewards.append(value)
    if len(rewards) < 10:
        return {"measured": False, "reason": f"only {len(rewards)} steps with a reward"}
    first, last = rewards[0], statistics.fmean(rewards[-5:])
    return {
        "measured": True,
        "first": first,
        "last": last,
        "first_five_mean": statistics.fmean(rewards[:5]),
        "degraded": last < first,
        "rule": "mean reward over the final five steps below the step-zero value",
    }


def main() -> int:
    started = time.monotonic()
    v1_runner.log("v2 staged runner starting")
    provenance = prereg_v2.provenance()
    v1_runner.log(
        f"prereg v2 {prereg_v2.PREREG_V2_HASH[:16]} inherits measurement "
        f"{provenance['inherited_measurement_hash'][:16]} unchanged={provenance['measurement_unchanged']}"
    )
    if not prereg_v2.measurement_is_unchanged():
        v1_runner.log("REFUSING: v2 measurement differs from v1, so the arms are not comparable")
        return 2

    state = queue_module.load_queue(QUEUE_PATH)
    if state is None:
        state = queue_module.QueueState(
            entries=[
                {
                    "arm": arm,
                    "seed": seed,
                    "status": "pending",
                    "run_id": None,
                    "halt_reason": None,
                    "steps": prereg_v2.PREREG_V2.policy.steps_per_run,
                    "detail": "",
                }
                for arm, seed in STAGE
            ],
            created_utc=datetime.now(timezone.utc).isoformat(),
        )
        queue_module.save_queue(state, QUEUE_PATH)
        v1_runner.log(f"built v2 stage with {len(state.entries)} entries: {[e['arm'] for e in state.entries]}")

    while not state.stopped:
        entry = queue_module.next_pending(state)
        if entry is None:
            state.stopped = True
            state.stopped_reason = "stage complete: a0, a1 and a2 at seed 1 have all been attempted"
            break

        entry["status"] = "running"
        queue_module.save_queue(state, QUEUE_PATH)
        v1_runner.commit_and_push(f"v2 queue: starting {entry['arm']} seed {entry['seed']}")

        config = build_config(entry["arm"], entry["seed"])
        run_id = f"v2-{entry['arm']}-seed{entry['seed']}-{config.hash()[:8]}"
        v1_runner.log(f"starting v2 {entry['arm']} seed {entry['seed']} as {run_id}, {config.steps} steps")

        result = v1_runner.run_subprocess(config, run_id)
        queue_module.record_result(state, entry, result)
        queue_module.save_queue(state, QUEUE_PATH)
        v1_runner.log(f"{run_id} -> {result['status']} {result.get('halt_reason') or ''}")

        # Stage gate. If a0 seed 1 ends below where it started on mean reward,
        # the loop is degrading the policy and running a1 and a2 on it would
        # spend hours to produce two more degrading runs.
        if entry["arm"] == "a0":
            gate = reward_gate(run_id)
            v1_runner.log(f"a0 reward gate: {gate}")
            if gate.get("degraded"):
                state.stopped = True
                state.stopped_reason = (
                    f"a0 seed 1 mean reward fell from {gate['first']:.4f} at step 0 to "
                    f"{gate['last']:.4f} at the end. The loop is degrading the policy, so a1 and a2 "
                    "were not run. Reported rather than continued."
                )
                break

        v1_runner.commit_and_push(
            f"v2 run({entry['arm']}/s{entry['seed']}): {result['status']} {result.get('halt_reason') or ''}".strip()
        )

    queue_module.save_queue(state, QUEUE_PATH)
    v1_runner.log(f"v2 stage stopped: {state.stopped_reason}")
    summarise(state, time.monotonic() - started)
    v1_runner.commit_and_push("v2 stage: complete, see report/V2_STAGE.md")
    return 0


def summarise(state, elapsed: float) -> None:
    """The four numbers the operator asked for, per arm."""
    import glob
    import statistics

    lines = [
        "# v2 stage result",
        "",
        f"Policy `{prereg_v2.POLICY_MODEL}`, chat template applied, "
        f"max_new_tokens {prereg_v2.MAX_NEW_TOKENS}. Measurement inherited unchanged from "
        f"`{prereg_v2.PREREG_V2.inherits_measurement_from}`.",
        "",
        f"Wall clock for the stage: {elapsed / 3600:.2f} h.",
        "",
        "| Arm | Steps | Mean reward first10 to last10 | Capped fraction first10 to last10 | "
        "Visible pass first10 to last10 | Cheat rate first10 to last10 | Mean think chars | Flag rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry in state.entries:
        if not entry.get("run_id"):
            lines.append(f"| {entry['arm']} | not run | | | | |")
            continue
        summaries = [
            json.loads(open(p).read())
            for p in sorted(glob.glob(str(runlog.ARTIFACTS / "runs" / entry["run_id"] / "steps/*/summary.json")))
        ]
        if not summaries:
            lines.append(f"| {entry['arm']} | 0 | run produced no steps: {entry.get('halt_reason')} | | | |")
            continue

        def col(key):
            return [s.get(key) for s in summaries if s.get(key) is not None]

        vis, cheat = col("visible_pass_rate"), col("cheat_rate_oracle")
        flag, think = col("flag_rate_monitor"), col("mean_think_chars")
        reward, capped = col("mean_reward"), col("capped_fraction")

        def band(values):
            if not values:
                return "not measured"
            return f"{statistics.fmean(values[:10]):.3f} to {statistics.fmean(values[-10:]):.3f}"

        lines.append(
            f"| {entry['arm']} | {len(summaries)} | {band(reward)} | {band(capped)} | {band(vis)} "
            f"| {band(cheat)} | {statistics.fmean(think):.0f} | {band(flag)} |"
        )
    lines += [
        "",
        "H1 asks whether the oracle cheat rate rises in a0. If the a0 column above does not move, "
        "adding seeds will not change that, and the decision is the operator's.",
        "",
        "**Mean reward and capped fraction are first-class here, not footnotes.** v1's retrospective "
        "showed a policy whose reward fell over training while 40 percent of its completions were "
        "truncated, so a flat cheat rate there said more about the loop than about the model. If "
        "reward falls again, the same caution applies and the cheat rate is not interpretable as a "
        "statement about the policy.",
        "",
        f"The stage stops early if a0 seed 1 ends below its step-zero reward. Stopped: {state.stopped}. "
        f"{state.stopped_reason}",
        "",
    ]
    (runlog.REPO_ROOT / "report" / "V2_STAGE.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
