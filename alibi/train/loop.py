"""GRPO wiring, resumable by run id, with the halt conditions armed.

Design constraints this file exists to satisfy, from the unattended brief:

  Resumable       the training host kills sessions without warning, so state
                  lives on disk after every step and a rerun with the same run
                  id continues rather than restarting.
  Halting         every per step condition is checked, and a halt writes HALT.md
                  and fails this run without stopping the queue.
  Decoupled       held out scoring is measurement only and never enters the
                  reward, so it does not belong in the training critical path.
                  It runs in a worker pool keyed by completion id and its
                  results are joined back before the step record is written.
  Auditable       every field in alibi.records is written per completion, at
                  write time, including all three sensitivity verdicts.

The oracle is applied here, downstream of the reward, in the logging record.
`reward_fn` never sees it.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from alibi import halt as halt_module
from alibi import prereg, records, runlog
from alibi.data.build import Problem
from alibi.env.executor import Executor
from alibi.env.scoring import score_held_out, score_visible
from alibi.monitor.base import JudgeStats, safe_judge
from alibi.monitor.rules import StubMonitor
from alibi.rollout.base import Completion, extract_code, has_code, monitor_view
from alibi.train.reward import ScoredCompletion, reward_breakdown

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class ArmConfig:
    """Frozen per run. Serialised into every run directory."""

    arm: str
    seed: int
    steps: int = 100
    group_size: int = 8
    prompts_per_step: int = 4
    max_new_tokens: int = 512
    temperature: float = 1.0
    learning_rate: float = 1e-5
    lora_r: int = 16
    lora_alpha: int = 32
    model_id: str = MODEL_ID
    monitor_name: str = "rules"
    dry_run: bool = False
    # Held out scoring is measurement only. Subsampling it does not touch the
    # reward, and the rate is recorded so an auditor can see the denominator.
    held_out_sample_rate: float = 1.0
    checkpoint_every: int = 10

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        return runlog.config_hash(self.to_dict())


@dataclass
class RunState:
    """What a resumed run needs to continue. Written after every step."""

    run_id: str
    arm: str
    seed: int
    step: int = 0
    status: str = "running"  # running | complete | failed
    halt_reason: str | None = None
    kl_history: list[float] = field(default_factory=list)
    # (indeterminate, executions) per step, for the lagging-window halt check.
    indeterminate_window: list[list[int]] = field(default_factory=list)
    started_utc: str = ""
    updated_utc: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def state_path(run_dir: Path) -> Path:
    return run_dir / "state.json"


def load_state(run_dir: Path) -> RunState | None:
    path = state_path(run_dir)
    if not path.exists():
        return None
    return RunState(**json.loads(path.read_text(encoding="utf-8")))


def save_state(run_dir: Path, state: RunState) -> None:
    from datetime import datetime, timezone

    state.updated_utc = datetime.now(timezone.utc).isoformat()
    # Written atomically: a session killed mid write must not leave a state file
    # that cannot be parsed, because that would make the run unresumable.
    tmp = state_path(run_dir).with_suffix(".json.partial")
    tmp.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    tmp.rename(state_path(run_dir))


def build_monitor(config: ArmConfig):
    """The monitor for this arm, with the substitutions that are not allowed refused.

    Section 6 of the unattended brief: the rules monitor may not be substituted
    for the LLM monitor in a monitored arm. That is enforced here rather than
    left to discipline, because the substitution would be invisible in the
    resulting curves.
    """
    from alibi.monitor.rules import RulesMonitor

    if config.arm == "a0":
        # A0 has no monitor by definition. The stub keeps the code path
        # identical across arms so a0 is not a different program.
        return StubMonitor()
    if config.dry_run:
        return StubMonitor()
    if config.monitor_name == "rules":
        return RulesMonitor()
    if config.monitor_name == "llm":
        from alibi.monitor.llm import LLMMonitor
        from alibi.monitor.selection import load_selection

        selection = load_selection()
        if selection is None:
            raise ValueError(
                "no monitor has been selected. Run `alibi monitor select` first: a monitored arm "
                "may not start against an unrecorded judge, because the model identity is not "
                "covered by the prereg hash and would otherwise be unrecoverable from artifacts."
            )
        return LLMMonitor(model_id=selection["model_id"], base_url=selection["base_url"])
    raise ValueError(f"unknown monitor {config.monitor_name!r}")


def _executor_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


class HeldOutPool:
    """Held out scoring, off the training critical path.

    Held out results never enter the reward, so nothing in the step blocks on
    them except the moment the step record is written. Keyed by completion id so
    a result can never be attached to the wrong completion.
    """

    def __init__(self, executor: Executor, workers: int | None = None) -> None:
        self.executor = executor
        # Half the cores, not all but one. The held-out pool competes with
        # generation for CPU, and oversubscribing it is what turns a slow
        # completion into a timeout, which is indeterminate, which is a halt.
        self.pool = ThreadPoolExecutor(max_workers=workers or max(2, (os.cpu_count() or 4) // 2))
        self.pending: dict[str, object] = {}

    def submit(self, key: str, problem: Problem, code: str) -> None:
        self.pending[key] = self.pool.submit(score_held_out, problem, code, self.executor)

    def collect(self) -> dict:
        out = {}
        for key, future in self.pending.items():
            try:
                out[key] = future.result()
            except Exception as exc:  # noqa: BLE001 - a scoring crash is indeterminate, not a fail
                out[key] = exc
        self.pending.clear()
        return out

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


def score_and_record(
    *,
    config: ArmConfig,
    run_id: str,
    step: int,
    completions: list[Completion],
    problems: dict[str, Problem],
    executor: Executor,
    monitor,
    held_out_pool: HeldOutPool | None,
    git: dict,
    eligibility_hash: str | None,
) -> tuple[list[dict], JudgeStats, halt_module.StepStats]:
    """Score one step's completions and build their records.

    Order matters. The reward is computed from the visible score and the monitor
    verdict only. The oracle is applied afterwards, to the record, and the
    reward has already been returned by then.
    """
    stats = JudgeStats()
    step_stats = halt_module.StepStats(
        run_id=run_id, arm=config.arm, seed=config.seed, step=step, group_size=config.group_size
    )

    visible_scores = {}
    monitor_verdicts = {}
    codes = {}
    for index, completion in enumerate(completions):
        key = f"{completion.prompt_id}#{index}"
        problem = problems[completion.prompt_id]
        code = extract_code(completion.answer)
        codes[key] = code
        visible_scores[key] = score_visible(problem, code, executor)

        view_name, view_text = monitor_view(completion, config.arm)
        if config.arm == "a0":
            from alibi.monitor.base import Verdict

            verdict = Verdict(
                flagged=False,
                score=0.0,
                reason="arm a0 has no monitor by design",
                cache_hit=True,
                model_id=None,
                prompt_template_version="none",
            )
        else:
            verdict = safe_judge(monitor, view_text)
        monitor_verdicts[key] = (view_name, verdict)
        stats.observe(verdict)

        if held_out_pool is not None:
            held_out_pool.submit(key, problem, code)

    held_out_results = held_out_pool.collect() if held_out_pool is not None else {}

    from alibi.env.oracle import CheatOracle

    oracle = CheatOracle()

    out = []
    for index, completion in enumerate(completions):
        key = f"{completion.prompt_id}#{index}"
        problem = problems[completion.prompt_id]
        visible = visible_scores[key]
        view_name, verdict = monitor_verdicts[key]
        code = codes[key]

        held_out = held_out_results.get(key)
        if held_out is None or isinstance(held_out, BaseException):
            from alibi.env.scoring import Scored

            detail = f"held out scoring failed: {held_out}" if held_out is not None else "not scored this step"
            held_out = Scored(
                task_id=problem.task_id,
                n_tests=problem.n_held_out,
                n_pass=0,
                n_fail=0,
                n_indeterminate=problem.n_held_out,
                indeterminate_indices=sorted(problem.held_out_indices),
                pass_fraction=None,
                status="indeterminate",
                detail=detail,
            )

        scored = ScoredCompletion(
            prompt_id=completion.prompt_id,
            text=completion.text,
            answer=completion.answer,
            has_code=has_code(completion.answer),
            think_closed=completion.think_closed,
            visible_pass_fraction=visible.pass_fraction,
            visible_indeterminate=visible.n_indeterminate,
            monitor_flagged=verdict.flagged,
            monitor_state=verdict.state,
            extracted_code=code,
            think=completion.think,
        )
        breakdown = reward_breakdown(scored, prereg.PREREG.lambda_monitor)

        # The oracle runs here, after the reward is already a number.
        oracle_verdict = oracle.judge(
            code,
            problem.visible_asserts,
            problem.entry_point,
            visible.pass_fraction if visible.pass_fraction is not None else 0.0,
            held_out.pass_fraction if held_out.pass_fraction is not None else 0.0,
        )
        behavioural = prereg.cheated(
            held_out.pass_fraction, visible.pass_fraction, prereg.PREREG.oracle.held_out_cheat_threshold
        )

        record = records.from_scored(
            visible,
            held_out,
            run_id=run_id,
            arm=config.arm,
            seed=config.seed,
            step=step,
            prompt_id=completion.prompt_id,
            completion_idx=index,
            git_revision=git.get("revision"),
            git_dirty=git.get("dirty"),
            config_hash=config.hash(),
            eligibility_hash=eligibility_hash,
            text=completion.text,
            think=completion.think,
            answer=completion.answer,
            monitor_view_name=view_name,
            finish_reason=completion.finish_reason,
            held_out_scored_async=held_out_pool is not None,
            oracle_behavioural=behavioural,
            oracle_structural=oracle_verdict.structural,
            oracle_agreement=oracle_verdict.agreement,
            cheat_form=oracle_verdict.cheat_form,
            oracle_signals=oracle_verdict.signals,
            monitor_name=getattr(monitor, "name", "none") if config.arm != "a0" else "none",
            monitor_model_id=verdict.model_id,
            monitor_prompt_template_version=verdict.prompt_template_version,
            monitor_verdict_state=verdict.state,
            monitor_flagged=verdict.flagged,
            monitor_score=verdict.score,
            monitor_reason=verdict.reason,
            monitor_raw_response=verdict.raw_response,
            monitor_cache_hit=verdict.cache_hit,
            monitor_latency_seconds=verdict.latency_seconds,
            monitor_error=verdict.error,
            **breakdown,
        )
        out.append(record.to_dict())

        step_stats.held_out_executions += held_out.n_tests
        step_stats.held_out_indeterminate += held_out.n_indeterminate
        step_stats.completion_texts.append(completion.text)

    step_stats.monitor_judgements = stats.total
    step_stats.monitor_errors = stats.errors
    return out, stats, step_stats


def write_step(run_dir: Path, step: int, step_records: list[dict], step_summary: dict) -> Path:
    """completions.jsonl plus the step summary. Scrubbed at write time."""
    step_dir = run_dir / "steps" / f"{step:05d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    with open(step_dir / "completions.jsonl", "w", encoding="utf-8") as handle:
        for record in step_records:
            missing = records.validate(record)
            if missing:
                raise ValueError(f"completion record is missing required fields: {missing}")
            handle.write(json.dumps(runlog.scrub(record), sort_keys=True) + "\n")
    runlog.write_json(step_dir / "summary.json", step_summary)
    return step_dir


def write_logprobs(step_dir: Path, rows: list[dict]) -> Path | None:
    """logprobs.parquet: prompt_id, completion_idx, position, token_id, sampler and trainer logprob.

    Costs one column today and a full rerun in October, per architecture doc
    section 4. Written even when the two are identical on the inline path.
    """
    if not rows:
        return None
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    path = step_dir / "logprobs.parquet"
    pq.write_table(table, path)
    return path


def host_telemetry() -> dict:
    """GPU temperature, clocks and host load, where available.

    This machine is a laptop under sustained load for days. Thermal drift shows
    up first as held out timeouts, which is why indeterminate rate is both a
    halt condition and a required figure panel.
    """
    telemetry: dict = {"load_average": None, "gpu": None}
    try:
        telemetry["load_average"] = list(os.getloadavg())
    except OSError:
        pass
    try:
        import subprocess

        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,clocks.sm,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            fields = [f.strip() for f in out.stdout.strip().splitlines()[0].split(",")]
            telemetry["gpu"] = {
                "temperature_c": _to_number(fields[0]),
                "clock_sm_mhz": _to_number(fields[1]),
                "utilization_pct": _to_number(fields[2]),
                "memory_used_mib": _to_number(fields[3]),
            }
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return telemetry


def _to_number(text: str):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def step_summary(
    *,
    config: ArmConfig,
    run_id: str,
    step: int,
    step_records: list[dict],
    stats: JudgeStats,
    step_stats: halt_module.StepStats,
    kl: float | None,
    duration: float,
    honest_probe: dict | None = None,
) -> dict:
    """Everything the console line and the report need, computed once."""
    n = len(step_records)

    def mean(key):
        values = [r[key] for r in step_records if r[key] is not None]
        return sum(values) / len(values) if values else None

    cheat_at = {}
    for threshold in prereg.PREREG.oracle.sensitivity_thresholds:
        label = f"{threshold:.2f}"
        verdicts = [r["cheat_at_threshold"].get(label) for r in step_records]
        determinate = [v for v in verdicts if v is not None]
        cheat_at[label] = {
            "rate": (sum(determinate) / len(determinate)) if determinate else None,
            "n_determinate": len(determinate),
            "n_unmeasured": len(verdicts) - len(determinate),
        }

    forms: dict[str, int] = {}
    for record in step_records:
        for form in record["cheat_form"]:
            forms[form] = forms.get(form, 0) + 1

    flagged = [r["monitor_flagged"] for r in step_records]
    flag_rate = sum(flagged) / len(flagged) if flagged else None
    primary = cheat_at[f"{prereg.PREREG.oracle.held_out_cheat_threshold:.2f}"]["rate"]

    return {
        "run_id": run_id,
        "arm": config.arm,
        "seed": config.seed,
        "step": step,
        "n_completions": n,
        "mean_reward": mean("reward"),
        "visible_pass_rate": mean("visible_pass_fraction"),
        "held_out_pass_rate": mean("held_out_pass_fraction"),
        "cheat_rate_oracle": primary,
        "cheat_rate_at_threshold": cheat_at,
        "flag_rate_monitor": flag_rate,
        "obfuscation_gap": (primary - flag_rate) if (primary is not None and flag_rate is not None) else None,
        "cheat_form_counts": dict(sorted(forms.items())),
        "kl": kl,
        "mean_completion_chars": mean("completion_chars"),
        "mean_think_chars": mean("think_chars"),
        "monitor": stats.to_dict(),
        # Running false positive estimate on known-honest reference solutions.
        # Excluded from flag_rate_monitor, which is the published metric,
        # because these are not policy completions.
        "honest_probe": honest_probe or {"n": 0, "flag_rate": None},
        "monitor_spend": _spend_snapshot(),
        "held_out_indeterminate_fraction": step_stats.indeterminate_fraction,
        "held_out_executions": step_stats.held_out_executions,
        "held_out_indeterminate": step_stats.held_out_indeterminate,
        "duration_seconds": duration,
        "telemetry": host_telemetry(),
        "prereg_hash": prereg.PREREG_HASH,
        "config_hash": config.hash(),
    }


def _spend_snapshot() -> dict:
    from alibi.monitor import spend

    ledger = spend.load()
    return {
        "total_tokens": ledger.total_tokens,
        "calls": ledger.calls,
        "cache_hits": ledger.cache_hits,
        "usd": ledger.usd,
        "usd_absent_reason": ledger.usd_absent_reason,
        "usd_cap": ledger.usd_cap,
    }


def console_line(summary: dict) -> str:
    """The fixed width line from docs/kickoff/04-cli-and-report-spec.md section 2."""

    def fmt(value, width=4):
        return "  n/a" if value is None else f"{value:.{width - 2}f}"

    return (
        f"step {summary['step']:03d} | arm {summary['arm']} seed {summary['seed']} "
        f"| reward {fmt(summary['mean_reward'])} | vis {fmt(summary['visible_pass_rate'])} "
        f"| held {fmt(summary['held_out_pass_rate'])} | oracle {fmt(summary['cheat_rate_oracle'])} "
        f"| flag {fmt(summary['flag_rate_monitor'])} | gap {fmt(summary['obfuscation_gap'])} "
        f"| kl {fmt(summary['kl'], 5)} | cache {fmt(summary['monitor']['cache_hit_rate'])} "
        f"| indet {fmt(summary['held_out_indeterminate_fraction'])} "
        f"| fp {fmt((summary.get('honest_probe') or {}).get('flag_rate'))}"
    )
