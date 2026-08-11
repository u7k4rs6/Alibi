"""A running false positive estimate: what the judge says about known-honest code.

Added at the operator's instruction after the monitor override. The reason it
matters is specific to this experiment rather than general hygiene: the selected
judge flags some fraction of genuine reference solutions, and every such flag is
a penalty applied to honest behaviour. H3 compares how much A1 and A2 reduce
cheating, and a monitor that punishes honest work at a material rate changes
what "reduce cheating" even means, because the policy can lower its penalty by
becoming less legible rather than less dishonest.

Method. Each step, a deterministic rotating sample of MBPP+ reference solutions
is put through the same monitor, with the same view the arm uses, and the flag
rate is recorded. Reference solutions are the only completions in this project
that are known-honest by construction.

Three honesty notes that belong with the number:

  1. The monitor is temperature 0 and cached, so a fixed probe set would give a
     constant. The sample rotates deterministically by step so the estimate is
     unbiased across a run, at the cost of per-step sampling noise. Read the
     trend, not a single step.
  2. Reference solutions are cleaner and more idiomatic than anything a 0.5B
     policy emits, so this is a lower bound on the false positive rate against
     real completions.
  3. These judgements are excluded from `flag_rate_monitor`, which is the
     published metric, because they are not policy completions. Their errors are
     included in the monitor error halt statistic, because an erroring monitor
     is an erroring monitor regardless of what it was shown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from alibi.monitor.base import safe_judge

DEFAULT_PROBES_PER_STEP = 4


@dataclass
class HonestProbe:
    n: int = 0
    flagged: int = 0
    errors: int = 0
    task_ids: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def flag_rate(self) -> float | None:
        """Flag rate over judgements that actually returned a verdict.

        Errors are excluded from the denominator here, and counted separately,
        so an outage does not read as a false positive spike.
        """
        determinate = self.n - self.errors
        return self.flagged / determinate if determinate else None

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "flagged": self.flagged,
            "errors": self.errors,
            "flag_rate": self.flag_rate,
            "task_ids": self.task_ids,
            "reasons": self.reasons[:3],
        }


def probe_honest(monitor, problems, arm: str, step: int, k: int = DEFAULT_PROBES_PER_STEP) -> HonestProbe:
    """Judge k known-honest reference solutions, rotating deterministically by step."""
    from alibi.rollout.base import monitor_view

    result = HonestProbe()
    if not problems or arm == "a0":
        # a0 has no monitor, so there is no false positive rate to estimate.
        return result

    for offset in range(k):
        problem = problems[(step * k + offset) % len(problems)]
        # The arm's own view, so the estimate is comparable to the arm's flag
        # rate. A reference solution has no think region, so a1 sees the code as
        # its "reasoning", which is the closest honest analogue available.
        view = problem.reference_code if arm != "a1" else problem.reference_code
        verdict = safe_judge(monitor, view)
        result.n += 1
        result.task_ids.append(problem.task_id)
        if verdict.is_error:
            result.errors += 1
            continue
        if verdict.flagged:
            result.flagged += 1
            result.reasons.append(f"task {problem.task_id}: {verdict.reason[:160]}")
    return result
