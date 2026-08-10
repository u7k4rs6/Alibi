"""The one place a candidate is scored against a problem.

Everything that needs a pass fraction goes through `score_candidate`, so the
timeout budgets, the three state outcome handling and the indeterminate
accounting cannot drift between the gate, the trainer and the evaluator.

Timeouts are asymmetric on purpose. The visible harness runs three asserts and
five seconds is plenty. The held out harness runs about 105 inputs, and giving
it the same flat budget was measured on day 1 to time out three reference
solutions, which would have been published as correct code that does not
generalise. Held out therefore gets a generous per test budget and a wall clock
derived from the test count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from alibi.data.build import Problem
from alibi.env._runner import FAIL, INDETERMINATE, PASS
from alibi.env.executor import ExecResult, Executor
from alibi.env.tests import held_out_harness, program, visible_harness

# Visible: three asserts in the prompt. The security doc's 5 second default.
VISIBLE_WALL_CLOCK_SECONDS = 5.0
VISIBLE_PER_TEST_SECONDS = 2.0

# Held out: about 105 inputs. Generous per test, and a wall clock that scales
# with the number of tests plus a fixed allowance for interpreter and numpy
# start up, which is roughly 0.2 s on the day one host.
HELD_OUT_PER_TEST_SECONDS = 2.0
HELD_OUT_WALL_CLOCK_BASE_SECONDS = 15.0
HELD_OUT_WALL_CLOCK_PER_TEST_SECONDS = 0.5
HELD_OUT_WALL_CLOCK_CAP_SECONDS = 120.0


def held_out_wall_clock(n_tests: int) -> float:
    budget = HELD_OUT_WALL_CLOCK_BASE_SECONDS + HELD_OUT_WALL_CLOCK_PER_TEST_SECONDS * n_tests
    return min(budget, HELD_OUT_WALL_CLOCK_CAP_SECONDS)


@dataclass(frozen=True)
class Scored:
    """One candidate scored on one problem, with the auditor's fields present.

    `pass_fraction` is over determinate tests only and is None when nothing was
    determinate. `indeterminate_indices` names which tests, not just how many,
    because an auditor with artifacts and no GPU cannot re-run to find out.
    """

    task_id: int
    n_tests: int
    n_pass: int
    n_fail: int
    n_indeterminate: int
    indeterminate_indices: list[int]
    pass_fraction: float | None
    status: str
    detail: str = ""
    duration_seconds: float = 0.0
    outcomes: list[str] = field(default_factory=list)

    @property
    def determinate(self) -> int:
        return self.n_pass + self.n_fail

    @property
    def indeterminate_fraction(self) -> float:
        return self.n_indeterminate / self.n_tests if self.n_tests else 0.0

    def to_record(self) -> dict:
        return {
            "task_id": self.task_id,
            "n_tests": self.n_tests,
            "n_pass": self.n_pass,
            "n_fail": self.n_fail,
            "n_indeterminate": self.n_indeterminate,
            "indeterminate_indices": list(self.indeterminate_indices),
            "indeterminate_fraction": self.indeterminate_fraction,
            "pass_fraction": self.pass_fraction,
            "status": self.status,
            "detail": self.detail[:500],
            "duration_seconds": round(self.duration_seconds, 4),
        }


def _to_scored(task_id: int, result: ExecResult, harness_indices: list[int] | None = None) -> Scored:
    indices = [i for i, outcome in enumerate(result.outcomes) if outcome == INDETERMINATE]
    if harness_indices is not None:
        # Report the harness's own input indices rather than positions in the
        # filtered list, so an auditor can point at the exact MBPP+ input.
        indices = [harness_indices[i] for i in indices if i < len(harness_indices)]
    return Scored(
        task_id=task_id,
        n_tests=result.n_tests,
        n_pass=result.n_pass,
        n_fail=result.n_fail,
        n_indeterminate=result.n_indeterminate,
        indeterminate_indices=indices,
        pass_fraction=result.pass_fraction,
        status=result.status,
        detail=result.detail,
        duration_seconds=result.duration_seconds,
        outcomes=list(result.outcomes),
    )


def score_visible(problem: Problem, code: str, executor: Executor) -> Scored:
    result = executor.run(
        program(problem.test_imports, code),
        visible_harness(problem.visible_asserts),
        problem.n_visible,
        timeout_seconds=VISIBLE_WALL_CLOCK_SECONDS,
        per_test_timeout_seconds=VISIBLE_PER_TEST_SECONDS,
    )
    return _to_scored(problem.task_id, result)


def score_held_out(problem: Problem, code: str, executor: Executor) -> Scored:
    result = executor.run(
        program(problem.test_imports, code),
        held_out_harness(problem.plus_test_src, problem.held_out_indices),
        problem.n_held_out,
        timeout_seconds=held_out_wall_clock(problem.n_held_out),
        per_test_timeout_seconds=HELD_OUT_PER_TEST_SECONDS,
    )
    return _to_scored(problem.task_id, result, sorted(problem.held_out_indices))


def score_candidate(problem: Problem, code: str, executor: Executor) -> tuple[Scored, Scored]:
    return score_visible(problem, code, executor), score_held_out(problem, code, executor)


__all__ = [
    "FAIL",
    "INDETERMINATE",
    "PASS",
    "Scored",
    "held_out_wall_clock",
    "score_candidate",
    "score_held_out",
    "score_visible",
]
