"""Halt conditions fire, and none of them is a warning."""

from __future__ import annotations

import pytest

from alibi import prereg
from alibi.halt import (
    DEGENERATE_POLICY,
    INDETERMINATE_RATE,
    KL_SPIKE,
    MONITOR_ERROR_RATE,
    Halt,
    StepStats,
    check_degenerate,
    check_indeterminate,
    check_kl,
    check_monitor_errors,
    check_step,
    should_stop_queue,
    write_halt,
)


def stats(**kwargs) -> StepStats:
    base = {"run_id": "test", "arm": "a1", "seed": 1, "step": 5}
    base.update(kwargs)
    return StepStats(**base)


def test_monitor_error_rate_above_two_percent_halts() -> None:
    with pytest.raises(Halt) as caught:
        check_monitor_errors(stats(monitor_judgements=100, monitor_errors=3))
    assert caught.value.reason == MONITOR_ERROR_RATE
    assert caught.value.evidence["fraction"] == 0.03


def test_monitor_error_rate_at_the_limit_does_not_halt() -> None:
    check_monitor_errors(stats(monitor_judgements=100, monitor_errors=2))


def test_indeterminate_above_five_percent_halts() -> None:
    with pytest.raises(Halt) as caught:
        check_indeterminate(stats(held_out_executions=100, held_out_indeterminate=6))
    assert caught.value.reason == INDETERMINATE_RATE


def test_indeterminate_at_the_limit_does_not_halt() -> None:
    check_indeterminate(stats(held_out_executions=100, held_out_indeterminate=5))


def test_kl_spike_halts_against_its_own_baseline() -> None:
    history = [0.01] * prereg.PREREG.halt.kl_baseline_steps
    check_kl(stats(kl=0.02), history)
    with pytest.raises(Halt) as caught:
        check_kl(stats(kl=0.04), history)
    assert caught.value.reason == KL_SPIKE
    assert caught.value.evidence["baseline_median"] == 0.01


def test_kl_does_not_halt_before_the_baseline_window_is_full() -> None:
    """Otherwise step 2 halts on noise."""
    check_kl(stats(kl=99.0), [0.01, 0.01])


def test_identical_completions_in_a_group_halt() -> None:
    same = ["def f():\n    return 1\n" * 3] * 4
    with pytest.raises(Halt) as caught:
        check_degenerate(stats(completion_texts=same, group_size=4))
    assert caught.value.reason == DEGENERATE_POLICY
    assert "identical" in caught.value.message


def test_a_varied_group_does_not_halt() -> None:
    texts = [f"def f():\n    return {i}\n{'x' * 40}" for i in range(4)]
    check_degenerate(stats(completion_texts=texts, group_size=4))


def test_short_completions_halt() -> None:
    with pytest.raises(Halt) as caught:
        check_degenerate(stats(completion_texts=["a", "b", "c", "d"], group_size=4))
    assert caught.value.reason == DEGENERATE_POLICY
    assert "length" in caught.value.message


def test_check_step_runs_every_condition() -> None:
    """A clean step passes all of them."""
    texts = [f"def solve():\n    return {i}\n{'#' * 60}" for i in range(4)]
    check_step(
        stats(
            monitor_judgements=8,
            monitor_errors=0,
            held_out_executions=8,
            held_out_indeterminate=0,
            kl=0.01,
            completion_texts=texts,
            group_size=4,
        ),
        kl_history=[],
    )


def test_three_consecutive_halts_for_the_same_reason_stop_the_queue() -> None:
    assert should_stop_queue([MONITOR_ERROR_RATE] * 3) is True
    assert should_stop_queue([MONITOR_ERROR_RATE, KL_SPIKE, MONITOR_ERROR_RATE]) is False
    assert should_stop_queue([MONITOR_ERROR_RATE, MONITOR_ERROR_RATE]) is False


def test_halt_writes_a_file_with_the_reason_run_id_step_and_numbers(tmp_path, monkeypatch) -> None:
    import alibi.halt as halt_module

    monkeypatch.setattr(halt_module, "HALT_PATH", tmp_path / "HALT.md")
    halt = Halt(MONITOR_ERROR_RATE, "too many monitor errors", {"fraction": 0.07, "limit": 0.02})
    path = write_halt(halt, run_id="a1-seed1-xyz", step=42)
    text = path.read_text(encoding="utf-8")
    assert MONITOR_ERROR_RATE in text
    assert "a1-seed1-xyz" in text
    assert "42" in text
    assert "0.07" in text
    assert "FAILED" in text


def test_halt_thresholds_come_from_the_frozen_registration() -> None:
    """The halt module decides whether, never how much."""
    assert prereg.PREREG.halt.max_monitor_error_fraction == 0.02
    assert prereg.PREREG.halt.max_indeterminate_fraction == 0.05
    assert prereg.PREREG.halt.kl_spike_multiple == 3.0
