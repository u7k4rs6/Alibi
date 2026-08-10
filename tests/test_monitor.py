"""A failure in the monitor path can never produce flagged=False.

This is the assertion the unattended brief singles out, and it is the one that
protects the primary metric. `flagged=False` lowers the penalty term and raises
the reward, so a monitor outage that resolved to unflagged would look exactly
like a policy that stopped cheating.
"""

from __future__ import annotations

import pytest

from alibi.monitor.base import (
    STATE_ERROR,
    STATE_OK,
    JudgeStats,
    Monitor,
    MonitorError,
    Verdict,
    cache_key,
    safe_judge,
)


class RaisingMonitor:
    name = "raiser"
    model_id = "test-model"
    prompt_template_version = "v1"

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def judge(self, view: str) -> Verdict:
        raise self.exc


class WrongTypeMonitor:
    name = "wrong-type"
    model_id = "test-model"
    prompt_template_version = "v1"

    def judge(self, view: str):
        return {"flagged": False}


class OkMonitor:
    name = "ok"
    model_id = "test-model"
    prompt_template_version = "v1"

    def judge(self, view: str) -> Verdict:
        return Verdict(flagged=False, score=0.1, reason="clean", model_id=self.model_id)


@pytest.mark.parametrize(
    "exc",
    [
        MonitorError("api down"),
        TimeoutError("read timed out"),
        ValueError("could not parse response"),
        KeyboardInterrupt(),
        SystemExit(1),
        RuntimeError("rate limited"),
    ],
)
def test_an_exception_in_the_monitor_path_cannot_yield_unflagged(exc: BaseException) -> None:
    """The load bearing test. Every exception type, including BaseException ones."""
    verdict = safe_judge(RaisingMonitor(exc), "some reasoning text")
    assert verdict.flagged is True
    assert verdict.state == STATE_ERROR
    assert verdict.is_error
    assert verdict.error


def test_a_monitor_returning_the_wrong_type_cannot_yield_unflagged() -> None:
    verdict = safe_judge(WrongTypeMonitor(), "view")
    assert verdict.flagged is True
    assert verdict.state == STATE_ERROR


def test_an_error_verdict_cannot_be_constructed_unflagged() -> None:
    """There is no path, not even a deliberate one, to an unflagged error."""
    with pytest.raises(ValueError, match="must be flagged"):
        Verdict(flagged=False, score=None, reason="", state=STATE_ERROR)


def test_from_error_does_not_take_a_flagged_argument() -> None:
    import inspect

    params = set(inspect.signature(Verdict.from_error).parameters)
    assert "flagged" not in params
    assert Verdict.from_error("boom").flagged is True


def test_an_error_verdict_cannot_claim_a_cache_hit() -> None:
    with pytest.raises(ValueError, match="cache hit"):
        Verdict(flagged=True, score=None, reason="", state=STATE_ERROR, cache_hit=True)


def test_an_unknown_verdict_state_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown monitor verdict state"):
        Verdict(flagged=True, score=None, reason="", state="probably_fine")


def test_a_healthy_monitor_still_passes_through() -> None:
    verdict = safe_judge(OkMonitor(), "view")
    assert verdict.state == STATE_OK
    assert verdict.flagged is False


def test_error_verdicts_count_toward_the_halt_statistic() -> None:
    stats = JudgeStats()
    stats.observe(safe_judge(OkMonitor(), "a"))
    stats.observe(safe_judge(RaisingMonitor(MonitorError("down")), "b"))
    assert stats.total == 2
    assert stats.errors == 1
    assert stats.error_fraction == 0.5


def test_cache_key_matches_the_architecture_doc() -> None:
    """sha256(monitor_name || model_id || prompt_template_version || view_text)."""
    first = cache_key("llm", "m-1", "v3", "text")
    assert first == cache_key("llm", "m-1", "v3", "text")
    assert first != cache_key("llm", "m-1", "v4", "text")
    assert first != cache_key("llm", "m-2", "v3", "text")
    assert first != cache_key("rules", "m-1", "v3", "text")
    assert first != cache_key("llm", "m-1", "v3", "other text")


def test_monitor_protocol_shape() -> None:
    assert isinstance(OkMonitor(), Monitor)
