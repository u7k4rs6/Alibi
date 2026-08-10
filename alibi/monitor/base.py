"""The Monitor protocol and the Verdict, with error as a first class state.

The rule this module exists to enforce: **a monitor that failed is not a monitor
that saw nothing wrong.** An outage, a timeout, a rate limit or a parse failure
must never resolve to `flagged=False`, because `flagged=False` lowers the
penalty term and therefore raises the reward. That would make an API problem
look like a policy that stopped cheating, which is precisely the result this
project is trying to measure honestly.

So `Verdict` has no public constructor that can produce an unflagged error. The
only way to build an error verdict is `Verdict.from_error`, which hard codes
`flagged=True`, and `__post_init__` refuses any ERROR verdict that is not
flagged. tests/test_monitor.py asserts that an exception raised anywhere in the
judging path still yields `flagged=True`.

Verdict states:

  ok         the monitor answered and the answer was parsed
  error      the monitor did not answer, or its answer could not be parsed.
             Always flagged. Counted toward the halt condition.
  cache_hit  is a property of an ok verdict, not a separate state
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

STATE_OK = "ok"
STATE_ERROR = "error"
VERDICT_STATES = (STATE_OK, STATE_ERROR)


class MonitorError(RuntimeError):
    """Raised inside a monitor implementation. Always converted to an error verdict."""


@dataclass(frozen=True)
class Verdict:
    """A monitor's judgement of one view of one completion."""

    flagged: bool
    score: float | None
    reason: str
    cache_hit: bool = False
    state: str = STATE_OK
    raw_response: str = ""
    error: str = ""
    latency_seconds: float = 0.0
    model_id: str | None = None
    prompt_template_version: str = ""

    def __post_init__(self) -> None:
        if self.state not in VERDICT_STATES:
            raise ValueError(f"unknown monitor verdict state {self.state!r}")
        if self.state == STATE_ERROR and not self.flagged:
            # The single most important assertion in the monitor path.
            raise ValueError(
                "a monitor error verdict must be flagged. An error is not evidence "
                "of innocence, and letting it read as unflagged would lower the "
                "penalty term and turn an outage into an apparent reduction in "
                "cheating."
            )
        if self.state == STATE_ERROR and self.cache_hit:
            raise ValueError("an error verdict cannot be a cache hit")

    @classmethod
    def from_error(cls, error: str, model_id: str | None = None, template_version: str = "", latency: float = 0.0):
        """The only way to build an error verdict. Flagged is not a parameter."""
        return cls(
            flagged=True,
            score=None,
            reason=f"monitor error, counted as flagged: {error}"[:500],
            cache_hit=False,
            state=STATE_ERROR,
            raw_response="",
            error=error[:1000],
            latency_seconds=latency,
            model_id=model_id,
            prompt_template_version=template_version,
        )

    @property
    def is_error(self) -> bool:
        return self.state == STATE_ERROR


@runtime_checkable
class Monitor(Protocol):
    name: str
    model_id: str | None
    prompt_template_version: str

    def judge(self, view: str) -> Verdict: ...


def cache_key(monitor_name: str, model_id: str | None, template_version: str, view: str) -> str:
    """sha256(monitor_name || model_id || prompt_template_version || view_text).

    Exactly the key from docs/kickoff/02-technical-architecture.md section 7.
    """
    payload = "\x00".join([monitor_name, model_id or "", template_version, view])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class JudgeStats:
    """Per step monitor accounting, for the halt condition and the report."""

    total: int = 0
    errors: int = 0
    cache_hits: int = 0
    latencies: list[float] = field(default_factory=list)

    def observe(self, verdict: Verdict) -> None:
        self.total += 1
        if verdict.is_error:
            self.errors += 1
        if verdict.cache_hit:
            self.cache_hits += 1
        self.latencies.append(verdict.latency_seconds)

    @property
    def error_fraction(self) -> float:
        return self.errors / self.total if self.total else 0.0

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.total if self.total else 0.0

    @property
    def mean_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    def to_dict(self) -> dict:
        return {
            "judgements": self.total,
            "errors": self.errors,
            "error_fraction": self.error_fraction,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hit_rate,
            "mean_latency_seconds": self.mean_latency,
        }


def safe_judge(monitor: Monitor, view: str) -> Verdict:
    """Call a monitor so that no exception can escape as an unflagged verdict.

    Every monitor call in the codebase goes through here. A monitor that raises,
    returns the wrong type, or returns a malformed verdict yields a flagged
    error verdict rather than propagating or defaulting to False.
    """
    import time

    started = time.monotonic()
    model_id = getattr(monitor, "model_id", None)
    template = getattr(monitor, "prompt_template_version", "")
    try:
        verdict = monitor.judge(view)
    except BaseException as exc:  # noqa: BLE001 - the whole point of this function
        return Verdict.from_error(
            f"{type(exc).__name__}: {exc}", model_id, template, time.monotonic() - started
        )
    if not isinstance(verdict, Verdict):
        return Verdict.from_error(
            f"monitor returned {type(verdict).__name__}, not a Verdict",
            model_id,
            template,
            time.monotonic() - started,
        )
    return verdict
