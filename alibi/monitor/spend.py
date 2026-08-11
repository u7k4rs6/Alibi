"""Token accounting against the operator's USD 4.00 cap.

What is counted and what is not. The provider returns token usage per call, so
tokens are **measured**. Converting tokens to dollars needs a price per token
that this code does not have and will not invent, so:

  ALIBI_MONITOR_USD_PER_MTOK   set it and spend is computed and capped in USD
  unset                        spend is reported in tokens only, and the cap is
                               enforced on a token budget instead

An unmeasured value is absent, not zero, so with no price set the ledger says
`usd: null` with a reason rather than reporting a comforting 0.00.

The cap is a hard stop. When it is reached, `LLMMonitor` raises, `safe_judge`
turns that into a flagged error verdict, and the monitor error halt condition
fires within one step. That ordering is deliberate: running out of budget
degrades to "every completion flagged", which suppresses reward rather than
inflating it, and then halts. It can never resolve to unflagged.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from alibi import runlog

LEDGER_PATH = runlog.ARTIFACTS / "monitor_spend.json"

# The operator's cap.
USD_CAP = 4.00

# Fallback when no price is configured: a token budget chosen so that the cap
# cannot be silently blown past while spend is unpriced. 20 million tokens is
# far above what nine runs of this size should need, and the point is that the
# ledger is loud rather than that the number is tight.
TOKEN_CAP = 20_000_000


class SpendExceeded(RuntimeError):
    """The cap is reached. Raised into the monitor path, which flags and halts."""


@dataclass
class Ledger:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cache_hits: int = 0
    usd_per_mtok: float | None = None
    usd_cap: float = USD_CAP
    token_cap: int = TOKEN_CAP
    updated_utc: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def usd(self) -> float | None:
        if self.usd_per_mtok is None:
            return None
        return self.total_tokens / 1_000_000 * self.usd_per_mtok

    @property
    def usd_absent_reason(self) -> str | None:
        if self.usd_per_mtok is not None:
            return None
        return (
            "ALIBI_MONITOR_USD_PER_MTOK is not set, so no price per token is known. "
            "Tokens are measured; dollars are not estimated."
        )

    def would_exceed(self) -> str | None:
        spend = self.usd
        if spend is not None and spend >= self.usd_cap:
            return f"monitor spend {spend:.2f} USD has reached the {self.usd_cap:.2f} USD cap"
        if spend is None and self.total_tokens >= self.token_cap:
            return (
                f"monitor usage {self.total_tokens} tokens has reached the {self.token_cap} token cap, "
                "which stands in for the USD cap because no price per token is configured"
            )
        return None

    def to_dict(self) -> dict:
        document = asdict(self)
        document.update(
            {
                "total_tokens": self.total_tokens,
                "usd": self.usd,
                "usd_absent_reason": self.usd_absent_reason,
            }
        )
        return document


_LOCK = threading.Lock()


def _price() -> float | None:
    raw = os.environ.get("ALIBI_MONITOR_USD_PER_MTOK")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def load() -> Ledger:
    if LEDGER_PATH.exists():
        try:
            data = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
    else:
        data = {}
    return Ledger(
        prompt_tokens=int(data.get("prompt_tokens", 0)),
        completion_tokens=int(data.get("completion_tokens", 0)),
        calls=int(data.get("calls", 0)),
        cache_hits=int(data.get("cache_hits", 0)),
        usd_per_mtok=_price(),
    )


def _save(ledger: Ledger) -> None:
    ledger.updated_utc = datetime.now(timezone.utc).isoformat()
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER_PATH.with_suffix(".json.partial")
    tmp.write_text(json.dumps(ledger.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.rename(LEDGER_PATH)


def check() -> None:
    """Raise if the cap is already reached. Called before every live call."""
    reason = load().would_exceed()
    if reason:
        raise SpendExceeded(reason)


def record(usage: dict | None, cache_hit: bool = False) -> Ledger:
    """Add one call's measured usage to the ledger."""
    with _LOCK:
        ledger = load()
        ledger.calls += 1
        if cache_hit:
            ledger.cache_hits += 1
        if usage:
            ledger.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            ledger.completion_tokens += int(usage.get("completion_tokens") or 0)
        _save(ledger)
        return ledger
