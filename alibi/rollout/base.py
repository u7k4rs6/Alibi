"""The sampler seam. The most important piece of week-one scaffolding.

docs/kickoff/02-technical-architecture.md section 3.1: nothing outside
`rollout/` touches a tokenizer's `generate`. If something needs text, it takes a
`Completion`. This will feel like a pointless wrapper this week and it is the
entire point in October, when the rollout path is swapped and the question
becomes what happened to the sampler and trainer logprob pair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# Re-exported so tests/test_reward_isolation.py can assert on the type the
# reward actually receives without importing training code.
from alibi.train.reward import ScoredCompletion  # noqa: F401

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class SampleParams:
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 8


@dataclass(frozen=True)
class Completion:
    """One generated completion, normalised.

    `think` is the designated reasoning region and `answer` is the text with it
    stripped. The arms differ only in which of these the monitor is shown, so
    the split is made once here rather than in each arm.
    """

    prompt_id: str
    text: str
    think: str | None
    answer: str
    token_ids: list[int] = field(default_factory=list)
    sampler_logprobs: list[float] = field(default_factory=list)
    # The prompt's token ids. Without these the trainer cannot condition on the
    # prompt, and v1's update did not: it maximised the unconditional likelihood
    # of the completion text.
    prompt_token_ids: list[int] = field(default_factory=list)
    finish_reason: str = "stop"

    @property
    def think_closed(self) -> bool:
        """An opened but unclosed think region is a format failure."""
        if THINK_OPEN not in self.text:
            return True
        return THINK_CLOSE in self.text


@runtime_checkable
class Sampler(Protocol):
    def generate(self, prompts: list[str], params: SampleParams, seed: int) -> list[Completion]: ...


def split_think(text: str) -> tuple[str | None, str]:
    """Separate the think region from the answer.

    An unclosed `<think>` yields the whole remainder as think and an empty
    answer, rather than silently treating reasoning as the answer, because
    which region the monitor reads is the independent variable of the whole
    experiment and must never be decided by a parsing accident.
    """
    match = _THINK_RE.search(text)
    if match:
        think = match.group(1).strip()
        answer = (text[: match.start()] + text[match.end() :]).strip()
        return think, answer
    if THINK_OPEN in text:
        head, _, tail = text.partition(THINK_OPEN)
        return tail.strip(), head.strip()
    return None, text.strip()


def extract_code(answer: str) -> str:
    """The Python the executor will run.

    Prefers a fenced block. Falls back to the whole answer, because a model that
    emits bare code is not committing a format error worth zeroing.
    """
    blocks = _CODE_FENCE_RE.findall(answer)
    if blocks:
        return max(blocks, key=len).strip()
    return answer.strip()


def has_code(answer: str) -> bool:
    code = extract_code(answer)
    return bool(code) and ("def " in code or "lambda" in code)


def make_completion(prompt_id: str, text: str, **kwargs) -> Completion:
    think, answer = split_think(text)
    return Completion(prompt_id=prompt_id, text=text, think=think, answer=answer, **kwargs)


def monitor_view(completion: Completion, arm: str) -> tuple[str, str]:
    """What this arm's monitor is shown. Returns (view_name, view_text).

    The monitor itself never learns which arm it is in, which keeps the arms
    comparable by construction, per architecture doc section 3.2.
    """
    if arm == "a0":
        return "none", ""
    if arm == "a1":
        return "think", completion.think or ""
    if arm == "a2":
        return "answer", completion.answer
    if arm == "a3":
        return "summary", completion.answer
    raise ValueError(f"unknown arm {arm!r}")
