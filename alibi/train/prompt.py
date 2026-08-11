"""Prompt construction, and the type that makes the v1 fault impossible to repeat.

The v1 matrix built prompts as a bare string and handed that to the sampler,
never applying the tokenizer's chat template. Measured consequence: Qwen3-0.6B,
a model post-trained to emit a designated thought region, emitted one on 0 of 16
completions under the raw string and 16 of 16 under its own template. Arm A1
would therefore have read an empty view even with a correctly chosen policy.

So the sampler no longer accepts a string. It accepts a `Prompt`, which records
whether the chat template was applied and refuses to be constructed by accident.
`tests/test_prompt_contract.py` fails if a bare string ever reaches the sampler.

This is deliberately a type-level guard rather than a comment or a review habit.
The v1 fault was invisible precisely because a string is a string.
"""

from __future__ import annotations

from dataclasses import dataclass


class RawStringPassedToSampler(TypeError):
    """Raised when a bare string reaches the sampler.

    Named so the failure says what went wrong rather than surfacing as an
    AttributeError three frames deeper.
    """


@dataclass(frozen=True)
class Prompt:
    """A prompt that knows how it was rendered."""

    prompt_id: str
    rendered: str
    chat_templated: bool
    source_text: str

    def __post_init__(self) -> None:
        if not isinstance(self.rendered, str) or not self.rendered:
            raise ValueError("a Prompt must carry non-empty rendered text")


def require_prompts(prompts) -> list[Prompt]:
    """Every entry must be a Prompt. A bare string is the v1 fault and is refused."""
    checked = []
    for index, item in enumerate(prompts):
        if isinstance(item, str):
            raise RawStringPassedToSampler(
                f"prompt {index} is a bare string. The sampler takes Prompt objects so that whether "
                "the chat template was applied is recorded rather than assumed. Building a raw "
                "string and sampling it is the v1 fault that made arm A1 read an empty view: "
                "Qwen3-0.6B emits a thought region on 0 of 16 completions from a raw string and "
                "16 of 16 through its own chat template."
            )
        if not isinstance(item, Prompt):
            raise RawStringPassedToSampler(
                f"prompt {index} is a {type(item).__name__}, not a Prompt"
            )
        checked.append(item)
    return checked


def task_text(problem, system_prompt: str) -> str:
    """The task as prose. Identical between v1 and v2, so only the rendering differs."""
    return (
        f"{system_prompt}\n\n"
        f"{problem.description}\n\n"
        f"Your solution must define `{problem.entry_point}` and pass these tests:\n\n"
        + "\n".join(problem.visible_asserts)
        + "\n"
    )


def build_prompt(problem, system_prompt: str, tokenizer=None, use_chat_template: bool = False) -> Prompt:
    """Render one problem into a Prompt.

    With `use_chat_template`, the text goes through the tokenizer's own chat
    template, which is what a post-trained thinking model needs in order to
    behave as it was trained to. Without it, the text is passed through
    unchanged, which is exactly what v1 did and is kept so v1's remaining runs
    stay byte-identical to the ones already completed.
    """
    text = task_text(problem, system_prompt)
    if not use_chat_template:
        return Prompt(
            prompt_id=str(problem.task_id), rendered=text, chat_templated=False, source_text=text
        )
    if tokenizer is None:
        raise ValueError("use_chat_template requires a tokenizer")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )
    return Prompt(
        prompt_id=str(problem.task_id), rendered=rendered, chat_templated=True, source_text=text
    )
