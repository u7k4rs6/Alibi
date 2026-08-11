"""A bare string must never reach the sampler.

This is the test the v1 fault earns. Arm A1 read an empty view for four complete
runs because `build_prompt` returned a raw string and nothing applied the
tokenizer's chat template. Measured: Qwen3-0.6B emits a designated thought region
on 0 of 16 completions from a raw string and 16 of 16 through its own template.

A string is a string, which is why the fault was invisible to review. The guard
is therefore at the type level and asserted here.
"""

from __future__ import annotations

import inspect

import pytest

from alibi import prereg, prereg_v2
from alibi.train.prompt import Prompt, RawStringPassedToSampler, build_prompt, require_prompts


class FakeTokenizer:
    """Records that the chat template was applied, and marks the output."""

    def __init__(self) -> None:
        self.calls = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.calls += 1
        return "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n"


class FakeProblem:
    task_id = 2
    description = "Write a function to find the shared elements from the given two lists."
    entry_point = "similar_elements"
    visible_asserts = ["assert similar_elements((1,2),(2,3)) == (2,)"]


def test_a_bare_string_is_refused() -> None:
    with pytest.raises(RawStringPassedToSampler, match="bare string"):
        require_prompts(["Write a function ..."])


def test_a_list_of_bare_strings_is_refused() -> None:
    prompt = build_prompt(FakeProblem(), "sys")
    with pytest.raises(RawStringPassedToSampler):
        require_prompts([prompt, "second one is a string"])


def test_other_wrong_types_are_refused() -> None:
    for value in ({"rendered": "x"}, 42, None, b"bytes"):
        with pytest.raises(RawStringPassedToSampler):
            require_prompts([value])


def test_prompts_pass_through() -> None:
    prompt = build_prompt(FakeProblem(), "sys")
    assert require_prompts([prompt]) == [prompt]


def test_v1_rendering_is_unchanged_by_the_guard() -> None:
    """v1's remaining runs must stay byte-identical to the completed ones."""
    prompt = build_prompt(FakeProblem(), "sys", use_chat_template=False)
    assert prompt.chat_templated is False
    assert prompt.rendered == prompt.source_text
    assert "<|im_start|>" not in prompt.rendered


def test_v2_rendering_applies_the_chat_template() -> None:
    tokenizer = FakeTokenizer()
    prompt = build_prompt(FakeProblem(), "sys", tokenizer=tokenizer, use_chat_template=True)
    assert tokenizer.calls == 1
    assert prompt.chat_templated is True
    assert prompt.rendered != prompt.source_text
    assert "<|im_start|>" in prompt.rendered
    # The task text itself is unchanged, so only the rendering differs between
    # v1 and v2 and the arms stay comparable on content.
    assert prompt.source_text in prompt.rendered


def test_chat_template_without_a_tokenizer_is_an_error_not_a_silent_fallback() -> None:
    """Falling back to a raw string here is exactly how v1 failed."""
    with pytest.raises(ValueError, match="requires a tokenizer"):
        build_prompt(FakeProblem(), "sys", tokenizer=None, use_chat_template=True)


def test_the_sampler_calls_the_guard() -> None:
    """Catches someone removing the guard while leaving the type annotation."""
    from alibi.train.grpo import InlineSampler

    source = inspect.getsource(InlineSampler.generate)
    assert "require_prompts" in source, "InlineSampler.generate no longer validates its prompts"


def test_grpo_build_prompt_returns_a_prompt_not_a_string() -> None:
    from alibi.train.grpo import build_prompt as grpo_build_prompt

    result = grpo_build_prompt(FakeProblem())
    assert isinstance(result, Prompt)
    assert not isinstance(result, str)


# --------------------------------------------------------------------------
# v2 registration.
# --------------------------------------------------------------------------


def test_v2_changes_the_policy_and_nothing_else() -> None:
    """The load-bearing claim: v2 is the same experiment on a different policy."""
    assert prereg_v2.measurement_is_unchanged() is True


def test_v2_inherits_the_v1_measurement_objects_by_reference() -> None:
    assert prereg_v2.PREREG_V2.oracle is prereg.PREREG.oracle or prereg_v2.PREREG_V2.oracle == prereg.PREREG.oracle
    assert prereg_v2.PREREG_V2.halt == prereg.PREREG.halt
    assert prereg_v2.PREREG_V2.run_order == prereg.PREREG.run_order
    assert prereg_v2.PREREG_V2.lambda_monitor == prereg.PREREG.lambda_monitor
    assert prereg_v2.PREREG_V2.inherited_measurement_hash == prereg.PREREG_HASH


def test_v2_declares_the_three_changes_and_only_those() -> None:
    policy = prereg_v2.PREREG_V2.policy
    assert policy.model_id == "Qwen/Qwen3-0.6B"
    assert policy.apply_chat_template is True
    assert policy.max_new_tokens == 3072
    # Unchanged from v1.
    assert policy.group_size == 8


def test_v2_hash_differs_from_v1() -> None:
    assert prereg_v2.PREREG_V2_HASH != prereg.PREREG_HASH


def test_v1_registration_was_not_mutated_by_importing_v2() -> None:
    assert prereg.PREREG.version == "alibi-prereg-v1.0"
    assert prereg.PREREG_HASH == prereg.Prereg().hash()
