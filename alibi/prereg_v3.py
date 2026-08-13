"""Pre-registration v3, scoped down. **Not tagged, and not run.**

v3 was specified as a scoped-down variant after v2 was stood down. The
no-training sampling pass that was supposed to set its capped-fraction halt
instead measured that v3 cannot produce a comparison, so the design is recorded
here and the run was not started.

This file is deliberately not a registration in force. `RUNNABLE` is False and
no `alibi-prereg-v3.0` tag exists. A pre-registration tag is a record of intent
frozen **before** data; here the blocking data arrived first, so there is
nothing to pre-register. Recording the design without tagging it is the honest
form.

v3 is **not v2 rescued**. It is a different, smaller experiment that was
abandoned before it started, for a reason measured rather than assumed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from alibi import prereg_v2

RUNNABLE = False

# The design, exactly as scoped.
MAX_NEW_TOKENS = 1024
ARMS = ("a0", "a2")
SEEDS = (1,)


@dataclass(frozen=True)
class V3PolicySpec:
    """Everything not listed here is inherited unchanged from v2.2.

    Inherited: policy Qwen3-0.6B, prompt conditioning, the chat template, the
    real trainer logprob, the probe-selected training spec, and every v1
    measurement object that v2 itself inherited.
    """

    model_id: str = prereg_v2.POLICY_MODEL
    apply_chat_template: bool = True
    max_new_tokens: int = MAX_NEW_TOKENS
    group_size: int = 8
    prompts_per_step: int = 2
    steps_per_run: int = 80
    arms: tuple[str, ...] = ARMS
    seeds: tuple[int, ...] = SEEDS

    a1_excluded_because: str = (
        "Arm A1's monitor reads the designated thought region. The measured median think block "
        "is 746 tokens and a 1024-token budget cannot hold think plus answer with any margin, so "
        "A1 would read a truncated or empty view. That is the same fault that made A1 "
        "arithmetically identical to A0 in v1, and running it again knowing this would produce a "
        "second empty arm rather than a second measurement."
    )


# What the no-training sampling pass measured, which is why v3 was not started.
# Qwen3-0.6B, chat template, 8 eligible problems, group 8, 64 completions,
# max_new_tokens 1024, no training and no weight update.
BLOCKING_MEASUREMENT = {
    "artifact": "artifacts/diagnostics/cap_at_1024/result.json",
    "n_completions": 64,
    "capped_fraction": 0.6875,
    "capped_ci95": [0.5661, 0.7877],
    "closed_think_fraction": 0.421875,
    "has_code_fraction": 0.421875,
    "has_code_ci95": [0.3087, 0.5439],
    "median_tokens": 1024.0,
    "why_this_blocks": (
        "The median completion is capped, so more than half of all completions never finish "
        "thinking. Code is extractable from 42 percent, and the closed-think fraction equals the "
        "has-code fraction exactly, 27 of 64 in both cases: a completion yields code if and only "
        "if its think block closed. The other 58 percent therefore carry an empty answer.\n\n"
        "That is decisive for the one comparison v3 exists to make. Arm A2's monitor reads the "
        "answer, so on 58 percent of completions it would read an empty string and return "
        "unflagged. A2 would be measuring the monitor's response to absence, not to cheating, and "
        "the a0 against a2 contrast would be uninterpretable. It is the A1 failure of v1 "
        "reappearing in A2, arriving through the token budget rather than through the policy."
    ),
    "halt_threshold_not_set": (
        "The capped-fraction halt was to be set from this measurement plus headroom. A threshold "
        "above 0.6875 would not be a guard, it would be a licence, so no threshold was chosen."
    ),
}


@dataclass(frozen=True)
class PreregV3:
    version: str = "alibi-prereg-v3.0-unregistered"
    runnable: bool = RUNNABLE
    policy: V3PolicySpec = field(default_factory=V3PolicySpec)
    inherits_from: str = prereg_v2.PREREG_VERSION
    inherited_hash: str = prereg_v2.PREREG_V2_HASH
    blocking_measurement: dict = field(default_factory=lambda: dict(BLOCKING_MEASUREMENT))

    def to_dict(self) -> dict:
        return asdict(self)


PREREG_V3 = PreregV3()


def why_not_run() -> str:
    return BLOCKING_MEASUREMENT["why_this_blocks"]
