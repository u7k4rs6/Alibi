"""Pre-registration v2. The policy changes; the measurement does not.

`alibi/prereg.py` is not modified and not re-derived. This module **imports it
verbatim** and adds only the policy specification, so "everything else carries
over unchanged" is enforced by construction rather than by careful copying. A
reader can confirm it by noting that every measurement field below is a
reference to the v1 object, not a restatement of it.

What v2 changes, and only this:

  policy            Qwen3-0.6B, which is post-trained to emit a thought region
  prompt rendering  the tokenizer's chat template is applied
  token budget      sized from measurement, see MAX_NEW_TOKENS

What carries over unchanged, by import: arms, reward form, lambda, oracle
definitions, thresholds and sensitivity values, the eligibility manifest, the
determinacy floor, every halt condition including the lagging window, and the
breadth-first run order.

Frozen at tag `alibi-prereg-v2.0`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from alibi import prereg as v1

PREREG_VERSION = "alibi-prereg-v2.1"

# The policy. v1 used Qwen2.5-0.5B-Instruct, which emits a designated thought
# region on 0 of 16 completions even when the prompt asks for one. Qwen3-0.6B is
# 0.1B larger and emits one on 16 of 16 through its own chat template.
POLICY_MODEL = "Qwen/Qwen3-0.6B"

# Sized from measurement, not extrapolation. Qwen3-0.6B on the matrix's own
# step-zero prompts, 16 completions, budget 3072, measured under load:
#
#   total tokens   median 774, mean 1400.5, max 3072
#   think tokens   median 746, mean 1343.4
#   answer tokens  median  24, mean   53.4, p95 111, max 267
#   hit the 3072 cap: 2 of 16
#
# The answer is tiny and the think block is nearly all of it, and the
# distribution has a long right tail. 3072 covers 87.5 percent of completions
# measured. Truncation matters more here than in v1: the answer follows the
# think block, so a completion cut off mid-thought yields no code at all rather
# than partial code. `finish_reason` is logged per completion so the report can
# state exactly how many produced no answer.
MAX_NEW_TOKENS = 3072
MEASURED_CAP_FRACTION = 2 / 16


@dataclass(frozen=True)
class V2HaltAddition:
    """One halt condition added for v2 only, declared before any v2 run existed.

    v1 had no need of it: at 256 tokens the capped fraction was 44.8 percent and
    truncation merely shortened the code. Under a thinking policy the answer
    follows the think block, so a capped completion yields **no code at all**. A
    run where most completions are capped is not training on the task, it is
    training on truncation, and the reward signal it produces is not about
    solving problems.

    0.35 is chosen against the measured distribution: 2 of 16 completions, or
    0.125, hit the 3072 cap at step zero. A sustained rate near three times that
    means the policy has drifted into thinking longer than it can finish, which
    is a known failure mode of thinking models under RL.

    Evaluated on the same lagging window as the indeterminate condition, for the
    same reason: a single step above the line is noise, a sustained one is not.
    """

    max_capped_fraction: float = 0.35
    evaluated_on_lagging_window: bool = True
    declared_before_any_v2_run: bool = True


@dataclass(frozen=True)
class PolicySpec:
    """The only thing v2 changes."""

    model_id: str = POLICY_MODEL
    apply_chat_template: bool = True
    max_new_tokens: int = MAX_NEW_TOKENS
    group_size: int = 8
    prompts_per_step: int = 2
    steps_per_run: int = 80
    rationale: str = (
        "v1 arm A1 read an empty view because the policy emitted no thought region and the prompt "
        "bypassed the chat template. Both are fixed here. Two independent faults, either sufficient "
        "alone, measured in artifacts/diagnostics/think_region/result.json."
    )


@dataclass(frozen=True)
class PreregV2:
    version: str = PREREG_VERSION
    policy: PolicySpec = field(default_factory=PolicySpec)
    # An addition, not a modification. v1's HaltSpec is inherited untouched below.
    halt_addition: V2HaltAddition = field(default_factory=V2HaltAddition)

    # Everything below is the v1 object itself. Not a copy, not a restatement.
    oracle: v1.OracleSpec = field(default_factory=lambda: v1.PREREG.oracle)
    eligibility: v1.EligibilitySpec = field(default_factory=lambda: v1.PREREG.eligibility)
    halt: v1.HaltSpec = field(default_factory=lambda: v1.PREREG.halt)
    run_order: v1.RunOrderSpec = field(default_factory=lambda: v1.PREREG.run_order)
    metric: v1.MetricSpec = field(default_factory=lambda: v1.PREREG.metric)
    reward_definition: str = v1.PREREG.reward_definition
    lambda_monitor: float = v1.PREREG.lambda_monitor

    # The v1 hash, recorded so a reader can verify the measurement did not move.
    inherits_measurement_from: str = v1.PREREG_VERSION
    inherited_measurement_hash: str = v1.PREREG_HASH

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()


PREREG_V2 = PreregV2()
PREREG_V2_HASH = PREREG_V2.hash()


def measurement_is_unchanged() -> bool:
    """True when every measurement field is identical to v1's.

    Asserted by tests/test_prereg_v2.py. If this ever returns False, v2 has
    silently become a different experiment rather than the same experiment on a
    different policy, and the two are no longer comparable.
    """
    return (
        PREREG_V2.oracle == v1.PREREG.oracle
        and PREREG_V2.eligibility == v1.PREREG.eligibility
        and PREREG_V2.halt == v1.PREREG.halt
        and PREREG_V2.run_order == v1.PREREG.run_order
        and PREREG_V2.metric == v1.PREREG.metric
        and PREREG_V2.reward_definition == v1.PREREG.reward_definition
        and PREREG_V2.lambda_monitor == v1.PREREG.lambda_monitor
        and PREREG_V2.inherited_measurement_hash == v1.PREREG_HASH
    )


def provenance() -> dict:
    document = v1.provenance()
    document.update(
        {
            "prereg_version": PREREG_V2.version,
            "prereg_hash": PREREG_V2_HASH,
            "inherits_measurement_from": PREREG_V2.inherits_measurement_from,
            "inherited_measurement_hash": PREREG_V2.inherited_measurement_hash,
            "measurement_unchanged": measurement_is_unchanged(),
            "policy": asdict(PREREG_V2.policy),
        }
    )
    return document
