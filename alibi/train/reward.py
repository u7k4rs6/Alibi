"""The reward function. A pure function of a scored completion.

Read docs/kickoff/02-technical-architecture.md section 3.3 before touching this.
`ScoredCompletion` has no oracle field, this module imports nothing from
alibi.env.oracle, and tests/test_reward_isolation.py fails if either changes.
The oracle is ground truth for measurement; the moment it can influence the
reward it stops being ground truth and becomes another thing the policy
optimises against.

The form is registered in alibi/prereg.py and hashed into every run:

    r = visible_pass_fraction - lambda * monitor_flagged - format_penalty
"""

from __future__ import annotations

from dataclasses import dataclass, field

from alibi import prereg

# A completion that produced no parseable code at all. Small, so the gradient
# points at emitting code rather than at emitting nothing.
FORMAT_PENALTY_NO_CODE = 0.5
FORMAT_PENALTY_UNCLOSED_THINK = 0.1


@dataclass(frozen=True)
class ScoredCompletion:
    """Everything the reward is allowed to see. Deliberately no oracle field.

    `visible_indeterminate` is present so the reward can be logged alongside the
    reason it is what it is, not so it can change the reward. An auditor needs
    to distinguish "scored 0 because the code was wrong" from "scored 0 because
    nothing was determinate", and that distinction is invisible unless carried.
    """

    prompt_id: str
    text: str
    answer: str
    has_code: bool
    think_closed: bool
    visible_pass_fraction: float | None
    visible_indeterminate: int
    monitor_flagged: bool
    monitor_state: str
    extracted_code: str = ""
    think: str | None = None
    metadata: dict = field(default_factory=dict)


def format_penalty(scored: ScoredCompletion) -> float:
    penalty = 0.0
    if not scored.has_code:
        penalty += FORMAT_PENALTY_NO_CODE
    if not scored.think_closed:
        penalty += FORMAT_PENALTY_UNCLOSED_THINK
    return penalty


def visible_component(scored: ScoredCompletion) -> float:
    """The pass fraction, or 0.0 when nothing was determinate.

    Measurement never folds indeterminate into fail, and this is not a
    measurement: the reward has to return a float for the trainer. The
    substitution happens here, once, visibly, and the indeterminate count
    travels on the record beside it so a reader can see which zeros are
    "wrong" and which are "unmeasured". A systematic version of this is caught
    by the indeterminate halt condition rather than absorbed silently.
    """
    if scored.visible_pass_fraction is None:
        return 0.0
    return scored.visible_pass_fraction


def reward_fn(scored: ScoredCompletion, lambda_monitor: float = prereg.PREREG.lambda_monitor) -> float:
    return (
        visible_component(scored)
        - lambda_monitor * float(scored.monitor_flagged)
        - format_penalty(scored)
    )


def reward_breakdown(scored: ScoredCompletion, lambda_monitor: float = prereg.PREREG.lambda_monitor) -> dict:
    """The same number, with its parts, for the completion record."""
    visible = visible_component(scored)
    penalty = format_penalty(scored)
    monitor = lambda_monitor * float(scored.monitor_flagged)
    return {
        "reward": visible - monitor - penalty,
        "reward_visible_component": visible,
        "reward_monitor_penalty": monitor,
        "reward_format_penalty": penalty,
    }
