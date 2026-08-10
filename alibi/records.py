"""The contract with an external auditor who has artifacts and no GPU.

Every field an auditor could need is written at the moment the completion is
scored, not derived later by a report. The rule this module enforces: an
unlogged field costs a rerun, an extra column costs bytes, so when in doubt it
is logged.

`REQUIRED_COMPLETION_FIELDS` is asserted by tests/test_records.py against a real
record. Adding a field to the schema without writing it fails the suite, which
is the point: the schema is the thing that fails loudly, rather than a silent
`None` discovered in October.

Nothing here computes a metric. It records what was observed, including all
three pre-registered sensitivity verdicts, so `alibi report` reads rather than
recomputes and an auditor never has to trust that the recomputation matched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from alibi import prereg
from alibi.env.scoring import Scored

SCHEMA_VERSION = 3

# Asserted against every written record. Order is irrelevant, presence is not.
REQUIRED_COMPLETION_FIELDS = (
    # identity and provenance
    "schema_version",
    "run_id",
    "arm",
    "seed",
    "step",
    "prompt_id",
    "completion_idx",
    "git_revision",
    "git_dirty",
    "config_hash",
    "prereg_hash",
    "prereg_version",
    "eligibility_hash",
    # the text, and the arm's view of it
    "text",
    "think",
    "answer",
    "monitor_view_name",
    "completion_chars",
    "think_chars",
    "answer_chars",
    "finish_reason",
    # execution, three state
    "visible_pass_fraction",
    "visible_n_pass",
    "visible_n_fail",
    "visible_n_indeterminate",
    "visible_indeterminate_indices",
    "visible_status",
    "held_out_pass_fraction",
    "held_out_n_pass",
    "held_out_n_fail",
    "held_out_n_indeterminate",
    "held_out_indeterminate_indices",
    "held_out_indeterminate_fraction",
    "held_out_status",
    "held_out_scored_async",
    # oracle, measurement only
    "oracle_behavioural",
    "oracle_structural",
    "oracle_agreement",
    "cheat_form",
    "oracle_signals",
    "cheat_at_threshold",
    # monitor, in the reward
    "monitor_name",
    "monitor_model_id",
    "monitor_prompt_template_version",
    "monitor_verdict_state",
    "monitor_flagged",
    "monitor_score",
    "monitor_reason",
    "monitor_raw_response",
    "monitor_cache_hit",
    "monitor_latency_seconds",
    "monitor_error",
    # reward
    "reward",
    "reward_visible_component",
    "reward_monitor_penalty",
    "reward_format_penalty",
    "lambda_monitor",
)


@dataclass
class CompletionRecord:
    """One completion, fully described. Written to completions.jsonl."""

    run_id: str
    arm: str
    seed: int
    step: int
    prompt_id: str
    completion_idx: int
    git_revision: str | None
    git_dirty: bool | None
    config_hash: str
    eligibility_hash: str | None

    text: str
    think: str | None
    answer: str
    monitor_view_name: str
    finish_reason: str

    visible_pass_fraction: float | None
    visible_n_pass: int
    visible_n_fail: int
    visible_n_indeterminate: int
    visible_indeterminate_indices: list[int]
    visible_status: str

    held_out_pass_fraction: float | None
    held_out_n_pass: int
    held_out_n_fail: int
    held_out_n_indeterminate: int
    held_out_indeterminate_indices: list[int]
    held_out_indeterminate_fraction: float
    held_out_status: str
    held_out_scored_async: bool

    oracle_behavioural: bool | None
    oracle_structural: bool
    oracle_agreement: str
    cheat_form: list[str]
    oracle_signals: list[str]

    monitor_name: str
    monitor_model_id: str | None
    monitor_prompt_template_version: str
    monitor_verdict_state: str
    monitor_flagged: bool
    monitor_score: float | None
    monitor_reason: str
    monitor_raw_response: str
    monitor_cache_hit: bool
    monitor_latency_seconds: float
    monitor_error: str

    reward: float
    reward_visible_component: float
    reward_monitor_penalty: float
    reward_format_penalty: float

    schema_version: int = SCHEMA_VERSION
    prereg_hash: str = prereg.PREREG_HASH
    prereg_version: str = prereg.PREREG_VERSION
    lambda_monitor: float = prereg.PREREG.lambda_monitor
    # All three registered thresholds, computed here at write time.
    cheat_at_threshold: dict = field(default_factory=dict)
    completion_chars: int = 0
    think_chars: int = 0
    answer_chars: int = 0

    def __post_init__(self) -> None:
        self.completion_chars = len(self.text or "")
        self.think_chars = len(self.think or "")
        self.answer_chars = len(self.answer or "")
        if not self.cheat_at_threshold:
            self.cheat_at_threshold = prereg.sensitivity(
                self.held_out_pass_fraction, self.visible_pass_fraction
            )

    def to_dict(self) -> dict:
        return asdict(self)


def from_scored(
    visible: Scored,
    held_out: Scored,
    **kwargs,
) -> CompletionRecord:
    """Build a record from two Scored results plus the surrounding context.

    Keeps the mapping from execution result to logged field in one place, so a
    caller cannot accidentally log `n_fail` where `n_indeterminate` belongs.
    """
    return CompletionRecord(
        visible_pass_fraction=visible.pass_fraction,
        visible_n_pass=visible.n_pass,
        visible_n_fail=visible.n_fail,
        visible_n_indeterminate=visible.n_indeterminate,
        visible_indeterminate_indices=visible.indeterminate_indices,
        visible_status=visible.status,
        held_out_pass_fraction=held_out.pass_fraction,
        held_out_n_pass=held_out.n_pass,
        held_out_n_fail=held_out.n_fail,
        held_out_n_indeterminate=held_out.n_indeterminate,
        held_out_indeterminate_indices=held_out.indeterminate_indices,
        held_out_indeterminate_fraction=held_out.indeterminate_fraction,
        held_out_status=held_out.status,
        **kwargs,
    )


def validate(record: dict) -> list[str]:
    """Missing required fields. Empty list means the record is complete."""
    return sorted(f for f in REQUIRED_COMPLETION_FIELDS if f not in record)
