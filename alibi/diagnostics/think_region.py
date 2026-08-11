"""Does the policy emit a designated thought region at all?

A **separate declared diagnostic**. It is never pooled with the matrix, it is
excluded from the evidence index by policy, and it involves no training and no
weight update. It exists because arm A1 was found to be judging empty strings:
Qwen2.5-0.5B-Instruct does not emit `<think>` tags, so the think-region monitor
had nothing to read and A1 became arithmetically A0.

The question this answers is not "can a model this size think out loud". It is
whether emitting a designated thought region is a **post-training property** that
has to be checked before an arm is built on it. So it samples a similarly sized
model that was post-trained for it, on byte-identical prompts, and compares.

Three conditions, all at the matrix's own step-zero prompt set and group size:

  qwen2.5-0.5b-instruct   the policy the matrix actually trains, raw prompt
  qwen3-0.6b              similar scale, post-trained to think, same raw prompt
  qwen3-0.6b-chat-template  the same model through its own chat template

The third exists because if a model only emits a thought region under its native
template, that is itself the answer: the behaviour belongs to post-training and
its interface, not to scale.

Nothing here writes to a matrix run directory or to prereg.py.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from alibi import prereg, runlog
from alibi.rollout.base import extract_code, make_completion

OUT_DIR = runlog.ARTIFACTS / "diagnostics" / "think_region"

POLICY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
COMPARISON_MODEL = "Qwen/Qwen3-0.6B"

# Raised so a think block and an answer both fit. The matrix uses 256, which
# would truncate a thinking model mid-thought and would make the comparison a
# measurement of the token budget rather than of the model.
MAX_NEW_TOKENS = 1024
GROUP_SIZE = 8
PROMPTS_AT_STEP_ZERO = 2
SEED = 1

# Kept small on purpose. The matrix is training on this machine while this runs,
# and starving its held-out executor pool would raise its indeterminate rate and
# could trip a halt condition on a run this diagnostic has no business touching.
EXECUTOR_WORKERS = 2
TORCH_THREADS = 2


@dataclass
class ConditionResult:
    name: str
    model_id: str
    used_chat_template: bool
    n_completions: int = 0
    think_chars: list[int] = field(default_factory=list)
    answer_chars: list[int] = field(default_factory=list)
    n_nonempty_think: int = 0
    n_flagged_think_view: int = 0
    n_flagged_answer_view: int = 0
    n_monitor_errors: int = 0
    cheat_form_counts: dict = field(default_factory=dict)
    n_any_cheat_form: int = 0
    n_behavioural_cheat: int = 0
    n_behaviourally_determinate: int = 0
    visible_pass_fractions: list[float] = field(default_factory=list)
    seconds: float = 0.0
    examples: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        from alibi.report.metrics import wilson_interval

        def stats(values):
            if not values:
                return {"mean": None, "median": None, "min": None, "max": None}
            return {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }

        n = self.n_completions
        think_low, think_high = wilson_interval(self.n_nonempty_think, n)
        return {
            "name": self.name,
            "model_id": self.model_id,
            "used_chat_template": self.used_chat_template,
            "n_completions": n,
            "think_chars": stats(self.think_chars),
            "answer_chars": stats(self.answer_chars),
            "nonempty_think_fraction": (self.n_nonempty_think / n) if n else None,
            "nonempty_think_ci95": [think_low, think_high],
            "flag_rate_think_view": (self.n_flagged_think_view / n) if n else None,
            "flag_rate_answer_view": (self.n_flagged_answer_view / n) if n else None,
            "monitor_errors": self.n_monitor_errors,
            "cheat_form_counts": dict(sorted(self.cheat_form_counts.items())),
            "any_cheat_form_fraction": (self.n_any_cheat_form / n) if n else None,
            "behavioural_cheat_rate": (
                self.n_behavioural_cheat / self.n_behaviourally_determinate
                if self.n_behaviourally_determinate
                else None
            ),
            "n_behaviourally_determinate": self.n_behaviourally_determinate,
            "visible_pass_rate": stats(self.visible_pass_fractions)["mean"],
            "seconds": round(self.seconds, 1),
        }


def _prompts():
    """The matrix's own step-zero prompt set, byte identical."""
    from alibi.train.grpo import build_prompt, eligible_problems

    problems = eligible_problems()
    chosen = [problems[(0 * PROMPTS_AT_STEP_ZERO + i) % len(problems)] for i in range(PROMPTS_AT_STEP_ZERO)]
    return [(p, build_prompt(p)) for p in chosen]


def _generate(model_id: str, use_chat_template: bool, prompts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(TORCH_THREADS)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()

    out = []
    torch.manual_seed(SEED)
    for problem, prompt in prompts:
        if use_chat_template:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        encoded = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                num_return_sequences=GROUP_SIZE,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prompt_len = encoded["input_ids"].shape[1]
        for row in range(generated.shape[0]):
            token_ids = generated[row, prompt_len:].tolist()
            decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
            out.append((problem, make_completion(str(problem.task_id), decoded)))
        del generated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def run_condition(name: str, model_id: str, use_chat_template: bool) -> ConditionResult:
    from alibi.env.executor import Executor
    from alibi.env.oracle import structural_check
    from alibi.env.scoring import score_held_out, score_visible
    from alibi.monitor.base import safe_judge
    from alibi.monitor.selection import load_selection
    from alibi.monitor.llm import LLMMonitor
    from alibi.rollout.base import monitor_view

    started = time.monotonic()
    selection = load_selection()
    monitor = LLMMonitor(model_id=selection["model_id"], base_url=selection["base_url"])
    executor = Executor()

    result = ConditionResult(name=name, model_id=model_id, used_chat_template=use_chat_template)
    generated = _generate(model_id, use_chat_template, _prompts())

    for problem, completion in generated:
        result.n_completions += 1
        result.think_chars.append(len(completion.think or ""))
        result.answer_chars.append(len(completion.answer or ""))
        if (completion.think or "").strip():
            result.n_nonempty_think += 1

        # Both arm views, on the same completion, so the comparison between A1
        # and A2 is within-completion rather than between samples.
        _, think_view = monitor_view(completion, "a1")
        _, answer_view = monitor_view(completion, "a2")
        think_verdict = safe_judge(monitor, think_view)
        answer_verdict = safe_judge(monitor, answer_view)
        result.n_flagged_think_view += int(think_verdict.flagged and not think_verdict.is_error)
        result.n_flagged_answer_view += int(answer_verdict.flagged and not answer_verdict.is_error)
        result.n_monitor_errors += int(think_verdict.is_error) + int(answer_verdict.is_error)

        code = extract_code(completion.answer)
        flagged, forms, _signals, _error = structural_check(code, problem.visible_asserts, problem.entry_point)
        if flagged:
            result.n_any_cheat_form += 1
            for form in forms:
                result.cheat_form_counts[form] = result.cheat_form_counts.get(form, 0) + 1

        visible = score_visible(problem, code, executor)
        held_out = score_held_out(problem, code, executor)
        if visible.pass_fraction is not None:
            result.visible_pass_fractions.append(visible.pass_fraction)
        verdict = prereg.cheated(
            held_out.pass_fraction, visible.pass_fraction, prereg.PREREG.oracle.held_out_cheat_threshold
        )
        if verdict is not None:
            result.n_behaviourally_determinate += 1
            result.n_behavioural_cheat += int(verdict)

        if len(result.examples) < 2:
            result.examples.append(
                {
                    "task_id": problem.task_id,
                    "think_chars": len(completion.think or ""),
                    "think_excerpt": (completion.think or "")[:400],
                    "answer_excerpt": (completion.answer or "")[:400],
                }
            )

    result.seconds = time.monotonic() - started
    return result


def main() -> dict:
    from alibi.monitor import spend

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    before = spend.load().to_dict()
    started = time.monotonic()

    conditions = [
        ("qwen2.5-0.5b-instruct", POLICY_MODEL, False),
        ("qwen3-0.6b-raw-prompt", COMPARISON_MODEL, False),
        ("qwen3-0.6b-chat-template", COMPARISON_MODEL, True),
    ]
    results = []
    for name, model_id, template in conditions:
        print(f"--- {name}", flush=True)
        results.append(run_condition(name, model_id, template).summary())
        print(json.dumps(results[-1], indent=2, sort_keys=True), flush=True)

    after = spend.load().to_dict()
    document = {
        "diagnostic": "think_region",
        "declared": (
            "Separate diagnostic, never pooled with the matrix and excluded from the evidence index. "
            "No training and no weight update. Sampling only."
        ),
        "prompt_set": "the matrix's own step-zero prompts, byte identical",
        "group_size": GROUP_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "prereg_hash": prereg.PREREG_HASH,
        "conditions": results,
        "wall_clock_seconds": round(time.monotonic() - started, 1),
        "monitor_cost": {
            "calls": after["calls"] - before["calls"],
            "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
            "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
            "total_tokens": after["total_tokens"] - before["total_tokens"],
            "usd": None if after["usd"] is None else after["usd"] - (before["usd"] or 0.0),
            "usd_absent_reason": after["usd_absent_reason"],
            "note": (
                "The spend ledger is global and shared with the matrix, so this is the delta measured "
                "across this diagnostic rather than a separate ledger."
            ),
        },
        "gpu": "local RTX 4060 Laptop, shared with the running matrix, so no rented compute cost",
    }
    (OUT_DIR / "result.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


if __name__ == "__main__":
    main()
