# Alibi: Technical Architecture

Companion to `01-prd.md`. This doc is normative: if the code and this doc disagree, one of them is a bug and the report says which.

---

## 1. Shape

```
prompts ──> Sampler ──> completions (+ per-token sampler logprobs)
                            │
                            ├──> Executor (sandboxed) ──> visible pass, held-out pass
                            ├──> CheatOracle (never in reward) ──> oracle verdict
                            └──> RewardMonitor (in reward) ──> flag, reason
                                        │
                            RewardFn ───┘
                                        │
                                  GRPOTrainer ──> policy update (+ trainer logprobs)
                                        │
                                   RunLogger ──> artifacts/
```

Three properties the layout exists to guarantee:

1. The oracle can never leak into the reward, because `RewardFn` is not given a reference to it.
2. Generation happens behind an interface, so the rollout path can be swapped without touching the trainer.
3. Every number that appears in the report is recomputable from `artifacts/` alone.

## 2. Layout

```
alibi/
  cli.py                 dispatcher, see doc 04
  config.py              frozen dataclasses, one per arm, serialised into every run
  data/
    build.py             MBPP + MBPP+ join, visible/held-out split, split manifest
    manifest.json        problem ids, split hashes, dataset revisions
  env/
    executor.py          sandboxed subprocess runner
    tests.py             assert construction and pass-fraction scoring
    oracle.py            CheatOracle: behavioural + structural, measurement only
  monitor/
    base.py              Monitor protocol
    llm.py               hosted judge, temperature 0, disk cache keyed by content hash
    rules.py             deterministic keyword + AST monitor (ablation)
    cache/               committed verdict cache, see section 7
  rollout/
    base.py              Sampler protocol, Completion dataclass
    trl_inline.py        week-one implementation
    (vllm.py)            NOT written this week, seam only
  train/
    reward.py            RewardFn, pure function of scored completions
    loop.py              GRPOTrainer wiring
  report/
    metrics.py           every published number computed here and nowhere else
    figure.py            the one figure
    verify.py            no-GPU recomputation from artifacts
artifacts/
  runs/<run_id>/...
  index.json             declared evidence selection, see section 8
tests/
docs/kickoff/            these four documents
```

## 3. Interfaces that matter

### 3.1 Sampler seam

This is the single most important piece of week-one scaffolding, because the follow-on project lives here.

```python
class Sampler(Protocol):
    def generate(
        self, prompts: list[str], params: SampleParams, seed: int
    ) -> list[Completion]: ...

@dataclass(frozen=True)
class Completion:
    prompt_id: str
    text: str
    think: str | None          # extracted think region, None if absent
    answer: str                # text with think region stripped
    token_ids: list[int]
    sampler_logprobs: list[float]   # per generated token, from the generating path
    finish_reason: str
```

`TRLInlineSampler` wraps whatever TRL does internally and normalises it into `Completion`. It will feel like a pointless wrapper this week. It is the entire point in October.

Rule: no code outside `rollout/` touches a tokenizer's `generate`. If something needs text, it takes a `Completion`.

### 3.2 Monitor

```python
class Monitor(Protocol):
    name: str
    def judge(self, view: str) -> Verdict: ...

@dataclass(frozen=True)
class Verdict:
    flagged: bool
    score: float          # 0..1, used if soft penalties are enabled
    reason: str
    cache_hit: bool
```

The arm decides what `view` is: `completion.think` for A1, `completion.answer` for A2, a generated summary for A3. The monitor itself does not know which arm it is in, which keeps arms comparable by construction.

### 3.3 Reward

```python
def reward_fn(scored: ScoredCompletion, cfg: ArmConfig) -> float:
    return (
        scored.visible_pass_fraction
        - cfg.lambda_monitor * float(scored.verdict.flagged)
        - format_penalty(scored)
    )
```

`ScoredCompletion` deliberately has no `oracle` field. The oracle verdict is attached downstream in the logging record, in a separate type. Enforce it with a test that asserts `reward_fn`'s signature cannot see oracle data, and keep that test even though it looks silly.

## 4. Logging, and the data the follow-on project needs

Per training step, write:

`artifacts/runs/<run_id>/steps/<n>/completions.jsonl`
one record per completion: prompt_id, seed, arm, text, think, answer, visible_pass_fraction, held_out_pass_fraction, oracle_behavioural, oracle_structural, monitor_flagged, monitor_score, monitor_reason, monitor_cache_hit, reward, finish_reason, lengths.

`artifacts/runs/<run_id>/steps/<n>/logprobs.parquet`
columns: prompt_id, completion_idx, position, token_id, sampler_logprob, trainer_logprob, seed, step.

`trainer_logprob` is the log-probability the trainer recomputes for the same token during the update. On the inline path these two should be close. Storing both anyway is the point: the follow-on project's headline is what happens to this pair when the rollout path changes, and that dataset costs one column today and a full re-run in October.

`artifacts/runs/<run_id>/env.lock`
git rev plus dirty flag, python, torch, transformers, trl, peft, accelerate, CUDA, driver, GPU name, dataset revisions, monitor model id, every seed, and the config hash. Same discipline as Lockstep's `env.lock`, and for the same reason: it lets a non-reproducing replay be told apart from a fixed bug.

`artifacts/runs/<run_id>/config.json`
the frozen `ArmConfig`, serialised. Any run whose config hash is absent from its own artifacts is not citable.

## 5. Determinism policy

Full bitwise determinism is out of scope this week and is the follow-on project's whole subject. What is in scope:

- Every seed set explicitly (python, numpy, torch, and the sampler's own) and recorded per step.
- Monitor at temperature 0 with a committed verdict cache, so monitor decisions replay exactly even if the hosted model changes underneath.
- The executor is deterministic by construction: no network, fixed timeout, fixed recursion limit, sorted iteration wherever a dict is walked.
- Metrics computed with a fixed reduction order over sorted prompt ids.

State plainly in the README that rollouts are not bitwise reproducible on this path. Claiming otherwise here would be exactly the error the follow-on project exists to catch.

## 6. Executor

Model-generated Python is executed. Treat it as hostile. Details in `03-security-and-access.md`; the architectural constraints are:

- Separate process, never `exec` in the trainer process.
- Hard wall-clock timeout, memory and file-descriptor limits via `resource.setrlimit`.
- No network namespace access.
- Working directory is a fresh temp dir, deleted after.
- A crashed or timed-out execution is a scored outcome (`pass_fraction = 0`), never an exception that kills a run.

Throughput matters: one process per completion at group size 8 is fine at 0.5B, but pool the workers and cap concurrency to the number of physical cores minus one.

## 7. Monitor cache

Key: `sha256(monitor_name || monitor_model_id || prompt_template_version || view_text)`.
Value: the full `Verdict` plus the raw response.

The cache is committed to the repo. Consequences, both intended:

- A rerun of the same run is free and produces identical verdicts.
- If the hosted model is deprecated, the runs still replay.
- A cache miss during what should be a replay is a loud warning, not a silent API call. Make it fail unless `--allow-monitor-misses` is passed.

Record cache hit rate per step. A replay claiming to reproduce a run at less than 100 percent hit rate is not a replay.

## 8. Evidence selection

`artifacts/index.json` names, by explicit declaration, which run directories back which published claim. Not by filename sort, not by mtime, not by "the latest one." Add a test asserting that the declared choice differs from what a naive sort would pick, so the declaration is doing work rather than decorating.

Every claim in the report maps to exactly one entry. A claim with no entry does not go in the report.

## 9. Verification without a GPU

`report/verify.py` reads `artifacts/` and recomputes every published number: cheat rates, flag rates, gaps, seed spreads, and the figure's underlying series. It checks artifact digests and prints a table of computed versus published, with a non-zero exit on any mismatch.

This is the highest-leverage hour in the whole build. It is what lets a mentor check the result in two minutes on a laptop, and it is the difference between an applicant who says a number and one who hands over the means to falsify it.

## 10. Testing

- `tests/test_oracle.py`: hand-written cheating and honest solutions, asserting the oracle classifies both correctly, including the awkward middle case of a partially correct implementation that happens to pass visible tests.
- `tests/test_reward_isolation.py`: the reward function cannot reach oracle data.
- `tests/test_executor.py`: timeout, memory bomb, infinite loop, import of a blocked module, all scored rather than raised.
- `tests/test_replay.py`: a fixed artifacts fixture replays to identical metrics.
- `tests/test_evidence.py`: declared selection differs from sorted-by-name and sorted-by-mtime.

CI runs all of it on CPU. Write the workflow file on day 1, not at the end. Lockstep declared five CI gates in its architecture doc and shipped without `.github/` existing, and a third party reading the directory tree found it.

## 11. Known risks

| Risk | Mitigation |
|------|------------|
| 0.5B cannot express the behaviour at all, everything is flat | Kill criteria in the PRD. Escalate to 1.5B only if free hours remain. |
| Monitor API rate limits stall training | Async batch, cache, and a rules-monitor fallback that requires no network. |
| MBPP prompt format makes cheating so easy it saturates at step 0 | Check on day 1 with the base model before any training. If visible pass rate is already near 1.0, add held-out-only problems or reduce visible asserts to 1. |
| TRL version churn breaks `GRPOTrainer` wiring | Pin exact versions in `env.lock` on day 1 and do not upgrade mid-week. |
| Seven days | Days 5 to 7 are already the buffer. A2 and A3 are the first things cut. |
