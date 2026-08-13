# Alibi

**A failed experiment with a complete forensic record.** It set out to measure whether a
policy penalised by a monitor stops reward hacking or just stops saying so. No such
measurement exists here. What exists instead is five distinct faults, each of which alone
would have produced a confident, publishable-looking curve that meant nothing, and each of
which was caught by instrumentation built before the runs.

That is the result, and it is the reason the repository is worth reading.

---

## The five faults

**No monitored-arm comparison exists at any version.**

1. **Arm A1's monitor read an empty view.** Qwen2.5-0.5B-Instruct emits no designated
   thought region despite the prompt asking, so A1's flag rate was zero by construction
   and its reward was arithmetically the unmonitored control's. Confirmed by a0/s1 and
   a1/s1 matching to three decimals.
2. **The chat template was never applied.** Qwen3-0.6B emits a thought region on 0 of 16
   completions from a raw prompt and 16 of 16 through its own template, same content.
   Either fault alone was sufficient to empty the arm.
3. **44.8 percent of completions hit the token cap**, scoring 0.1053 visible pass against
   0.2254 for those that stopped naturally.
4. **The objective had no prompt conditioning.** The gradient maximised the unconditional
   likelihood of completion text. Reward fell 0.2176 to 0.0990 while entropy rose 0.6384
   to 0.8354.
5. **`trainer_logprob` was a copy of `sampler_logprob`**, 3430 of 3430 rows. The column
   collected to detect sampler-trainer divergence could not have shown any.

Details, including what each would have produced undetected and what caught it, are in
[report/REPORT.md](report/REPORT.md).

## Why the faults surfaced as numbers

The load-bearing claim. Each was caught because a specific piece of instrumentation
existed before the runs did:

- The reward monitor is separated from the behavioural oracle **by construction**, with a
  test that fails if anything connects them.
- Timeouts and crashes are a third outcome, never a fail, so infrastructure failure cannot
  read as a policy that does not generalise.
- A monitor error is always flagged, so an outage raises the penalty rather than lowering
  it. No exception in the judging path can yield `flagged=False`.
- Structural precision was measured on honest code **before** the panel was trusted, and
  found 5 of 365 reference solutions falsely flagged.
- A vindication claim was **withdrawn as circular** once the eligibility filter was seen
  to use the behavioural rule's own criterion.

## What this says about experiments of this shape

- An arm that reads thought tokens presupposes a policy post-trained to emit them **and**
  an interface that lets it. Neither is implied by model size.
- An empty view and a clean policy are indistinguishable at the monitor's output. A flag
  rate of exactly zero is a defect signal until shown otherwise.
- The token budget silently decides which arms are comparable.
- A flat cheat rate is uninformative unless the loop is shown to have been optimising the
  intended objective.

## What was never tested

H1, H2 and H3 are all unevaluated. v2 was abandoned because two of its own registered
values are jointly infeasible on an 8 GB card. v3 was stopped by its own qualifying
measurement: at a 1024-token budget the capped fraction is 0.6875 and only 42 percent of
completions yield code, so an answer-reading arm would read an empty string on 58 percent
of them. See [BLOCKED.md](BLOCKED.md) and [BLOCKED-v2.md](BLOCKED-v2.md).

## Unresolved

Sampler and trainer logprobs diverge by 0.2150 mean and 1.8477 max on identical weights,
past what bf16 noise explains, and it is not diagnosed.

## Checking the claims

```
alibi verify --no-gpu
```

Recomputes every published number from `artifacts/` alone. No GPU, no network.

## Layout

| Path | What it is |
|---|---|
| `report/REPORT.md` | the study, numbers injected from artifacts |
| `DECISIONS.md` | every autonomous decision, including those later withdrawn |
| `BLOCKED.md`, `BLOCKED-v2.md` | the A1 collision and the v2 infeasibility, with evidence |
| `BUDGET.md` | measured throughput and the step-count arithmetic |
| `alibi/prereg.py` | the frozen v1 measurement, hashed into every run |
| `alibi/prereg_v2.py`, `alibi/prereg_v3.py` | later versions, inheriting by import; v3 is `RUNNABLE = False` |
| `artifacts/index.json` | evidence by explicit declaration, never by sort or mtime |

## What this is not

Not frontier scale. Not novel. Not bitwise reproducible on this rollout path. No claim
generalises beyond the 160 problems any single run samples, which are a deterministic
lowest-task-id prefix rather than a random draw, and which are almost entirely MBPP's test
split plus several of its few-shot exemplars.

## Dual use

This trains a small model to evade a text-based monitor on a toy coding task. The artifact
has no capability worth misusing, and the technique it demonstrates is already published.
It is included for the reason the safety literature publishes such things, which is that
the mitigation depends on knowing the failure mode exists.

## Licenses and citations

See [LICENSES.md](LICENSES.md).
