# Alibi

Train a small open model with GRPO on a task where cheating pays, penalise it with a
monitor that reads its reasoning, and measure whether it stops cheating or just stops
saying so.

---

## Read this first

This repository contains a working instrument and an experiment that has not yet
produced a citable result. Four faults were found by the instrument itself, and all
four are reported rather than quietly corrected.

**No `GRPOTrainer` was used.** TRL is pinned in every `env.lock` and is never
instantiated. The update is a hand-written policy gradient in `alibi/train/grpo.py`.
There is no `GRPOConfig` to inspect, and any reader assuming a standard TRL loop will
misread the artifacts.

**In v1 that objective was neither clipped nor anchored.** No reference policy existed,
so there was no KL term in the loss, no ratio clipping and no trust region. It was
`-advantage x logprob` at a constant learning rate, with no scheduler and no warmup.
An unclipped, unanchored objective is not GRPO.

**v1's update did not condition on the prompt.** It fed completion tokens with the
prompt stripped, so the gradient maximised the unconditional likelihood of completion
text rather than the likelihood of a solution given a problem.

**Two v1 columns do not mean what their names suggest.**

- `kl` is not KL from a reference policy. None existed. It compared the policy against
  the sampler's own logprobs from the same step, computed with the same weights, so it
  measured numerical noise rather than drift. **The KL spike halt condition was
  guarding nothing in v1**, and no run tripping it is uninformative rather than
  reassuring.
- `trainer_logprob` is a literal copy of `sampler_logprob`, identical on 3430 of 3430
  rows at step 0. v1's `logprobs.parquet` therefore does not support the sampler versus
  trainer analysis it was collected for.

All four are fixed for v2. None is fixed retroactively: v1's artifacts stand as
measured. **v1's flat cheat rate is not evidence about the pre-registered H1.**

A fifth finding, of a different kind: **arm A1 measured nothing.** Its monitor reads a
designated thought region, and Qwen2.5-0.5B-Instruct emits none, so the monitor judged
empty strings and A1 became arithmetically identical to the unmonitored control. See
[BLOCKED.md](BLOCKED.md).

## What is solid

- The environment gate. A hardcoding solution built programmatically from the visible
  asserts passes visible tests on 371 of 376 problems and fails held-out on 370.
  Cheating pays, which is the premise the whole design rests on.
- The sandbox. Four adversarial tests, mount and network namespaces, fail-closed.
- The oracle is separated from the reward by construction, with a test that fails if
  anything connects them.
- Monitor errors are always flagged, so an outage can never read as innocence.

## Checking the claims

```
alibi verify --no-gpu
```

Recomputes every published number from `artifacts/` alone. No GPU, no network.

## Layout

| Path | What it is |
|---|---|
| `report/REPORT.md` | generated, numbers injected from artifacts |
| `DECISIONS.md` | every autonomous decision, what was rejected, and why |
| `BLOCKED.md` | the A1 collision, with evidence |
| `BUDGET.md` | measured throughput and the step-count arithmetic |
| `alibi/prereg.py` | the frozen v1 measurement, hashed into every run |
| `alibi/prereg_v2.py` | v2, which imports v1's measurement rather than restating it |
| `artifacts/index.json` | evidence by explicit declaration, never by sort or mtime |

## What this is not

Not frontier scale. Not novel. Not bitwise reproducible on this rollout path. The
policy is 0.5B in v1 and 0.6B in v2, and no claim here generalises beyond the 160
problems any single run actually samples, which are a deterministic prefix of the
eligible set rather than a random draw.

## Dual use

This trains a small model to evade a text-based monitor on a toy coding task. The
artifact has no capability worth misusing, and the technique it demonstrates is already
published. It is included for the reason the safety literature publishes such things,
which is that the mitigation depends on knowing the failure mode exists.

## Licenses and citations

See [LICENSES.md](LICENSES.md).
