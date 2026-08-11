# BLOCKED: v2 as registered does not fit this GPU

**Found:** 2026-08-11, before any v2 stage run. No v2 evidence exists.

## The collision

Two registered v2 values are individually justified and jointly infeasible on an
8 GB RTX 4060.

**`max_new_tokens = 3072`**, from `alibi/prereg_v2.py`, sized from measurement:
Qwen3-0.6B through its chat template produced total tokens with median 774 and
mean 1400.5, and 2 of 16 completions hit the 3072 cap. Under a thinking policy
the answer follows the think block, so a truncated completion yields **no code at
all**, which is why the budget was set generously.

**`max_capped_fraction = 0.35`**, the v2-only halt, declared before any v2 run
against that same measured 0.125 cap rate.

## What was measured

Backward pass through a prompt-plus-completion sequence, Qwen3-0.6B, LoRA r16,
gradient checkpointing enabled, one completion at a time, each budget in its own
process so nothing leaks between them:

| Budget | Result | Peak allocated |
|---|---|---|
| 3072 | **OOM** | 7.05 GB |
| 2048 | **OOM** | 6.45 GB |
| 1536 | **OOM** | 6.62 GB |
| 1024 | **OK** | 5.54 GB |

The floor is the logits tensor. At 3272 positions and a 151936-token vocabulary
that is about 1 GB in bf16 before the backward graph, and log_softmax over it in
float32 is another 2 GB if taken whole. Chunking the softmax and enabling
gradient checkpointing were both applied and are not sufficient: the full-vocab
logits for every position are materialised by the forward regardless.

An earlier bisection reported OOM at every budget including 1024. That run was
invalid: it reused one process and did not free the previous model, so the
baseline was already several GB. The table above is the corrected measurement.

## Why this cannot be resolved by choosing a smaller budget

At 1024 the measured median completion of 774 tokens fits, but the mean of 1400
does not, so a substantial fraction would be truncated. **The exact fraction at
1024 was not measured** and is not estimated here. What is certain is the
direction: truncation would rise well above the 0.125 measured at 3072, and the
v2 halt fires at 0.35 on a lagging window. A budget chosen to fit memory would
plausibly trip the halt that exists to catch exactly that failure.

Both numbers are registered in `alibi-prereg-v2.2`. Changing either is a
registration change, not an implementation choice, so it is not made here.

## Options, none taken

1. **Fused chunked LM head.** Compute the head and log_softmax in tiles over
   time and backward each tile, so full-vocab logits are never materialised for
   the whole sequence. Peak becomes one tile. This changes **no registered
   value** and would let 3072 run as registered. It is a real rewrite of the
   update with correctness risk around the clipping and anchor terms, and it is
   the option I would take first.
2. **Rent a larger GPU.** The PRD allows roughly 0.30 USD per hour against a 15
   USD ceiling. A 24 GB card removes the constraint outright.
3. **Lower `max_new_tokens`** and accept a higher capped fraction, which means
   amending both the budget and the halt in a v2.3 registration, with the
   truncation fraction measured first rather than assumed.
4. **A policy that thinks less.** Changes v2's policy, which is the one thing v2
   exists to change, so this is the least attractive.

## State

v1 is complete and untouched: four declared runs, five failures, no monitored-arm
evidence because arm A1 measured nothing. See `BLOCKED.md`.

v2 has produced no run. The probes have produced no run. Nothing has been
silently reconfigured to make something fit.
