# BUDGET

**Status: not measured.** The calibration run has not been executed, so every
number below is absent rather than estimated. An unmeasured value is absent, not
zero, and a projected wall clock invented here would be exactly the kind of
fabricated number the project forbids.

## What calibration must fill in

Run a short A0 calibration at seed 1, 20 to 40 steps or 20 minutes, whichever
comes first. It is calibration, not evidence, and `artifacts/index.json` already
excludes `calib-*` from the evidence index by policy.

| Measure | Value | How |
|---|---|---|
| Seconds per training step, mean | not measured | `duration_seconds` in each step summary |
| Seconds per training step, p90 | not measured | same |
| Executor throughput, visible, under load | not measured | visible `duration_seconds` per completion |
| Executor throughput, held-out, under load | not measured | held-out `duration_seconds` per completion |
| Monitor latency | not measured | `monitor.mean_latency_seconds` per step |
| Monitor cache hit rate | not measured | `monitor.cache_hit_rate` per step |
| Monitor error rate | not measured | `monitor.error_fraction` per step |
| Indeterminate rate, held-out | not measured | `held_out_indeterminate_fraction` per step |
| p99 reference test time under load | not measured | needed for the timeout headroom ratio |
| Timeout headroom ratio | not measured | per-test held-out timeout (2.0 s) divided by p99 above |
| Disk per step for logprobs.parquet | not measured | measured: the 3-step smoke run wrote 3 files for 384 token rows |
| Projected disk, full matrix | not measured | per-step disk times steps times 9 runs |
| Projected wall clock, full matrix | not measured | per-step seconds times steps times 9 runs |

## Timeout headroom

The per-test held-out timeout is 2.0 s. If the measured ratio to the p99
observed reference test time under load is under 10x, the timeout is raised and
the change is recorded in DECISIONS.md. Timeout headroom is infrastructure, not
measurement, so it may be changed. The visible and held-out split, and the
existence of a per-test timeout, may not.

## Choosing the step count

Once the table above is filled in, choose the per-run step count so the full
nine-run matrix fits with 25 percent slack, and record the arithmetic here.
Every run in the matrix uses the same step count. Fewer steps across the whole
matrix beats a complete run of one arm and nothing else.

## Measured so far

From the 3-step smoke run on this host (2 prompts, group 2, 96 max new tokens,
so **not** representative of a real step):

- 384 token rows written to logprobs.parquet across 3 steps
- held-out indeterminate rate 0.00
- monitor cache hit rate 1.00 (rules monitor is deterministic)

These are recorded because they were observed. They are not a projection.
