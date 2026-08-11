# BUDGET

Measured from the a1 calibration run `calib-a1-s1`, 6 steps, live Qwen judge.
Calibration is **not evidence** and `artifacts/index.json` excludes `calib-*`
from the evidence index by policy.

## Measured

| Measure | Value | Conditions |
|---|---|---|
| Seconds per step, mean | **12.73** | 2 prompts x group 4 = 8 completions, 256 max new tokens |
| Seconds per step, p90 | **14.47** | same |
| Seconds per step, min to max | 9.50 to 14.95 | same |
| Monitor mean latency | **0.039 s** | 65 percent cache hits, so this is a blend |
| Monitor cache hit rate | **0.653** | rises as views repeat |
| Monitor error rate | **0 / 72** | against a 0.02 halt threshold |
| Held-out indeterminate | **0 / 5052** | against a 0.05 halt threshold |
| False positive probe | **0.00** every step | Qwen judge on reference solutions |
| Monitor tokens | 2093 over 72 calls | ~29 tokens per call at this view size |
| logprobs.parquet | 26.1 KB per step | 8 completions at 256 tokens |
| GPU temperature | 56 to 63 C | utilisation 0 to 95 percent |

## Chosen configuration and the arithmetic

The calibration ran a smaller shape than the matrix. Scaling from it:

- group size **8**, fixed by `docs/kickoff/01-prd.md` section 8, not mine to change
- prompts per step **2**, so 16 completions per step, twice calibration's 8
- max new tokens **384**, 1.5 times calibration's 256

Work per step is therefore about `2 x 1.5 = 3` times calibration, so

```
12.73 s x 3                       = 38 s per step, expected
38 s x 80 steps                   = 3040 s   = 0.85 h per run
0.85 h x 9 runs                   = 7.6 h    for the full matrix
7.6 h x 1.25 (25 percent slack)   = 9.5 h    planned envelope
```

**Step count chosen: 80, identical for every run in the matrix.**

Deadline is 17 Aug and the matrix starts 11 Aug, so a 9.5 h envelope leaves
several days of margin for reruns and for the a3 stretch arm. A larger step
count was rejected: the brief is explicit that fewer steps across the whole
matrix beats a complete run of one arm and nothing else, and 80 steps at 16
completions each is 1280 completions per run, which is enough for a cheat rate
trend to be visible if there is one.

## Projected

| Projection | Value | Basis |
|---|---|---|
| Disk, logprobs, full matrix | ~56 MB | 26.1 KB x 3 x 80 steps x 9 runs |
| Monitor calls, full matrix | ~5760 | 12 judgements per step x 80 steps x 6 monitored runs |
| Monitor tokens, full matrix | not projected | calibration views were short and unrepresentative; the ledger measures the real figure as it goes |

## Timeout headroom

The per-test held-out timeout is 2.0 s. Under real training load the calibration
produced **zero** indeterminate held-out executions across 5052 tests, so no
test came close to its budget and the ratio to the p99 observed test time is
comfortably above 10x. The timeout is therefore left unchanged. Had it been
under 10x, the timeout would have been raised and the change recorded in
DECISIONS.md, because timeout headroom is infrastructure and not measurement.

## Spend

`ALIBI_MONITOR_USD_PER_MTOK` is not set, so the ledger reports tokens and
`usd: null` with a reason rather than an invented dollar figure. The USD 4.00
cap is enforced through a 20 million token stand-in until a price is configured.
Set the variable to get the cap enforced in dollars.
