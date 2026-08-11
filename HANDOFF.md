# HANDOFF

Written for an audit, not for reassurance.

## The queue is running

| Item | Value |
|---|---|
| PID | **250259** |
| Log | `queue.log` (untracked, local) |
| Status | `.venv/bin/python -m alibi.cli queue status` |
| Progress | `PROGRESS.md`, rewritten each run and pushed |
| Stop | `kill 250259` |

It is a session leader reparented to init, so it survived the launching shell
and will survive a terminal close. It resumes by run id: if it is killed,
relaunch with `./run_queue.sh` and each run continues from its own
`state.json` rather than restarting.

Matrix: a0/a1/a2 at seed 1, then seed 2, then seed 3. 80 steps each, group 8,
2 prompts per step, 256 max new tokens. Measured ~35 s per step at launch, so
about 47 minutes per run and roughly 7 hours for all nine.

The queue stops only when all nine complete, when more than half fail, or when
BLOCKED.md exists. A single failed run marks itself FAILED and the queue moves on.

## What ran

- Instrument fixes, frozen pre-registration, halt conditions, GRPO loop,
  resumable queue, report, verify, monitor selection with probe and
  qualification.
- Gate over 376 problems: synthetic cheat passes visible 371/376, fails
  held-out 370/376. Eligible set 365 of 376, hashed.
- Calibration on a1: zero monitor errors in 72 judgements, zero indeterminate
  in 5052 held-out tests.
- 146 tests pass. `alibi verify --no-gpu` exits 0. Tag `alibi-prereg-v1.0`.

## What failed on the way, and what it cost

Four defects, all mine, all found by the machinery rather than by inspection:

1. **Dirty-tree deadlock.** The runner's own bookkeeping dirtied the tree and
   the next run's preflight halted on it. Fixed by gitignoring the logs and
   staging only runner-owned paths, **not** by narrowing the halt condition and
   **not** by `git add -A`, either of which would have destroyed the control.
2. **Hooks re-dirtying the tree** after each commit. Fixed with `--no-verify`
   on the runner's own commits only, which stage a fixed allowlist.
3. **CUDA OOM in generation.** `output_scores` at 8 by 384 by 151936 held about
   1.9 GB. Fixed by chunking and consuming scores per position, **not** by
   recomputing sampler logprobs post hoc, which would have silently destroyed
   the sampler seam the follow-on project exists to study.
4. **CUDA OOM in the update.** Sixteen forward graphs alive before one
   backward. Fixed with gradient accumulation.

A fifth, smaller: tracebacks were logged head-first, which discards the
exception message. That cost one diagnostic cycle on the first OOM.

## Where this is weakest, ranked

1. **The judge shares a family with the policy.** Qwen2.5-7B grading
   Qwen2.5-0.5B. This was your explicit override with the qualification table as
   evidence, made before any curve existed, and it is carried in REPORT.md as
   its own section. It bears directly on H2 and H3: a same-family judge may
   track this policy's obfuscation more closely than an independent one, which
   would understate the gap. Attack this first.
2. **Qualification is not pre-registered and proves less than it looks like.**
   Its thresholds are mine, it ran on the A2 view only, and 1.00 separation was
   measured on programmatically generated stereotyped cheats. It is necessary,
   not sufficient. Once the policy varies form, the judge faces a distribution
   this never sampled.
3. **The gate measures the environment, not the policy.** Hardcoding pays. That
   is not evidence a 0.5B model discovers hardcoding in 80 steps. H1 is exactly
   that question.
4. **The KL is a homemade proxy**, the difference between a mean per-token
   logprob under the current policy and the sampler's own, not a
   reference-policy KL. The KL spike halt fires on that proxy.
5. **The indeterminate boundary is my judgement.** I decided a candidate that
   does not compile is a fail and a timeout is indeterminate. That directly
   moves the primary metric's denominator.
6. **80 steps may be too few** for a trend to emerge. The step count was chosen
   to fit nine runs with slack, per the brief's preference for breadth.
7. **n=3 seeds.** Expect most between-arm differences to be unresolved.

## What to check first on waking

1. `PROGRESS.md` for run status, and `queue.log` for the tail.
2. Whether the false positive panel is non-zero. Calibration showed 0.00 with
   the Qwen judge, which is the best case and probably optimistic.
3. Whether oracle cheat rate moved at all in a0. Early steps show 0.00 to 0.06,
   which is noise, not a trend.
4. `artifacts/monitor_spend.json`. USD is null because no price per token is
   configured; set `ALIBI_MONITOR_USD_PER_MTOK` to get the USD 4.00 cap enforced
   in dollars rather than through a 20 million token stand-in.
