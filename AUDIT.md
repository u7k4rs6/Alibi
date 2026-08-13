# AUDIT

Adversarial audit of the repository's claims against its artifacts, conducted
2026-08-13. The auditor's brief: find claims the artifacts do not support. The
priority target was REPORT.md section 3, "what the instrument got right", since
it is the load-bearing claim and was written by the same process that produced
the five faults.

Method: `alibi verify --no-gpu` and the full test suite were run; every fault
measurement was recomputed from stored artifacts; the reward-isolation and
Verdict invariants were attacked directly; git history was checked for the
timing claims; every numeric literal in REPORT.md was classified by backing.

Findings first, worst first. What verified as sound is at the end.

---

## Findings

### F1. Section 6's headline numbers have no backing artifact — HIGH

**Claim.** REPORT.md section 6 (Unresolved): after the fix, sampler and trainer
logprobs "differ on 0 of 96 rows, with a mean absolute difference of 0.2150 and
a maximum of 1.8477."

**Checked.** `artifacts/runs/selftest-logprob/` does not exist; it was deleted
after the measurement. No other artifact contains a non-copy `trainer_logprob`:
`memcheck-v2` and `timing-v2` exist but hold zero steps, and no diagnostics file
carries the pair.

**Found.** The numbers cannot be recomputed from anything in the repository.
Worse, the run that produced them used **Qwen2.5-0.5B-Instruct at a 48-token
budget** (a self-test configuration), not the v2 policy or budget, and section 6
does not say so. The open question is real, but its quantitative content — the
specific 0.2150/1.8477 magnitudes said to exceed bf16 noise — rests on a deleted
artifact from a different configuration than the text implies.

### F2. The report's closing verification claim is false — HIGH

**Claim.** REPORT.md's final line: "Numbers are injected from `artifacts/`,
never typed. `alibi verify --no-gpu` recomputes every claim above." README:
"Recomputes every published number from `artifacts/` alone."

**Checked.** `report/published.json` contains **12 claims**, all of them
terminal cheat/flag/gap min/max for arms a0 and a1 — every one of which is
0.0000. REPORT.md contains **36 distinct multi-decimal numeric literals**.
Exactly **1** of the 36 matches a verify-covered value.

**Found.** Verify covers roughly **3 percent** of the report's published
numbers. The rest divide into: injected live at build time from artifacts
(retrospective, precision, behavioural estimate, pooled step zero, think-region
table — these reproduce but verify does not check them), hardcoded prose that
reproduces from artifacts (fault 2.1/2.3 numbers, the 3430 count, the v3
table), and hardcoded prose with **no artifact at all** (F1, F3). "Never typed"
is also false as a blanket statement: the v2 bisection table, the fault 2.1
numbers and section 6's numbers are typed into the generator. The honest
statement would be "verify covers the declared-run terminal statistics;
everything else is recomputed at build time or cited to a named artifact, and
two tables have no artifact."

### F3. The v2 infeasibility table exists only as prose — HIGH

**Claim.** REPORT.md section 5 and BLOCKED-v2.md: OOM at 3072 (7.05 GB peak),
2048 (6.45), 1536 (6.62), fits at 1024 (5.54).

**Checked.** `artifacts/diagnostics/` contains `cap_at_1024`, `think_region`
and `v1_retrospective.json` — no bisection artifact. The numbers appear only in
BLOCKED-v2.md and REPORT.md prose.

**Found.** The measurement that justified abandoning v2 is not reproducible
from the repository and was run once, on a GPU shared with other processes at
the time (the OOM traces themselves show 70-130 MB of co-resident usage). A
prior bisection attempt on the same question was invalid (single-process leak,
disclosed in BLOCKED-v2.md), which raises rather than lowers the bar the
surviving measurement should meet. The direction of the result is plausible;
the specific peaks are unsupported.

### F4. "Declared before the measurement" rests on assertion — MEDIUM-HIGH

**Claim.** Section 3: the 0.01 near-zero bar for structural false positives was
"declared before the measurement was run". DECISIONS.md D-24 makes the same
claim, and cites it as exemplary practice.

**Checked.** `git log -S NEAR_ZERO_FALSE_POSITIVE_RATE` shows the bar, the
measurement code, and the DECISIONS entry describing the result all landed in
**one commit** (79c2d69). The same pattern holds for the v2 capped-fraction
halt's "declared before any v2 run" (true at the tag level, since v2.1 predates
any v2 run) and for the bootstrap seed.

**Found.** For the near-zero bar specifically, history cannot corroborate the
ordering: within a single commit, nothing distinguishes "bar fixed, then
measured" from "measured, then bar written". The claim may well be true; it is
not evidenced. A practice worth adopting: commit the threshold before running
the measurement, so the ordering is in history rather than in testimony.

### F5. The determinacy floor's mandated output was dropped — MEDIUM

**Claim.** Section 3: "The determinacy floor ... is applied at report time and
recomputed from stored completions, so an auditor can check it without a GPU."
The operator's original instruction: report the excluded fraction per step per
arm as its own series, and report the headline gap both with and without the
floor, floor primary where they differ.

**Checked.** `metrics.floored_series` and `recompute_step` exist and work; the
figure panel exists. But the rewritten REPORT.md contains **no with/without-
floor comparison and no excluded-fraction table** — one sentence in section 3
is the floor's entire presence in the report body.

**Found.** The computation exists; the publication mandated for it does not.
The rewrite instruction was "same content ... nothing gets softened", and this
output was lost. Section 3 cites the floor as something the instrument got
right while the report no longer shows the floor doing anything.

### F6. Oracle/reward separation is test-enforced with an enumerable hole — MEDIUM

**Claim.** Section 3: "separated ... **by construction**", with a test that
fails if anything connects them, including through a call site.

**Checked.** Attacked directly. `reward.py` imports no oracle module (verified
by AST, not text). `reward_fn`'s signature cannot receive oracle data by name,
and the call-site AST test scans argument text for oracle terms. But
`ScoredCompletion` carries a **`metadata: dict`** field. A `ScoredCompletion`
was constructed in the audit with `metadata={"oracle_behavioural": True,
"held_out_pass_fraction": 0.0}` and passed to `reward_fn` with no error and no
test failure, because the smuggling happens at construction, not at the reward
call site, and the dict's contents are invisible to every isolation test.

**Found.** Current code never reads `metadata` inside the reward (verified),
so the isolation **holds today**. But it holds by the absence of a read, not by
construction: the architecture doc's own standard is "ScoredCompletion
deliberately has no oracle field", and a generic dict field is an oracle field
waiting for an assignment. Two lesser bypasses also exist and should be named
for honesty rather than fixed: `object.__setattr__` defeats the frozen Verdict
(sabotage-level, inherent to Python dataclasses), and monitor cache entries are
unsigned, so a tampered committed cache file would replay `flagged: false`
verdicts through the legitimate path.

### F7. The code still calls itself "TRL GRPOTrainer wiring" — MEDIUM

**Claim.** The fold: "No `GRPOTrainer` was used ... nothing anywhere implies
otherwise" (the audit brief's phrasing of what should hold).

**Checked.** `alibi/train/grpo.py` line 1: `"""TRL GRPOTrainer wiring, and the
smoke test that proves the loop is real.` The module that contains the
hand-written update opens by describing itself as the thing the report denies.

**Found.** The report's disclosure is correct and prominent; the code's own
docstring contradicts it. A reader who opens the module before the report gets
the wrong claim first.

### F8. The indeterminate halt selectively killed arm A2, unanalysed — MEDIUM

**Claim.** Section 5, H3: "no A2 run ever completed: every A2 attempt failed,
three of them on the indeterminate held-out halt." Stated as fact, not analysed.

**Checked.** Run states: a2/s1 failed on the indeterminate halt at step 64,
mid-morning, under the same conditions in which a0/s1 and a1/s1 completed;
a2/s3 and a0/s3 failed on the same halt late in the wall-clock day. **A2
completed zero of five attempts across the entire project.** Additionally,
three `a2-seed2-*` run directories remain in `status: running` — killed
processes whose states were never marked failed.

**Found.** Two candidate mechanisms are visible in the artifacts and neither is
discussed: A2's steps are longer (live monitor calls plus honest probes), which
widens the window in which held-out scoring competes for CPU and times out; and
failures cluster late in the day, consistent with the thermal-drift threat the
report itself names elsewhere but never connects to the A2 attrition. If the
halt's firing probability depends on the arm, surviving runs are selected for
being cheap arms, and any future matrix run under this design inherits that
bias. This is the strongest candidate for a **sixth fault visible in the
artifacts but unreported**. (The v1 `kl` column measuring noise is arguably
also a sixth fault, but it is disclosed above the fold; it is excluded from the
five-count by framing, not concealed.)

### F9. The "non-circular" population differs by ten problems — LOW-MEDIUM

**Claim.** Section 3: the behavioural false-positive estimate (0/375, upper
bound 0.0101) "rests on a genuinely non-circular population."

**Checked.** The pre-eligibility set is 375 evaluable problems; the eligible
set is 365. The excluded ref-fails-held-out class is task 255 at h=0.9908
(nowhere near the 0.10 cheat threshold) and task 596 at h=None — **unmeasurable,
counted as "fails held-out"** in the eligibility rule and in prose that says "2
whose reference fails held-out."

**Found.** The estimate is genuinely non-circular in construction: the removed
problems could not have been behavioural false positives (h=0.99 is not ≤0.10,
and None is excluded from the estimate's denominator anyway). But the
non-circular population shares 97 percent of its members with the circular one,
so it adds about ten problems of independent information; the upper bound of
0.0101 is driven by n, not by independence. The report presents the limits
honestly but not this one. Also, counting an unmeasurable reference as "fails
held-out" is a small misdescription repeated in the gate output.

### F10. Provenance disclosures are partial — LOW

**Checked.** The four declared evidence runs ran at **four different git
revisions** (74812eda, cd77823d, 8e6935a2, ed48c335). Diffing them shows the
changes touch only report-layer code and monitor cache files — `reward.py`,
`scoring.py`, `oracle.py`, `tests.py`, `executor.py` and `prereg.py` are
byte-identical across all four, and all four carry the same prereg and
eligibility hashes. The report discloses the config-hash discontinuity and both
halt amendments (both check out against history) but not the multi-revision
matrix.

**Found.** Harmless in substance, incomplete in disclosure. The claim "an
auditor diffing config hashes will see a discontinuity" understates what an
auditor actually sees, which is four revisions, and makes them do the
measurement-path diff themselves to learn it does not matter.

### F11. Qualification ran on the training prefix — LOW

**Checked.** `qualify()` samples `eligible_problems()[:sample]` — the same
lowest-task-id prefix the training runs sample. Section 3 lists qualification's
limits (not pre-registered, A2 view only, stereotyped cheats) but not that the
judge was qualified on the exact problems it would later judge in training.

**Found.** One more limit that belongs in the stated list, given the report
elsewhere establishes the prefix is not representative of MBPP.

---

## Verified as sound

- `alibi verify --no-gpu` exits 0; all 148 tests pass.
- **Tag ordering holds.** `alibi-prereg-v1.0` was committed 2026-08-10 23:21 UTC;
  the earliest declared run started 2026-08-11 05:40 UTC. The v2.x tags all
  predate any v2 run; no v2 or v3 run ever executed.
- **The prereg and eligibility hashes are identical across all four evidence
  runs**, and the measurement-path code is byte-identical across their four
  revisions (verified by diff, not asserted).
- **Structural precision reproduces exactly**: 5/365, rate 0.0137, same five
  task ids (84, 167, 577, 626, 801).
- **The behavioural estimate reproduces**: 0/375, upper bound 0.0101, from the
  stored datacheck artifact.
- **Fault 2.1 reproduces**: a1/s1 first 20 steps, 320 completions, mean
  `think_chars` 3.2, 0/320 flagged; a2/s1 flagged 21/320. The a0/s1-a1/s1
  identity holds with **zero mismatches at three decimals across five series
  times 80 steps** — stronger than the report claims.
- **Fault 2.3 reproduces**: capped 574/1280 = 0.4484; visible pass 0.1053
  capped vs 0.2254 natural.
- **Fault 2.4 reproduces**: reward 0.2176 to 0.0990, entropy 0.6384 to 0.8354,
  zero-variance mean 0.2375, from stored artifacts via the retrospective.
- **Fault 2.5's premise reproduces**: 3430/3430 identical rows in a0/s1 step 0.
  (The post-fix numbers are F1.)
- **The v3 qualifying measurement reproduces from its artifact**: capped
  0.6875, closed-think = has-code = 27, in
  `artifacts/diagnostics/cap_at_1024/result.json`.
- **The think-region table is injected live** from
  `artifacts/diagnostics/think_region/result.json`, and its 0/16, 0/16, 16/16
  values match the artifact.
- **The Verdict invariant holds against non-sabotage attack**: direct
  construction of an unflagged error verdict raises, and
  `dataclasses.replace(..., flagged=False)` raises because `__post_init__`
  re-runs.
- **`reward.py` imports no oracle module** (AST-verified; the word "oracle"
  appears only in comments explaining the prohibition).
- The report and README state plainly, above the fold, that no `GRPOTrainer`
  was used and the update is hand-written (F7 concerns the code's own
  docstring, not the report).
- The five faults are five: no two subsections describe the same defect. 2.1
  (policy emits no think region) and 2.2 (template never applied) are the
  closest pair and are genuinely independent — the think-region diagnostic
  shows either alone empties the view.

---

## Best guess at the claim most likely wrong that could not be checked

**The v2 memory bisection (F3), and specifically "1024 fits at 5.54 GB peak."**
It was measured once, in a single process per budget but on a GPU carrying
70-130 MB of unrelated resident memory, after a first attempt at the same
measurement produced invalid results for a subtle reason. It is load-bearing —
it is the factual basis for abandoning v2 and scoping v3's budget — it has no
artifact, and a peak-memory measurement on a shared consumer GPU with a
fragmenting allocator is exactly the kind of number that fails to reproduce.
If any number in this repository is wrong, it is probably one of the four in
that table.

---

## Remediation, 2026-08-13

Every finding above is left as written. Status per finding, with the remedy
chosen and, where a claim could not be supported, the withdrawal.

- **F1 — remediated by re-measurement, prior numbers withdrawn.** The
  divergence was re-measured on the v2 policy and budget, per token, no
  training, and committed to `artifacts/diagnostics/logprob_divergence/`
  (declared in the index). The result, mean 0.0175 and max 0.354 over 19,347
  pairs, is an order of magnitude below the withdrawn 0.2150/1.8477, which
  confirms the finding: the deleted-run figures did not survive a provenanced
  re-measurement. Section 6 and the README now carry the withdrawal and the new
  numbers.
- **F2 — remediated by counting.** 33 fault-measurement claims were added to
  `published.json` (45 total), all recomputed from artifacts by
  `alibi.report.metrics.fault_measurements`. Verify now covers 21 of the
  report's 42 distinct numeric literals, and the closing line states that
  counted figure, the prior false claim, and the residual, instead of claiming
  full coverage.
- **F3 — remediated by re-measurement; the audit's best guess was wrong.** The
  bisection was re-run with a fresh process per repeat, three repeats per
  budget, and the 613 MiB unevictable GPU baseline recorded. Every peak
  replicated bit-identically and matches the prose table, so the v2 abandonment
  stands. The artifact is committed and declared in the index, and section 5
  now states that a load-bearing decision rested for a time on a single
  unartifacted measurement. The audit's closing guess, that this table was the
  number most likely wrong, did not survive its own re-measurement, which is
  recorded here rather than quietly dropped.
- **F4 — claim withdrawn.** The report now discloses that the bar, the
  measurement and the result landed in one commit, that the ordering rests on
  the author's word, and that the pre-declaration claim is withdrawn rather
  than reworded.
- **F5 — remediated.** The with/without-floor gap table is restored under the
  floor bullet in section 3, with the excluded-fraction series in its figure
  panel.
- **F6 — remediated by removing the channel.** The `metadata` dict was deleted
  from `ScoredCompletion` rather than guarded, because a generic container on
  that type is an oracle field waiting for an assignment, and two tests now
  fail on the exact smuggling path the audit used and on any container-typed
  field reappearing. The `object.__setattr__` and unsigned-cache bypasses are
  accepted as inherent to Python and to an editable repository, and are named
  in this file rather than papered over.
- **F7 — remediated.** The `grpo.py` docstring no longer calls itself TRL
  GRPOTrainer wiring, and records why the line changed.
- **F8 — promoted to fault 2.6.** With the recomputed numbers: a2 completed 0
  of 5 attempts at a mean step of 58.97 seconds against a0's 40.85, and the
  report states the generalisation, that a halt whose firing probability
  depends on a treatment variable is a selection filter, and states that the
  audit found it, not the instrument.
- **F9 — disclosed.** The 97 percent population overlap and the
  sample-size-driven upper bound are stated beside the estimate.
- **F10 — disclosed.** The four-revision matrix and the diff result are in
  provenance.
- **F11 — disclosed.** Qualification-on-the-training-prefix is added to the
  stated limits.

The report also gained a section, "The audit, and where this report failed its
own standard", naming the four section-3 failures in the report's own voice and
citing what the audit verified as sound.
