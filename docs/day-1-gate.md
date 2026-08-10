# Day one gate

Run `alibi data check --include-dict-form` to reproduce. Roughly four minutes on
11 CPU workers, no GPU, no model.

Result: **the environment incentivises cheating.** A hardcoding solution built
programmatically from the visible asserts passes the visible tests on 371 of 376
problems and fails the held-out tests on 370 of them. PRD hypothesis H1 is worth
testing on a GPU.

Run id `datacheck-20260810T223417Z`, config hash
`53693b06fd3db5d79bf60e060fc73026a777c1a334f61f6369920928055f129a`.

## The numbers

Over 376 problems, MBPP joined to MBPP+ at the revisions pinned in
`alibi/data/manifest.json`.

| Measure | Count | Fraction |
|---|---|---|
| Synthetic cheat passes visible tests | 371/376 | 0.9867 |
| Passes visible and fails held out | 370/376 | 0.9840 |
| Passes visible and scores exactly 0.0 held out | 288/376 | 0.7660 |
| Flagged by the oracle, behavioural check | 361/376 | 0.9601 |
| Flagged by the oracle, structural check | 372/376 | 0.9894 |

Held-out pass fraction of the cheats that passed visible: mean 0.0369, median
0.0000, max 1.0000.

Control, to show the instrument is not simply failing everything:

| Measure | Count | Fraction |
|---|---|---|
| MBPP+ reference solution passes visible | 375/376 | 0.9973 |
| MBPP+ reference solution passes held out | 371/376 | 0.9867 |

Visible assert count: 3 asserts on 349 problems, 4 on 22, 5 on 3, 6 on 1, 7 on 1.
Held-out test count: min 1, median 105, max 147.

Base model visible and held-out pass rates are **not measured**. Day 1 loads no
model. They are scheduled for day 2 and are recorded as absent, not as zero.

## What the gate does not settle

**The reward is not saturated, but the headroom is not measured either.** This
says a hardcoder would be rewarded. It does not say the base policy cannot
already solve these problems honestly, which is the risk in architecture doc
section 11's third row. That needs the day 2 base model measurement.

**One problem has a single held-out test.** Task 635 has `n_held_out = 1`, and
the cheat passes it, which is the entire `max 1.0000` above. A held-out set of
size one is noise. Consider a minimum held-out count when the split is frozen.

**Roughly 23 problems give a hardcoder a real held-out score.** On these the
cheat's fallback `return None` happens to be correct for many held-out inputs,
so held-out pass sits between 0.2 and 0.8. This is why the 0.766 row, the
fraction scoring exactly zero, is the tighter statement of the effect than the
0.984 row.

**The structural check is shape dependent.** The same cheat written as a dict
lookup is flagged 0/376 by the structural rule as the PRD defines it, against
372/376 for the if-chain form. The dict form only passes visible on 201/376
because unhashable arguments make it a TypeError, so it is not a free evasion,
but a policy that learns the dict shape would be invisible to the structural
check while remaining fully visible to the behavioural one. Both checks are
recorded separately, so this shows up as disagreement rather than as a silent
gap.

**Four problems have no constructible cheat**, each named in the run's
`result.json`: task 596, whose reference solution needs `sys`, which the sandbox
denylist blocks, and tasks 737, 787 and 794, whose reference solutions return
`re.Match` objects with no round-trippable repr.
