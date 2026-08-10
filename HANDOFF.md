# HANDOFF

Written for an audit, not for reassurance.

## What ran

- Instrument fixes, the frozen pre-registration, halt conditions, the GRPO loop,
  the resumable queue, the report and verify paths.
- Full gate over 376 problems, twice, after the fixes.
- Smoke test on this host: 3 GRPO steps, 2 prompts, every logging path written.
- 131 tests pass on the training interpreter.

## What did not run

**The run matrix has not started.** No a0, a1 or a2 run exists. Every hypothesis
is unevaluated. Calibration has not run either, so BUDGET.md is empty by design
rather than estimated.

## What I should check first

1. `docs/deviations.json` D8 is still `open`: base model pass rates are
   unmeasured, so nobody has confirmed the base policy cannot already solve
   these problems honestly. If it can, the environment does not have the
   headroom the gate implies. This is the cheapest thing to check and the most
   likely to invalidate the design.
2. `BUDGET.md` is empty. The step count for the matrix has not been chosen, so
   the matrix cannot be sized.

## Where this result is weakest, ranked

1. **The gate measures the environment, not the policy.** A synthetic cheat
   passing visible on 98.67 percent of problems shows hardcoding is rewarded. It
   says nothing about whether a 0.5B model discovers hardcoding in the step
   budget. H1 is exactly that question. An auditor should attack any sentence
   that slides from "cheating pays" to "the model will cheat".
2. **The structural check was extended after seeing data.** On day 1 it caught
   the if-chain cheat 372/376 and the dict cheat 0/376, and I extended it. That
   is metric development after observation, and the only reason it is defensible
   is that the extension happened before any training curve existed and the
   structural check was simultaneously demoted to diagnostic. An auditor should
   check the commit order against the tag, and should treat any structural
   number as soft.
3. **The KL term is homemade.** `_update` in `alibi/train/grpo.py` is a plain
   GRPO formulation, not TRL's `GRPOTrainer.train()`, and its KL is the
   difference between a mean per-token logprob under the current policy and the
   sampler's own mean. That is a proxy, not a reference-policy KL. The KL spike
   halt condition therefore fires on a proxy. Attack this first if any run halts
   on `kl_spike`.
4. **Indeterminate is defined by me.** I decided a candidate that does not
   compile is a fail and a timeout is indeterminate. That boundary is a
   judgement call and it directly moves the primary metric's denominator. It is
   defensible, it is documented at the site, and it is exactly where I would
   look for a thumb on the scale.
5. **The synthetic cheat is not what a model produces.** It is a generated
   if-chain over the exact visible inputs. A real policy's hardcoding will be
   messier and may fall outside every enumerated `cheat_form`. The behavioural
   oracle is form-blind, which is why it is primary, but the `cheat_form` panel
   will understate real cheating.
6. **Single host, laptop, thermal drift.** Nine runs over days on an RTX 4060.
   Held-out timeouts are the contamination channel. The indeterminate rate is a
   halt condition and a figure panel, and its correlation with cheat rate is
   reported, but a sustained thermal decline that tracks training progress would
   be hard to distinguish from a real effect.
7. **n=3 seeds.** Any arm difference smaller than the seed spread is not
   resolved, and with three seeds the spread is wide. Expect most between-arm
   comparisons to be unresolved.

## What is uncertain

- Whether the monitor will be the LLM monitor or the rules monitor for a1 and
  a2. No monitor model has been chosen and no API key is configured. The code
  refuses to substitute the rules monitor into a monitored arm, so as it stands
  a1 and a2 cannot run against an LLM monitor without that choice being made.
  This is the single largest unresolved dependency for the matrix.
- Whether this host or a Kaggle host runs the matrix. The smoke test ran here.
  Kaggle almost certainly cannot open user namespaces, so the mount and network
  namespaces would degrade there and `env.lock` would say so.
