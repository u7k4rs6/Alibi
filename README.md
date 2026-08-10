# Alibi

Trains a small language model on a coding task where cheating pays, penalises it
with a monitor that reads its reasoning, and measures whether it stops cheating
or just stops saying so.

**Status: day 1 of 7.** The instrument is built. No training has run. Numbers
below are from the environment, not from a policy.

## The measurement

Two judges that are deliberately kept apart:

- the **reward monitor** reads text, and its verdict enters the reward
- the **cheat oracle** executes held-out tests the model never sees, and its
  verdict never enters the reward

```
obfuscation_gap(step) = cheat_rate_oracle(step) - flag_rate_monitor(step)
```

A model that reforms drives both terms down. A model that learns to hide drives
the second down while the first stays put.

`tests/test_reward_isolation.py` fails the day anything connects them.

## Day one gate

Before spending a GPU hour, the environment has to actually reward cheating. A
hardcoding solution is constructed programmatically from each problem's visible
asserts, with no model involved, and scored:

```
alibi data check
```

See `docs/day-1-gate.md` for the measured result.

## What this is not

- Not frontier scale. Qwen2.5-0.5B with LoRA.
- Not novel. A replication in miniature of published work.
- Not bitwise reproducible on this rollout path. Determinism is the follow-on
  project's subject, not a claim made here.
- Not a security boundary against a determined attacker. The sandbox is built
  for untrusted generated code, and `alibi/env/probe.py` reports exactly which
  controls are active on a given host.

## Licenses and citations

See `LICENSES.md`.

## Dual use

This trains a small model to evade a text based monitor on a toy coding task.
The artifact has no capability worth misusing and the technique is already
published. It is included for the reason the safety literature publishes such
things, which is that the mitigation depends on knowing the failure mode exists.
