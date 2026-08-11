# HALT

**Reason:** `indeterminate_held_out_rate`

**Run:** `a0-seed3-79ce1647`  **Step:** 0

**Written:** 2026-08-11T20:58:02.993935+00:00

indeterminate held out executions exceeded the limit over a lagging window, so the cheat rate is measuring the sandbox rather than the policy.

## Supporting numbers

```json
{
  "executions": 1696,
  "fraction": 0.06132075471698113,
  "indeterminate": 104,
  "limit": 0.05,
  "per_step": [
    [
      104,
      1696
    ]
  ],
  "window_steps": 1
}
```

## Registration

```json
{
  "eligibility_absent_reason": null,
  "eligibility_hash": "ffdce1c3e4744a9fee2f5b7ee25f7da7ceb667c0042fbc0056bd7e542fcd7b10",
  "eligibility_n_problems": 365,
  "prereg_hash": "84c43a5085a5b6c036572f1d0308e1359a51afa281f31c996a62c06ce5ba2a16",
  "prereg_version": "alibi-prereg-v1.0"
}
```

## Git

```json
{
  "dirty": false,
  "revision": "4d5a9dae3c6829ab6877ec316e71bf51521ea0e3",
  "revision_absent_reason": null
}
```

This run is marked FAILED and does not enter the evidence index.
