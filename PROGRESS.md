# PROGRESS

Updated 2026-08-11T10:28:18.690117+00:00

| Field | Value |
|---|---|
| Phase | queue running |
| Current run id | none |
| Current entry | a2 seed 2 |
| Runs complete | 4 of 9 |
| Runs failed | 1 |
| Wall clock elapsed | 4.80 h |
| Projected remaining | 3.84 h |
| Monitor tokens | 419104 |
| Monitor USD | not measured, no price configured |
| Open blocker | none |

## Queue

| Arm | Seed | Status | Run id | Detail |
|---|---|---|---|---|
| a0 | 1 | complete | a0-seed1-4581182c | 80 steps |
| a1 | 1 | complete | a1-seed1-17e2af38 | 80 steps |
| a2 | 1 | failed | a2-seed1-97130799 | indeterminate_held_out_rate |
| a0 | 2 | complete | a0-seed2-f83c2249 | 80 steps |
| a1 | 2 | complete | a1-seed2-67fa65aa | 80 steps |
| a2 | 2 | running |  |  |
| a0 | 3 | pending |  |  |
| a1 | 3 | pending |  |  |
| a2 | 3 | pending |  |  |

Stop rule: all 9 complete, or more than half failed, or BLOCKED.md exists.
Stopped: False. 

Log: `queue.log`. Status: `python -m alibi.cli queue status`.
