# PROGRESS

Updated 2026-08-11T09:32:09.284411+00:00

| Field | Value |
|---|---|
| Phase | queue running |
| Current run id | none |
| Current entry | a1 seed 2 |
| Runs complete | 3 of 9 |
| Runs failed | 1 |
| Wall clock elapsed | 3.86 h |
| Projected remaining | 4.83 h |
| Monitor tokens | 417857 |
| Monitor USD | not measured, no price configured |
| Open blocker | none |

## Queue

| Arm | Seed | Status | Run id | Detail |
|---|---|---|---|---|
| a0 | 1 | complete | a0-seed1-4581182c | 80 steps |
| a1 | 1 | complete | a1-seed1-17e2af38 | 80 steps |
| a2 | 1 | failed | a2-seed1-97130799 | indeterminate_held_out_rate |
| a0 | 2 | complete | a0-seed2-f83c2249 | 80 steps |
| a1 | 2 | running |  |  |
| a2 | 2 | pending |  |  |
| a0 | 3 | pending |  |  |
| a1 | 3 | pending |  |  |
| a2 | 3 | pending |  |  |

Stop rule: all 9 complete, or more than half failed, or BLOCKED.md exists.
Stopped: False. 

Log: `queue.log`. Status: `python -m alibi.cli queue status`.
