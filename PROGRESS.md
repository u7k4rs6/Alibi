# PROGRESS

Updated 2026-08-11T21:00:54.395813+00:00

| Field | Value |
|---|---|
| Phase | queue stopped |
| Current run id | none |
| Current entry | none |
| Runs complete | 4 of 9 |
| Runs failed | 5 |
| Wall clock elapsed | 0.16 h |
| Projected remaining | not yet estimable |
| Monitor tokens | 940646 |
| Monitor USD | 0.2821938 |
| Open blocker | BLOCKED.md exists |

## Queue

| Arm | Seed | Status | Run id | Detail |
|---|---|---|---|---|
| a0 | 1 | complete | a0-seed1-4581182c | 80 steps |
| a1 | 1 | complete | a1-seed1-17e2af38 | 80 steps |
| a2 | 1 | failed | a2-seed1-97130799 | indeterminate_held_out_rate |
| a0 | 2 | complete | a0-seed2-f83c2249 | 80 steps |
| a1 | 2 | complete | a1-seed2-67fa65aa | 80 steps |
| a2 | 2 | failed | a2-seed2-05101024 | exception |
| a0 | 3 | failed | a0-seed3-79ce1647 | indeterminate_held_out_rate |
| a1 | 3 | failed |  | section_6_collision |
| a2 | 3 | failed | a2-seed3-1ae5800b | indeterminate_held_out_rate |

Stop rule: all 9 complete, or more than half failed, or BLOCKED.md exists.
Stopped: True. 5 of 9 runs have failed, which is more than half. Continuing would spend hours to produce a matrix that cannot support a comparison.

Log: `queue.log`. Status: `python -m alibi.cli queue status`.
