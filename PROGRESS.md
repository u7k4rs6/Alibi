# PROGRESS

Updated 2026-08-11T00:40:26.318998+00:00

| Field | Value |
|---|---|
| Phase | queue stopped |
| Current run id | none |
| Current entry | none |
| Runs complete | 0 of 9 |
| Runs failed | 5 |
| Wall clock elapsed | 0.22 h |
| Projected remaining | 0.17 h |
| Monitor tokens | 10444 |
| Monitor USD | not measured, no price configured |
| Open blocker | none |

## Queue

| Arm | Seed | Status | Run id | Detail |
|---|---|---|---|---|
| a0 | 1 | failed | a0-seed1-4581182c | indeterminate_held_out_rate |
| a1 | 1 | failed | a1-seed1-17e2af38 | exception |
| a2 | 1 | failed | a2-seed1-97130799 | dirty_git_tree |
| a0 | 2 | failed | a0-seed2-f83c2249 | dirty_git_tree |
| a1 | 2 | failed | a1-seed2-67fa65aa | dirty_git_tree |
| a2 | 2 | pending |  |  |
| a0 | 3 | pending |  |  |
| a1 | 3 | pending |  |  |
| a2 | 3 | pending |  |  |

Stop rule: all 9 complete, or more than half failed, or BLOCKED.md exists.
Stopped: True. 5 of 9 runs have failed, which is more than half. Continuing would spend hours to produce a matrix that cannot support a comparison.

Log: `queue.log`. Status: `python -m alibi.cli queue status`.
