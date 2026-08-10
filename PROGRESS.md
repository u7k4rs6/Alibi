# PROGRESS

One file to read on waking. Updated at least every 15 minutes while the queue runs.

| Field | Value |
|---|---|
| Phase | preparation complete, matrix NOT started |
| Current run id | none |
| Step | n/a |
| Wall clock elapsed | preparation session only |
| Projected finish | not projected: calibration has not run, see BUDGET.md |
| Open blocker | the run matrix has not been started, see "How to start" below |

## State

- Preparation session finished and printed GO.
- Tag `alibi-prereg-v1.0` exists.
- `alibi verify --no-gpu` exits 0.
- Smoke test passed on this host: 3 GRPO steps, 2 prompts, all logging paths written.
- Queue: 9 entries (a0/a1/a2 x seeds 1/2/3), all `pending`.

## How to start

Calibration first, then the matrix. Neither has run.

```
.venv/bin/python -m alibi.cli train --arm a0 --seed 1 --steps 30   # calibration, not evidence
.venv/bin/python -m alibi.cli train --arm a0 --seed 1              # matrix
```

Runs are resumable by run id: rerunning the same command continues from
`artifacts/runs/<run_id>/state.json` rather than restarting.

## Halt state

No HALT.md. No BLOCKED.md.
