#!/usr/bin/env bash
# Waits for the v1 queue to drain, then runs the v2 stage. Never runs both at
# once: two policies on one 8 GB card is the OOM that cost v1 two runs, and CPU
# contention is what raises v1's indeterminate rate into a halt.
set -a; . ./.env; set +a
export PYTORCH_ALLOC_CONF=expandable_segments:True

while pgrep -f "alibi.cli queue run" > /dev/null 2>&1; do sleep 60; done
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] v1 queue has drained, starting v2 stage"
exec .venv/bin/python -u -m alibi.train.runner_v2
