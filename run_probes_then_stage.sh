#!/usr/bin/env bash
# Waits for v1 to drain, runs the three probes, then launches the v2 stage with
# whichever probe held reward flat or rising. Stops and says so if none did.
set -a; . ./.env; set +a
export PYTORCH_ALLOC_CONF=expandable_segments:True

while pgrep -f "alibi.cli queue run" > /dev/null 2>&1; do sleep 60; done
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] v1 drained, running probes"
.venv/bin/python -u -m alibi.diagnostics.probes || exit 1

CHOSEN=$(.venv/bin/python -c "import json;print(json.load(open('artifacts/diagnostics/probes/result.json'))['chosen'] or '')")
if [ -z "$CHOSEN" ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] NO PROBE HELD REWARD FLAT OR RISING. Stage not launched."
  exit 0
fi
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] probe $CHOSEN selected, launching v2 stage"
exec .venv/bin/python -u -m alibi.train.runner_v2
