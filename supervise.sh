#!/usr/bin/env bash
# One supervisor for all three phases. Written because detached runners have
# been killed three times mid-run by session churn, and a chain that assumes a
# previous process is still alive silently skips its phase when it is not.
#
#   phase 1  drain the v1 queue, restarting it if it dies
#   phase 2  the ten-step hyperparameter probes
#   phase 3  the v2 stage, with whichever probe held reward flat or rising
#
# Every phase is resumable, so a kill costs the current step and nothing else.
set -a; . ./.env; set +a
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

v1_done() {
  $PY - <<'PYEOF'
import json, sys
try:
    d = json.load(open("artifacts/queue.json"))
except Exception:
    sys.exit(1)
pending = [e for e in d["entries"] if e["status"] in ("pending", "running")]
sys.exit(0 if (d.get("stopped") or not pending) else 1)
PYEOF
}

echo "[$(stamp)] supervisor starting"
for attempt in $(seq 1 40); do
  if v1_done; then echo "[$(stamp)] v1 queue drained"; break; fi
  echo "[$(stamp)] phase 1: v1 queue, attempt $attempt"
  rm -f artifacts/queue.lock
  $PY -u -m alibi.cli queue run >> queue.log 2>&1
  sleep 5
done

if ! v1_done; then
  echo "[$(stamp)] v1 did not drain after 40 attempts, stopping rather than starting v2 on a partial control"
  exit 1
fi

if [ ! -f artifacts/diagnostics/probes/result.json ]; then
  echo "[$(stamp)] phase 2: hyperparameter probes"
  $PY -u -m alibi.diagnostics.probes || { echo "[$(stamp)] probes failed"; exit 1; }
fi

CHOSEN=$($PY -c "import json;print(json.load(open('artifacts/diagnostics/probes/result.json'))['chosen'] or '')" 2>/dev/null)
if [ -z "$CHOSEN" ]; then
  echo "[$(stamp)] NO PROBE HELD REWARD FLAT OR RISING. v2 stage NOT launched. See report/PROBES.md"
  exit 0
fi
echo "[$(stamp)] phase 3: v2 stage with probe $CHOSEN"
exec $PY -u -m alibi.train.runner_v2
