#!/usr/bin/env bash
# Detached queue runner. Survives the launching shell's exit.
set -a; . ./.env; set +a
# Reduces allocator fragmentation on an 8 GB card under a long run.
export PYTORCH_ALLOC_CONF=expandable_segments:True
exec .venv/bin/python -u -m alibi.cli queue run
