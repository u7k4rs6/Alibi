#!/usr/bin/env bash
# Detached queue runner. Survives the launching shell's exit.
set -a; . ./.env; set +a
exec .venv/bin/python -u -m alibi.cli queue run
