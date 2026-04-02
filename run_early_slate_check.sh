#!/usr/bin/env bash
# Run before 9:00 AM MT cron: if StatsAPI shows an early slate and no predictions archive yet,
# invoke tools/run_predictor_if_no_archive.py (same guard as 9:00/10:00). No-op otherwise.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

./venv/bin/python3 tools/run_early_slate_predictor.py
