#!/usr/bin/env bash
# Slate-idempotent predictor: skips if archive/predictions_${SLATE}_*.csv already exists.
# Point 9:00 AM and 10:00 AM (fallback) cron jobs at this script instead of model/overgang_model.py.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

./venv/bin/python3 tools/run_predictor_if_no_archive.py
