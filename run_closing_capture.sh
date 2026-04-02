#!/usr/bin/env bash
# Same-day Closing_Line capture: recurring cron calls this; gated script picks active slate,
# earliest first pitch (statsapi), matching archive, pregame buffer window, and once-per-slate lock.
# Nightly run_nightly_reports.sh continues to grade/export; CLV runs there after this fill.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

./venv/bin/python3 tools/run_closing_capture_gated.py
