#!/usr/bin/env bash
# Same-day Closing_Line capture: run via cron while The Odds API still lists today's slate
# (e.g. late afternoon/evening on game day). Updates the latest archive in place.
# Nightly run_nightly_reports.sh continues to grade/export; CLV runs there after this fill.
set -e

cd /home/ubuntu/over-gang-mlb-ai

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

LATEST=$(/bin/ls -1t archive/predictions_*.csv 2>/dev/null | /usr/bin/head -1)
[ -n "$LATEST" ] || { echo "No predictions archive found"; exit 0; }

./venv/bin/python3 tools/fill_closing_lines.py "$LATEST"
