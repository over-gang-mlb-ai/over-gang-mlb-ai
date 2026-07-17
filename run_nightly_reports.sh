#!/usr/bin/env bash
set -e

cd /home/ubuntu/over-gang-mlb-ai

# Cron/minimal shells often lack project env; export .env so child Python sees ODDS_API_KEY (fill_closing_lines / fetch_mlb_odds).
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# Nightly grading/reporting always targets the completed prior slate in America/Denver.
TARGET_SLATE=$(./venv/bin/python3 -c "from datetime import datetime, timedelta; from zoneinfo import ZoneInfo; now_mt = datetime.now(ZoneInfo('America/Denver')); print((now_mt.date() - timedelta(days=1)).strftime('%Y-%m-%d'))")
LATEST=$(OVERGANG_TARGET_DATE="$TARGET_SLATE" ./venv/bin/python3 tools/select_slate_predictions_archive.py)
[ -n "$LATEST" ] || { echo "No predictions archive found for prior slate $TARGET_SLATE"; exit 0; }

./venv/bin/python3 tools/fill_closing_lines.py "$LATEST"
./venv/bin/python3 tools/update_clv.py "$LATEST"
./venv/bin/python3 scripts/auto_grade_predictions_csv.py "$LATEST"
./venv/bin/python3 scripts/append_season_tracker.py "$LATEST" --season 2026
./venv/bin/python3 scripts/season_tracker_summary.py --season 2026 > tracking/season_summary_2026.txt

# Grade the prior slate's F5/K boards without allowing an
# unavailable auxiliary board to block the O/U + ML report.
GRADED_F5=""
if ./venv/bin/python3 scripts/grade_k_f5_boards.py --date "$TARGET_SLATE"; then
  TARGET_COMPACT=${TARGET_SLATE//-/}
  GRADED_F5=$(ls -1t archive/graded_f5_board_${TARGET_COMPACT}_*.csv 2>/dev/null | head -n 1 || true)
else
  echo "F5/K board grading unavailable for $TARGET_SLATE; continuing O/U + ML report."
fi

CLIENT_REPORT="archive/client_$(basename "$LATEST")"

./venv/bin/python3 scripts/export_client_report.py "$LATEST" -o "$CLIENT_REPORT"

SUMMARY_ARGS=(
  "$LATEST"
  tracking/season_summary_2026.txt
)

if [ -n "$GRADED_F5" ]; then
  SUMMARY_ARGS+=(--graded-f5 "$GRADED_F5")
fi

./venv/bin/python3 scripts/send_summary_telegram.py "${SUMMARY_ARGS[@]}"
./venv/bin/python3 -c "from model.overgang_model import send_telegram_file; send_telegram_file('$CLIENT_REPORT', caption='📊 Client Report')"
