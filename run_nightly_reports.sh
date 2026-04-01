#!/usr/bin/env bash
set -e

cd /home/ubuntu/over-gang-mlb-ai

LATEST=$(/bin/ls -1t archive/predictions_*.csv 2>/dev/null | /usr/bin/head -1)
[ -n "$LATEST" ] || { echo "No predictions archive found"; exit 0; }

./venv/bin/python3 tools/fill_closing_lines.py "$LATEST"
./venv/bin/python3 tools/update_clv.py "$LATEST"
./venv/bin/python3 scripts/auto_grade_predictions_csv.py "$LATEST"
./venv/bin/python3 scripts/append_season_tracker.py "$LATEST" --season 2026
./venv/bin/python3 scripts/season_tracker_summary.py --season 2026 > tracking/season_summary_2026.txt

LATEST_GRADED=$(./venv/bin/python3 - <<'PY'
import csv, glob, os
for path in sorted(glob.glob("archive/predictions_*.csv"), key=os.path.getmtime, reverse=True):
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        for row in csv.DictReader(f):
            ou = (row.get("OU_Result") or "").strip().upper()
            ml = (row.get("ML_Result") or "").strip().upper()
            if ou in ("WIN", "LOSS", "PUSH") or ml in ("WIN", "LOSS", "PUSH"):
                print(path)
                raise SystemExit(0)
PY
)

[ -n "$LATEST_GRADED" ] || { echo "No graded archive found"; exit 0; }

CLIENT_REPORT="archive/client_$(basename "$LATEST_GRADED")"

./venv/bin/python3 scripts/export_client_report.py "$LATEST_GRADED" -o "$CLIENT_REPORT"
./venv/bin/python3 scripts/send_summary_telegram.py "$LATEST_GRADED" tracking/season_summary_2026.txt
./venv/bin/python3 -c "from model.overgang_model import send_telegram_file; send_telegram_file('$CLIENT_REPORT', caption='📊 Client Report')"
