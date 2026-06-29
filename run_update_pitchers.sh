#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/over-gang-mlb-ai
source venv/bin/activate

# Make sure logs dir exists first
mkdir -p logs

# Safe PYTHONPATH export (won’t error if unset)
export PYTHONPATH="${PYTHONPATH:-}:/home/ubuntu/over-gang-mlb-ai"

python -u scripts/update_pitchers.py >> logs/update_pitchers_$(date +%F).log 2>&1

# Refresh K-prop pitcher stats used by archive/pitcher_k_board_*.csv.
# Best-effort: do not block the broader pitcher/bullpen refresh chain if this
# auxiliary K-prop stats feed fails.
if python -u scripts/build_pitcher_k_stats.py >> logs/update_pitchers_$(date +%F).log 2>&1; then
  echo "$(date -Is) ✅ Refreshed data/pitcher_k_stats.csv" >> logs/update_pitchers_$(date +%F).log
else
  echo "$(date -Is) ⚠️ pitcher_k_stats.csv refresh failed; keeping previous file" >> logs/update_pitchers_$(date +%F).log
fi
