#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/over-gang-mlb-ai
source venv/bin/activate

# Make sure logs dir exists first
mkdir -p logs

# Safe PYTHONPATH export (won’t error if unset)
export PYTHONPATH="${PYTHONPATH:-}:/home/ubuntu/over-gang-mlb-ai"

python -u scripts/update_pitchers.py >> logs/update_pitchers_$(date +%F).log 2>&1
