#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/over-gang-mlb-ai
source venv/bin/activate
python -u scripts/update_bullpen.py
