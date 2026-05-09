#!/usr/bin/env bash
# Refresh Statcast-backed velocity_data.csv before the morning predictor run.
# Does not change model code; invokes scripts/update_velocity.py only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

YEAR="$(TZ=America/Denver date +%Y)"
echo "[velocity] START $(date -u '+%Y-%m-%dT%H:%M:%SZ') (America/Denver calendar year=${YEAR})"

./venv/bin/python3 scripts/update_velocity.py \
  --year "${YEAR}" \
  --recent-days 30 \
  --min-pitches 1

echo "[velocity] END   $(date -u '+%Y-%m-%dT%H:%M:%SZ') OK"
