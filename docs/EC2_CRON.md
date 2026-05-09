# EC2 cron notes (Over Gang MLB AI)

## Velocity refresh (before morning predictor)

Run **before** `run_predictions_guarded.sh` so `data/velocity_data.csv` is current for starter velo trims.

Example (8:47 AM America/Denver ≈ 14:47 UTC during MDT):

```cron
47 14 * * * cd /home/ubuntu/over-gang-mlb-ai && /home/ubuntu/over-gang-mlb-ai/run_update_velocity.sh >> logs/morning_velocity.log 2>&1
```

Adjust the minute/hour if your predictor cron changes; keep velocity **earlier** than the predictor job.

Wrapper: `run_update_velocity.sh` (repo root) — loads `.env` if present, runs  
`scripts/update_velocity.py --year $(TZ=America/Denver date +%Y) --recent-days 30 --min-pitches 1`.
