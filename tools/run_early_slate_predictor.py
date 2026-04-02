#!/usr/bin/env python3
"""
Conditional early run of run_predictions when earliest first pitch (StatsAPI) is before
a configurable America/Denver threshold (default 9:00 AM), and no archive exists yet for that slate.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from statsapi import schedule  # noqa: E402

from core.public_betting_scraper import active_slate_date_mt  # noqa: E402

LOG_PREFIX = "[EARLY_SLATE]"
MT = ZoneInfo("America/Denver")


def _game_mt_date(g, slate: date) -> bool:
    try:
        raw = g.get("game_datetime") or ""
        dt_utc = datetime.strptime(
            str(raw).strip(), "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=ZoneInfo("UTC"))
    except (KeyError, TypeError, ValueError):
        return False
    return dt_utc.astimezone(MT).date() == slate


def _parse_earliest_pitch_utc(games: list) -> datetime | None:
    best: datetime | None = None
    for g in games:
        raw = g.get("game_datetime")
        if not raw:
            continue
        try:
            dt = datetime.strptime(
                str(raw).strip(), "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if best is None or dt < best:
            best = dt
    return best


def _archive_exists_for_slate(slate_yyyymmdd: str) -> bool:
    archive = ROOT / "archive"
    if not archive.is_dir():
        return False
    return any(archive.glob(f"predictions_{slate_yyyymmdd}_*.csv"))


def _threshold_time_mt() -> time:
    h_raw = os.environ.get("OG_EARLY_SLATE_THRESHOLD_HOUR_MT", "9").strip()
    m_raw = os.environ.get("OG_EARLY_SLATE_THRESHOLD_MINUTE_MT", "0").strip()
    try:
        h = int(h_raw)
        m = int(m_raw)
    except ValueError:
        print(
            f"{LOG_PREFIX} invalid OG_EARLY_SLATE_THRESHOLD_HOUR_MT / MINUTE_MT",
            file=sys.stderr,
        )
        sys.exit(1)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        print(
            f"{LOG_PREFIX} threshold hour/minute out of range (got {h}:{m})",
            file=sys.stderr,
        )
        sys.exit(1)
    return time(h, m)


def _is_earliest_before_threshold(earliest_utc: datetime, threshold_t: time) -> bool:
    earliest_local = earliest_utc.astimezone(MT)
    line = datetime.combine(earliest_local.date(), threshold_t, tzinfo=MT)
    return earliest_local < line


def main() -> None:
    threshold_t = _threshold_time_mt()
    slate_date, _meta = active_slate_date_mt()
    slate_yyyymmdd = slate_date.strftime("%Y%m%d")

    if _archive_exists_for_slate(slate_yyyymmdd):
        print(
            f"{LOG_PREFIX} archive already exists for slate {slate_yyyymmdd} "
            f"(archive/predictions_{slate_yyyymmdd}_*.csv); skipping"
        )
        sys.exit(0)

    date_str = slate_date.strftime("%Y-%m-%d")
    raw_games = schedule(start_date=date_str, end_date=date_str) or []
    slate_games = [g for g in raw_games if _game_mt_date(g, slate_date)]

    if not slate_games:
        print(
            f"{LOG_PREFIX} no games on slate {date_str} "
            f"(statsapi empty or no games matching active slate MT date)"
        )
        sys.exit(0)

    earliest = _parse_earliest_pitch_utc(slate_games)
    if earliest is None:
        print(f"{LOG_PREFIX} could not parse game_datetime for earliest first pitch")
        sys.exit(0)

    earliest_local = earliest.astimezone(MT)
    line = datetime.combine(earliest_local.date(), threshold_t, tzinfo=MT)
    print(
        f"{LOG_PREFIX} earliest_first_pitch_mt={earliest_local.isoformat()} "
        f"threshold_mt={line.isoformat()}"
    )

    if not _is_earliest_before_threshold(earliest, threshold_t):
        print(f"{LOG_PREFIX} not an early slate (earliest is at or after threshold); skipping")
        sys.exit(0)

    guarded = ROOT / "tools" / "run_predictor_if_no_archive.py"
    if not guarded.is_file():
        print(f"{LOG_PREFIX} guarded runner not found: {guarded}", file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, str(guarded)]
    print(f"{LOG_PREFIX} early slate + no archive; invoking guarded runner: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(ROOT))
    print(f"{LOG_PREFIX} guarded runner exit status={r.returncode}")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
