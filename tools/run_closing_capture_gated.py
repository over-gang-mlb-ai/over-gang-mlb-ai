#!/usr/bin/env python3
"""
Gated same-day Closing_Line capture: active slate date (same rules as run_predictions),
earliest first pitch from statsapi.schedule, archive glob for that slate, time window + lock.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# Project root (parent of tools/)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from statsapi import schedule  # noqa: E402

from core.public_betting_scraper import active_slate_date_mt  # noqa: E402

LOG_PREFIX = "[CLOSING_CAPTURE]"


def _game_mt_date(g, slate: date) -> bool:
    try:
        raw = g.get("game_datetime") or ""
        dt_utc = datetime.strptime(
            str(raw).strip(), "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=ZoneInfo("UTC"))
    except (KeyError, TypeError, ValueError):
        return False
    return dt_utc.astimezone(ZoneInfo("America/Denver")).date() == slate


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


def _buffer_minutes() -> int:
    raw = os.environ.get("OG_CLOSING_CAPTURE_BUFFER_MINUTES", "90").strip()
    try:
        m = int(raw)
    except ValueError:
        print(
            f"{LOG_PREFIX} OG_CLOSING_CAPTURE_BUFFER_MINUTES={raw!r} invalid; need integer minutes",
            file=sys.stderr,
        )
        sys.exit(1)
    if m <= 0:
        print(
            f"{LOG_PREFIX} OG_CLOSING_CAPTURE_BUFFER_MINUTES must be positive, got {m}",
            file=sys.stderr,
        )
        sys.exit(1)
    return m


def _lock_path(slate_yyyymmdd: str) -> Path:
    base = os.environ.get("OG_CLOSING_CAPTURE_LOCK_DIR", "").strip()
    if base:
        return Path(base) / f".closing_capture_done_{slate_yyyymmdd}"
    return ROOT / "data" / f".closing_capture_done_{slate_yyyymmdd}"


def _newest_archive_for_slate(slate_yyyymmdd: str) -> Path | None:
    pat = f"predictions_{slate_yyyymmdd}_*.csv"
    archive = ROOT / "archive"
    if not archive.is_dir():
        return None
    matches = list(archive.glob(pat))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def main() -> None:
    buffer_min = _buffer_minutes()
    slate_date, _meta = active_slate_date_mt()
    slate_yyyymmdd = slate_date.strftime("%Y%m%d")
    lock_file = _lock_path(slate_yyyymmdd)

    if lock_file.is_file():
        print(
            f"{LOG_PREFIX} already captured for slate {slate_yyyymmdd} (lock: {lock_file})"
        )
        sys.exit(0)

    csv_path = _newest_archive_for_slate(slate_yyyymmdd)
    if csv_path is None:
        print(
            f"{LOG_PREFIX} no archive file for active slate "
            f"(expected archive/predictions_{slate_yyyymmdd}_*.csv)"
        )
        sys.exit(0)

    date_str = slate_date.strftime("%Y-%m-%d")
    raw_games = schedule(start_date=date_str, end_date=date_str) or []
    # Same slate membership as run_predictions (MT calendar date); include all statuses
    # so earliest first pitch is the true first scheduled pitch, not only pregame rows.
    slate_games = [g for g in raw_games if _game_mt_date(g, slate_date)]

    if not slate_games:
        print(
            f"{LOG_PREFIX} no games on slate {date_str} "
            f"(statsapi schedule empty or no games matching active slate MT date)"
        )
        sys.exit(0)

    earliest = _parse_earliest_pitch_utc(slate_games)
    if earliest is None:
        print(f"{LOG_PREFIX} could not parse game_datetime for earliest first pitch")
        sys.exit(0)

    now_utc = datetime.now(timezone.utc)
    window_start = earliest - timedelta(minutes=buffer_min)

    if now_utc < window_start:
        print(
            f"{LOG_PREFIX} too early: now_utc={now_utc.isoformat()} "
            f"earliest_first_pitch_utc={earliest.isoformat()} "
            f"window=[{window_start.isoformat()}, {earliest.isoformat()}) buffer_min={buffer_min}"
        )
        sys.exit(0)

    if now_utc >= earliest:
        print(
            f"{LOG_PREFIX} past earliest first pitch: now_utc={now_utc.isoformat()} "
            f"earliest_first_pitch_utc={earliest.isoformat()} — not running"
        )
        sys.exit(0)

    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            str(lock_file),
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o644,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as lf:
            lf.write(f"slate={slate_yyyymmdd}\nstarted_utc={now_utc.isoformat()}\n")
    except FileExistsError:
        print(
            f"{LOG_PREFIX} already captured or in progress (lock appeared: {lock_file})"
        )
        sys.exit(0)

    fill_script = ROOT / "tools" / "fill_closing_lines.py"
    cmd = [sys.executable, str(fill_script), str(csv_path)]
    print(
        f"{LOG_PREFIX} running fill for {csv_path.name} "
        f"(earliest_first_pitch_utc={earliest.isoformat()}, buffer_min={buffer_min})"
    )
    try:
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    except subprocess.CalledProcessError as e:
        try:
            lock_file.unlink(missing_ok=True)
        except OSError:
            pass
        print(f"{LOG_PREFIX} fill_closing_lines failed: {e}", file=sys.stderr)
        sys.exit(e.returncode if e.returncode else 1)

    print(f"{LOG_PREFIX} success; lock={lock_file}")


if __name__ == "__main__":
    main()
