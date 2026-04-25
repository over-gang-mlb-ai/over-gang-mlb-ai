#!/usr/bin/env python3
"""
Repeatable same-day Closing_Line capture for the active slate: earliest first pitch opens
the window, canonical archive selection chooses the target file, and a transient lock
prevents overlapping runs without enforcing one-shot-per-slate behavior.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

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


def _closing_line_filled(val) -> bool:
    """True when Closing_Line should not be refilled (aligned with fill_closing_lines.py)."""
    if pd.isna(val):
        return False
    s = str(val).strip()
    return bool(s) and s.lower() != "nan"


def _parse_row_game_start_utc(val) -> datetime | None:
    """Parse archive row Datetime (ISO Z) for gating on still-missing Closing_Line rows."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return None


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
        return Path(base) / f".closing_capture_running_{slate_yyyymmdd}"
    return ROOT / "data" / f".closing_capture_running_{slate_yyyymmdd}"


def _canonical_archive_for_slate(slate_date: date) -> Path | None:
    selector = ROOT / "tools" / "select_slate_predictions_archive.py"
    if not selector.is_file():
        return None
    env = os.environ.copy()
    env["OVERGANG_TARGET_DATE"] = slate_date.strftime("%Y-%m-%d")
    try:
        res = subprocess.run(
            [sys.executable, str(selector)],
            check=True,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"{LOG_PREFIX} selector failed: {e}", file=sys.stderr)
        return None
    out = (res.stdout or "").strip()
    if not out:
        return None
    return Path(out)


def main() -> None:
    buffer_min = _buffer_minutes()
    slate_date, _meta = active_slate_date_mt()
    slate_yyyymmdd = slate_date.strftime("%Y%m%d")
    lock_file = _lock_path(slate_yyyymmdd)

    if lock_file.is_file():
        print(
            f"{LOG_PREFIX} capture already running for slate {slate_yyyymmdd} (lock: {lock_file})"
        )
        sys.exit(0)

    csv_path = _canonical_archive_for_slate(slate_date)
    if csv_path is None:
        print(
            f"{LOG_PREFIX} no archive file for active slate "
            f"(expected archive/predictions_{slate_yyyymmdd}_*.csv)"
        )
        sys.exit(0)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"{LOG_PREFIX} could not read {csv_path}: {e}", file=sys.stderr)
        sys.exit(1)

    if "Closing_Line" not in df.columns:
        print(f"{LOG_PREFIX} missing Closing_Line column in {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows_needing_closing = sum(
        1 for i in range(len(df)) if not _closing_line_filled(df.at[i, "Closing_Line"])
    )
    if rows_needing_closing == 0:
        print(
            f"{LOG_PREFIX} skip: all Closing_Line values already filled in "
            f"{csv_path.name} ({len(df)} row(s)); no fill run."
        )
        sys.exit(0)

    date_str = slate_date.strftime("%Y-%m-%d")
    row_starts: list[datetime] = []
    if "Datetime" in df.columns:
        for i in range(len(df)):
            if _closing_line_filled(df.at[i, "Closing_Line"]):
                continue
            dt = _parse_row_game_start_utc(df.at[i, "Datetime"])
            if dt is not None:
                row_starts.append(dt)

    if row_starts:
        earliest = min(row_starts)
        gate_source = "earliest_unfilled_row_datetime"
    else:
        raw_games = schedule(start_date=date_str, end_date=date_str) or []
        slate_games = [g for g in raw_games if _game_mt_date(g, slate_date)]
        if not slate_games:
            print(
                f"{LOG_PREFIX} no games on slate {date_str} "
                f"(statsapi schedule empty or no games matching active slate MT date); "
                f"cannot infer start for {rows_needing_closing} unfilled row(s) without Datetime"
            )
            sys.exit(0)
        earliest = _parse_earliest_pitch_utc(slate_games)
        if earliest is None:
            print(
                f"{LOG_PREFIX} could not parse game_datetime for earliest first pitch "
                f"(fallback for {rows_needing_closing} unfilled row(s) without parseable Datetime)"
            )
            sys.exit(0)
        gate_source = "earliest_slate_first_pitch_fallback"

    now_utc = datetime.now(timezone.utc)
    window_start = earliest - timedelta(minutes=buffer_min)

    if now_utc < window_start:
        print(
            f"{LOG_PREFIX} too early: now_utc={now_utc.isoformat()} "
            f"earliest_first_pitch_utc={earliest.isoformat()} "
            f"window=[{window_start.isoformat()}, {earliest.isoformat()}) buffer_min={buffer_min}"
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
            f"{LOG_PREFIX} capture already running (lock appeared: {lock_file})"
        )
        sys.exit(0)

    fill_script = ROOT / "tools" / "fill_closing_lines.py"
    cmd = [sys.executable, str(fill_script), str(csv_path)]
    print(
        f"{LOG_PREFIX} running fill for {csv_path.name} "
        f"({gate_source}, earliest_utc={earliest.isoformat()}, "
        f"unfilled_rows={rows_needing_closing}, buffer_min={buffer_min})"
    )
    try:
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    except subprocess.CalledProcessError as e:
        print(f"{LOG_PREFIX} fill_closing_lines failed: {e}", file=sys.stderr)
        sys.exit(e.returncode if e.returncode else 1)
    finally:
        try:
            lock_file.unlink(missing_ok=True)
        except OSError:
            pass

    print(f"{LOG_PREFIX} pass complete for {csv_path.name}")


if __name__ == "__main__":
    main()
