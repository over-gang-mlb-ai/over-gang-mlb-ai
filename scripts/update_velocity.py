#!/usr/bin/env python3
"""
Regenerate data/velocity_data.csv from Statcast-compatible Baseball Savant exports.

Data sources (read-only HTTP; no prediction code touched):
  • Season_Velo: Savant Pitch Arsenals leaderboard (avg_speed), four-seam (FF) preferred,
    sinker (SI) used when FF is absent — same Statcast pitch-speed definitions as the site.
  • Recent_Velo: Statcast Search pitch-level CSV (release_speed) for FF|SI over a rolling
    calendar window, aggregated per pitcher (MLBAM id) as the mean of qualifying pitches.

Name keys are written to match VelocityTracker.load_velocity_csv():
  unidecode(DataManager.normalize_name(raw_name).lower().strip())

Run manually (no cron wiring in this repo):
  python -u scripts/update_velocity.py
  python -u scripts/update_velocity.py --year 2025 --recent-days 30 --min-pitches 1
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from unidecode import unidecode

# Project root (parent of scripts/)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.data_manager import DataManager  # noqa: E402

_OUT_CSV = _ROOT / "data" / "velocity_data.csv"

# Savant Pitch Arsenals — same family of endpoint as pybaseball.statcast_pitcher_pitch_arsenal.
_PITCH_ARSENAL_URLS = (
    "https://baseballsavant.mlb.com/leaderboard/pitch-arsenals?year={year}&min={min_p}&type=avg_speed&hand=&csv=true",
)

# Statcast Search — pitch-level details; FF + SI (primary hard stuff) for rolling average velocity.
_STATCAST_SEARCH = (
    "https://baseballsavant.mlb.com/statcast_search/csv?all=true"
    "&hfPT=FF%7CSI&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones="
    "&hfGT=R%7CPO%7CS%7C=&hfSea=&hfSit=&player_type=pitcher&hfOuts=&opponent="
    "&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={gt}&game_date_lt={lt}"
    "&team=&position=&hfRO=&home_road=&hfFlag=&metric_1=&hfInn=&min_pitches=0"
    "&min_results=0&sort_col=pitches&sort_order=desc&min_abs=0&type=details&"
)

_REQUEST_HEADERS = {
    "User-Agent": "OverGangVelocityRefresh/1.0 (+https://github.com/local)",
    "Accept": "text/csv,*/*",
}

_FASTBALL_PRIORITY = ("FF", "SI", "FT")  # FF first; SI/FT as fallbacks for season row


def _looks_like_html(text: str) -> bool:
    s = text.lstrip()[:500].lower()
    return s.startswith("<!doctype") or s.startswith("<html")


def _fetch_csv(url: str, timeout: int = 120) -> pd.DataFrame:
    r = requests.get(url, headers=_REQUEST_HEADERS, timeout=timeout)
    r.raise_for_status()
    text = r.text
    if _looks_like_html(text):
        raise RuntimeError(f"Savant returned HTML instead of CSV (url={url[:120]}…)")
    df = pd.read_csv(io.StringIO(text))
    df.columns = df.columns.str.strip()
    return df


def _first_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _season_table(year: int, min_pitches: int) -> pd.DataFrame:
    """Return DataFrame with columns: mlb_id (int), raw_name (str), season_velo (float)."""
    last_err: Exception | None = None
    for tmpl in _PITCH_ARSENAL_URLS:
        url = tmpl.format(year=year, min_p=min_pitches)
        try:
            df = _fetch_csv(url)
        except Exception as e:
            last_err = e
            continue
        if df.empty:
            last_err = RuntimeError("Pitch arsenals CSV empty")
            continue

        # Wide format: ff_avg_speed / si_avg_speed style (Savant custom naming varies by year).
        ff_w = _first_col(df, ("ff_avg_speed", "ff_avg_velo", "ff_velo", "ff_avg_speed_mph"))
        si_w = _first_col(df, ("si_avg_speed", "si_avg_velo", "si_velo"))
        id_w = _first_col(df, ("player_id", "pitcher_id", "id"))
        name_w = _first_col(df, ("player_name", "name", "pitcher"))

        if ff_w and id_w:
            rows = []
            for _, row in df.iterrows():
                pid = row.get(id_w)
                if pd.isna(pid):
                    continue
                try:
                    mlb_id = int(pid)
                except (TypeError, ValueError):
                    continue
                raw = row.get(name_w, "")
                if not isinstance(raw, str) or not raw.strip():
                    raw = str(mlb_id)
                v_ff = row.get(ff_w)
                v_si = row.get(si_w) if si_w else float("nan")
                velo = None
                if pd.notna(v_ff):
                    velo = float(v_ff)
                elif si_w and pd.notna(v_si):
                    velo = float(v_si)
                if velo is None:
                    continue
                rows.append({"mlb_id": mlb_id, "raw_name": raw.strip(), "season_velo": velo})
            if rows:
                return pd.DataFrame(rows)

        # Long format: one row per pitcher per pitch type.
        pt = _first_col(df, ("pitch_type", "pitch", "type"))
        av = _first_col(df, ("avg_speed", "velocity", "avg_velo", "mph"))
        # Prefer player_id / pitcher_id; "pitcher" is sometimes MLBAM id on Savant exports (non-numeric rows drop out).
        pid_c = _first_col(df, ("player_id", "pitcher_id", "mlbam", "pitcher"))
        pname = _first_col(df, ("player_name", "name", "pitcher_name"))
        if not all([pt, av, pid_c]):
            last_err = RuntimeError(
                f"Unrecognized pitch-arsenals columns: {list(df.columns)[:25]}…"
            )
            continue

        work = df[[pid_c, pt, av]].copy()
        work.columns = ["mlb_id", "pitch_type", "avg_speed"]
        work = work.dropna(subset=["mlb_id", "pitch_type", "avg_speed"])
        work["pitch_type"] = work["pitch_type"].astype(str).str.upper().str.strip()
        work["mlb_id"] = pd.to_numeric(work["mlb_id"], errors="coerce")
        work = work.dropna(subset=["mlb_id"])
        work["mlb_id"] = work["mlb_id"].astype(int)

        name_map = {}
        if pname:
            for _, r in df[[pid_c, pname]].drop_duplicates(subset=[pid_c]).iterrows():
                try:
                    mid = int(r[pid_c])
                    name_map[mid] = str(r[pname]).strip() if pd.notna(r[pname]) else ""
                except (TypeError, ValueError):
                    continue

        out_rows = []
        for mlb_id, g in work.groupby("mlb_id"):
            raw = name_map.get(mlb_id, str(mlb_id))
            sub = g.groupby("pitch_type")["avg_speed"].mean()
            velo = None
            for ft in _FASTBALL_PRIORITY:
                if ft in sub.index and pd.notna(sub.loc[ft]):
                    velo = float(sub.loc[ft])
                    break
            if velo is None:
                continue
            out_rows.append({"mlb_id": mlb_id, "raw_name": raw, "season_velo": velo})
        if out_rows:
            return pd.DataFrame(out_rows)

        last_err = RuntimeError("Could not parse pitch arsenals schema")

    raise RuntimeError(f"Season pitch-arsenals fetch failed: {last_err}")


def _recent_velocity_by_id(recent_days: int, season_year: int) -> pd.Series:
    """
    Mean Statcast release_speed (FF|SI) per pitcher MLBAM id over the last `recent_days` days.
    Skips offseason-ish gaps by simply requesting each day (empty days are ignored).
    """
    today = date.today()
    start = today - timedelta(days=max(1, recent_days))
    # Do not query before season start for season_year (rough guard).
    season_start = date(season_year, 3, 15)
    if start < season_start:
        start = season_start

    sums: dict[int, float] = {}
    counts: dict[int, int] = {}

    d = start
    while d <= today:
        gt = d.strftime("%Y-%m-%d")
        lt = (d + timedelta(days=1)).strftime("%Y-%m-%d")
        url = _STATCAST_SEARCH.format(gt=gt, lt=lt)
        try:
            df = _fetch_csv(url, timeout=180)
        except Exception:
            d += timedelta(days=1)
            continue
        if df.empty:
            d += timedelta(days=1)
            continue

        pid_c = _first_col(df, ("pitcher", "player_id"))
        rs_c = _first_col(df, ("release_speed", "release_speed_mph"))
        if not pid_c or not rs_c:
            d += timedelta(days=1)
            continue

        chunk = df[[pid_c, rs_c]].copy()
        chunk = chunk.dropna(subset=[pid_c, rs_c])
        chunk[pid_c] = pd.to_numeric(chunk[pid_c], errors="coerce")
        chunk = chunk.dropna(subset=[pid_c])
        chunk[pid_c] = chunk[pid_c].astype(int)
        chunk[rs_c] = pd.to_numeric(chunk[rs_c], errors="coerce")
        chunk = chunk.dropna(subset=[rs_c])

        for pid, g in chunk.groupby(pid_c):
            s = float(g[rs_c].sum())
            c = int(len(g))
            sums[pid] = sums.get(pid, 0.0) + s
            counts[pid] = counts.get(pid, 0) + c

        d += timedelta(days=1)

    out = {}
    for pid in sums:
        c = counts.get(pid, 0)
        if c > 0:
            out[pid] = sums[pid] / c
    return pd.Series(out, name="recent_velo")


def _format_name_key(raw: str) -> str:
    """Match VelocityTracker: unidecode(normalize_name).lower().strip() after normalize."""
    can = DataManager.normalize_name(raw)
    if not can:
        can = raw
    return unidecode(can.lower().strip())


def _build_output(season_df: pd.DataFrame, recent: pd.Series) -> pd.DataFrame:
    rows = []
    for _, r in season_df.iterrows():
        mid = int(r["mlb_id"])
        raw = str(r["raw_name"])
        season_v = float(r["season_velo"])
        recent_v = float(recent[mid]) if mid in recent.index else season_v
        key = _format_name_key(raw)
        if not key:
            continue
        rows.append(
            {
                "Name": key,
                "Season_Velo": round(season_v, 1),
                "Recent_Velo": round(recent_v, 1),
            }
        )
    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["Name"], keep="last")
    out = out.sort_values("Name").reset_index(drop=True)
    return out


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=str(path.parent))
    os.close(fd)
    try:
        tmp_path = Path(tmp)
        df.to_csv(tmp_path, index=False, lineterminator="\n")
        os.replace(tmp_path, path)
    finally:
        if Path(tmp).exists():
            try:
                os.remove(tmp)
            except OSError:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild data/velocity_data.csv from Savant Statcast data.")
    ap.add_argument(
        "--year",
        type=int,
        default=None,
        help="Season year for pitch-arsenals (default: current calendar year).",
    )
    ap.add_argument(
        "--recent-days",
        type=int,
        default=30,
        help="Rolling window (days) for Recent_Velo via Statcast pitch search (default: 30).",
    )
    ap.add_argument(
        "--min-pitches",
        type=int,
        default=1,
        help="Savant pitch-arsenals min pitches filter (default: 1 for broad coverage).",
    )
    args = ap.parse_args()

    year = args.year if args.year is not None else date.today().year

    print(f"📊 Fetching season pitch-arsenals ({year}, min_pitches={args.min_pitches})…")
    season_df = _season_table(year, args.min_pitches)
    print(f"   Season rows (with FF/SI speed): {len(season_df)}")

    print(f"📊 Aggregating Statcast FF|SI release_speed (last {args.recent_days} days)…")
    recent = _recent_velocity_by_id(args.recent_days, year)
    print(f"   Pitchers with recent pitch data: {len(recent)}")

    out = _build_output(season_df, recent)
    print(f"✅ Writing {len(out)} rows → {_OUT_CSV}")

    _atomic_write_csv(_OUT_CSV, out)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
