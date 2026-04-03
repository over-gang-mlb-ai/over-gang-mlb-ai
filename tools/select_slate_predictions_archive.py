#!/usr/bin/env python3
"""
Pick the canonical predictions CSV for the active slate when multiple
archive/predictions_YYYYMMDD_HHMM.csv files exist.

Preference order (among files matching the slate date from active_slate_date_mt):
1. More non-empty Closing_Line cells (downstream capture quality)
2. If tie: more non-empty CLV cells
3. If tie: newest mtime
4. If no file has any Closing_Line: newest mtime for that slate only

Prints a single absolute path, or nothing if no candidate exists.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.public_betting_scraper import active_slate_date_mt  # noqa: E402


def _nonempty_cell_count(series: pd.Series | None) -> int:
    if series is None:
        return 0
    n = 0
    for v in series:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            n += 1
    return n


def main() -> int:
    slate_date, _ = active_slate_date_mt()
    slate = slate_date.strftime("%Y%m%d")
    archive = ROOT / "archive"
    if not archive.is_dir():
        return 0

    paths = sorted(glob.glob(str(archive / f"predictions_{slate}_*.csv")))
    if not paths:
        return 0

    scored: list[tuple[str, int, int, float]] = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        closing = df["Closing_Line"] if "Closing_Line" in df.columns else None
        clv = df["CLV"] if "CLV" in df.columns else None
        nc = _nonempty_cell_count(closing)
        nv = _nonempty_cell_count(clv)
        mtime = os.path.getmtime(p)
        scored.append((p, nc, nv, mtime))

    if not scored:
        return 0

    max_nc = max(s[1] for s in scored)
    if max_nc == 0:
        best = max(scored, key=lambda s: s[3])
        print(best[0])
        return 0

    scored.sort(key=lambda s: (-s[1], -s[2], -s[3]))
    print(scored[0][0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
