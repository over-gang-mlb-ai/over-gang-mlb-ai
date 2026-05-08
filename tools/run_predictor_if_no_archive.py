#!/usr/bin/env python3
"""
Slate-idempotent predictor entry: run model/overgang_model.py unless a per-slate
success marker exists (logs/predictor_guard_${SLATE}.done). Scheduled 9:00 / 10:00 MT
runs write the marker on success; prior archive CSVs alone do not skip (manual
pre-cron exports no longer block official runs).
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.public_betting_scraper import active_slate_date_mt  # noqa: E402

LOG_PREFIX = "[PREDICTOR_GUARD]"


def _marker_path_for_slate(slate_yyyymmdd: str) -> Path:
    return ROOT / "logs" / f"predictor_guard_{slate_yyyymmdd}.done"


def _marker_exists_for_slate(slate_yyyymmdd: str) -> bool:
    return _marker_path_for_slate(slate_yyyymmdd).is_file()


def _write_success_marker(slate_yyyymmdd: str, cmd: list[str], return_code: int) -> None:
    path = _marker_path_for_slate(slate_yyyymmdd)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts_utc = datetime.now(timezone.utc)
    ts_mt = datetime.now(ZoneInfo("America/Denver"))
    body = (
        f"timestamp_utc={ts_utc.isoformat()}\n"
        f"timestamp_mt={ts_mt.isoformat()}\n"
        f"slate={slate_yyyymmdd}\n"
        f"command={' '.join(cmd)}\n"
        f"return_code={return_code}\n"
    )
    path.write_text(body, encoding="utf-8")


def _is_official_guard_window_mt() -> bool:
    now = datetime.now(ZoneInfo("America/Denver"))
    return now.hour in (9, 10)


def _archive_exists_for_slate(slate_yyyymmdd: str) -> bool:
    archive = ROOT / "archive"
    if not archive.is_dir():
        return False
    return any(archive.glob(f"predictions_{slate_yyyymmdd}_*.csv"))


def main() -> None:
    slate_date, _meta = active_slate_date_mt()
    slate_yyyymmdd = slate_date.strftime("%Y%m%d")

    if _marker_exists_for_slate(slate_yyyymmdd):
        print(
            f"{LOG_PREFIX} skip: success marker exists for slate {slate_yyyymmdd} "
            f"({_marker_path_for_slate(slate_yyyymmdd)})"
        )
        sys.exit(0)

    if _archive_exists_for_slate(slate_yyyymmdd):
        print(
            f"{LOG_PREFIX} informational: archive file(s) exist for slate {slate_yyyymmdd} "
            f"(archive/predictions_{slate_yyyymmdd}_*.csv); not used as skip condition"
        )
    else:
        print(
            f"{LOG_PREFIX} informational: no archive/predictions_{slate_yyyymmdd}_*.csv "
            f"(archive may be created by this run)"
        )

    pred = ROOT / "model" / "overgang_model.py"
    if not pred.is_file():
        print(f"{LOG_PREFIX} predictor not found: {pred}", file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, str(pred)]
    print(f"{LOG_PREFIX} running predictor for slate {slate_yyyymmdd}: {' '.join(cmd)} (cwd={ROOT})")
    r = subprocess.run(cmd, cwd=str(ROOT))
    print(f"{LOG_PREFIX} predictor exit status={r.returncode}")

    if r.returncode != 0:
        sys.exit(r.returncode)

    if _is_official_guard_window_mt():
        _write_success_marker(slate_yyyymmdd, cmd, r.returncode)
        print(f"{LOG_PREFIX} wrote success marker: {_marker_path_for_slate(slate_yyyymmdd)}")
    else:
        print(
            f"{LOG_PREFIX} predictor succeeded outside official 9–10 AM America/Denver hour; "
            f"not writing success marker"
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
