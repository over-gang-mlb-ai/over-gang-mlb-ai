#!/usr/bin/env python3
"""
Slate-idempotent predictor entry: run model/overgang_model.py only if no archive exists
for the active slate (archive/predictions_${SLATE}_*.csv). Use for scheduled 9:00 / 10:00
runs and as the final step after early-slate detection.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.public_betting_scraper import active_slate_date_mt  # noqa: E402

LOG_PREFIX = "[PREDICTOR_GUARD]"


def _archive_exists_for_slate(slate_yyyymmdd: str) -> bool:
    archive = ROOT / "archive"
    if not archive.is_dir():
        return False
    return any(archive.glob(f"predictions_{slate_yyyymmdd}_*.csv"))


def main() -> None:
    slate_date, _meta = active_slate_date_mt()
    slate_yyyymmdd = slate_date.strftime("%Y%m%d")

    if _archive_exists_for_slate(slate_yyyymmdd):
        print(
            f"{LOG_PREFIX} archive already exists for slate {slate_yyyymmdd} "
            f"(archive/predictions_{slate_yyyymmdd}_*.csv); skipping predictor run"
        )
        sys.exit(0)

    pred = ROOT / "model" / "overgang_model.py"
    if not pred.is_file():
        print(f"{LOG_PREFIX} predictor not found: {pred}", file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, str(pred)]
    print(f"{LOG_PREFIX} no archive for slate {slate_yyyymmdd}; running: {' '.join(cmd)} (cwd={ROOT})")
    r = subprocess.run(cmd, cwd=str(ROOT))
    print(f"{LOG_PREFIX} predictor exit status={r.returncode}")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
