#!/usr/bin/env python3
"""
Targeted starter-fallback refresh lane.

Reads today's normal full-slate predictions archive, finds only starter fallback
games, rechecks MLB feed for those Game_ID/side combinations, and if any starter
names are now available, reruns the model only for those Game_IDs.

Targeted model output is prefixed with starter_refresh_ so it cannot be mistaken
for the normal full-slate predictions file.
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "archive"
LOGS = ROOT / "logs"
DATA = ROOT / "data"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.public_betting_scraper import active_slate_date_mt  # noqa: E402

LOG_PREFIX = "[STARTER_REFRESH]"


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "y", "on"}


def norm_name(value: object) -> str:
    s = str(value or "").strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s'.-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_dotenv_if_needed() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw = line.split("=", 1)
        key = key.strip()
        val = raw.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def read_model_telegram_constants() -> Tuple[str, str]:
    model_path = ROOT / "model" / "overgang_model.py"
    text = model_path.read_text(encoding="utf-8", errors="ignore") if model_path.exists() else ""
    token = ""
    chat = ""

    m = re.search(r"^\s*TELEGRAM_BOT_TOKEN\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
    if m:
        token = m.group(1).strip()

    m = re.search(r"^\s*TELEGRAM_CHAT_ID\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
    if m:
        chat = m.group(1).strip()

    return token, chat


def telegram_creds() -> Tuple[str, str]:
    load_dotenv_if_needed()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if token and chat:
        return token, chat

    model_token, model_chat = read_model_telegram_constants()
    return token or model_token, chat or model_chat


def telegram_send_message(text: str) -> bool:
    token, chat = telegram_creds()
    if not token or not chat:
        print(f"{LOG_PREFIX} Telegram credentials missing; message not sent", file=sys.stderr)
        return False

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat, "text": text},
            timeout=20,
        )
        resp.raise_for_status()
        print(f"{LOG_PREFIX} Telegram message sent")
        return True
    except Exception as exc:
        print(f"{LOG_PREFIX} Telegram message failed: {exc}", file=sys.stderr)
        return False


def telegram_send_file(path: Path, caption: str) -> bool:
    token, chat = telegram_creds()
    if not token or not chat:
        print(f"{LOG_PREFIX} Telegram credentials missing; file not sent", file=sys.stderr)
        return False

    try:
        with path.open("rb") as fh:
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendDocument",
                data={"chat_id": chat, "caption": caption},
                files={"document": fh},
                timeout=30,
            )
        resp.raise_for_status()
        print(f"{LOG_PREFIX} Telegram file sent: {path}")
        return True
    except Exception as exc:
        print(f"{LOG_PREFIX} Telegram file failed: {exc}", file=sys.stderr)
        return False


def latest_normal_predictions(slate: str) -> Optional[Path]:
    if not ARCHIVE.is_dir():
        return None
    files = [
        p for p in ARCHIVE.glob(f"predictions_{slate}_*.csv")
        if not p.name.startswith("starter_refresh_")
    ]
    if not files:
        return None
    return sorted(files, key=lambda p: p.stat().st_mtime)[-1]


def latest_refresh_predictions(slate: str, since_ts: float) -> Optional[Path]:
    files = list(ARCHIVE.glob(f"starter_refresh_predictions_{slate}_*.csv"))
    files = [p for p in files if p.stat().st_mtime >= since_ts]
    if not files:
        return None
    return sorted(files, key=lambda p: p.stat().st_mtime)[-1]


def load_name_sets() -> Dict[str, set]:
    out: Dict[str, set] = {"main": set(), "lowip": set(), "k": set()}
    paths = {
        "main": DATA / "pitcher_stats.csv",
        "lowip": DATA / "pitcher_stats_lowip.csv",
        "k": DATA / "pitcher_k_stats.csv",
    }

    for key, path in paths.items():
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if "Name" in df.columns:
                out[key] = set(df["Name"].astype(str).map(norm_name))
        except Exception as exc:
            print(f"{LOG_PREFIX} failed reading {path}: {exc}", file=sys.stderr)

    return out


def add_candidate(cands: List[Tuple[str, str]], label: str, value: object) -> None:
    v = str(value or "").strip()
    if not v:
        return
    if v.upper() == "TBD":
        return
    if "league avg" in v.lower():
        return
    cands.append((label, v))


def feed_candidates(game_id: str, side: str) -> List[Tuple[str, str]]:
    url = f"https://statsapi.mlb.com/api/v1.1/game/{int(float(game_id))}/feed/live"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    cands: List[Tuple[str, str]] = []

    try:
        node = (data.get("gameData", {}).get("probablePitchers", {}) or {}).get(side) or {}
        if isinstance(node, dict):
            add_candidate(cands, "gameData.probablePitchers.fullName", node.get("fullName"))
            add_candidate(cands, "gameData.probablePitchers.name", node.get("name"))
            add_candidate(cands, "gameData.probablePitchers.boxscoreName", node.get("boxscoreName"))
    except Exception:
        pass

    try:
        team_node = (data.get("gameData", {}).get("teams", {}) or {}).get(side) or {}
        node = team_node.get("probablePitcher") or {}
        if isinstance(node, dict):
            add_candidate(cands, "gameData.teams.probablePitcher.fullName", node.get("fullName"))
            add_candidate(cands, "gameData.teams.probablePitcher.name", node.get("name"))
            add_candidate(cands, "gameData.teams.probablePitcher.boxscoreName", node.get("boxscoreName"))
        elif isinstance(node, str):
            add_candidate(cands, "gameData.teams.probablePitcher.str", node)
    except Exception:
        pass

    try:
        box_team = (data.get("liveData", {}).get("boxscore", {}).get("teams", {}) or {}).get(side) or {}
        pitchers = box_team.get("pitchers") or []
        players = box_team.get("players") or {}
        if pitchers:
            pid = str(pitchers[0])
            pnode = players.get(f"ID{pid}") or players.get(pid) or {}
            person = pnode.get("person") or {}
            add_candidate(cands, "boxscore.pitchers[0].person.fullName", person.get("fullName"))
            add_candidate(cands, "boxscore.pitchers[0].fullName", pnode.get("fullName"))
    except Exception:
        pass

    seen = set()
    final: List[Tuple[str, str]] = []
    for label, name in cands:
        key = norm_name(name)
        if key and key not in seen:
            seen.add(key)
            final.append((label, name))

    return final


def fallback_sides_from_predictions(df: pd.DataFrame) -> List[dict]:
    rows: List[dict] = []

    for _, row in df.iterrows():
        dq = str(row.get("Data_Quality_Flag") or "").lower()
        away_fb = truthy(row.get("Away_Starter_Fallback_Used"))
        home_fb = truthy(row.get("Home_Starter_Fallback_Used"))

        if "fallback_pitcher" in dq and not (away_fb or home_fb):
            away_fb = True
            home_fb = True

        game_id = str(row.get("Game_ID") or "").strip()
        if not game_id or game_id.lower() == "nan":
            continue

        if away_fb:
            rows.append(
                {
                    "game_id": game_id,
                    "game": str(row.get("Game") or ""),
                    "side": "away",
                    "old": "League Avg Away",
                }
            )
        if home_fb:
            rows.append(
                {
                    "game_id": game_id,
                    "game": str(row.get("Game") or ""),
                    "side": "home",
                    "old": "League Avg Home",
                }
            )

    return rows


def marker_path(slate: str) -> Path:
    return LOGS / f"starter_fallback_refresh_{slate}.json"


def load_marker(slate: str) -> dict:
    p = marker_path(slate)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_marker(slate: str, source: Path, updates: List[dict], output: Optional[Path]) -> None:
    p = marker_path(slate)
    p.parent.mkdir(parents=True, exist_ok=True)

    existing = load_marker(slate)
    history = existing.get("updates", [])
    history.extend(updates)

    body = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_mt": datetime.now(ZoneInfo("America/Denver")).isoformat(),
        "slate": slate,
        "source_predictions": str(source),
        "output_predictions": str(output) if output else "",
        "updates": history,
    }
    p.write_text(json.dumps(body, indent=2), encoding="utf-8")


def update_key(update: dict) -> str:
    return f"{update.get('game_id')}|{update.get('side')}|{norm_name(update.get('new'))}"


def main() -> int:
    slate_date, _meta = active_slate_date_mt()
    slate = slate_date.strftime("%Y%m%d")
    slate_iso = slate_date.strftime("%Y-%m-%d")

    src = latest_normal_predictions(slate)
    if src is None:
        print(f"{LOG_PREFIX} no normal predictions archive found for slate {slate}; no-op")
        return 0

    df = pd.read_csv(src)
    sides = fallback_sides_from_predictions(df)
    print(f"{LOG_PREFIX} source={src} fallback_side_count={len(sides)}")

    if not sides:
        print(f"{LOG_PREFIX} no starter fallback sides found; no-op")
        return 0

    local_sets = load_name_sets()
    marker = load_marker(slate)
    already = {
        update_key(u)
        for u in marker.get("updates", [])
        if isinstance(u, dict)
    }

    updates: List[dict] = []

    for item in sides:
        try:
            cands = feed_candidates(item["game_id"], item["side"])
        except Exception as exc:
            print(f"{LOG_PREFIX} feed check failed for {item}: {exc}", file=sys.stderr)
            continue

        if not cands:
            print(f"{LOG_PREFIX} still missing starter: {item['game']} {item['side']}")
            continue

        label, name = cands[0]
        n = norm_name(name)
        update = {
            "game_id": item["game_id"],
            "game": item["game"],
            "side": item["side"],
            "old": item["old"],
            "new": name,
            "feed_label": label,
            "in_main_stats": n in local_sets["main"],
            "in_lowip": n in local_sets["lowip"],
            "in_k_stats": n in local_sets["k"],
        }

        # A starter name alone is not enough to justify a targeted rerun.
        # The model's starter resolution path depends on data/pitcher_stats.csv.
        # If the name is not present there, rerunning only recreates a League Avg row.
        if not update["in_main_stats"]:
            print(
                f"{LOG_PREFIX} starter named but missing main pitcher_stats row; "
                f"no targeted rerun: {update['game']} {update['side']} {name} "
                f"(lowip={'yes' if update['in_lowip'] else 'no'}, "
                f"k_stats={'yes' if update['in_k_stats'] else 'no'})"
            )
            continue

        if update_key(update) in already:
            print(f"{LOG_PREFIX} already alerted: {update['game']} {update['side']} {name}")
            continue

        updates.append(update)

    if not updates:
        print(f"{LOG_PREFIX} no new starter updates detected; no-op")
        return 0

    game_ids = sorted({u["game_id"] for u in updates})

    lines = [
        "🔁 Starter Fallback Update Detected",
        "",
        "Updated starter data found for targeted refresh:",
        "",
    ]

    for u in updates:
        side_label = "Away" if u["side"] == "away" else "Home"
        local = (
            f"main_stats={'yes' if u['in_main_stats'] else 'no'}, "
            f"lowip={'yes' if u['in_lowip'] else 'no'}, "
            f"k_stats={'yes' if u['in_k_stats'] else 'no'}"
        )
        lines.extend(
            [
                f"{u['game']}",
                f"{side_label} starter: {u['old']} → {u['new']}",
                f"Local data: {local}",
                "",
            ]
        )

    lines.append("Running Over Gang targeted refresh for these Game_IDs only.")
    telegram_send_message("\n".join(lines))

    env = os.environ.copy()
    env["OVERGANG_TARGET_DATE"] = slate_iso
    env["OVERGANG_TARGET_GAME_IDS"] = ",".join(game_ids)
    env["OVERGANG_ARCHIVE_PREFIX"] = "starter_refresh"
    env["OVERGANG_SUPPRESS_MODEL_TELEGRAM"] = "1"

    cmd = [sys.executable, str(ROOT / "model" / "overgang_model.py")]
    start_ts = time.time()

    print(f"{LOG_PREFIX} running targeted model: {' '.join(cmd)} game_ids={game_ids}")
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    print(f"{LOG_PREFIX} targeted model exit status={r.returncode}")

    if r.returncode != 0:
        return r.returncode

    out = latest_refresh_predictions(slate, start_ts)
    write_marker(slate, src, updates, out)

    if out is not None and out.exists():
        telegram_send_file(out, caption=f"🔁 Starter Refresh Predictions — {slate_iso}")
    else:
        telegram_send_message(
            f"⚠️ Starter refresh completed for {slate_iso}, but no starter_refresh_predictions file was found."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
