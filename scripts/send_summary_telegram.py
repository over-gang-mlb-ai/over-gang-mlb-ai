#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Send a plain-text Telegram summary: today's W/L/P from a graded archive CSV
and season W/L/P lines read from a season summary text file.
Does not modify input files. No CSV output.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root for optional Telegram config import
ROOT = Path(__file__).resolve().parent.parent


def _is_graded_result(val: Any) -> bool:
    if val is None:
        return False
    s = str(val).strip().upper()
    return s in ("WIN", "LOSS", "PUSH")


def _result_to_wlp(s: str) -> Optional[Tuple[int, int, int]]:
    u = s.strip().upper()
    if u == "WIN":
        return (1, 0, 0)
    if u == "LOSS":
        return (0, 1, 0)
    if u == "PUSH":
        return (0, 0, 1)
    return None


def _read_archive_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            return []
        return [dict(row) for row in r]


def _tally_today(rows: List[Dict[str, str]]) -> Tuple[int, int, int, int, int, int, int, int, int]:
    """Returns ou_w, ou_l, ou_p, ml_w, ml_l, ml_p, c_w, c_l, c_p."""
    ou_w = ou_l = ou_p = 0
    ml_w = ml_l = ml_p = 0
    c_w = c_l = c_p = 0
    for row in rows:
        ou = (row.get("OU_Result") or "").strip().upper()
        if _is_graded_result(ou):
            t = _result_to_wlp(ou)
            if t:
                ou_w += t[0]
                ou_l += t[1]
                ou_p += t[2]
                c_w += t[0]
                c_l += t[1]
                c_p += t[2]
        ml = (row.get("ML_Result") or "").strip().upper()
        if _is_graded_result(ml):
            t = _result_to_wlp(ml)
            if t:
                ml_w += t[0]
                ml_l += t[1]
                ml_p += t[2]
                c_w += t[0]
                c_l += t[1]
                c_p += t[2]
    return ou_w, ou_l, ou_p, ml_w, ml_l, ml_p, c_w, c_l, c_p


def _fmt_wlp(w: int, l: int, p: int) -> str:
    return f"{w}-{l}-{p}"


def _parse_season_simple_lines(text: str) -> Optional[Tuple[str, str, str]]:
    """Parse lines like 'O/U: 5-3-1', 'ML: 4-2-0', 'Combined: 9-5-1'."""
    ou = ml = cb = None
    pat = re.compile(
        r"^(?i)(O/U|ML|Combined)\s*:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*$"
    )
    for line in text.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        label = m.group(1).upper()
        w, l_, p = int(m.group(2)), int(m.group(3)), int(m.group(4))
        s = _fmt_wlp(w, l_, p)
        if label == "O/U":
            ou = s
        elif label == "ML":
            ml = s
        elif label == "COMBINED":
            cb = s
    if ou and ml and cb:
        return ou, ml, cb
    return None


def _parse_season_tracker_style(text: str) -> Optional[Tuple[str, str, str]]:
    """Parse season_tracker_summary.py stdout: --- O/U --- / --- ML --- / --- Combined ... --- blocks."""

    def extract_block(header: str) -> Optional[Tuple[int, int, int]]:
        m = re.search(
            rf"---\s*{header}\s*---\s*\n(.*?)(?=\n---|\Z)",
            text,
            re.S | re.I,
        )
        if not m:
            return None
        block = m.group(1)
        w = l_ = p = 0
        for line in block.splitlines():
            mm = re.match(r"\s*WIN:\s*(\d+)", line, re.I)
            if mm:
                w = int(mm.group(1))
                continue
            mm = re.match(r"\s*LOSS:\s*(\d+)", line, re.I)
            if mm:
                l_ = int(mm.group(1))
                continue
            mm = re.match(r"\s*PUSH:\s*(\d+)", line, re.I)
            if mm:
                p = int(mm.group(1))
        return (w, l_, p)

    ou = extract_block("O/U")
    ml = extract_block("ML")
    cb = extract_block(r"Combined \(O/U \+ ML\)") or extract_block("Combined")

    if ou is None or ml is None or cb is None:
        return None
    return (
        _fmt_wlp(*ou),
        _fmt_wlp(*ml),
        _fmt_wlp(*cb),
    )


def _load_season_triples(path: Path) -> Tuple[str, str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    t = _parse_season_simple_lines(text)
    if t:
        return t
    t = _parse_season_tracker_style(text)
    if t:
        return t
    raise ValueError(
        f"Could not parse season file {path}: expected lines like "
        "'O/U: W-L-P', 'ML: W-L-P', 'Combined: W-L-P' or season_tracker_summary-style blocks."
    )


def _telegram_send_plain(text: str) -> bool:
    sys.path.insert(0, str(ROOT))
    try:
        from model.overgang_model import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    except Exception as e:
        print(f"Failed to load Telegram config: {e}", file=sys.stderr)
        return False

    token = (TELEGRAM_BOT_TOKEN or "").strip()
    chat = (TELEGRAM_CHAT_ID or "").strip()
    if not token or not chat:
        print("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing.", file=sys.stderr)
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = json.dumps(
        {"chat_id": chat, "text": text},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                if 200 <= resp.status < 300:
                    return True
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            print(f"Telegram HTTP {e.code}: {err_body}", file=sys.stderr)
        except Exception as e:
            print(f"Telegram request failed: {e}", file=sys.stderr)
    return False


def _build_message(
    today_ou: str,
    today_ml: str,
    today_cb: str,
    season_ou: str,
    season_ml: str,
    season_cb: str,
) -> str:
    return (
        "📊 OVER GANG DAILY RESULTS\n"
        "\n"
        "Today:\n"
        f"O/U: {today_ou}\n"
        f"ML: {today_ml}\n"
        f"Combined: {today_cb}\n"
        "\n"
        "Season:\n"
        f"O/U: {season_ou}\n"
        f"ML: {season_ml}\n"
        f"Combined: {season_cb}\n"
        "\n"
        "📎 Full Report Attached"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send nightly Telegram summary (today + season W/L/P)."
    )
    parser.add_argument(
        "archive_csv",
        type=Path,
        help="Graded archive predictions CSV path",
    )
    parser.add_argument(
        "season_summary_txt",
        type=Path,
        help="Season summary text file (e.g. tracking/season_summary_2026.txt)",
    )
    args = parser.parse_args()
    arch = args.archive_csv.resolve()
    season_path = args.season_summary_txt.resolve()

    if not arch.is_file():
        print(f"Archive not found: {arch}", file=sys.stderr)
        return 1
    if not season_path.is_file():
        print(f"Season summary not found: {season_path}", file=sys.stderr)
        return 1

    try:
        rows = _read_archive_rows(arch)
    except Exception as e:
        print(f"Failed to read archive: {e}", file=sys.stderr)
        return 1

    ou_w, ou_l, ou_p, ml_w, ml_l, ml_p, c_w, c_l, c_p = _tally_today(rows)
    today_ou = _fmt_wlp(ou_w, ou_l, ou_p)
    today_ml = _fmt_wlp(ml_w, ml_l, ml_p)
    today_cb = _fmt_wlp(c_w, c_l, c_p)

    try:
        season_ou, season_ml, season_cb = _load_season_triples(season_path)
    except Exception as e:
        print(f"{e}", file=sys.stderr)
        return 1

    msg = _build_message(today_ou, today_ml, today_cb, season_ou, season_ml, season_cb)
    if _telegram_send_plain(msg):
        print("Telegram summary sent.")
        return 0
    print("Telegram summary failed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
