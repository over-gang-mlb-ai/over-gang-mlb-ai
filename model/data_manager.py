import os
import re
import io
import json
import time
import requests
import pandas as pd
from datetime import datetime
from unidecode import unidecode
from rapidfuzz import fuzz, process
from itertools import islice
from typing import Tuple
from urllib.parse import urlencode

# ================================
# Paths & Config
# ================================
DATA_DIR = "data"
STATS_FILE = os.path.join(DATA_DIR, "pitcher_stats.csv")

# Name matching threshold for fuzzy match
NAME_MATCH_THRESHOLD = 85

# Dynamic IP thresholds (same semantics your predictor expects)
MIN_PITCHER_IP_EARLY = 10
MIN_PITCHER_IP_MID = 20
MIN_PITCHER_IP_LATE = 15

# Refuse to overwrite pitcher_stats.csv if new pull has fewer than this many rows
MIN_PITCHER_SAVE_COUNT = 100

# Early-season blend: w26 = min(max(ip_current, 0), EARLY_SEASON_BLEND_IP_CAP) / EARLY_SEASON_BLEND_IP_CAP
# so ~50 IP of current season fully weights toward live xERA/WHIP vs prior-year carryover.
EARLY_SEASON_BLEND_IP_CAP = 50.0


class DataManager:
    # ----------------------------
    # Helpers
    # ----------------------------
    @staticmethod
    def normalize_name(name: str) -> str:
        if not isinstance(name, str):
            return ""
        name = unidecode(name)
        name = re.sub(r"[^a-zA-Z ]", "", name.lower().strip())

        # strip suffixes
        for suffix in ["jr", "sr", "ii", "iii", "iv"]:
            name = re.sub(fr"\s*{suffix}$", "", name)

        nick = {
            "nick": "nicholas", "nate": "nathan", "mike": "michael",
            "matt": "matthew", "jim": "james", "chris": "christopher",
            "joe": "joseph", "andy": "andrew", "dan": "daniel",
            "tom": "thomas", "tim": "timothy", "bob": "robert",
            "rob": "robert", "dave": "david", "will": "william",
            "alex": "alexander", "josh": "joshua", "sam": "samuel"
        }
        parts = [p for p in name.split() if len(p) > 0]
        return " ".join([nick.get(p, p) for p in parts])

    @staticmethod
    def _read_csv_with_column_normalization(path):
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            if "Name" not in df.columns:
                print(f"❌ 'Name' column missing in file: {path}")
                return pd.DataFrame()
            return df
        except Exception as e:
            print(f"🚨 Failed to load CSV at {path}: {e}")
            return pd.DataFrame()

    # ----------------------------
    # MLB + Savant fetchers
    # ----------------------------
    @staticmethod
    def _savant_body_looks_like_html(text: str) -> bool:
        """Reject error pages / HTML served instead of Savant CSV export."""
        if not text or not str(text).strip():
            return True
        head = str(text).lstrip()[:1200].lower()
        if head.startswith("<!doctype") or head.startswith("<html"):
            return True
        if head.startswith("<?xml") and "<html" in head[:500]:
            return True
        if head.startswith("<") and ("<table" in head[:600] or "<body" in head[:600] or "<head" in head[:600]):
            return True
        return False

    @staticmethod
    def _savant_xera_single_year(year: int) -> pd.DataFrame:
        """
        Return DataFrame with columns: player_id, xERA for one season.
        Multi-stage (canonical real xERA from Savant CSV first):
          1) CSV export — validated body, broad column aliases
          2) Derive xERA from est_wOBA when CSV has wOBA but not xERA (not official Savant xERA)
        HTML read_html path removed: brittle on current Savant JS pages; does not improve real xERA.
        """
        os.makedirs("debug", exist_ok=True)
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        }

        def _pick_col(cols, *candidates):
            norm = {c.lower().strip().replace(" ", "").replace("_", ""): c for c in cols}
            normalized_candidates = [c.lower().strip().replace(" ", "").replace("_", "") for c in candidates]
            for key, original in norm.items():
                if key in normalized_candidates:
                    return original
            for c in cols:
                cl = str(c).lower().replace(" ", "").replace("_", "")
                if any(cl.startswith(k) for k in normalized_candidates):
                    return c
            return None

        _PID = (
            "player_id", "playerid", "player id", "mlbam", "mlb_am_id", "mlbam_id",
            "pitcher_id", "mlbamid",
        )
        _XERA = (
            "xERA", "xera", "expectedera", "exera", "expected_era", "x_era",
            "expectedrunavg", "expected_run_avg",
        )

        csv_urls = [
            f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=pitcher&year={year}&season={year}&csv=true",
            f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=pitcher&year={year}&csv=true",
        ]

        last_err = None
        for i, url in enumerate(csv_urls, 1):
            try:
                r = requests.get(url, headers=headers, timeout=25)
                r.raise_for_status()
                csv_text = r.text.lstrip("\ufeff")
                path = f"debug/savant_xera_{year}_csv_attempt{i}.csv"
                with open(path, "w", encoding="utf-8") as f:
                    f.write(csv_text)

                if DataManager._savant_body_looks_like_html(csv_text):
                    last_err = f"CSV attempt {i}: response body looks like HTML, not CSV ({url})"
                    continue

                try:
                    df = pd.read_csv(io.StringIO(csv_text))
                except Exception as e:
                    last_err = f"CSV attempt {i} parse failed: {e}"
                    continue

                if df is None or df.empty or len(df.columns) < 2:
                    last_err = f"CSV attempt {i}: empty or single-column parse ({url})"
                    continue

                pid_col = _pick_col(df.columns, *_PID)
                xera_col = _pick_col(df.columns, *_XERA)

                if pid_col and xera_col:
                    out = df[[pid_col, xera_col]].rename(columns={pid_col: "player_id", xera_col: "xERA"})
                    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
                    out["xERA"] = pd.to_numeric(out["xERA"], errors="coerce")
                    out = out.dropna(subset=["player_id", "xERA"]).astype({"player_id": int})
                    if not out.empty:
                        print(f"✅ xERA source: Savant CSV (rows={len(out)}, year={year}, attempt={i})")
                        return out
                    last_err = f"CSV attempt {i}: no rows with valid player_id+xERA after dropna ({url})"
                else:
                    hdr = ",".join(map(str, df.columns))
                    with open(f"debug/savant_headers_{year}.csv", "w", encoding="utf-8") as f:
                        f.write(hdr + "\n")
                    last_err = (
                        f"Missing columns on CSV attempt {i}. id='{pid_col}', xERA='{xera_col}' from {url} (preview saved)."
                    )
            except Exception as e:
                last_err = f"CSV attempt {i} failed: {e}"

        # ---------- Derive xERA from est_wOBA (secondary; not official Savant xERA)
        # Approach: linear transform anchored to league averages
        #   xERA ≈ L_ERA + SLOPE * (est_wOBA - L_wOBA)
        # Defaults are adjustable via env vars.
        try:
            # pick a previously saved CSV/HTML table to read est_wOBA from
            source = None
            for p in [
                f"debug/savant_xera_{year}_csv_attempt1.csv",
                f"debug/savant_xera_{year}_csv_attempt2.csv",
                f"debug/savant_xera_{year}_html_table.csv",
            ]:
                if os.path.exists(p):
                    source = p
                    break
            if source is None:
                # as a last attempt, fetch CSV once more for est_wOBA
                url = f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=pitcher&year={year}&csv=true"
                r = requests.get(url, headers=headers, timeout=25)
                r.raise_for_status()
                fb_text = r.text.lstrip("\ufeff")
                if DataManager._savant_body_looks_like_html(fb_text):
                    raise ValueError("Fallback CSV fetch returned HTML, not Savant CSV.")
                source = f"debug/savant_xera_{year}_csv_attempt_fallback.csv"
                with open(source, "w", encoding="utf-8") as f:
                    f.write(fb_text)

            if source.endswith("_html_table.csv"):
                df = pd.read_csv(source, encoding="utf-8-sig")
            else:
                df = pd.read_csv(source, encoding="utf-8-sig")

            cols = [str(c) for c in df.columns]
            pid_col = _pick_col(cols, *_PID)
            estwoba_col = _pick_col(
                cols, "est_wOBA", "estwoba", "expectedwoba", "xwoba", "x_woba", "xwoba_con", "est_woba"
            )
            if not (pid_col and estwoba_col):
                raise ValueError(
                    f"Derivation failed: could not find est_wOBA. id='{pid_col}', est_wOBA='{estwoba_col}'"
                )

            work = df[[pid_col, estwoba_col]].rename(columns={pid_col: "player_id", estwoba_col: "est_wOBA"})
            work["player_id"] = pd.to_numeric(work["player_id"], errors="coerce").astype("Int64")
            work["est_wOBA"] = pd.to_numeric(work["est_wOBA"], errors="coerce")
            work = work.dropna(subset=["player_id", "est_wOBA"]).astype({"player_id": int})

            # league anchors (tunable via env)
            L_ERA = float(os.getenv("OGP_LEAGUE_ERA", "4.25"))
            L_WOBA = float(os.getenv("OGP_LEAGUE_WOBA", "0.310"))
            SLOPE = float(os.getenv("OGP_WOBA_TO_ERA_SLOPE", "20.0"))
            # (rule of thumb: ~0.010 wOBA ≈ 0.20 ERA ⇒ slope ≈ 20)

            work["xERA"] = L_ERA + SLOPE * (work["est_wOBA"] - L_WOBA)
            work["xERA"] = work["xERA"].clip(lower=2.0, upper=8.5)

            out = work[["player_id", "xERA"]].dropna(subset=["xERA"])
            out.to_csv(f"debug/savant_xera_{year}_derived_from_estwoba.csv", index=False)

            if out.empty:
                raise ValueError("Derived xERA frame is empty.")

            print(
                f"ℹ️ Derived xERA from est_wOBA for {len(out)} pitchers "
                f"(anchors: L_ERA={L_ERA}, L_wOBA={L_WOBA}, slope={SLOPE})."
            )
            if last_err:
                print(f"  Previous errors: {last_err}")

            return out

        except Exception as e:
            msg = (
                f"Savant xERA fetch failed for year {year}: {last_err or e}. "
                f"See debug/savant_xera_{year}_*.csv/.html for raw responses."
            )
            raise ValueError(msg)

    @staticmethod
    def _savant_xera_by_id(year: int) -> pd.DataFrame:
        """
        Return DataFrame with columns: player_id, xERA (real Savant leaderboard values when CSV succeeds).
        Tries current season first, then prior season (same MLBAM ids) so early-season / thin
        leaderboards still receive last year's Savant xERA before merge falls back to ERA.
        """
        errs = []
        for y in (year, year - 1) if year > 2015 else (year,):
            if y < 2015:
                break
            try:
                return DataManager._savant_xera_single_year(y)
            except ValueError as e:
                errs.append(f"{y}: {e}")
                continue
        raise ValueError(
            "Savant xERA unavailable: " + (" | ".join(errs) if errs else "unknown")
        )

    @staticmethod
    def _mlb_pitching_stats_by_id(
        year: int,
        *,
        prior_year_fallback: bool = True,
        apply_safe_mode: bool = True,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Pull ERA/WHIP/IP for all active MLB pitchers via MLB StatsAPI (free).
        Returns (DataFrame with columns mlb_id, Name, ERA, WHIP, IP, used_prior_year_fallback).
        used_prior_year_fallback is True when the requested season had 0 regular-season splits and prior year was used.

        prior_year_fallback: if False, do not retry with year-1 when the requested year has no splits (returns empty).
        apply_safe_mode: if False, do not reject small current-year universes (< MIN_PITCHER_SAVE_COUNT) when not using
        the prior-year fallback path — used so update_pitcher_stats can blend a thin live year with full prior season.
        """
        headers = {"User-Agent": "Mozilla/5.0"}

        # 1) active MLB team IDs
        teams = requests.get(
            "https://statsapi.mlb.com/api/v1/teams?sportId=1&activeStatus=Y",
            headers=headers, timeout=15
        ).json().get("teams", [])
        team_ids = [t["id"] for t in teams]

        # 2) collect active pitcher IDs
        pitcher_ids = set()
        for tid in team_ids:
            try:
                roster = requests.get(
                    f"https://statsapi.mlb.com/api/v1/teams/{tid}/roster?rosterType=active",
                    headers=headers, timeout=15
                ).json().get("roster", [])
                for r in roster:
                    pos = r.get("position", {}) or {}
                    # pitcher's position check
                    if (pos.get("code") == "1") or (pos.get("abbreviation") == "P") or (pos.get("name") == "Pitcher"):
                        pid = r.get("person", {}).get("id")
                        if pid:
                            pitcher_ids.add(int(pid))
            except Exception:
                continue

        if not pitcher_ids:
            raise ValueError("No pitcher IDs found from MLB rosters.")

        # 3) fetch season pitching stats from stats endpoint (returns ERA/WHIP/IP per player)
        stats_base = "https://statsapi.mlb.com/api/v1/stats"
        params = {
            "stats": "season",
            "group": "pitching",
            "season": year,
            "gameType": "R",
            "limit": 1000,
        }
        print(f"[Pitcher update] Source: MLB StatsAPI stats endpoint (year={year}, roster_pitchers={len(pitcher_ids)})")
        print(f"[Pitcher update] Request URL: {stats_base}?{urlencode(params)}")

        try:
            resp = requests.get(stats_base, params=params, headers=headers, timeout=25)
            resp.raise_for_status()
            body = resp.json()
        except Exception as e:
            raise ValueError(f"MLB StatsAPI stats request failed: {e}") from e

        stats_list = body.get("stats") if isinstance(body, dict) else None
        if not isinstance(stats_list, list) or not stats_list:
            raise ValueError(
                "MLB StatsAPI returned no stats array (stats endpoint). Check endpoint or response shape."
            )
        splits = stats_list[0].get("splits") if isinstance(stats_list[0], dict) else []
        if not isinstance(splits, list):
            splits = []
        total_raw_splits = len(splits)
        used_fallback = False
        rows = []
        for sp in splits:
            if not isinstance(sp, dict):
                continue
            player = sp.get("player") or {}
            st = sp.get("stat") or {}
            pid = player.get("id")
            name_full = (player.get("fullName") or "").strip()
            if pid is None:
                continue
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                continue
            era_raw = st.get("era") if st.get("era") is not None else st.get("ERA")
            whip_raw = st.get("whip") if st.get("whip") is not None else st.get("WHIP")
            ip_raw = st.get("inningsPitched") if st.get("inningsPitched") is not None else st.get("ip") or st.get("IP")
            era = None
            if era_raw not in (None, ""):
                try:
                    era = float(era_raw)
                except (TypeError, ValueError):
                    pass
            whip = None
            if whip_raw not in (None, ""):
                try:
                    whip = float(whip_raw)
                except (TypeError, ValueError):
                    pass
            ip = None
            if ip_raw is not None and ip_raw != "":
                try:
                    if isinstance(ip_raw, (int, float)):
                        ip = float(ip_raw)
                    else:
                        s = str(ip_raw).strip()
                        if "." in s:
                            whole, frac = s.split(".", 1)
                        else:
                            whole, frac = s, "0"
                        frac_dec = {"0": 0.0, "1": 1/3, "2": 2/3}.get(frac, 0.0)
                        ip = float(whole) + frac_dec
                except Exception:
                    try:
                        ip = float(ip_raw)
                    except Exception:
                        pass
            if any(v is not None for v in (era, whip, ip)):
                rows.append({
                    "mlb_id": pid,
                    "Name": name_full,
                    "ERA": era,
                    "WHIP": whip,
                    "IP": ip
                })
        # Preseason fallback: if no regular-season splits yet, retry with previous year
        if prior_year_fallback and total_raw_splits == 0 and year is not None:
            used_fallback = True
            fallback_year = year - 1
            print(f"[Pitcher update] No regular-season splits for {year}; retrying with {fallback_year}")
            params["season"] = fallback_year
            print(f"[Pitcher update] Request URL: {stats_base}?{urlencode(params)}")
            try:
                resp = requests.get(stats_base, params=params, headers=headers, timeout=25)
                resp.raise_for_status()
                body = resp.json()
            except Exception as e:
                raise ValueError(f"MLB StatsAPI stats request failed (fallback {fallback_year}): {e}") from e
            stats_list = body.get("stats") if isinstance(body, dict) else None
            if isinstance(stats_list, list) and stats_list:
                splits = stats_list[0].get("splits") if isinstance(stats_list[0], dict) else []
                if not isinstance(splits, list):
                    splits = []
                total_raw_splits = len(splits)
                rows = []
                for sp in splits:
                    if not isinstance(sp, dict):
                        continue
                    player = sp.get("player") or {}
                    st = sp.get("stat") or {}
                    pid = player.get("id")
                    name_full = (player.get("fullName") or "").strip()
                    if pid is None:
                        continue
                    try:
                        pid = int(pid)
                    except (TypeError, ValueError):
                        continue
                    era_raw = st.get("era") if st.get("era") is not None else st.get("ERA")
                    whip_raw = st.get("whip") if st.get("whip") is not None else st.get("WHIP")
                    ip_raw = st.get("inningsPitched") if st.get("inningsPitched") is not None else st.get("ip") or st.get("IP")
                    era = None
                    if era_raw not in (None, ""):
                        try:
                            era = float(era_raw)
                        except (TypeError, ValueError):
                            pass
                    whip = None
                    if whip_raw not in (None, ""):
                        try:
                            whip = float(whip_raw)
                        except (TypeError, ValueError):
                            pass
                    ip = None
                    if ip_raw is not None and ip_raw != "":
                        try:
                            if isinstance(ip_raw, (int, float)):
                                ip = float(ip_raw)
                            else:
                                s = str(ip_raw).strip()
                                if "." in s:
                                    whole, frac = s.split(".", 1)
                                else:
                                    whole, frac = s, "0"
                                frac_dec = {"0": 0.0, "1": 1/3, "2": 2/3}.get(frac, 0.0)
                                ip = float(whole) + frac_dec
                        except Exception:
                            try:
                                ip = float(ip_raw)
                            except Exception:
                                pass
                    if any(v is not None for v in (era, whip, ip)):
                        rows.append({
                            "mlb_id": pid,
                            "Name": name_full,
                            "ERA": era,
                            "WHIP": whip,
                            "IP": ip
                        })
        # SAFE MODE: only for true current-year responses (splits existed). Preseason prior-year fallback
        # must not be rejected for having < MIN_PITCHER_SAVE_COUNT rows — empty/malformed still fails below.
        if apply_safe_mode and not used_fallback and len(rows) < MIN_PITCHER_SAVE_COUNT:
            print("[Pitcher update] SAFE MODE: insufficient live pitcher data; keeping existing pitcher_stats.csv")
            raise ValueError(
                "[Pitcher update] SAFE MODE: insufficient live pitcher data; keeping existing pitcher_stats.csv"
            )
        if used_fallback and len(rows) > 0:
            print(
                "[Pitcher update] Prior-year regular-season fallback accepted "
                f"({len(rows)} pitchers with ERA/WHIP/IP; preseason — not gated by MIN_PITCHER_SAVE_COUNT={MIN_PITCHER_SAVE_COUNT})."
            )
        rows_after_filter = len(rows)
        print(f"[Pitcher update] Raw splits returned: {total_raw_splits}, Rows with ERA/WHIP/IP: {rows_after_filter}")

        df = pd.DataFrame(rows)
        if df.empty:
            if total_raw_splits == 0:
                if not apply_safe_mode:
                    return pd.DataFrame(), False
                raise ValueError(
                    "MLB StatsAPI returned no pitcher rows (stats endpoint returned 0 splits). "
                    "Check endpoint or network."
                )
            if not apply_safe_mode:
                return pd.DataFrame(), used_fallback
            raise ValueError(
                f"MLB StatsAPI returned no pitcher rows (raw splits: {total_raw_splits}, "
                "rows with ERA/WHIP/IP: 0). Check API response structure for stat/player."
            )
        return df, used_fallback

    @staticmethod
    def _mlb_savant_join(mlb_df: pd.DataFrame, savant_df: pd.DataFrame) -> pd.DataFrame:
        """Merge MLB season ERA/WHIP/IP with Savant xERA (same rules as update_pitcher_stats)."""
        if mlb_df is None or mlb_df.empty:
            return pd.DataFrame(columns=["mlb_id", "Name", "ERA", "WHIP", "IP", "xERA"])
        merged = mlb_df.merge(savant_df, left_on="mlb_id", right_on="player_id", how="left")
        merged["xERA"] = merged["xERA"].where(~merged["xERA"].isna(), merged["ERA"])
        merged["xERA"] = merged["xERA"].fillna(4.25)
        merged["IP"] = pd.to_numeric(merged["IP"], errors="coerce")
        merged["WHIP"] = pd.to_numeric(merged["WHIP"], errors="coerce")
        merged["xERA"] = pd.to_numeric(merged["xERA"], errors="coerce")
        return merged[["mlb_id", "Name", "ERA", "WHIP", "IP", "xERA"]]

    @staticmethod
    def _blend_cur_prev_season_pitchers(
        cur: pd.DataFrame,
        prev: pd.DataFrame,
        min_ip: float,
        blend_ip_cap: float = EARLY_SEASON_BLEND_IP_CAP,
    ) -> pd.DataFrame:
        """
        Early-season carryover: for each mlb_id, blend current-season xERA/WHIP with prior full season.

        w26 = min(max(ip_current, 0), blend_ip_cap) / blend_ip_cap
        blended xERA = w26 * xERA_cur + (1 - w26) * xERA_prev (with coalesce when one side is missing).
        Final IP is current-season IP when a current row exists, else prior IP. LowIP uses current IP when present.
        """
        cols = ["mlb_id", "Name", "xERA", "WHIP", "IP", "LowIP"]
        if cur.empty and prev.empty:
            return pd.DataFrame(columns=cols)
        if cur.empty:
            o = prev.copy()
            o["LowIP"] = o["IP"] < float(min_ip)
            return o[cols]
        if prev.empty:
            o = cur.copy()
            o["LowIP"] = o["IP"] < float(min_ip)
            return o[cols]

        cur = cur.sort_values("IP", ascending=False).drop_duplicates(subset=["mlb_id"], keep="first")
        prev = prev.sort_values("IP", ascending=False).drop_duplicates(subset=["mlb_id"], keep="first")
        joined = cur.merge(prev, on="mlb_id", how="outer", suffixes=("_cur", "_prev"))

        def _nz(a, b, default: float) -> float:
            if a is not None and pd.notna(a):
                try:
                    return float(a)
                except (TypeError, ValueError):
                    pass
            if b is not None and pd.notna(b):
                try:
                    return float(b)
                except (TypeError, ValueError):
                    pass
            return default

        out_rows = []
        for _, r in joined.iterrows():
            has_cur = pd.notna(r.get("IP_cur"))
            has_prev = pd.notna(r.get("IP_prev"))
            ip_cur = float(r["IP_cur"]) if has_cur else None
            ip_prev = float(r["IP_prev"]) if has_prev else None

            xc = r.get("xERA_cur")
            xp = r.get("xERA_prev")
            wc = r.get("WHIP_cur")
            wp = r.get("WHIP_prev")

            if has_cur and has_prev:
                xc_eff = _nz(xc, xp, 4.25)
                xp_eff = _nz(xp, xc, 4.25)
                wc_eff = _nz(wc, wp, 1.30)
                wp_eff = _nz(wp, wc, 1.30)
                ip26 = float(ip_cur) if ip_cur is not None else 0.0
                w26 = min(max(ip26, 0.0), float(blend_ip_cap)) / float(blend_ip_cap)
                x_bl = w26 * xc_eff + (1.0 - w26) * xp_eff
                whip_bl = w26 * wc_eff + (1.0 - w26) * wp_eff
                ip_final = ip_cur if ip_cur is not None else 0.0
                low_ip = ip_final < float(min_ip)
                name = (
                    str(r["Name_cur"]).strip()
                    if pd.notna(r.get("Name_cur")) and str(r.get("Name_cur")).strip()
                    else str(r["Name_prev"]).strip()
                )
            elif has_cur:
                x_bl = _nz(xc, xp, 4.25)
                whip_bl = _nz(wc, wp, 1.30)
                ip_final = ip_cur if ip_cur is not None else 0.0
                low_ip = ip_final < float(min_ip)
                name = str(r["Name_cur"]).strip()
            else:
                x_bl = _nz(xp, xc, 4.25)
                whip_bl = _nz(wp, wc, 1.30)
                ip_final = ip_prev if ip_prev is not None else 0.0
                low_ip = ip_final < float(min_ip)
                name = str(r["Name_prev"]).strip()

            mid = r["mlb_id"]
            try:
                mid = int(mid)
            except (TypeError, ValueError):
                continue
            out_rows.append(
                {"mlb_id": mid, "Name": name, "xERA": x_bl, "WHIP": whip_bl, "IP": ip_final, "LowIP": low_ip}
            )

        return pd.DataFrame(out_rows)

    # ----------------------------
    # Public API
    # ----------------------------
    @staticmethod
    def update_pitcher_stats():
        """
        Nightly updater using MLB StatsAPI (ERA/WHIP/IP) + Savant (xERA).
        Writes data/pitcher_stats.csv with columns: Name, xERA, WHIP, IP, LowIP

        Early season: fetches both the current regular season and the prior full season, then blends xERA/WHIP
        toward live stats using current-season IP (see EARLY_SEASON_BLEND_IP_CAP). Thin current-year pulls no
        longer trigger SAFE MODE alone because breadth comes from prior-year carryover in the blend.
        """
        try:
            print("🔄 Updating pitcher stats from MLB + Savant...")
            current_year = datetime.now().year
            prior_year = current_year - 1
            month = datetime.now().month

            if month < 6:
                min_ip = MIN_PITCHER_IP_EARLY
            elif month < 8:
                min_ip = MIN_PITCHER_IP_MID
            else:
                min_ip = MIN_PITCHER_IP_LATE

            # Current year: no inline prior-year substitution — we blend explicitly with prior_year below.
            # apply_safe_mode=False so a small live universe does not abort before carryover merge.
            mlb_cur, _ = DataManager._mlb_pitching_stats_by_id(
                current_year, prior_year_fallback=False, apply_safe_mode=False
            )
            # Prior full season: keep API fallback to year-2 if the season has no splits yet; no row-count SAFE MODE.
            mlb_prev, mlb_prior_year_fallback = DataManager._mlb_pitching_stats_by_id(
                prior_year, prior_year_fallback=True, apply_safe_mode=False
            )

            if mlb_cur.empty and mlb_prev.empty:
                raise ValueError(
                    "Pitcher update: no MLB pitching rows for current or prior season; cannot build pitcher table."
                )

            try:
                savant_cur = DataManager._savant_xera_by_id(current_year)
            except Exception as e:
                print(f"⚠️ Savant xERA fetch failed (current year), proceeding without xERA: {e}")
                savant_cur = pd.DataFrame(columns=["player_id", "xERA"])
                savant_cur["player_id"] = pd.Series(dtype=int)
                savant_cur["xERA"] = pd.Series(dtype=float)

            try:
                savant_prev = DataManager._savant_xera_by_id(prior_year)
            except Exception as e:
                print(f"⚠️ Savant xERA fetch failed (prior year), proceeding without xERA: {e}")
                savant_prev = pd.DataFrame(columns=["player_id", "xERA"])
                savant_prev["player_id"] = pd.Series(dtype=int)
                savant_prev["xERA"] = pd.Series(dtype=float)

            cur_joined = DataManager._mlb_savant_join(mlb_cur, savant_cur)
            prev_joined = DataManager._mlb_savant_join(mlb_prev, savant_prev)
            blended = DataManager._blend_cur_prev_season_pitchers(cur_joined, prev_joined, min_ip)
            if not mlb_cur.empty and not mlb_prev.empty:
                print(
                    f"[Pitcher update] Early-season blend: current_year rows={len(mlb_cur)} | "
                    f"prior_year rows={len(mlb_prev)} | blended_rows={len(blended)} "
                    f"(w_live = min(IP_cur, {EARLY_SEASON_BLEND_IP_CAP:g}) / {EARLY_SEASON_BLEND_IP_CAP:g})"
                )

            blended["Name"] = blended["Name"].astype(str).str.strip()
            blended["norm_name"] = blended["Name"].apply(DataManager.normalize_name)
            blended = blended.sort_values(["norm_name", "IP"], ascending=[True, False])
            blended = blended.drop_duplicates(subset=["norm_name"], keep="first")

            out = blended[["norm_name", "xERA", "WHIP", "IP", "LowIP"]].rename(columns={"norm_name": "Name"})
            out["Name"] = out["Name"].astype(str).str.strip().apply(DataManager.normalize_name)
            out = out.drop_duplicates(subset="Name", keep="first")
            refresh_n = len(out)

            # Load existing canonical base (broad universe) before row-count guard and merge.
            existing_canon = pd.DataFrame(columns=["Name", "xERA", "WHIP", "IP", "LowIP"])
            existing_n = 0
            if os.path.exists(STATS_FILE):
                _ex = DataManager._read_csv_with_column_normalization(STATS_FILE)
                if _ex is not None and not _ex.empty and "Name" in _ex.columns:
                    for c in ("xERA", "WHIP", "IP", "LowIP"):
                        if c not in _ex.columns:
                            _ex[c] = pd.NA
                    _ex["Name"] = _ex["Name"].astype(str).str.strip().apply(DataManager.normalize_name)
                    _ex = _ex.drop_duplicates(subset="Name", keep="last")
                    existing_canon = _ex[["Name", "xERA", "WHIP", "IP", "LowIP"]].copy()
                    existing_n = len(existing_canon)

            # Strict row-count guard only when there is no canonical file to merge into (first bootstrap).
            if refresh_n < MIN_PITCHER_SAVE_COUNT:
                if mlb_prior_year_fallback and refresh_n > 0:
                    print(
                        "[Pitcher update] Final save allowed: row count below MIN_PITCHER_SAVE_COUNT "
                        f"({refresh_n} < {MIN_PITCHER_SAVE_COUNT}) but merge used accepted prior-year "
                        "MLB StatsAPI data — will merge into canonical (or create baseline)."
                    )
                elif existing_n > 0:
                    print(
                        "[Pitcher update] Refresh row count below MIN_PITCHER_SAVE_COUNT "
                        f"({refresh_n} < {MIN_PITCHER_SAVE_COUNT}); merging into existing canonical "
                        f"({existing_n} rows) — breadth preserved."
                    )
                else:
                    print(
                        f"[Pitcher update] Refusing initial write: new row count too small ({refresh_n}) "
                        f"and no existing canonical at {STATS_FILE}"
                    )
                    raise ValueError(
                        f"Pitcher update aborted: new row count ({refresh_n}) below minimum ({MIN_PITCHER_SAVE_COUNT}); "
                        "existing file not overwritten."
                    )

            # Merge: refresh rows update matching names; all other canonical rows are kept.
            refresh_names = set(out["Name"].tolist())
            if existing_n > 0:
                only_in_existing = existing_canon[~existing_canon["Name"].isin(refresh_names)].copy()
                final_out = pd.concat([out, only_in_existing], ignore_index=True)
            else:
                final_out = out.copy()
            final_out = final_out.drop_duplicates(subset="Name", keep="first")
            final_n = len(final_out)

            narrow_refresh = existing_n > 0 and refresh_n < existing_n
            print(
                "[Pitcher update] Canonical merge: "
                f"refresh_rows={refresh_n} | existing_canonical_rows={existing_n} | "
                f"final_merged_rows={final_n} | "
                f"narrow_refresh_merged_without_dropping_breadth={bool(narrow_refresh)}"
            )

            os.makedirs(DATA_DIR, exist_ok=True)
            final_out.to_csv(STATS_FILE, index=False)

            print(f"✅ Saved {final_n} pitchers to {STATS_FILE} (min_ip={min_ip})")

            # also save LowIP fallback file
            fallback_file = os.path.join(DATA_DIR, "pitcher_stats_lowip.csv")
            final_out[final_out["LowIP"]].to_csv(fallback_file, index=False)

            return final_out.set_index("Name")

        except Exception as e:
            print(f"❌ Update failed: {e}")
            return DataManager.load_pitcher_stats()

    @staticmethod
    def load_pitcher_stats():
        try:
            if not os.path.exists(STATS_FILE):
                print(f"⚠️ Pitcher stats file not found: {STATS_FILE}")
                return pd.DataFrame()

            df = DataManager._read_csv_with_column_normalization(STATS_FILE)
            if len(df) < 50:
                raise ValueError("⚠️ Pitcher data too short")

            df["Name"] = df["Name"].astype(str).str.strip().apply(DataManager.normalize_name)
            df = df.drop_duplicates(subset="Name", keep="last").set_index("Name")

            print(f"✅ Loaded pitcher_stats.csv with {len(df)} pitchers")
            print(f"📋 Sample index: {list(df.index[:10])}")
            return df
        except Exception as e:
            print(f"⚠️ Failed to load pitcher stats: {e}")
            return pd.DataFrame()

    @staticmethod
    def scrape_fangraphs_pitcher(name):
        # kept as last-resort fallback (FG often blocks; use sparingly)
        try:
            print(f"🌐 Scraping FanGraphs for: {name}")
            search_url = f"https://www.fangraphs.com/api/players/find-pitcher?q={name}"
            res = requests.get(search_url, timeout=15)
            results = res.json()
            if not results:
                print("❌ No FanGraphs match found.")
                return None

            best_match = results[0]
            player_id = best_match["playerid"]
            full_name = best_match["playername"].strip().lower()

            stats_url = f"https://www.fangraphs.com/players/id/{player_id}/stats?position=P"
            dfs = pd.read_html(stats_url)
            stats_df = dfs[0]

            current_year = str(datetime.now().year)
            current_season = stats_df[stats_df["Season"].astype(str).str.startswith(current_year)]
            if current_season.empty:
                print("⚠️ No stats found for this season.")
                return None

            row = current_season.iloc[0]
            xera = float(row.get("xERA", 4.50))
            whip = float(row.get("WHIP", 1.30))
            ip = float(row.get("IP", 0.0))
            low_ip = ip < 60

            return {"Name": full_name, "xERA": xera, "WHIP": whip, "IP": ip, "LowIP": low_ip}
        except Exception as e:
            print(f"❌ Scraper error: {e}")
            return None

    @staticmethod
    def match_pitcher_row(df: pd.DataFrame, pitcher_name: str, alias_log=None):
        print(f"🔎 match_pitcher_row() called with: {pitcher_name}")
        if df.empty or not pitcher_name:
            return None

        print(f"📋 DataFrame index sample: {list(df.index)[:10]}")
        print(f"🎯 Trying to match: {pitcher_name} → {DataManager.normalize_name(pitcher_name)}")
        clean_name = DataManager.normalize_name(pitcher_name)
        print(f"🎯 Trying to match: {pitcher_name} → {clean_name}")

        # ✅ direct support for league-average fallbacks (no fuzzy)
        if clean_name in {"league avg away", "league avg home"}:
            if clean_name in df.index:
                # always return a Series with expected fields
                row = df.loc[clean_name]
                if isinstance(row, dict):
                    row = pd.Series(row, name=clean_name)
                # ensure expected columns exist
                for k, v in {"xERA": 4.20, "WHIP": 1.30, "IP": 150.0, "LowIP": False}.items():
                    if k not in row.index:
                        row[k] = v
                return row[["xERA", "WHIP", "IP", "LowIP"]]
            # safe default if row is somehow missing
            return pd.Series({"xERA": 4.20, "WHIP": 1.30, "IP": 150.0, "LowIP": False}, name=clean_name)

        alias_file = os.path.join(DATA_DIR, "pitcher_aliases.json")
        if os.path.exists(alias_file):
            try:
                with open(alias_file, "r") as f:
                    alias_map = json.load(f)

                alias_key = pitcher_name.strip().lower()
                reversed_aliases = {}
                for official, variations in alias_map.items():
                    if isinstance(variations, list):
                        for a in variations:
                            reversed_aliases[a.lower()] = official.lower()
                    elif isinstance(variations, str) and variations.strip():
                        reversed_aliases[variations.lower()] = official.lower()

                if alias_key in reversed_aliases:
                    clean_name = DataManager.normalize_name(reversed_aliases[alias_key])
                    print(f"🔁 Alias matched: {pitcher_name} → {clean_name}")
                    if alias_log is not None:
                        alias_log.append(f"{pitcher_name} → {clean_name}")
            except Exception as e:
                print(f"⚠️ Alias file error: {e}")

        # direct match
        if clean_name in df.index:
            row = df.loc[clean_name]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if row.isnull().any():
                print(f"🚫 Skipping '{clean_name}' due to NaN values:\n{row}")
                return None
            return row

        # fuzzy match with last-name guard
        choices = df.index.tolist()
        result = process.extractOne(clean_name, choices, scorer=fuzz.WRatio, score_cutoff=NAME_MATCH_THRESHOLD)
        if result:
            best_match, score = result[0], result[1]
            if clean_name.split()[-1] == best_match.split()[-1]:
                print(f"🟨 Fuzzy matched: {pitcher_name} → {best_match} ({score}%)")
                return df.loc[best_match]
            else:
                print(f"⛔ Reject fuzzy match due to last-name mismatch: {pitcher_name} → {best_match}")

        # manual fallback file
        manual_fallback_path = os.path.join(DATA_DIR, "manual_fallback_pitchers.csv")
        if os.path.exists(manual_fallback_path):
            try:
                fdf = pd.read_csv(manual_fallback_path)
                fdf.set_index(fdf["Name"].str.lower(), inplace=True)
                key = pitcher_name.lower()
                if key in fdf.index:
                    print(f"📦 Using manual fallback for: {pitcher_name}")
                    return fdf.loc[key].to_dict()
            except Exception as e:
                print(f"⚠️ Failed to load manual fallback for {pitcher_name}: {e}")

        # as a last resort, try scraping FG
        scraped = DataManager.scrape_fangraphs_pitcher(pitcher_name)
        if scraped:
            try:
                # append and de-dupe
                add = pd.DataFrame([scraped])
                if os.path.exists(STATS_FILE):
                    existing = pd.read_csv(STATS_FILE)
                    existing = pd.concat([existing, add], ignore_index=True)
                    existing["Name"] = existing["Name"].apply(DataManager.normalize_name)
                    existing = existing.drop_duplicates(subset="Name", keep="last")
                    existing.to_csv(STATS_FILE, index=False)
                else:
                    add["Name"] = add["Name"].apply(DataManager.normalize_name)
                    add.to_csv(STATS_FILE, index=False)
                print(f"✅ Auto-added {scraped['Name']} to pitcher_stats.csv")
            except Exception:
                pass
            return scraped

        # final league-average fallback
        print(f"⚠️ Falling back to league average for: {pitcher_name}")
        try:
            # Optional: only works if caller defined this function
            from model.overgang_model import send_telegram_alert  # type: ignore
            send_telegram_alert(f"⚠️ *Over Gang Alert*: No pitcher match for `{pitcher_name}`, using *league average* ⚾")
        except Exception:
            pass

        return {"xERA": 4.50, "WHIP": 1.30, "LowIP": True}

    @staticmethod
    def load_manual_fallback_pitchers():
        path = os.path.join(DATA_DIR, "manual_fallback_pitchers.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.set_index(df["Name"].str.lower(), inplace=True)
            return df
        return pd.DataFrame()
