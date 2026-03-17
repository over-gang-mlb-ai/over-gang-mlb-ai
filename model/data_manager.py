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
    def _savant_xera_by_id(year: int) -> pd.DataFrame:
        """
        Return DataFrame with columns: player_id, xERA
        Multi-stage:
          1) CSV export (fast, preferred)
          2) HTML table scrape (robust when CSV omits xERA)
          3) Derive xERA from est_wOBA (last-resort, transparent)
        Saves debug artifacts under ./debug for troubleshooting.
        """
        os.makedirs("debug", exist_ok=True)
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        }

        # ---------- helper: choose column by fuzzy/normalized name
        def _pick_col(cols, *candidates):
            norm = {c.lower().strip().replace(" ", "").replace("_", ""): c for c in cols}
            normalized_candidates = [c.lower().strip().replace(" ", "").replace("_", "") for c in candidates]
            for key, original in norm.items():
                if key in normalized_candidates:
                    return original
            # soft search (prefix match)
            for c in cols:
                cl = str(c).lower().replace(" ", "").replace("_", "")
                if any(cl.startswith(k) for k in normalized_candidates):
                    return c
            return None

        # ---------- 1) CSV attempt(s)
        csv_urls = [
            f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=pitchers&year={year}&season={year}&csv=true",
            f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=pitchers&year={year}&csv=true",
        ]

        last_err = None
        for i, url in enumerate(csv_urls, 1):
            try:
                r = requests.get(url, headers=headers, timeout=25)
                r.raise_for_status()
                csv_text = r.text
                # save a sample for debugging
                path = f"debug/savant_xera_{year}_csv_attempt{i}.csv"
                with open(path, "w", encoding="utf-8") as f:
                    f.write(csv_text)

                df = pd.read_csv(io.StringIO(csv_text))
                pid_col = _pick_col(df.columns, "player_id", "playerid", "player id")
                xera_col = _pick_col(df.columns, "xERA", "xera", "expectedera", "exera")

                if pid_col and xera_col:
                    out = df[[pid_col, xera_col]].rename(columns={pid_col: "player_id", xera_col: "xERA"})
                    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
                    out["xERA"] = pd.to_numeric(out["xERA"], errors="coerce")
                    out = out.dropna(subset=["player_id", "xERA"]).astype({"player_id": int})
                    if not out.empty:
                        print(f"✅ xERA source: CSV (rows={len(out)}, attempt={i})")
                        return out
                else:
                    # keep a header snapshot for support
                    hdr = ",".join(map(str, df.columns))
                    with open(f"debug/savant_headers_{year}.csv", "w", encoding="utf-8") as f:
                        f.write(hdr + "\n")
                    last_err = (
                        f"Missing columns on CSV attempt {i}. id='{pid_col}', xERA='{xera_col}' from {url} (preview saved)."
                    )
            except Exception as e:
                last_err = f"CSV attempt {i} failed: {e}"

        # ---------- 2) HTML table scrape
        try:
            html_url = f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=pitchers&year={year}"
            r = requests.get(html_url, headers=headers, timeout=25)
            r.raise_for_status()
            html_path = f"debug/savant_xera_{year}_page.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(r.text)

            # read all tables; pick the one that has player_id and xERA (or est_wOBA for fallback)
            tables = pd.read_html(io.StringIO(r.text), flavor=["lxml", "bs4"], header=0)
            chosen = None
            for t in tables:
                cols = [str(c) for c in t.columns]
                pid_col = _pick_col(cols, "player_id", "playerid", "player id")
                xera_col = _pick_col(cols, "xERA", "xera", "expectedera", "exera")
                estwoba_col = _pick_col(cols, "est_wOBA", "estwoba", "expectedwoba", "xwoba")
                if pid_col and (xera_col or estwoba_col):
                    chosen = t
                    break

            if chosen is not None:
                cols = [str(c) for c in chosen.columns]
                pid_col = _pick_col(cols, "player_id", "playerid", "player id")
                xera_col = _pick_col(cols, "xERA", "xera", "expectedera", "exera")

                if pid_col and xera_col:
                    out = chosen[[pid_col, xera_col]].rename(columns={pid_col: "player_id", xera_col: "xERA"})
                    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
                    out["xERA"] = pd.to_numeric(out["xERA"], errors="coerce")
                    out = out.dropna(subset=["player_id", "xERA"]).astype({"player_id": int})
                    if not out.empty:
                        print(f"✅ xERA source: HTML (rows={len(out)})")
                        return out
                # keep table preview for debugging/derivation
                chosen.to_csv(f"debug/savant_xera_{year}_html_table.csv", index=False)
            else:
                last_err = "HTML scrape found no suitable table."
        except Exception as e:
            last_err = f"HTML scrape failed: {e}"

        # ---------- 3) Derive xERA from est_wOBA (transparent fallback)
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
                url = f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=pitchers&year={year}&csv=true"
                r = requests.get(url, headers=headers, timeout=25)
                r.raise_for_status()
                source = f"debug/savant_xera_{year}_csv_attempt_fallback.csv"
                with open(source, "w", encoding="utf-8") as f:
                    f.write(r.text)

            if source.endswith("_html_table.csv"):
                df = pd.read_csv(source)
            else:
                df = pd.read_csv(source)

            cols = [str(c) for c in df.columns]
            pid_col = _pick_col(cols, "player_id", "playerid", "player id")
            estwoba_col = _pick_col(cols, "est_wOBA", "estwoba", "expectedwoba", "xwoba")
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
    def _mlb_pitching_stats_by_id(year: int) -> pd.DataFrame:
        """
        Pull ERA/WHIP/IP for all active MLB pitchers via MLB StatsAPI (free).
        Returns columns: mlb_id, Name, ERA, WHIP, IP
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

        def chunks(iterable, size):
            it = iter(iterable)
            while True:
                batch = list(islice(it, size))
                if not batch:
                    return
                yield batch

        rows = []
        total_raw_people = 0
        debug_stats_logged = [False]  # one sample per run
        debug_raw_person_logged = [False]  # first raw person even if stats empty
        debug_request_logged = [False]  # log final request URL once per run
        people_base = "https://statsapi.mlb.com/api/v1/people"
        hydrate_val = f"stats(group=pitching,type=season,season={year},gameType=R)"
        print(f"[Pitcher update] Source: MLB StatsAPI people endpoint (year={year}, pitcher_ids={len(pitcher_ids)})")

        for batch in chunks(sorted(pitcher_ids), 50):
            ids = ",".join(map(str, batch))
            params = {
                "personIds": ids,
                "hydrate": hydrate_val,
            }
            if not debug_request_logged[0]:
                debug_request_logged[0] = True
                print(f"[Pitcher update] Request URL (first batch): {people_base}?{urlencode(params)}")
            try:
                resp = requests.get(people_base, params=params, headers=headers, timeout=20)
                body = resp.json()
                data = body.get("people") if isinstance(body, dict) else None
                if not isinstance(data, list):
                    data = []
            except Exception as e:
                print(f"[Pitcher update] Batch request failed: {e}")
                continue

            raw_count = len(data)
            total_raw_people += raw_count
            batch_rows_before = len(rows)

            for person in data:
                # One-time debug: first raw person (even if stats empty) to see response shape
                if not debug_raw_person_logged[0]:
                    debug_raw_person_logged[0] = True
                    pid_debug = person.get("id", "?")
                    keys_list = list(person.keys()) if isinstance(person, dict) else []
                    stats_val = person.get("stats")
                    stats_repr = repr(stats_val)
                    if len(stats_repr) > 200:
                        stats_repr = stats_repr[:200] + "..."
                    extra_keys = ["pitchHand", "primaryNumber", "primaryPosition", "currentTeam"]
                    has_extra = {k: k in person for k in extra_keys}
                    print(f"[Pitcher update] DEBUG first raw person: id={pid_debug}, top_level_keys={keys_list}")
                    print(f"[Pitcher update] DEBUG 'stats' exists={'stats' in person}, stats type={type(stats_val).__name__}, stats preview={stats_repr}")
                    print(f"[Pitcher update] DEBUG keys present: {has_extra}")

                pid = int(person["id"])
                name_full = (person.get("fullName") or "").strip()
                stats = person.get("stats")
                if stats is None:
                    stats = []
                if not isinstance(stats, list):
                    stats = [stats] if stats else []

                # One-time debug: first person with non-empty stats
                if stats and not debug_stats_logged[0]:
                    debug_stats_logged[0] = True
                    first = stats[0] if stats else {}
                    splits_val = first.get("splits") if isinstance(first, dict) else None
                    first_split = (splits_val[0] if isinstance(splits_val, list) and splits_val else first)
                    stat_val = first_split.get("stat", {}) if isinstance(first_split, dict) else {}
                    print(f"[Pitcher update] DEBUG first person with stats: id={pid}, person_keys={list(person.keys())}")
                    print(f"[Pitcher update] DEBUG stats type={type(stats).__name__}, len(stats)={len(stats)}")
                    if isinstance(first, dict):
                        print(f"[Pitcher update] DEBUG first_stats_item keys={list(first.keys())}")
                    print(f"[Pitcher update] DEBUG splits type={type(splits_val).__name__}, stat keys={list(stat_val.keys()) if isinstance(stat_val, dict) else 'n/a'}")

                era = whip = ip = None
                stats_list = stats if isinstance(stats, list) else []
                for statgrp in stats_list:
                    if not isinstance(statgrp, dict):
                        continue
                    splits = statgrp.get("splits") or statgrp.get("split") or []
                    if not isinstance(splits, list):
                        splits = [splits] if splits else []
                    # Some responses put stat at group level (no splits list)
                    if not splits and statgrp.get("stat") is not None:
                        splits = [statgrp]
                    for sp in splits:
                        if not isinstance(sp, dict):
                            continue
                        st = sp.get("stat") or sp
                        if not isinstance(st, dict):
                            continue
                        e = st.get("era") if st.get("era") is not None else st.get("ERA")
                        w = st.get("whip") if st.get("whip") is not None else st.get("WHIP")
                        ip_raw = st.get("inningsPitched") if st.get("inningsPitched") is not None else st.get("ip") or st.get("IP")
                        if e is not None:
                            era = e
                        if w is not None:
                            whip = w
                        if ip_raw is not None:
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

                era = float(era) if era not in (None, "") else None
                whip = float(whip) if whip not in (None, "") else None

                if any(v is not None for v in (era, whip, ip)):
                    rows.append({
                        "mlb_id": pid,
                        "Name": name_full,
                        "ERA": era,
                        "WHIP": whip,
                        "IP": ip
                    })

            batch_rows_added = len(rows) - batch_rows_before
            if raw_count > 0 and batch_rows_added == 0:
                print(f"[Pitcher update] Batch: raw_people={raw_count}, rows_with_stats=0 (check response structure)")

        rows_after_filter = len(rows)
        print(f"[Pitcher update] Raw people returned: {total_raw_people}, Rows with ERA/WHIP/IP: {rows_after_filter}")

        df = pd.DataFrame(rows)
        if df.empty:
            if total_raw_people == 0:
                raise ValueError(
                    "MLB StatsAPI returned no pitcher rows (people endpoint returned 0 records). "
                    "Check endpoint or network."
                )
            raise ValueError(
                f"MLB StatsAPI returned no pitcher rows (raw people: {total_raw_people}, "
                "rows with ERA/WHIP/IP: 0). Check API response structure for stats/splits/stat."
            )
        return df

    # ----------------------------
    # Public API
    # ----------------------------
    @staticmethod
    def update_pitcher_stats():
        """
        Nightly updater using MLB StatsAPI (ERA/WHIP/IP) + Savant (xERA).
        Writes data/pitcher_stats.csv with columns: Name, xERA, WHIP, IP, LowIP
        """
        try:
            print("🔄 Updating pitcher stats from MLB + Savant...")
            current_year = datetime.now().year
            month = datetime.now().month

            if month < 6:
                min_ip = MIN_PITCHER_IP_EARLY
            elif month < 8:
                min_ip = MIN_PITCHER_IP_MID
            else:
                min_ip = MIN_PITCHER_IP_LATE

            mlb_df = DataManager._mlb_pitching_stats_by_id(current_year)

            try:
                savant_df = DataManager._savant_xera_by_id(current_year)
            except Exception as e:
                print(f"⚠️ Savant xERA fetch failed, proceeding without xERA: {e}")
                savant_df = pd.DataFrame(columns=["player_id", "xERA"])
                savant_df["player_id"] = pd.Series(dtype=int)
                savant_df["xERA"] = pd.Series(dtype=float)

            merged = mlb_df.merge(savant_df, left_on="mlb_id", right_on="player_id", how="left")
            # fallback: if missing xERA, use ERA; if ERA missing too, 4.25
            merged["xERA"] = merged["xERA"].where(~merged["xERA"].isna(), merged["ERA"])
            merged["xERA"] = merged["xERA"].fillna(4.25)

            merged["Name"] = merged["Name"].astype(str).str.strip()
            merged["norm_name"] = merged["Name"].apply(DataManager.normalize_name)

            merged["IP"] = pd.to_numeric(merged["IP"], errors="coerce")
            merged["WHIP"] = pd.to_numeric(merged["WHIP"], errors="coerce")
            merged["xERA"] = pd.to_numeric(merged["xERA"], errors="coerce")
            merged["LowIP"] = merged["IP"] < float(min_ip)

            # de-dupe by normalized name; keep highest IP row
            merged = merged.sort_values(["norm_name", "IP"], ascending=[True, False])
            merged = merged.drop_duplicates(subset=["norm_name"], keep="first")

            out = merged[["norm_name", "xERA", "WHIP", "IP", "LowIP"]].rename(columns={"norm_name": "Name"})
            os.makedirs(DATA_DIR, exist_ok=True)
            out.to_csv(STATS_FILE, index=False)

            print(f"✅ Saved {len(out)} pitchers to {STATS_FILE} (min_ip={min_ip})")

            # also save LowIP fallback file
            fallback_file = os.path.join(DATA_DIR, "pitcher_stats_lowip.csv")
            out[out["LowIP"]].to_csv(fallback_file, index=False)

            return out.set_index("Name")

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
