# Public betting / Vegas CSV — "No columns to parse from file" audit

## 1. File(s) involved

| File | Role |
|------|------|
| **core/public_betting_loader.py** | Defines `PUBLIC_BETTING_FILE`, calls `pd.read_csv()` in `load_public_betting_data()`. |
| **data/public_betting.csv** | CSV read by the loader (and by `VegasLines.get_vegas_line()` in overgang_model.py). Path is **relative**: `"data/public_betting.csv"`. |
| **model/overgang_model.py** (VegasLines.get_vegas_line) | Also reads `"data/public_betting.csv"` directly (line 276); same file, same error when CSV is empty. |

---

## 2. Exact root cause

**Pandas error:** `"No columns to parse from file"` is raised when `pd.read_csv()` is called on a file that has **no parseable column headers** — typically when the file is **empty** (0 bytes) or contains only blank lines.

On EC2, **data/public_betting.csv exists but is empty** (or effectively empty). That happens when:

1. **Scraper wrote an empty DataFrame:** `core/public_betting_scraper.py` builds `rows` from the Covers page. If no games are parsed (page structure change, no games, or exceptions in the loop), `rows = []`. Then `df = pd.DataFrame(rows)` has no columns, and `df.to_csv(OUTPUT_FILE, index=False)` writes a file with no header row (or 0 bytes). A later `pd.read_csv()` on that file then raises "No columns to parse from file".
2. **File created empty:** Something else (e.g. touch, or a failed write) created `data/public_betting.csv` with 0 bytes.

The loader only checks `os.path.exists(PUBLIC_BETTING_FILE)`. It does **not** check file size or handle “exists but empty,” so it still calls `pd.read_csv()`, which raises. Same for `VegasLines.get_vegas_line()` in overgang_model.py.

---

## 3. Exact suspicious code blocks

### A. core/public_betting_loader.py — no empty-file check before read_csv

**Lines 94–104:**

```python
def load_public_betting_data():
    if not os.path.exists(PUBLIC_BETTING_FILE):
        logging.warning("⚠️ Public betting CSV not found.")
        return {}

    try:
        df = pd.read_csv(PUBLIC_BETTING_FILE)
    except Exception as e:
        logging.error(f"❌ Failed to load public betting CSV: {e}")
        return {}
```

- **Path:** `PUBLIC_BETTING_FILE = "data/public_betting.csv"` (line 5) — relative; depends on CWD (e.g. project root on EC2).
- **Issue:** If the file exists but is empty, `pd.read_csv()` is still called and raises "No columns to parse from file". The exception is caught and `{}` is returned, but the error is logged and could be avoided by treating empty files as “no data.”

### B. core/public_betting_scraper.py — writes CSV even when rows is empty

**Lines 64–66:**

```python
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved {len(df)} games to {OUTPUT_FILE}")
```

- **Issue:** When `rows` is empty, `pd.DataFrame(rows)` has no columns. `df.to_csv(OUTPUT_FILE, index=False)` writes a file with no header (or 0 bytes). That file then causes "No columns to parse from file" when the loader (or overgang_model) reads it.

### C. model/overgang_model.py — VegasLines.get_vegas_line() reads same CSV with no empty check

**Lines 273–277 (approx):**

```python
    def get_vegas_line(home_team, away_team):
        game_key = ...
        try:
            df = pd.read_csv("data/public_betting.csv")
```

- **Issue:** Same relative path and same `read_csv()`; no check for empty file. Same pandas error when the CSV is empty.

---

## 4. Why "No columns to parse from file"

- **Not wrong path:** If the path were wrong, you’d get `FileNotFoundError`, not this pandas error.
- **Not wrong column names:** With an empty file there are no columns to name.
- **Cause:** The file **exists** and is **empty** (or has no valid header row). Pandas has nothing to parse as column names, so it raises "No columns to parse from file".

---

## 5. Minimal patch plan

1. **core/public_betting_loader.py — handle empty file before read_csv**  
   After the `os.path.exists(PUBLIC_BETTING_FILE)` check, add a check for empty file, e.g.  
   `if os.path.getsize(PUBLIC_BETTING_FILE) == 0: logging.warning("⚠️ Public betting CSV is empty."); return {}`  
   Then only call `pd.read_csv(PUBLIC_BETTING_FILE)` when the file has size > 0. Optionally also handle `df.empty` or `len(df.columns) == 0` after reading and return `{}` in that case.

2. **core/public_betting_scraper.py — don’t write CSV when there are no games**  
   Before `df.to_csv(OUTPUT_FILE, index=False)`, add:  
   `if not rows: print("⚠️ No games scraped; not writing empty CSV."); return`  
   So the scraper never overwrites the CSV with an empty file. If the file doesn’t exist, the loader already returns `{}`.

3. **model/overgang_model.py — VegasLines.get_vegas_line()**  
   Either (a) add the same empty-file check before `pd.read_csv("data/public_betting.csv")`, or (b) have `get_vegas_line()` use the same `load_public_betting_data()` result (or a shared helper that reads the CSV once and handles empty file) so only one place reads the CSV and handles empty/file-missing. Prefer (b) long-term to avoid two code paths that can both hit the empty file.

**Recommended order:** (1) loader empty-file check, (2) scraper skip-write when no games, (3) overgang_model either use loader or add the same empty check. That addresses the root cause and prevents the error without changing other behavior.
