# Over Gang v4.0 Projection Model — Audit Report

## 1. Issues Found

### A. Double-counting / incorrect use of inputs

| Issue | Location | Detail |
|-------|----------|--------|
| **Lineup double-count** | Lines 737–738, 851 | `vegas_line_adj = vegas_line + 0.3 * lineup_delta` is passed as `vegas_data` into `generate_prediction`. Lineup already affects the projection via `away_lineup_impact` / `home_lineup_impact` in `project_team_runs`. Comparing `projected_total` to the adjusted line double-counts lineup and shrinks edge. |
| **Vegas line adjusted instead of compared** | 851, 371 | Model is supposed to compare projection to the actual Vegas total. Passing `vegas_line_adj` makes the comparison target lineup-adjusted; edge becomes `projection - adjusted_line` instead of `projection - vegas_line`. |

### B. Confidence re-using projection inputs (bias)

| Issue | Location | Detail |
|-------|----------|--------|
| **Bullpen ERA in confidence** | 395–402 | When pick is OVER, confidence is multiplied by 1.05/1.03 for bad bullpens. Bullpen ERA already raised the projection in `project_team_runs`. Same input drives both projection and confidence → overconfident when projection is high. |
| **Park factors in confidence** | 401–404 | `confidence *= over_boost` (or under_boost). Park already multiplies runs in `project_team_runs`. Re-using for confidence reinforces the same signal. |

### C. Missing public betting data and skip logic

| Issue | Location | Detail |
|-------|----------|--------|
| **Skip only on type, not “missing”** | 326–334 | `skip=True` only when `public_data` is not a dict. Caller always passes `public = {}` when a game is missing from public data, so we never skip for missing public. That’s correct: we should still output projection and edge. |
| **Fragile if None is passed** | 326 | If `generate_prediction(..., public_data=None)` were ever called, we’d skip. Safer to treat `None` as `{}` at the top of `generate_prediction`. |

### D. Edge thresholds and unit sizing

| Issue | Location | Detail |
|-------|----------|--------|
| **Aggressive thresholds** | 93–94, 449–455 | `EDGE_THRESHOLD = 0.25` and `EDGE_FOR_FULL_UNIT = 0.5` mean we recommend a bet for a 0.25-run edge; vig and noise can erase that. Consider 0.35–0.4 for threshold and 0.6–0.7 for full unit (tunable via constants). |

### E. Naming / consistency / fragile logic

| Issue | Location | Detail |
|-------|----------|--------|
| **Away_Runs + Home_Runs ≠ Projected_Total** | 873–878, 894–896 | After applying `bat_mult` we set `projected_total = projected_total_adj` and `edge = edge_adj`, but we never update `away_runs` or `home_runs`. So `game_data` shows pre-multiplier away/home and post-multiplier total → displayed sum doesn’t match Projected_Total. |
| **Vegas_Line fallback** | 895 | `Vegas_Line` uses `total_current if total_current else vegas_line_adj`. For a true projection model the comparison line should be the real Vegas total; when public is missing we should fall back to raw `vegas_line`, not `vegas_line_adj`. |
| **Dead code** | 513–514 | Duplicate assignment `total_open = game_data.get('total_open', '?')` in `format_alert`. |

---

## 2. Specific Lines / Blocks

- **Lineup / Vegas:** 737–738 (`vegas_line_adj`), 851 (`vegas_data=vegas_line_adj`), 371 (`edge = projected_total - vegas_line`).
- **Confidence:** 395–404 (bullpen and park multipliers).
- **Skip / public:** 326–334 (`if not isinstance(public_data, dict)`).
- **Units / edge:** 93–94 (constants), 449–455 (recommended_units).
- **Consistency:** 873–878 (bat_mult applied to total/edge only), 894–896 (game_data Projected_Total, Away_Runs, Home_Runs, Vegas_Line).
- **format_alert:** 513–514 (duplicate total_open).

---

## 3. Minimal Code Changes (in order)

1. **Use raw Vegas line for projection comparison**  
   In `run_predictions`, call `generate_prediction(..., vegas_data=vegas_line)` (not `vegas_line_adj`).  
   When setting `game_data["Vegas_Line"]`, use fallback `vegas_line` (raw) instead of `vegas_line_adj`.

2. **Keep away_runs/home_runs consistent with Projected_Total**  
   When applying `bat_mult`, scale both team runs:  
   `away_runs = round(away_runs * bat_mult, 2)`, `home_runs = round(home_runs * bat_mult, 2)`,  
   then `projected_total = round(away_runs + home_runs, 2)` and recompute `edge` from that total vs `total_current or vegas_line`.

3. **Stop re-using projection inputs in confidence**  
   In `generate_prediction`, remove (or set to 1.0) the bullpen ERA and park factor confidence multipliers (lines 395–404). Keep confidence from `|edge|`, public %, and line movement only.

4. **Treat missing public as empty dict**  
   At the start of `generate_prediction`, add:  
   `if public_data is None: public_data = {}`  
   so we never skip solely because public was None.

5. **Remove dead code in format_alert**  
   Delete the duplicate `total_open = game_data.get('total_open', '?')` line (one of 513 or 514).

6. **Optional: less aggressive edge/units**  
   Change constants (or document): e.g. `EDGE_THRESHOLD = 0.35`, `EDGE_FOR_FULL_UNIT = 0.6`, with a short comment that they can be tuned for vig/risk.

---

## 4. Patch Plan (order of importance)

| Priority | Fix | Rationale |
|----------|-----|-----------|
| 1 | Pass raw `vegas_line` and fix Vegas_Line fallback | Core model correctness: compare projection to actual Vegas, no lineup double-count. |
| 2 | Scale away_runs/home_runs with bat_mult and recompute total/edge | Output consistency and correct edge after batter adjustment. |
| 3 | Remove bullpen/park from confidence | Reduces bias from re-using projection inputs. |
| 4 | `if public_data is None: public_data = {}` | Robustness; avoids skipping when caller passes None. |
| 5 | Remove duplicate total_open in format_alert | Dead code cleanup. |
| 6 | Tune or document EDGE_THRESHOLD / EDGE_FOR_FULL_UNIT | Less aggressive sizing; optional. |
