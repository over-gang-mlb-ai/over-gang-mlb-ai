# Audit: Low projected totals (4.7–5.3 per game, all UNDER)

## 1. Suspicious blocks in `model/overgang_model.py`

### A. **project_team_runs() — inverted ERA ratio (root cause)**

**Lines 139–144:**

```python
    effective_era = (
        STARTER_IP_SHARE * opponent_starter_xera + BULLPEN_IP_SHARE * opponent_bullpen_era
    )
    effective_era = max(2.5, min(7.0, effective_era))
    runs = LEAGUE_RUNS_PER_TEAM * (LEAGUE_ERA / effective_era)
```

**Why it causes low totals:**  
The formula uses `LEAGUE_ERA / effective_era`. So:

- **Worse pitcher** (high `effective_era`, e.g. 5.5) → **fewer** runs: `4.25 * (4.25/5.5) ≈ 3.28`
- **Better pitcher** (low `effective_era`, e.g. 3.5) → **more** runs: `4.25 * (4.25/3.5) ≈ 5.14`

So the direction is reversed: we give more runs against good pitchers and fewer against bad ones.  
Standard approach: expected runs ∝ pitcher’s ERA (worse pitcher → more runs). So the ratio should be **effective_era / LEAGUE_ERA**, not LEAGUE_ERA / effective_era.  
With the current formula, anything above league-average ERA (e.g. 4.5–6.0) pushes per-team runs into the mid‑2s to low‑3s; two teams then land in the 4.7–5.3 total range you see.

---

### B. **project_team_runs() — WHIP multiplier (secondary suppression)**

**Lines 154–156:**

```python
    whip_mult = opponent_starter_whip / WHIP_LEAGUE
    whip_mult = max(0.90, min(1.15, whip_mult))
    runs *= whip_mult
```

**Why it can suppress:**  
When the opposing starter has **good** WHIP (e.g. 1.15), `whip_mult = 1.15/1.30 ≈ 0.88` → clamped to **0.90**. So we always apply at least a 10% cut in those cases. Direction (good WHIP → fewer runs) is correct, but the floor 0.90 compounds with the inverted-ERA effect and further lowers totals. Not the main cause, but it reinforces low numbers.

---

### C. **generate_prediction() — lineup and offense always neutral when data fails**

**Lines 363–364 (away), 374–375 (home):**

```python
        lineup_impact=away_lineup_impact,
        ...
        offense_mult=away_offense_mult,
```

**Lines 712–718 (call site):**  
`away_impact`, `home_impact` are set to 0.0 before the try; if `get_team_best9()` or `safe_lineup_impacts()` fails or returns zeros, they stay 0.0.  
`away_offense_mult` / `home_offense_mult` come from `Batters.offense_vs_hand_dict`; when batter data is missing or wrong they default to 1.0.

**Why they “do nothing”:**  
- **lineup_impact = 0** → `lineup_mult = 1.0 + 0 = 1.0` (no boost or cut).  
- **offense_mult = 1.0** → no change.  
So when lineup or batter data is broken/missing, both inputs are neutral and never add runs. They don’t actively suppress, but they don’t help; combined with the inverted ERA, the model has no upside from offense/lineup.

---

### D. **generate_prediction() — Vegas fallback 8.5**

**Lines 331–333, 419–420:**

```python
        vegas_line = float(vegas_data)
    except Exception as e:
        ...
        vegas_line = 8.5
    ...
    edge = round(projected_total - vegas_line, 2)
```

When the public/Vegas CSV fails, `vegas_line` is 8.5. With projected totals 4.7–5.3, every game has a large negative edge and becomes UNDER. So the “all UNDER” result is a direct consequence of low projections plus fixed 8.5 line, not a separate bug.

---

## 2. Why away_runs and home_runs sit around 2.3–2.8

- **Inverted ERA:** For `effective_era` in the 5–6 range (or higher with defaults), `4.25 * (4.25 / 5.5) ≈ 3.28` before other multipliers. After park (e.g. 1.0), offense (1.0), lineup (1.0), and WHIP (e.g. 0.90–1.0), you get into the **2.3–2.8** band per team.
- **No upside from lineup/offense:** With `lineup_impact = 0` and `offense_mult = 1.0`, there is no boost; only the inverted base and WHIP (and possibly park < 1.0) can reduce runs.

So the combination of (1) wrong ERA ratio and (2) neutral lineup/offense and (3) WHIP floor 0.90 explains the consistently low per-team runs.

---

## 3. Inputs and multipliers that suppress the most

| Input / multiplier | Effect | Severity |
|--------------------|--------|----------|
| **LEAGUE_ERA / effective_era** | Inverted: worse pitchers reduce runs. Single largest cause of low totals. | **Critical** |
| **WHIP clamp floor 0.90** | Up to 10% cut when opponent has good WHIP. | **Moderate** |
| **lineup_impact = 0** | No boost when lineup scoring fails; doesn’t lower, but no offset to low base. | **Context** |
| **offense_mult = 1.0** | No boost when batter data missing; same as above. | **Context** |
| **Park** | If venue unknown, 1.0; if pitcher’s park, &lt; 1.0, so can suppress. | **Minor** |

---

## 4. Whether offense_mult and lineup_impact “do nothing”

- **lineup_impact:** When lineup impacts are all 0.000, `lineup_mult` is 1.0 for every game, so lineup has no effect. That’s consistent with lineup scoring failing or returning zero (e.g. empty/broken batter table or `get_team_best9()` returning empty).
- **offense_mult:** When batter data is missing or defaults are used, `away_offense_mult` and `home_offense_mult` are 1.0, so they don’t change the projection.  
So in the current failure mode, both are effectively no-ops: they don’t suppress by themselves, but they don’t add runs either.

---

## 5. Minimal patch plan (order)

1. **Fix the ERA ratio in `project_team_runs()` (lines 143–144)**  
   Change  
   `runs = LEAGUE_RUNS_PER_TEAM * (LEAGUE_ERA / effective_era)`  
   to  
   `runs = LEAGUE_RUNS_PER_TEAM * (effective_era / LEAGUE_ERA)`  
   so that worse pitchers (higher ERA) produce more runs. This is the single change that will move totals from ~5 to the ~8–10 range for average matchups.

2. **(Optional) Soften WHIP floor in `project_team_runs()` (line 155)**  
   Consider raising the clamp floor from 0.90 to 0.95 so that good-WHIP pitchers don’t apply a full 10% cut, reducing secondary suppression. Only after verifying totals with step 1.

3. **(Optional) Defensive checks**  
   Add a sanity check or log when `lineup_impact` is always 0 or when `offense_mult` is always 1.0 so you can confirm batter/lineup data is loading (no code change to projection math required for this audit).

No other file edits required for this audit; all suspicious blocks and the minimal patch plan are confined to `model/overgang_model.py` as above.
