# CLAUDE.md — Allied Health Nudge

Project context for Claude Code. Read this entire file before touching anything.

---

## What This Project Is and Why It Exists

This is a **data science portfolio project** built for job hunting in the healthcare analytics
space. The goal is to show a hiring team a complete, real-world ML pipeline — not a tutorial
notebook, but a project with clinical reasoning, proper label engineering, interpretable outputs,
and documented decisions.

**What it demonstrates:**
- Translating a clinical problem into a modelling task without data leakage
- Feature engineering including interaction terms with business meaning
- LightGBM with class imbalance handling, calibration, and early stopping discipline
- SHAP for interpretability
- Scoring pipeline with tiered outreach logic and operationally useful flags
- Validation gates between every stage

---

## Business Context (The Story)

**Fictional client:** A third-party administrator (TPA) managing employer group health funds.

**The clinical problem:**
```
Member has unmanaged MSK / metabolic / mental health condition
        ↓
Member skips allied health (physio, dietetics, psychology, chiro)
        ↓
Condition worsens without managed care
        ↓
Presents to Emergency Department or gets admitted
        ↓
High acute spend + poor member outcome
```

**The intervention:** Send a personalised nudge — "You have 8 physiotherapy sessions
remaining and haven't used them yet" — to members who are high-risk AND not using their
entitlements. The business asks: *who should we call first?*

**Plan tiers — benefit entitlements per year:**

| Plan   | Physio | Chiro | Dietetics | Psychology | Pop. share |
|--------|--------|-------|-----------|------------|------------|
| Bronze | 6      | 4     | 4         | 6          | 35%        |
| Silver | 10     | 6     | 6         | 10         | 45%        |
| Gold   | 15     | 10    | 10        | 15         | 20%        |

**Population:** 50,000 synthetic members, working-age (25–65), employer group book.


---

## Pipeline Status

| Stage | Notebook | Output | Status |
|-------|----------|--------|--------|
| 0 | Setup | venv, PostgreSQL, raw data | ✅ Complete |
| 1 | `01_data_build.ipynb` | `members.csv`, `benefits.csv`, `claims.csv`, `acute_events.csv` | ✅ Complete |
| 2 | `02_feature_engineering.ipynb` | `features.parquet` (50,000 × 34) | ✅ Complete |
| 3 | `03_label_engineering.ipynb` | `labels.parquet` | ✅ Complete |
| 4 | `04_model_training.ipynb` | `model.pkl`, `calibrator.pkl`, `eval_metrics.json` | ✅ Complete |
| 5 | `05_scoring_output.ipynb` | `scored_members.csv`, `scoring_summary.json` | ✅ Complete |
| 6 | `06_report.ipynb` | `report.html`, `report.pbix`, Streamlit app | ⏳ Not started |
| 7 | Operationalisation | Docker, MLflow, CI/CD | ⏳ Not started |

---

## The Three Architectural Decisions That Shape Everything

Understanding these three decisions is more important than reading any individual notebook.
They explain why the code is structured the way it is and what would break if you changed them.

---

### Decision 1 — Label DGP (How the target variable is built)

**The problem this solved:**
The original Stage 3 computed the label `high_acute_risk` by calling `simulate_acute_risk(row)`
on the feature matrix from Stage 2. That function multiplied together feature columns like
`zero_allied_health_flag`, `benefit_utilisation_rate`, and `comorbidity_count` to produce a
Bernoulli probability. The model was then asked to predict a label that was *derived from its
own training features*. This is circular — the model just had to learn a noisy version of the
amplifier formula, not a genuine clinical pattern. Result: early stopping at iteration 11,
SHAP ≈ 0 for all nudge-related features.

**The solution — `src/acute_events.py`:**
The label is now generated at the END of Stage 1, BEFORE Stage 2 ever runs, using only two
things: (1) demographics and condition cluster from `members.csv`, and (2) raw session counts
from `benefits.csv`. These are the source-of-truth files — not engineered features.

The DGP has two layers:
```
Layer 1 — Background risk (demographics + condition only):
  background_risk = BASE_RATE[condition_cluster]
                    × exp(0.014 × (age − 40))       ← ages risk up continuously
                    × COMORBIDITY_AMP[comorbidity_count]

  BASE_RATES:        Healthy=0.04, MSK=0.18, Metabolic=0.14, MH=0.12, Mixed=0.26
  COMORBIDITY_AMP:   {0: 1.00, 1: 1.30, 2: 1.75, 3: 2.20}

Layer 2 — Utilisation protection (raw session data only):
  overall_utilisation_rate = total_sessions_used / total_sessions_entitled
  protection = 1 − 0.70 × clip(overall_utilisation_rate, 0, 1)

Final:
  P(acute_event) = min(background_risk × protection, 0.90)
  acute_event    ~ Bernoulli(P),  rng = np.random.default_rng(seed=99)
```

**Why this works cleanly:** The `overall_utilisation_rate` in the DGP is computed from raw
`benefits.csv` session counts. In Stage 2, `allied_health_utilisation_rate` is also computed
from those same raw counts. So the feature has a genuine causal path to the label — but through
real data, not through its own engineered value. The model can now learn the relationship.

**Stage 3 is now trivial as a result:** It just loads `acute_events.csv`, renames the
`acute_event` column to `high_acute_risk`, and saves `labels.parquet`. No computation at all.

---

### Decision 2 — Interaction Features (Stage 2 Step 5b)

Six interaction features were added on top of the base 27 features from Stage 2.
They encode clinically meaningful combinations — things like "MSK patient who hasn't
used any physio" — that the model can learn from directly rather than having to discover
the combination itself.

| Feature | Formula | What it means clinically |
|---------|---------|--------------------------|
| `msk_zero_allied` | `has_msk_flag × zero_allied_health_flag` | MSK member not using any allied health |
| `metabolic_zero_allied` | `has_metabolic_flag × zero_allied_health_flag` | Metabolic member not using any allied health |
| `mh_zero_allied` | `has_mh_flag × zero_allied_health_flag` | Mental health member not using any allied health |
| `comorbid_zero_allied` | `comorbidity_count × zero_allied_health_flag` | Worse conditions + no allied health use |
| `age_zero_allied` | `age × zero_allied_health_flag` | Older + not using allied health |
| `bronze_high_comorbid` | `(plan_type=='Bronze').astype(int) × comorbidity_count` | Lowest benefit entitlement + highest clinical need |

**Total features after Stage 2:** 33 features + `member_id` = 34 columns in `features.parquet`.

**REMOVED — `comorbid_sessions_gap`** (`comorbidity_count × sessions_remaining_total`): This
feature was tried and caused the model to stop at iteration 4 with score range 0.20–0.23. It
directly encoded the DGP's protection ceiling. Do not re-add it.

---

### Decision 3 — Score Calibration and What It Is Used For

LightGBM raw scores on this synthetic dataset are compressed into a narrow range (0.15–0.45)
rather than spreading across 0–1. This is normal for a model trained on synthetic data with a
limited number of informative features.

**Calibration:** After training, isotonic regression is fitted on the validation set in Stage 4:
```python
from sklearn.isotonic import IsotonicRegression
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(y_val_preds, y_val)
```
Calibrated scores span 0–1 and are interpretable as approximate probabilities.

**Critical rule — what each score type is used for:**
- `risk_score` (raw) → used for RANKING and tier assignment (percentile thresholds)
- `risk_score_calibrated` → stored in output for business reporting ONLY, never used for ranking

**Why raw scores for tiers:** Isotonic regression creates flat plateaus — many members get
identical calibrated scores (e.g., 1.0 for the top 200 members). Percentile thresholds on
a flat plateau don't split cleanly. Raw scores are continuous → clean 10%/10%/80% splits.


---

## Features — All 33

This exact list must appear in FEATURE_COLS in every notebook (01 through 05). If FEATURE_COLS
in any notebook does not match, the model.num_feature() assertion in Stage 5 will catch it.

```python
FEATURE_COLS = [
    # Demographics (6)
    'age', 'age_band', 'gender', 'plan_type', 'employer_group_size', 'tenure_months',
    # Condition flags (5)
    'has_msk_flag', 'has_metabolic_flag', 'has_mh_flag', 'comorbidity_count',
    'condition_cluster',
    # Claims — 6-month window (10)
    'total_claims_6m', 'total_spend_6m',
    'gp_visits_6m', 'specialist_visits_6m', 'allied_health_claims_6m',
    'days_since_last_allied', 'allied_health_utilisation_rate',
    'specialist_to_gp_ratio', 'zero_allied_health_flag', 'high_gp_low_allied',
    # Benefits (6)
    'sessions_remaining_physio', 'sessions_remaining_chiro',
    'sessions_remaining_dietetics', 'sessions_remaining_psychology',
    'any_benefits_remaining', 'benefit_utilisation_rate',
    # Interaction features — Step 5b (6)
    'msk_zero_allied', 'metabolic_zero_allied', 'mh_zero_allied',
    'comorbid_zero_allied', 'age_zero_allied', 'bronze_high_comorbid',
]

CAT_COLS = ['age_band', 'gender', 'plan_type', 'employer_group_size', 'condition_cluster']
```

**Dtype rules — apply in every notebook that loads features.parquet:**
- `condition_cluster` is saved as pandas `category` dtype in parquet.
  **Always cast immediately after load:** `features['condition_cluster'] = features['condition_cluster'].astype(str)`
  Without this cast, dict `.get()` lookups silently fail for some cluster values.
- `age_band` is an ordered Categorical. Leave it as-is — LightGBM handles it correctly.

---

## Model (Stage 4)

**Algorithm:** LightGBM binary classifier
**Feature count:** 33 (assert with `model.num_feature() == 33`)
**Target:** `high_acute_risk` (from `labels.parquet`)
**Split:** 60% train / 20% val / 20% test, stratified, seed=42
**Positive rate:** ~20.6% → scale_pos_weight ≈ 3.85

**Locked hyperparameters:**
```python
params = {
    'objective':         'binary',
    'metric':            ['auc', 'binary_logloss'],  # auc must be FIRST
    'learning_rate':     0.01,
    'num_leaves':        31,
    'min_child_samples': 50,
    'scale_pos_weight':  spw,   # NEVER hardcode — always compute from labels.parquet
    'feature_fraction':  0.7,
    'bagging_fraction':  0.8,
    'bagging_freq':      5,
    'reg_alpha':         0.1,
    'reg_lambda':        0.1,
    'seed':              42,
}
callbacks = [
    lgb.early_stopping(100, first_metric_only=True),  # CRITICAL — see bugs section
    lgb.log_evaluation(50),
]
```

**`first_metric_only=True` explained:** This tells early stopping to watch ONLY the first
metric in the list (AUC) and ignore logloss. Without it, early stopping fires when logloss
degrades even if AUC is still climbing — which is common in imbalanced datasets. This
parameter is the difference between stopping at iteration 11 and stopping at iteration 52.

**Performance (current locked model):**
| Metric | Value |
|--------|-------|
| Best iteration | 52 |
| Train AUC | 0.7671 |
| Val AUC | 0.7449 |
| Train-Val gap | 0.022 — healthy, no overfitting |
| ROC-AUC (test) | 0.7326 |
| PR-AUC | 0.4023 |
| Recall @ top 20% | 0.4270 |
| Precision @ top 20% | 0.4400 |

**Val AUC ceiling is ~0.745.** This is a property of the synthetic label, not the model.
The label was built from a small number of variables with fixed multipliers. Once the model
learns those ~6 relationships in ~52 trees, no more signal exists in the remaining features.
Do not retrain trying to push past 0.75 on val — it won't work on this synthetic data.

**SHAP — top contributors:**
`comorbidity_count` (53.1%), `condition_cluster` (9.4%),
`allied_health_utilisation_rate` (7.2%), `age` (6.3%),
then `sessions_remaining_psychology`, `sessions_remaining_physio` in top 15.

**Calibration:**
```python
# Stage 4 Step 7b — fitted on val set, applied to full 50k in Stage 5
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(model.predict(X_val), y_val)
```
Saved as `outputs/calibrator.pkl`. Raw scores: 0.1522–0.4536 → Calibrated: 0.0000–1.0000.

**Saved outputs:**
- `outputs/model.pkl` — LightGBM model object
- `outputs/calibrator.pkl` — isotonic calibrator (new — did not exist before)
- `outputs/eval_metrics.json` — all metrics including `_calibrated` variants
- `outputs/shap_summary.png` — SHAP beeswarm, top 15 features
- `outputs/feature_importance.csv` — all 33 features ranked by mean |SHAP|


---

## Scoring Output (Stage 5)

**Tier assignment — percentile thresholds on RAW scores:**
```python
p90 = np.percentile(raw_scores, 90)   # → 0.4256
p80 = np.percentile(raw_scores, 80)   # → 0.4088
# High tier = above p90, Medium = p80 to p90, Low = below p80
```
Result: High 5,017 (10.0%) / Medium 4,984 (10.0%) / Low 39,999 (80.0%)

**Modality recommendation — condition-cluster first:**

The recommended modality is determined primarily by the member's condition cluster,
not by which benefit has the most sessions remaining. This is clinically appropriate:
an MSK member should be directed to physiotherapy even if they have more psychology
sessions remaining.

```python
CLUSTER_PREFERRED = {
    'MSK':       'sessions_remaining_physio',
    'Metabolic': 'sessions_remaining_dietetics',
    'MH':        'sessions_remaining_psychology',
    'Mixed':     None,   # fallback: largest remaining gap
    'Healthy':   None,   # fallback: largest remaining gap
}
# If cluster-preferred modality is exhausted → fallback to largest remaining
# Tiebreak priority: psychology > physio > dietetics > chiro
```

Result: Dietetics 40% / Psychology 34% / Physiotherapy 25% / Chiropractic 1%

**nudge_signal — requires BOTH conditions:**
```python
nudge_signal = (sessions_remaining_total > 0) & (features['zero_allied_health_flag'] == 1)
```
- `sessions_remaining_total > 0` — member has unused entitlement (something to offer them)
- `zero_allied_health_flag == 1` — member has made zero allied health claims in the past 6m
- Result: 4,758 members (9.5%) — operationally meaningful outreach target

The old definition was just `benefit_gap_flag` (any unused sessions) which fired for 100%
of members. That's useless. The new definition targets members who are both entitled AND
completely disengaged from allied health.

**plan_design_flag:**
```python
plan_design_flag = (risk_tier == 'High') & (sessions_remaining_total == 0)
```
High-risk member who has already exhausted all their benefit sessions. Cannot be nudged —
flagged for plan design review instead. Currently 0 in the synthetic data (avg utilisation
is 0.26, so almost nobody exhausts all four types). Will fire in production at year-end.

**Output file: `scored_members.csv` — 28 columns:**
```
member_id, age, gender, plan_type, employer_group_id, state,
condition_cluster, comorbidity_count,
has_msk_flag, has_metabolic_flag, has_mh_flag,
gp_visits_6m, allied_health_claims_6m, allied_health_utilisation_rate,
days_since_last_allied, zero_allied_health_flag,
sessions_remaining_physio, sessions_remaining_chiro,
sessions_remaining_dietetics, sessions_remaining_psychology,
sessions_remaining_total, recommended_modality,
risk_score,              ← RAW score — primary ranking column
risk_score_calibrated,   ← calibrated probability — informational only
risk_tier, benefit_gap_flag, nudge_signal, plan_design_flag
```

**Output file: `scoring_summary.json` — key fields:**
```json
{
  "total_scored":        50000,
  "high_risk":           5017,
  "medium_risk":         4984,
  "low_risk":            39999,
  "nudge_signal":        4758,
  "nudge_rate":          0.0952,
  "plan_design_flag":    0,
  "threshold_high":      0.4256,
  "threshold_medium":    0.4088,
  "model_best_iteration": 52,
  "roc_auc":             0.7326,
  "precision_top20pct":  0.44,
  "recall_top20pct":     0.427
}
```

**Mean risk score by tier (validation):**
- High tier mean: 0.4384
- Low tier mean: 0.2704
- Ratio: 1.62× — confirms tiers are meaningfully separated


---

## Bugs Fixed — Do Not Reintroduce

Read this section before adding any feature, changing any seed, or modifying scoring logic.

### Bug 1: MSK acute event rate = 0.0%
**Symptom:** Stage 3 sanity check showed MSK cluster had 0% positive rate.
**Root cause:** `src/synthesis.py` called `np.random.seed(42)` globally. The old
`simulate_acute_events()` also called `np.random.seed(42)`. Same seed → same random
state → the Bernoulli draws for the 3,814 MSK members (which happened to be generated
sequentially) all landed on zero.
**Fix:** `src/acute_events.py` uses `rng = np.random.default_rng(seed=99)` — an isolated
random generator with its own internal state that cannot be contaminated by global
`np.random.seed()` calls elsewhere.
**Rule:** Never use `np.random.seed()` in `src/acute_events.py`. Always use `rng = np.random.default_rng(seed=99)`.

### Bug 2: `comorbid_sessions_gap` feature
**Symptom:** Model stopped at iteration 4. Score range 0.20–0.23 (flat, useless).
**Root cause:** Feature was `comorbidity_count × sessions_remaining_total`. This is a
function of two variables that both appear in the DGP's protection formula. The model
recovered the DGP formula trivially in 4 trees and had nothing left to learn.
**Fix:** Removed from Stage 2 Step 5b and from FEATURE_COLS entirely.
**Rule:** Never add interaction features that multiply any `sessions_remaining_*` column
with any condition severity column. These directly encode the DGP.

### Bug 3: Calibrated scores used for tier thresholds
**Symptom:** Tiers were 11.1% High / 14.0% Medium / 74.9% Low — far from target 10/10/80.
**Root cause:** Isotonic regression creates flat plateaus in calibrated scores. Many members
get exactly the same calibrated score (e.g. 1.0 for the top ~200 members). Taking the
90th percentile of a flat distribution doesn't put 10% above it.
**Fix:** Always compute tier thresholds from raw scores (continuous, no plateaus).
**Rule:** `risk_score` column = raw score. `risk_score_calibrated` = calibrated, informational.

### Bug 4: nudge_signal fired for 100% of members
**Symptom:** Scoring summary showed nudge_signal = 49,993 (100%). Operationally useless.
**Root cause:** Old definition was `nudge_signal = benefit_gap_flag` (has any unused
sessions). With average utilisation rate 0.26, virtually no member exhausts all four
benefit types, so benefit_gap_flag fires for everyone.
**Fix:** `nudge_signal = benefit_gap_flag AND zero_allied_health_flag`. Now 4,758 (9.5%).
**Rule:** nudge_signal must always require BOTH unused sessions AND zero allied health usage.

### Bug 5: Psychology monoculture — 60% of modality recommendations
**Symptom:** 60% of members got Psychology as recommended modality.
**Root cause:** Old logic picked the benefit type with most sessions remaining.
Psychology has the highest per-plan session limits (6/10/15 for Bronze/Silver/Gold)
→ idxmax() nearly always returned psychology.
**Fix:** Condition-cluster-first logic — MSK→Physio, Metabolic→Dietetics, MH→Psychology.
**Rule:** CLUSTER_PREFERRED must always be applied before the fallback logic.

### Bug 6: FEATURE_COLS mismatch between notebooks
**Symptom:** Stage 5 crashed at predict() — model expected 33 features, got 27.
**Root cause:** Stage 5 was copied from an earlier version with the original 27-feature list.
**Fix:** Assertion guard in Stage 5 Step 1:
`assert len(FEATURE_COLS) == model.num_feature()`
**Rule:** FEATURE_COLS must be identical (same 33 features, same order) in notebooks 02, 04, 05.

### Bug 7: Early stopping fired on logloss instead of AUC
**Symptom:** Model stopped at iteration 11 even though AUC was still improving.
**Root cause:** `lgb.early_stopping(100)` without `first_metric_only=True` watches ALL
metrics. Logloss often degrades earlier than AUC on imbalanced datasets.
**Fix:** `lgb.early_stopping(100, first_metric_only=True)` — only watches metric[0] (AUC).
**Rule:** `first_metric_only=True` is always required. Also, AUC must be the FIRST metric
in the `'metric'` list (`['auc', 'binary_logloss']`).

### Bug 8: `scale_pos_weight` hardcoded as 3.69
**Symptom:** If the positive rate in labels.parquet changes between runs, the imbalance
weight silently stays at 3.69.
**Fix:** Always compute dynamically:
`spw = (y_train == 0).sum() / (y_train == 1).sum()`
Then cross-check: `assert abs(spw - labels_spw) < 0.05`.
**Rule:** Never hardcode scale_pos_weight. Compute from the training split every time.


---

## Data File Schemas

### `data/raw/acute_events.csv` — KEY NEW FILE (label source)
Generated by `src/acute_events.py`, called from notebook 01 Step 10.
Stage 3 reads this file. It is the ONLY source for labels. Never derive labels from features.parquet.
```
member_id              str (UUID)
condition_cluster      str
comorbidity_count      int (0–3)
age                    int (25–65)
overall_utilisation_rate  float (0–1)  ← raw sessions_used / sessions_entitled
background_risk        float           ← Layer 1 DGP output
protection_factor      float           ← Layer 2 DGP output (1 - 0.70 × util)
acute_event_prob       float           ← min(background × protection, 0.90)
acute_event            int (0 | 1)     ← Bernoulli draw, renamed to high_acute_risk in Stage 3
```

### `data/raw/members.csv`
Generated by `src/synthesis.py`, enriched by notebook 01 Steps 5–9.
```
member_id, age (25–65), gender (M|F), employer_group_id (EMP_001–EMP_012),
plan_type (Bronze|Silver|Gold), state, enrolment_start_date,
employer_group_size (Small|Mid|Large), tenure_months (1–36),
has_msk_flag (0|1), has_metabolic_flag (0|1), has_mh_flag (0|1),
comorbidity_count (0–3), condition_cluster (Healthy|MSK|Metabolic|MH|Mixed),
gp_visits_6m, specialist_visits_6m, ed_visits_6m,
allied_health_claims_6m, days_since_last_allied (1–180, 999=never),
total_claims_6m, total_spend_6m
```

### `data/raw/benefits.csv`
200,000 rows — 50,000 members × 4 benefit types each.
```
member_id, benefit_type (physio|chiro|dietetics|psychology),
sessions_entitled, sessions_used, sessions_remaining
```

### `data/raw/claims.csv`
50,000 rows — one row per member.
```
member_id, gp_visits_6m, specialist_visits_6m, ed_visits_6m,
allied_health_claims_6m, days_since_last_allied, total_claims_6m, total_spend_6m
```

### `data/features/features.parquet`
50,000 rows × 34 columns (member_id + 33 features). Output of Stage 2.
- `condition_cluster` stored as pandas `category` dtype — always `.astype(str)` on load.
- `age_band` stored as ordered Categorical — leave as-is for LightGBM.

### `data/features/labels.parquet`
50,000 rows × 2 columns: `member_id`, `high_acute_risk`. Output of Stage 3.

---

## Machines and Infrastructure

| Machine | Role | Address |
|---------|------|---------|
| T14s | Dev — runs all notebooks, Power BI Desktop, Streamlit | local |
| V530s | Data store + MLOps infra — PostgreSQL, MLflow, Docker | `100.68.3.15` (Tailscale) |

**PostgreSQL on V530s:**
- Database: `allied_health`, Schema: `allied_health`
- Tables: `features` (Stage 2), `labels` (Stage 3), `scored_members` (Stage 5)
- Connection: `postgresql://ds_user:password@100.68.3.15:5432/allied_health`
- Note: Stage 2 loads to PostgreSQL. Stages 3–5 read from Parquet, not PostgreSQL.

**Python venv:** `C:\venvs\allied-health-nudge\` — stored OUTSIDE OneDrive to avoid sync conflicts.
Never create venvs inside `C:\Users\alexl\OneDrive\...` — OneDrive sync corrupts them.

**Notebook path convention:**
```python
from pathlib import Path
data_dir    = Path().resolve().parent / 'data'     # → project_root/data
outputs_dir = Path().resolve().parent / 'outputs'  # → project_root/outputs
```
All notebooks use `Path().resolve().parent` to navigate from `notebooks/` to project root.
Never use bare string paths like `'data/features/features.parquet'` — they break when
the working directory is not `notebooks/`.

---

## Run Order

**Full pipeline (start fresh or after regenerating data):**
```
python src/synthesis.py           → creates members.csv (spine only)
01_data_build.ipynb               → all 10 steps — creates members.csv (enriched),
                                     benefits.csv, claims.csv, acute_events.csv
02_feature_engineering.ipynb      → creates features.parquet (33 features + member_id)
03_label_engineering.ipynb        → creates labels.parquet
04_model_training.ipynb           → creates model.pkl, calibrator.pkl, eval_metrics.json
05_scoring_output.ipynb           → creates scored_members.csv, scoring_summary.json
```

**Partial re-run rules:**
- Changed scoring logic only → re-run notebook 05 only
- Changed model params → re-run notebooks 04, 05
- Changed features → re-run notebooks 02, 03, 04, 05
- Changed label DGP (src/acute_events.py) → re-run from notebook 01 Step 10 onward
- Changed member spine → re-run everything from `src/synthesis.py`

---

## Repo Structure

```
allied-health-nudge/
├── CLAUDE.md                          ← you are here
├── src/
│   ├── acute_events.py                ← DGP — generates acute_events.csv (label source)
│   ├── synthesis.py                   ← generates 50k member spine
│   └── test_postgres_connection.py
├── notebooks/
│   ├── 01_data_build.ipynb            ← 10 steps — Step 10 calls src/acute_events.py
│   ├── 02_feature_engineering.ipynb   ← Step 5b adds 6 interaction features (33 total)
│   ├── 03_label_engineering.ipynb     ← loads acute_events.csv → renames → labels.parquet
│   ├── 04_model_training.ipynb        ← 33 features, iter=52, AUC=0.7326, + calibrator
│   ├── 05_scoring_output.ipynb        ← raw score tiers, cluster-first modality, nudge 9.5%
│   └── 06_report.ipynb                ← PENDING (Stage 6)
├── data/
│   ├── raw/
│   │   ├── members.csv                ← 50,000 rows with condition flags + claims
│   │   ├── benefits.csv               ← 200,000 rows (50k × 4 benefit types)
│   │   ├── claims.csv                 ← 50,000 rows
│   │   ├── acute_events.csv           ← 50,000 rows — sole label source, do not delete
│   │   ├── insurance.csv              ← cost calibration reference
│   │   ├── cost_reference.csv
│   │   ├── cms/                       ← CMS SynPUF files (read-only, large)
│   │   └── synthea/                   ← Synthea 5k CSV exports
│   └── features/
│       ├── features.parquet           ← (50,000 × 34) — 33 features + member_id
│       └── labels.parquet             ← (50,000 × 2) — member_id + high_acute_risk
└── outputs/
    ├── model.pkl                      ← LightGBM, best_iteration=52
    ├── calibrator.pkl                 ← IsotonicRegression fitted on val set
    ├── eval_metrics.json              ← ROC-AUC=0.7326, PR-AUC=0.4023
    ├── feature_importance.csv         ← 33 features ranked by mean |SHAP|
    ├── shap_summary.png               ← SHAP beeswarm, top 15 features
    ├── scored_members.csv             ← 50,000 × 28 cols — primary output
    └── scoring_summary.json           ← run summary with all key metrics
```

---

## Tech Stack

```
Python 3.13 — venv at C:\venvs\allied-health-nudge\
pandas, numpy, lightgbm 4.6.0, scikit-learn (IsotonicRegression, train_test_split, metrics)
shap 0.51.0, pyarrow 24.0.0, sqlalchemy 2.0.49, psycopg2-binary 2.9.12
matplotlib (Agg backend for SHAP plots)
```

---

## Obsidian Notes Status (00–07)

The Obsidian notes in `Allied-Health-Nudge/` are the detailed stage-by-stage documentation.
Current sync status after pipeline redesign:

| Note | Status |
|------|--------|
| 00-Project-Index.md | ✅ Updated |
| 00a-Data-Sources-and-Setup.md | ✅ No changes needed |
| 01-Data-Build.md | ✅ Updated (Step 10 added) |
| 02-Feature-Engineering.md | ✅ Updated (Step 5b, 33 features) |
| 03-Label-Engineering.md | ✅ Rewritten (new DGP from acute_events.py) |
| 04-Model-Training.md | ✅ Updated (iter=52, AUC=0.7326, calibrator) |
| 05-Scoring-and-Output.md | ✅ Rewritten (raw scores, cluster-first modality, nudge 9.5%) |
| 06-Business-Intelligence.md | ⏳ Not started |
| 07-Operationalisation.md | ⏳ Not started |
