# Allied Health Nudge

Identifying health insurance members at risk of acute escalation who are **not**
using their allied health benefits — to prioritise proactive outreach before they
reach the ED.

---

## The Problem

Health insurers fund allied health sessions (physiotherapy, chiropractic,
dietetics, psychology) that members routinely leave unused. Members with
unmanaged chronic conditions — MSK, metabolic, mental health — tend to bypass
allied health and escalate directly to costly acute care. This project builds a
model to surface those members early, so they can be nudged toward preventive
care they're already entitled to.

---

## Pipeline

| Stage | Notebook | What it does |
|-------|----------|--------------|
| 1 | `01_data_build` | Builds a 50,000-member synthetic population using CMS SynPUF claims + Synthea prevalence rates. Generates members, benefits entitlement, and claims tables. |
| 2 | `02_feature_engineering` | Constructs 33 features across demographics, condition clusters, 6-month claims history, and benefits utilisation. Loads to PostgreSQL. |
| 3 | `03_label_engineering` | Defines a binary `high_acute_risk` label via a two-layer DGP: background risk (condition cluster + demographics) moderated by a protective allied-health utilisation factor. |
| 4 | `04_model_training` | Trains a LightGBM classifier, evaluates on a held-out test set, and generates SHAP feature importance. |
| 5 | `05_scoring_output` | Scores the full population and produces a ranked outreach list. |
| 6 | `06_report` | Summary report of findings. |

---

## Key Signals

The label design centres on a two-layer causal structure:

- **Layer 1 — Background Risk**: condition cluster base rates, age amplifier,
  comorbidity burden (stepwise). Only demographics + conditions.
- **Layer 2 — Protective Moderator**: allied health utilisation reduces risk via
  `exp(-1.5 × utilisation_rate)`. Zero utilisation → no protection.

This structure ensures that nudge-relevant features (zero_allied_health_flag,
sessions_remaining_*, high_gp_low_allied) correlate with the label through the
protective mechanism rather than being directly baked into the formula. The
model must discover these relationships empirically.

**Top features by mean |SHAP| (test set):**

| Rank | Feature | Mean \|SHAP\| | Signal % |
|------|---------|--------------|----------|
| 1 | `comorbidity_count` | 0.240 | 53.1% |
| 2 | `condition_cluster` | 0.043 | 9.4% |
| 3 | `allied_health_utilisation_rate` | 0.033 | 7.2% |
| 4 | `age` | 0.029 | 6.3% |
| 5 | `has_msk_flag` | 0.014 | 3.2% |

Top 15 features capture 96.7% of total SHAP signal.

---

## Model Performance

**LightGBM** — binary classification, 60/20/20 stratified split,
`scale_pos_weight = 3.85`, early stopping at iteration 136.

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| ROC-AUC | 0.74 | > 0.75 |
| PR-AUC | 0.37 | > 0.55 |
| Recall @ top 20% | 0.44 | > 0.65 |
| Precision @ top 20% | 0.39 | > 0.35 |

Isotonic calibration applied post-hoc; ranking metrics are monotone-invariant
so ROC-AUC and Recall are unaffected. PR-AUC after calibration: 0.3912.

---

## Limitations

### 1. Synthetic Label Problem
The label is entirely synthetic — simulated via a Bernoulli draw on
probabilities derived from a hand-crafted DGP. This is fundamentally limited:
the model cannot outperform the signal-to-noise ratio baked into the simulation.
Real-world labels (actual ED visits, hospital admissions) would contain richer
patterns that a model could learn.

### 2. Tier Collapse
Plan tiers (Bronze/Silver/Gold) were designed to be meaningful via different
benefit entitlements, but in practice all three tiers produce near-identical
SHAP importance because `plan_type` information is fully mediated through
`sessions_remaining_*` and `benefit_utilisation_rate` features. The model
discovers these derived features carry the signal and ignores the parent
categorical. This is correct behaviour but limits the business narrative around
plan differentiation.

### 3. Single-Point-in-Time Features
All features are computed from a 6-month lookback window. There is no temporal
dimension — no trend features, no rolling windows, no change-over-time signals.
A member who has always had zero allied health use looks identical to one who
just stopped using it. Real-world outreach would benefit from knowing whether
behaviour is stable or changing.

### 4. Causal vs Correlational
The DGP creates a controlled causal structure, but the model learns
correlations, not causes. SHAP values tell us which features the model
*weights*, not which features would change the outcome if intervened upon.
Causal inference (e.g., double ML, IV) would be needed for policy decisions.

### 5. No Model Comparison
Only LightGBM was used. XGBoost, logistic regression with interactions, or a
simple rules-based baseline would provide context for whether the complexity is
justified. A model with 33 features and an AUC of 0.73 may not outperform a
simpler approach.

### 6. Portability
Model is saved via Python's `pickle`, which is version-sensitive. See
`outputs/model_version.json` for the exact environment. Switching Python or
LightGBM versions may break deserialisation. A `joblib` migration is noted in
the notebook.

---

## What I Learned

### The Data Generation Bottleneck
Building a realistic synthetic population was harder than training the model.
CMS SynPUF provides Medicare claims (65+ population), but the target population
is working-age (25-65). Synthea helped bridge the gap for condition prevalence,
but forcing these two sources to produce coherent synthetic members required
substantial ad-hoc mapping and calibration. If I were to do this again, I'd
invest more upfront in a single coherent data generation framework rather than
stitching together multiple sources.

### Label Design Is Everything
The first version of this project had a circular label design: the label was
computed directly from feature columns via explicit multipliers. The model
stopped training at iteration 136 and SHAP values were zero for all the
"interesting" nudge features. The model was correctly learning that its
features were a noisy function of themselves. Restructuring into the two-layer
DGP (background risk + protective moderator) broke the circularity and let the
model run 52 iterations — but the fundamental ceiling of synthetic labels
remains.

### SHAP Tells the Story, Not Just the Rankings
`comorbidity_count` dominates SHAP at 53% — this isn't a bug, it's the DGP
working as designed. But a hiring manager looking at this might ask: "so you
built a comorbidity counter?" The real story is in features 3-10:
`allied_health_utilisation_rate`, `benefit_utilisation_rate`,
`sessions_remaining_*` — these are where the intervention lives. The SHAP
ranking alone doesn't tell the business story; you need to narrate it.

### Portfolio Projects Need Honest Limitations
Early drafts of this README had aspirational metrics and no limitations
section. That's a red flag for hiring managers — it signals either inexperience
or dishonesty. The current version is honest about what worked and what didn't.
A model that doesn't meet targets but explains *why* is more compelling than
one that claims success on fabricated metrics.

### Infrastructure Is a Time Sink
Setting up PostgreSQL over Tailscale, managing Syncthing sync between two
machines, version-controlling notebooks, and keeping a Python venv consistent
across environments consumed more time than the modelling itself. For a solo
portfolio project, a single-machine SQLite + parquet pipeline would have been
faster and just as convincing.

---

## Data Sources

- **CMS SynPUF** — Medicare synthetic claims: beneficiary summaries, carrier
  claims, inpatient, outpatient, prescription drug events (2008–2010, Sample 1)
- **Synthea** — synthetic patient generator used to extract realistic condition
  prevalence and encounter distributions for a working-age population (25–65)
- **Insurance reference** — publicly available insurance cost dataset for
  member spend calibration

All data is synthetic. No real patient information is used.

---

## Tech Stack

- **Python** — pandas, numpy, LightGBM, scikit-learn, SHAP, matplotlib,
  SQLAlchemy
- **Storage** — Parquet (intermediate), PostgreSQL (feature store, self-hosted
  via Tailscale)
- **Data generation** — Synthea (Java-based synthetic patient simulator)

---

## Repo Structure

```
notebooks/
  01_data_build.ipynb
  02_feature_engineering.ipynb
  03_label_engineering.ipynb
  04_model_training.ipynb
  05_scoring_output.ipynb
  06_report.ipynb
src/
  synthesis.py              # 50k member spine generator
  acute_events.py           # Two-layer DGP for label generation
  test_postgres_connection.py
data/
  raw/                      # CMS SynPUF + Synthea outputs (gitignored)
  features/                 # features.parquet, labels.parquet (gitignored)
outputs/
  model.pkl                 # Trained LightGBM model
  model_version.json        # Version info for pickle portability
  calibrator.pkl            # Isotonic calibration model
  eval_metrics.json         # Test set metrics
  shap_summary.png          # SHAP feature importance plot
  feature_importance.csv    # Feature importance table
  scored_members.csv        # Full population with risk scores
```
