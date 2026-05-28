# Allied Health Nudge

Identifying health insurance members at risk of acute escalation who are not using their allied health benefits — to prioritise proactive outreach before they reach the ED.

---

## The Problem

Health insurers fund allied health sessions (physiotherapy, chiropractic, dietetics, psychology) that members routinely leave unused. Members with unmanaged chronic conditions — MSK, metabolic, mental health — tend to bypass allied health and escalate directly to costly acute care. This project builds a model to surface those members early, so they can be nudged toward preventive care they're already entitled to.

---

## Pipeline

| Stage | Notebook | What it does |
|-------|----------|--------------|
| 1 | `01_data_build` | Builds a 50,000-member synthetic population using CMS SynPUF claims + Synthea prevalence rates. Generates members, benefits entitlement, and claims tables. |
| 2 | `02_feature_engineering` | Constructs 27 features across demographics, condition clusters, 6-month claims history, and benefits utilisation. Loads to PostgreSQL. |
| 3 | `03_label_engineering` | Defines a binary `high_acute_risk` label using clinically-informed base rates per condition cluster, with multiplicative amplifiers for age, zero allied health use, plan tier, and comorbidity burden. |
| 4 | `04_model_training` | Trains a LightGBM classifier, evaluates on a held-out test set, and generates SHAP feature importance. |
| 5 | `05_scoring_output` | Scores the full population and produces a ranked outreach list. |
| 6 | `06_report` | Summary report of findings. |

---

## Key Signals

The label design centres on the core business hypothesis: members with chronic conditions who are **not using allied health** are accumulating unmanaged risk.

**Engineered features that capture this:**
- `zero_allied_health_flag` — no allied health claims in the past 6 months
- `high_gp_low_allied` — high GP visit frequency with zero allied health use (seeking care, but through the wrong channel)
- `sessions_remaining_*` — unused entitlement by benefit type (physio, chiro, dietetics, psychology)
- `benefit_utilisation_rate` — overall proportion of entitled sessions consumed
- `comorbidity_count` / `condition_cluster` — MSK, Metabolic, MH, Mixed

**Top features by mean |SHAP| (test set):**

| Rank | Feature | Mean \|SHAP\| |
|------|---------|--------------|
| 1 | `comorbidity_count` | 0.119 |
| 2 | `has_msk_flag` | 0.019 |
| 3 | `condition_cluster` | 0.016 |
| 4 | `allied_health_claims_6m` | 0.016 |
| 5 | `plan_type` | 0.015 |

---

## Model Performance

**LightGBM** — binary classification, 60/20/20 stratified split, `scale_pos_weight = 3.69`.

| Metric | Result | Target |
|--------|--------|--------|
| ROC-AUC | 0.71 | > 0.75 |
| PR-AUC | 0.39 | > 0.55 |
| Recall @ top 20% | 0.40 | > 0.65 |
| Precision @ top 20% | 0.42 | > 0.35 |

**On the gap:** The label is simulated via a Bernoulli draw on probabilities derived from the same features the model is trained on. The Bernoulli noise introduces an irreducible ceiling — the model cannot perfectly recover a stochastic function of its own inputs. Early stopping at iteration 11 (of 1000) reflects this sparse signal. Next steps: (1) calibrate label noise, (2) introduce cross-feature interaction terms to give the model more to learn from, (3) experiment with isotonic regression calibration on the output probabilities.

---

## Data Sources

- **CMS SynPUF** — Medicare synthetic claims: beneficiary summaries, carrier claims, inpatient, outpatient, prescription drug events (2008–2010, Sample 1)
- **Synthea** — synthetic patient generator used to extract realistic condition prevalence and encounter distributions for a working-age population (25–65)
- **Insurance reference** — a publicly available insurance cost dataset used to calibrate member spend estimates

All data is synthetic. No real patient information is used.

---

## Tech Stack

- **Python** — pandas, numpy, LightGBM, scikit-learn, SHAP, matplotlib, SQLAlchemy
- **Storage** — Parquet (intermediate), PostgreSQL (feature store, self-hosted via Tailscale)
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
  synthesis.py          # 50k member spine generator
data/
  raw/                  # CMS SynPUF + Synthea outputs
  features/             # features.parquet, labels.parquet
outputs/
  model.pkl
  eval_metrics.json
  shap_summary.png
  feature_importance.csv
```
