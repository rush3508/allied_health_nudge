# Allied Health Nudge

**Predicting which health insurance members will have an acute event — so they can
be nudged toward preventive allied health care before they reach the ED.**

A complete ML pipeline portfolio project: synthetic data generation → feature
engineering → model training → calibration → scored outreach list. Built for
healthcare analytics roles.

---

## The Clinical Problem

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

**The intervention:** Identify high-risk members who aren't using their entitled
allied health sessions, and send a personalised nudge — *"You have 8 physiotherapy
sessions remaining and haven't used them yet."*

The business question: **who should we call first?**

---

## Fictional Client

A third-party administrator (TPA) managing employer group health funds. 50,000
synthetic members, working-age (25–65), across three plan tiers:

| Plan   | Physio | Chiro | Dietetics | Psychology | Pop. share |
|--------|--------|-------|-----------|------------|------------|
| Bronze | 6      | 4     | 4         | 6          | 35%        |
| Silver | 10     | 6     | 6         | 10         | 45%        |
| Gold   | 15     | 10    | 10        | 15         | 20%        |

---

## Pipeline

| Stage | Notebook | Output |
|-------|----------|--------|
| 1 — Data Build | `01_data_build.ipynb` | 50k members, benefits, claims, acute events |
| 2 — Features | `02_feature_engineering.ipynb` | 33 features (demographics, claims, benefits, interactions) |
| 3 — Labels | `03_label_engineering.ipynb` | Binary `high_acute_risk` target |
| 4 — Training | `04_model_training.ipynb` | LightGBM model + isotonic calibrator |
| 5 — Scoring | `05_scoring_output.ipynb` | Ranked outreach list with tiered recommendations |
| 6 — Report | `06_report.ipynb` | Summary report (in progress) |

---

## Label Design

The target variable (`high_acute_risk`) is generated via a two-layer data-generating
process that separates background clinical risk from the protective effect of
allied health utilisation:

**Layer 1 — Background Risk** (demographics + conditions only):
- Condition cluster base rates (Healthy 4%, MSK 18%, Metabolic 14%, MH 12%, Mixed 26%)
- Age amplifier: `exp(0.014 × (age − 40))`
- Comorbidity burden: stepwise multipliers (1.0× → 2.2×)

**Layer 2 — Utilisation Protection** (raw session data only):
- `protection = 1 − 0.70 × clip(utilisation_rate, 0, 1)`
- Members who use their allied health benefits get a risk reduction

The label is generated from source-of-truth tables (`members.csv` + `benefits.csv`)
*before* feature engineering runs — preventing data leakage between features
and labels. See `src/acute_events.py`.

---

## Model

**LightGBM** binary classifier with 33 features, 60/20/20 stratified split,
`scale_pos_weight` computed dynamically from label prevalence (~3.85).

| Metric | Value |
|--------|-------|
| ROC-AUC (test) | 0.7326 |
| PR-AUC | 0.4023 |
| Recall @ top 20% | 0.4270 |
| Precision @ top 20% | 0.4400 |
| Best iteration | 52 |
| Train-val AUC gap | 0.022 |

Isotonic calibration applied post-hoc. Raw scores are used for ranking and tier
assignment; calibrated scores are provided for business interpretability.

---

## What the Model Found

**Top 5 features by mean |SHAP| (test set):**

| Rank | Feature | Mean \|SHAP\| | Signal % |
|------|---------|--------------|----------|
| 1 | `comorbidity_count` | 0.240 | 53.1% |
| 2 | `condition_cluster` | 0.043 | 9.4% |
| 3 | `allied_health_utilisation_rate` | 0.033 | 7.2% |
| 4 | `age` | 0.029 | 6.3% |
| 5 | `has_msk_flag` | 0.014 | 3.2% |

Top 15 features capture 96.7% of total SHAP signal. The intervention-relevant
features — `allied_health_utilisation_rate`, `benefit_utilisation_rate`,
`sessions_remaining_*` — appear in positions 3–15, confirming the model has
learned the utilisation-protection relationship the DGP was designed to create.

---

## Scoring Output

The full 50,000 members are scored and tiered using percentile thresholds on
raw model scores:

| Tier | Count | Threshold | Description |
|------|-------|-----------|-------------|
| High | 5,017 (10.0%) | > p90 | Priority outreach |
| Medium | 4,984 (10.0%) | p80–p90 | Secondary outreach |
| Low | 39,999 (80.0%) | < p80 | Monitor |

**Nudge signal:** 4,758 members (9.5%) qualify — they have unused benefit
sessions AND have made zero allied health claims in the past 6 months. These
are the operationally actionable outreach targets.

Modality recommendations are condition-cluster-first: MSK → physiotherapy,
Metabolic → dietetics, MH → psychology.

---

## Running It Yourself

```bash
# 1. Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Generate synthetic data (~30 seconds)
python src/synthesis.py
python src/acute_events.py

# 3. Run the pipeline (in order)
jupyter notebook notebooks/01_data_build.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_label_engineering.ipynb
jupyter notebook notebooks/04_model_training.ipynb
jupyter notebook notebooks/05_scoring_output.ipynb
```

All data is synthetic. No real patient information is used.

---

## Tech Stack

- **Python:** pandas, numpy, LightGBM, scikit-learn, SHAP, matplotlib, SQLAlchemy
- **Storage:** Parquet (intermediate), PostgreSQL (feature store)
- **Environment:** Python 3.12, Ubuntu 24.04

---

## Repo Structure

```
notebooks/           # 6-stage pipeline (Jupyter)
src/                 # Data generation (synthesis.py, acute_events.py)
data/
  raw/               # Generated CSVs (gitignored — run synthesis.py to create)
  features/          # features.parquet, labels.parquet (gitignored)
outputs/             # model.pkl, calibrator.pkl, eval_metrics.json, etc. (gitignored)
requirements.txt     # Python dependencies
```

---

## Why This Project?

Built to demonstrate the full ML pipeline skillset for healthcare analytics roles:

- Translating a clinical problem into a modelling task without data leakage
- Feature engineering with clinically meaningful interaction terms
- LightGBM with class imbalance handling, calibration, and early stopping discipline
- SHAP interpretability with business narrative, not just feature rankings
- Tiered outreach scoring with operationally useful flags (nudge signal, plan design review)
- Isolation of label generation from feature engineering to prevent circularity
