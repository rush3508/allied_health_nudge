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
| 6 — Report | `06_report.ipynb` | Summary report (`outputs/report.html`) |

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

## Feature Engineering

### Interaction Features

Six clinically meaningful interaction features were engineered on top of the base
27 features. These encode combinations the model would otherwise have to discover
on its own — a member with an MSK condition who hasn't used any allied health, for
example — giving the model direct access to patterns a clinician would recognise:

| Feature | Formula | Clinical meaning |
|---------|---------|------------------|
| `msk_zero_allied` | `has_msk_flag × zero_allied_health_flag` | MSK member not using any allied health |
| `metabolic_zero_allied` | `has_metabolic_flag × zero_allied_health_flag` | Metabolic member not using any allied health |
| `mh_zero_allied` | `has_mh_flag × zero_allied_health_flag` | Mental health member not using any allied health |
| `comorbid_zero_allied` | `comorbidity_count × zero_allied_health_flag` | Worse conditions + no allied health use |
| `age_zero_allied` | `age × zero_allied_health_flag` | Older + not using allied health |
| `bronze_high_comorbid` | `(plan_type=='Bronze') × comorbidity_count` | Lowest benefit entitlement + highest clinical need |

**Total:** 33 features across demographics, claims history, benefit utilisation,
and interaction terms.

---

## Model

**LightGBM** binary classifier with 33 features, 60/20/20 stratified split,
`scale_pos_weight` computed dynamically from label prevalence (4.67 at 17.6%
positive class rate). Early stopping on validation logloss with 50-round patience.

| Metric | Value |
|--------|-------|
| ROC-AUC (test) | 0.7417 |
| PR-AUC | 0.3720 |
| Recall @ top 20% | 0.4458 |
| Precision @ top 20% | 0.3930 |
| NDCG @ 5% | 0.5330 |
| NDCG @ 10% | 0.4875 |
| NDCG @ 20% | 0.4732 |
| Best iteration | 362 |
| Positive class rate | 17.6% |

Isotonic calibration applied post-hoc. Raw scores are used for ranking and tier
assignment; calibrated scores are provided for business interpretability.

### Why Raw Scores for Ranking?

Isotonic regression creates flat plateaus — many members get identical calibrated
scores (e.g., all top-200 members at 1.0). Percentile thresholds on a flat plateau
don't split cleanly. Raw LightGBM scores are continuous and produce clean 10%/10%/80%
tier boundaries. **Raw scores for ranking, calibrated scores for reporting.**

---

## What the Model Found

**Top 5 features by mean |SHAP| (test set):**

| Rank | Feature | Mean \|SHAP\| | Signal % |
|------|---------|--------------|----------|
| 1 | `comorbidity_count` | 0.493 | 37.4% |
| 2 | `condition_cluster` | 0.169 | 12.8% |
| 3 | `allied_health_utilisation_rate` | 0.148 | 11.2% |
| 4 | `age` | 0.088 | 6.7% |
| 5 | `benefit_utilisation_rate` | 0.061 | 4.6% |

Top 15 features capture 94.4% of total SHAP signal. The intervention-relevant
features — `allied_health_utilisation_rate`, `benefit_utilisation_rate`,
`sessions_remaining_*` — appear in positions 3–15, confirming the model has
learned the utilisation-protection relationship the DGP was designed to create.

A full SHAP summary plot and additional evaluation charts are available in
`outputs/report.html`.

---

## Condition-Cluster Stratification

The model's tier assignments across condition clusters demonstrate clinically
sensible stratification — not just statistical ranking:

| Cluster | High Priority | Medium Priority | Low Priority |
|---------|:------------:|:--------------:|:------------:|
| Healthy | 958 | 958 | 7,664 |
| MSK | 382 | 382 | 3,054 |
| Mental Health | 120 | 119 | 956 |
| Metabolic | 2,136 | 2,136 | 17,084 |
| Mixed | 1,406 | 1,405 | 11,240 |

Mixed-comorbidity members (the highest clinical risk) appear in the High tier at
a rate 3.4× their population share. MSK members are elevated 1.5×. Healthy members
are underrepresented in High priority by 4.5×. The model respects clinical priors
without being told to do so explicitly.

---

## Scoring Output

The full 50,000 members are scored and tiered using percentile thresholds on
raw model scores:

| Tier | Count | Threshold | Description |
|------|-------|-----------|-------------|
| High | 5,000 (10.0%) | > 0.7276 | Priority outreach |
| Medium | 5,000 (10.0%) | 0.6534 – 0.7276 | Secondary outreach |
| Low | 40,000 (80.0%) | < 0.6534 | Monitor |

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

# 2. Set up a database (PostgreSQL or any SQLAlchemy-compatible backend)
#    Create a database and configure the connection string as DB_URL
#    environment variable (see .env.example)

# 3. Generate synthetic data (~30 seconds)
python src/synthesis.py
python src/acute_events.py

# 4. Run the pipeline (in order)
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
- **Storage:** Parquet (intermediate), SQL database (feature store)
- **Environment:** Python 3.12+

---

## Repo Structure

```
notebooks/           # 6-stage pipeline (Jupyter)
src/                 # Data generation & utilities
  synthesis.py       # Synthetic member/benefit/claim generation
  acute_events.py    # Label DGP (two-layer risk + protection)
data/
  raw/               # Generated CSVs (gitignored — run synthesis.py to create)
  features/          # features.parquet, labels.parquet (gitignored)
outputs/             # model.pkl, calibrator.pkl, eval_metrics.json, report.html (gitignored)
requirements.txt     # Python dependencies
```

---

## Why This Project?

Built to demonstrate the full ML pipeline skillset for healthcare analytics roles:

- Translating a clinical problem into a modelling task without data leakage
- Label generation isolated from feature engineering — a two-layer DGP that
  separates background risk from the protective effect of allied health utilisation
- Feature engineering with clinically meaningful interaction terms
- LightGBM with class imbalance handling, calibration, and early stopping discipline
- SHAP interpretability with business narrative, not just feature rankings
- Tiered outreach scoring with operationally useful flags (nudge signal)
- Condition-cluster stratification validating clinical sensibility of model rankings

---

## Contact

**Alex Lim** — [LinkedIn](https://www.linkedin.com/in/alex-lim-bb7526232/)

*Portfolio project. Not a licensed financial or medical adviser.*
