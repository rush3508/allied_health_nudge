"""
acute_events.py
---------------
Data Generating Process (DGP) for the allied health nudge project.

Causal Structure
----------------
This module replaces the old label design in 03_label_engineering.ipynb, which
derived labels directly from engineered feature columns (zero_allied_flag, plan_type,
high_gp_low_allied etc.) via explicit multipliers. That design created a circular
dependency: the model was recovering a noisy function of its own inputs, causing
early stopping at iteration 11 and zero SHAP for the core nudge features.

The new DGP has two distinct causal layers:

  Layer 1 -- Background Risk (condition cluster + demographics ONLY):
    background_risk = BASE_RATE[condition_cluster]
                      x age_amplifier(age)           # continuous exponential
                      x comorbidity_amplifier(count)  # stepwise, reflects compounding burden

  Layer 2 -- Protective Moderator (allied health utilisation):
    protection = exp(-PROTECTION_K * clip(overall_utilisation_rate, 0, 1))
    Diminishing-returns curve: early sessions provide disproportionate benefit.
    -> util = 0.00 -> protection = 1.000 (no mitigation -- full background risk)
    -> util = 0.25 -> protection = 0.687 (31.3% risk reduction)
    -> util = 0.50 -> protection = 0.472 (52.8% risk reduction)
    -> util = 1.00 -> protection = 0.223 (77.7% maximum protection)

  Final outcome:
    P(acute_event) = background_risk x protection
    acute_event    ~ Bernoulli(P(acute_event))

Why This Breaks the Old Circularity
-------------------------------------
Nudge features (zero_allied_health_flag, sessions_remaining_*, high_gp_low_allied,
benefit_utilisation_rate) are NOT in the DGP formula. They correlate with
overall_utilisation_rate, giving them genuine indirect causal paths to the label
through the protective moderator. The model must discover these relationships
empirically, not recover a formula.

Expected outcomes vs old design:
  - Model runs significantly more boosting iterations (was 11)
  - AUC improves toward 0.78-0.83
  - Previously-zero SHAP features (zero_allied_flag, sessions_remaining, etc) become non-zero
  - SHAP values tell a coherent business story about the intervention

Dependencies
------------
Reads from:  data/raw/members.csv  (requires condition flags from Stage 1 Step 7)
             data/raw/benefits.csv (requires Stage 1 Step 8)
Writes to:   data/raw/acute_events.csv
Called by:   notebooks/01_data_build.ipynb (Step 10)
Consumed by: notebooks/03_label_engineering.ipynb
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ── DGP Parameters ────────────────────────────────────────────────────────────

# Base probability of acute escalation by condition cluster.
# Represents the unconditional rate for a 40-year-old who consumes ZERO allied health.
# Calibrated to target ~20% population-level positive rate given Synthea cluster
# distribution: Metabolic 42.7%, Mixed 28.1%, Healthy 19.1%, MSK 7.6%, MH 2.4%.
CLUSTER_BASE_RATES: dict = {
    'Healthy':   0.04,   # background noise -- no managed chronic conditions
    'MSK':       0.18,   # unmanaged musculoskeletal -> acute pain -> ED bypass
    'Metabolic': 0.14,   # unmanaged DM/obesity -> acute metabolic complications
    'MH':        0.12,   # untreated mental health -> crisis escalation
    'Mixed':     0.26,   # compounding burden of 2+ condition clusters
}

# Stepwise comorbidity amplifier -- super-linear, reflects compounding burden.
# A member with 3 conditions faces 2.2x the base risk of a single-condition member.
COMORBIDITY_AMP: dict = {
    0: 1.00,
    1: 1.30,
    2: 1.75,
    3: 2.20,
}

# Diminishing-returns protection coefficient (k in exp(-k * util)).
# Based on published evidence (Roblin 2005, Kroon 2012, Cheung 2016) showing
# allied health reduces acute events by 15-45%. With k=1.5:
#   util=0.00 → protection=1.000 (no risk reduction)
#   util=0.25 → protection=0.687 (31% reduction — early sessions highest impact)
#   util=0.50 → protection=0.472 (53% reduction)
#   util=1.00 → protection=0.223 (78% max reduction)
# This diminishing-returns curve reflects clinical reality where the first few
# sessions provide disproportionate benefit (Benell et al., 2011).
PROTECTION_K: float = 1.5


def age_amplifier(age: float) -> float:
    """
    Continuous exponential age amplifier. Anchored at 1.0 at age 40.
    Capped at 2.0 to prevent extreme amplification at very advanced ages.

    Gives the model a genuine continuous signal rather than discrete age-band
    steps, which would create artificial discontinuities in the learnable gradient.

    Examples:
        age 25 -> exp(0.014 x -15) = 0.81
        age 40 -> exp(0.000)       = 1.00
        age 55 -> exp(0.014 x 15)  = 1.23
        age 65 -> exp(0.014 x 25)  = 1.42
        age 90 -> min(exp(0.014 x 50), 2.0) = 2.00 (capped)
    """
    return float(min(np.exp(0.014 * (float(age) - 40.0)), 2.0))


def comorbidity_amplifier(count: int) -> float:
    """Stepwise multiplier for comorbidity burden. Clamps at 3-condition value."""
    return COMORBIDITY_AMP.get(int(count), COMORBIDITY_AMP[3])


def utilisation_protection(overall_util_rate: float) -> float:
    """
    Protective effect of allied health utilisation on acute event probability.

    Exponential decay from 1.0 to ~0.223, reflecting diminishing returns: the first
    few sessions provide disproportionate clinical benefit, while later sessions yield
    smaller marginal risk reduction. This better mirrors real-world allied health
    dynamics where initial engagement has the largest impact.

    The model will learn:
        allied_health_utilisation_rate  -> this protective factor (direct signal)
        zero_allied_health_flag         -> whether protection is entirely absent
        sessions_remaining_*            -> proxy for unfulfilled utilisation potential
        high_gp_low_allied              -> correlates with low allied utilisation
    """
    return float(np.exp(-PROTECTION_K * float(np.clip(overall_util_rate, 0.0, 1.0))))


# ── Core Simulation Function ──────────────────────────────────────────────────

def simulate_acute_events(
    members: pd.DataFrame,
    benefits: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate binary acute events for the full member population.

    Parameters
    ----------
    members  : DataFrame with [member_id, age, comorbidity_count, condition_cluster].
               Condition flags must be set (Stage 1 Step 7 must have run).
    benefits : DataFrame with [member_id, sessions_used, sessions_entitled].
               Stage 1 Step 8 must have run.
    seed     : Random seed for reproducibility.

    Returns
    -------
    DataFrame with columns:
        member_id                -- join key
        condition_cluster        -- cluster used in DGP (for validation)
        comorbidity_count        -- comorbidity count used in DGP
        age                      -- age used in DGP
        overall_utilisation_rate -- sessions_used / sessions_entitled across all 4 types
        background_risk          -- BASE_RATE x age_amp x comorbidity_amp (pre-moderation)
        protection_factor        -- exp(-1.5 x utilisation_rate) (diminishing-returns protection)
        acute_event_prob         -- background_risk x protection_factor
        acute_event              -- Bernoulli(acute_event_prob), the label
    """
    # Use default_rng (isolated state) rather than np.random.seed() (global state).
    # The legacy global RNG shares state with any np.random.seed(42) calls in
    # notebook 01 Step 7 (condition flag assignment). Reusing the same stream
    # creates a systematic correlation between which members receive condition flags
    # and their Bernoulli outcome — causing entire clusters (MSK) to return all-zero
    # events even when their acute_event_prob is correctly computed as ~0.15-0.22.
    # default_rng is fully isolated and unaffected by external seed resets.
    rng = np.random.default_rng(seed)

    # 1. Compute overall utilisation rate per member across all 4 benefit types.
    #    Aggregates physio + chiro + dietetics + psychology into one rate.
    #    Corresponds directly to allied_health_utilisation_rate in Stage 2 features,
    #    giving that feature a direct causal path to the label.
    util_agg = (
        benefits
        .groupby('member_id')
        .agg(
            sessions_used_total=('sessions_used', 'sum'),
            sessions_entitled_total=('sessions_entitled', 'sum'),
        )
        .reset_index()
    )
    util_agg['overall_utilisation_rate'] = (
        util_agg['sessions_used_total']
        / util_agg['sessions_entitled_total'].replace(0, np.nan)
    ).fillna(0.0)
    util_agg = util_agg[['member_id', 'overall_utilisation_rate']]


    # 2. Merge member demographics + condition + utilisation
    df = (
        members[['member_id', 'age', 'comorbidity_count', 'condition_cluster']]
        .merge(util_agg, on='member_id', how='left')
    )
    df['overall_utilisation_rate'] = df['overall_utilisation_rate'].fillna(0.0)

    # 3. Compute DGP components
    df['_base_rate']  = df['condition_cluster'].map(CLUSTER_BASE_RATES).fillna(CLUSTER_BASE_RATES['Healthy'])
    df['_age_amp']    = df['age'].apply(age_amplifier)
    df['_comorb_amp'] = df['comorbidity_count'].apply(comorbidity_amplifier)

    # Background risk: purely condition + demographics -- no utilisation features
    df['background_risk'] = df['_base_rate'] * df['_age_amp'] * df['_comorb_amp']

    # Protective moderator: utilisation suppresses background risk
    df['protection_factor'] = df['overall_utilisation_rate'].apply(utilisation_protection)

    # 4. Final probability
    df['acute_event_prob'] = df['background_risk'] * df['protection_factor']

    # 5. Bernoulli draw — uses isolated rng, not global numpy state
    df['acute_event'] = rng.binomial(1, df['acute_event_prob'].values)

    return df.drop(columns=['_base_rate', '_age_amp', '_comorb_amp']).reset_index(drop=True)


# ── Convenience Entry Point ───────────────────────────────────────────────────

def run(data_dir: Path = None, seed: int = 42) -> pd.DataFrame:
    """
    Loads raw files, runs simulation, saves acute_events.csv, prints diagnostics.

    Requires:
      data/raw/members.csv  -- with condition flags (Stage 1 Step 7 must have run)
      data/raw/benefits.csv -- Stage 1 Step 8 must have run
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / 'data' / 'raw'

    members  = pd.read_csv(data_dir / 'members.csv')
    benefits = pd.read_csv(data_dir / 'benefits.csv')

    out = simulate_acute_events(members, benefits, seed=seed)

    out_path = data_dir / 'acute_events.csv'
    out.to_csv(out_path, index=False)

    pos_count = int(out['acute_event'].sum())
    neg_count = len(out) - pos_count
    pos_rate  = out['acute_event'].mean()
    spw       = neg_count / pos_count

    print(f"Acute events simulated: {len(out):,} members")
    print(f"  Positive class rate:  {pos_rate:.1%}  ({pos_count:,} events)")
    print(f"  scale_pos_weight:     {spw:.2f}")
    print(f"  Saved to: {out_path}")

    # Protective effect sanity check
    q1_rate = out.nsmallest(int(len(out) * 0.25), 'overall_utilisation_rate')['acute_event'].mean()
    q4_rate = out.nlargest(int(len(out) * 0.25), 'overall_utilisation_rate')['acute_event'].mean()
    assert q1_rate > q4_rate, "BUG: Protective effect not present in simulation output"
    print(f"  Protection check: Q1 util={q1_rate:.1%} vs Q4 util={q4_rate:.1%} ({q1_rate/q4_rate:.2f}x)")

    return out


if __name__ == '__main__':
    run()
