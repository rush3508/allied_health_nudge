# src/synthesis.py
# Synthesise 50K member population
#
# Run from project root:
#   python src/synthesis.py
# Or from anywhere — paths are relative to this file.

import pandas as pd
import numpy as np
from faker import Faker
from pathlib import Path
import uuid

fake = Faker()
np.random.seed(42)
N = 50_000

# Project-relative paths — safe to run from any CWD
_PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW  = _PROJ_ROOT / 'data' / 'raw'
DATA_RAW.mkdir(parents=True, exist_ok=True)

# --- Member spine ---
employer_groups = [f"EMP_{i:03d}" for i in range(1, 13)]  # 12 groups
states = ['NSW', 'VIC', 'QLD', 'WA', 'SA', 'ACT']
plan_types = ['Bronze', 'Silver', 'Gold']
plan_weights = [0.35, 0.45, 0.20]  # Bronze-heavy, realistic for employer book

members = pd.DataFrame({
    'member_id': [str(uuid.uuid4()) for _ in range(N)],
    'age': np.random.normal(loc=42, scale=10, size=N).clip(25, 65).round().astype(int),
    'gender': np.random.choice(['M', 'F'], size=N),
    'employer_group_id': np.random.choice(employer_groups, size=N),
    'plan_type': np.random.choice(plan_types, size=N, p=plan_weights),
    'state': np.random.choice(states, size=N),
    'enrolment_start_date': pd.to_datetime([
        fake.date_between(start_date='-3y', end_date='today') for _ in range(N)
    ]),
})

# Employer group sizes (used in feature engineering)
group_size_map = {g: np.random.choice(['Small', 'Mid', 'Large'], p=[0.3, 0.5, 0.2])
                  for g in employer_groups}
members['employer_group_size'] = members['employer_group_id'].map(group_size_map)

# Tenure in months
members['tenure_months'] = ((pd.Timestamp('today') - members['enrolment_start_date'])
                             .dt.days / 30.44).round().astype(int).clip(1, 36)

members.to_csv(DATA_RAW / 'members.csv', index=False)
print(f"Members generated: {len(members):,}")
print(members['plan_type'].value_counts())
print(f"Age: mean={members['age'].mean():.1f}, min={members['age'].min()}, max={members['age'].max()}")

# Benefits entitlement table
# Each member gets a row per benefit type based on their plan

BENEFIT_ENTITLEMENTS = {
    'Bronze':  {'physio': 6,  'chiro': 4,  'dietetics': 4,  'psychology': 6},
    'Silver':  {'physio': 10, 'chiro': 6,  'dietetics': 6,  'psychology': 10},
    'Gold':    {'physio': 15, 'chiro': 10, 'dietetics': 10, 'psychology': 15},
}

benefit_rows = []
for _, member in members.iterrows():
    entitlements = BENEFIT_ENTITLEMENTS[member['plan_type']]
    for benefit_type, sessions_entitled in entitlements.items():
        
        # Simulate utilisation — most members use 0–40% of entitlement
        # High utilisation (>80%) is rare (~10% of members per benefit type)
        util_prob = np.random.random()
        if util_prob < 0.45:
            sessions_used = 0  # never claimed (45% of members)
        elif util_prob < 0.75:
            sessions_used = np.random.randint(1, max(2, int(sessions_entitled * 0.4) + 1))
        elif util_prob < 0.90:
            sessions_used = np.random.randint(
                int(sessions_entitled * 0.4), int(sessions_entitled * 0.8) + 1)
        else:
            sessions_used = sessions_entitled  # exhausted
        
        benefit_rows.append({
            'member_id': member['member_id'],
            'benefit_type': benefit_type,
            'sessions_entitled': sessions_entitled,
            'sessions_used': sessions_used,
            'sessions_remaining': sessions_entitled - sessions_used,
        })

benefits = pd.DataFrame(benefit_rows)
benefits.to_csv(DATA_RAW / 'benefits.csv', index=False)
print(f"Benefits rows: {len(benefits):,}")
print(f"Members with zero physio use: {(benefits[benefits.benefit_type=='physio']['sessions_used']==0).mean():.1%}")