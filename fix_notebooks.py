"""
Fix notebooks per requested changes:
- 04: extract assertion to own cell before training
- 04: extract bootstrap CI to own cell after eval_metrics
- 05: add NDCG cell after tier assignment; clean up c2d3e4f5
"""
import json
import copy
import uuid

def new_code_cell(source_lines, cell_id=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id or uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }

# ─── NOTEBOOK 04 ──────────────────────────────────────────────────────────────
path04 = r'notebooks\04_model_training.ipynb'
with open(path04, 'r', encoding='utf-8') as f:
    nb04 = json.load(f)

cells04 = nb04['cells']

# --- Task 2: extract assertion from training cell a93ee1dc to its own cell ---
train_idx = next(i for i, c in enumerate(cells04) if c.get('id') == 'a93ee1dc')
train_cell = cells04[train_idx]

# The assert line sits between the params dict and model = lgb.train(...)
# Find and remove it from the training cell source
assert_line = "assert params['metric'][0] == 'auc', 'AUC must be first metric for early stopping'\n"
blank_after  = "\n"

src = train_cell['source']
assert_pos = next((i for i, l in enumerate(src) if "assert params['metric'][0]" in l), None)
assert assert_pos is not None, "Assertion line not found in training cell"

# Remove assertion line and the blank line that follows it (if any)
new_src = src[:assert_pos]
after = src[assert_pos + 1:]
if after and after[0] == '\n':
    after = after[1:]
new_src += after
train_cell['source'] = new_src

# Insert a new assertion cell BEFORE the training cell
assertion_cell = new_code_cell(
    [assert_line[:-1]],  # drop trailing \n — Write adds it
    cell_id="asrt_metric"
)
# Rebuild source correctly
assertion_cell['source'] = [
    "assert params['metric'][0] == 'auc', 'AUC must be first metric for early stopping'"
]
cells04.insert(train_idx, assertion_cell)

print("Task 2: assertion extracted to own cell before training cell.")

# --- Task 3: extract bootstrap CI from eval cell cd5f6f52 to its own cell ---
# After inserting the assertion cell, the indices shifted by 1
eval_idx = next(i for i, c in enumerate(cells04) if c.get('id') == 'cd5f6f52')
eval_cell = cells04[eval_idx]

src = eval_cell['source']
# Find the bootstrap comment line
boot_start = next(
    (i for i, l in enumerate(src) if '# ── Bootstrap 95%' in l),
    None
)
assert boot_start is not None, "Bootstrap comment not found in eval cell"

# Split the source
eval_only_src = src[:boot_start]
# Remove trailing blank line at end of eval_only_src if present
while eval_only_src and eval_only_src[-1].strip() == '':
    eval_only_src.pop()
eval_cell['source'] = eval_only_src

boot_src = src[boot_start:]
boot_cell = new_code_cell(boot_src, cell_id="boot_ci95")

cells04.insert(eval_idx + 1, boot_cell)
print("Task 3: bootstrap CI extracted to own cell after eval cell.")

nb04['cells'] = cells04
with open(path04, 'w', encoding='utf-8') as f:
    json.dump(nb04, f, ensure_ascii=False, indent=1)
print("Saved", path04)


# ─── NOTEBOOK 05 ──────────────────────────────────────────────────────────────
path05 = r'notebooks\05_scoring_output.ipynb'
with open(path05, 'r', encoding='utf-8') as f:
    nb05 = json.load(f)

cells05 = nb05['cells']

# --- Task 5: add NDCG cell after tier assignment cell e7f8a9b0 ---
# Also clean up c2d3e4f5 to remove duplicate NDCG block (it will reference
# ndcg_5pct, ndcg_10pct, ndcg_20pct variables computed in the new cell).

tier_idx = next(i for i, c in enumerate(cells05) if c.get('id') == 'e7f8a9b0')

ndcg_src = [
    "# NDCG@k — ranking quality on full 50,000-member population\n",
    "from sklearn.metrics import ndcg_score\n",
    "\n",
    "labels_ndcg = pd.read_parquet(data_dir / 'features' / 'labels.parquet')\n",
    "y_true_all = labels_ndcg.set_index('member_id').loc[features['member_id'], 'high_acute_risk'].values\n",
    "\n",
    "y_true_2d = y_true_all.reshape(1, -1)\n",
    "y_pred_2d = raw_scores.reshape(1, -1)\n",
    "\n",
    "ndcg_5pct  = ndcg_score(y_true_2d, y_pred_2d, k=int(len(raw_scores) * 0.05))\n",
    "ndcg_10pct = ndcg_score(y_true_2d, y_pred_2d, k=int(len(raw_scores) * 0.10))\n",
    "ndcg_20pct = ndcg_score(y_true_2d, y_pred_2d, k=int(len(raw_scores) * 0.20))\n",
    "\n",
    "print('NDCG ranking quality (full 50k population, raw scores):')\n",
    "print(f'  NDCG@ 5%   {ndcg_5pct:.4f}')\n",
    "print(f'  NDCG@10%   {ndcg_10pct:.4f}')\n",
    "print(f'  NDCG@20%   {ndcg_20pct:.4f}')",
]

ndcg_cell = new_code_cell(ndcg_src, cell_id="ndcg_rank")
cells05.insert(tier_idx + 1, ndcg_cell)
print("Task 5: NDCG cell inserted after tier assignment cell.")

# Update c2d3e4f5: replace the NDCG computation block with references to the
# ndcg_5pct / ndcg_10pct / ndcg_20pct variables already computed above.
sum_idx = next(i for i, c in enumerate(cells05) if c.get('id') == 'c2d3e4f5')
sum_cell = cells05[sum_idx]
src = sum_cell['source']

# Find start of NDCG block in c2d3e4f5
ndcg_block_start = next(
    (i for i, l in enumerate(src) if '# ── NDCG@k evaluation' in l),
    None
)
# Find end — the line after the loop (the line with json.dump, i.e. first occurrence after ndcg_block_start)
if ndcg_block_start is not None:
    # Find the line that saves the JSON (first line after the block)
    save_line_idx = next(
        (i for i, l in enumerate(src) if i > ndcg_block_start and 'scoring_summary.json' in l and 'json.dump' in l),
        None
    )
    assert save_line_idx is not None, "Could not find json.dump line after NDCG block"

    # Replace NDCG block with compact reference lines
    replacement = [
        "# NDCG values computed in the NDCG cell above\n",
        "summary['ndcg_at_5pct']  = round(ndcg_5pct, 4)\n",
        "summary['ndcg_at_10pct'] = round(ndcg_10pct, 4)\n",
        "summary['ndcg_at_20pct'] = round(ndcg_20pct, 4)\n",
        "\n",
    ]
    new_src = src[:ndcg_block_start] + replacement + src[save_line_idx:]
    sum_cell['source'] = new_src
    print("Task 5: scoring summary cell updated to reference ndcg_*pct variables.")

nb05['cells'] = cells05
with open(path05, 'w', encoding='utf-8') as f:
    json.dump(nb05, f, ensure_ascii=False, indent=1)
print("Saved", path05)

print("\nAll done.")
