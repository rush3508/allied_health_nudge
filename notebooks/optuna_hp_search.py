"""
Optuna Hyperparameter Search — allied-health-nudge
===================================================
Purpose: Validate or refute the AUC ceiling claim (~0.745).
         A rigorous 50-trial Optuna search probes whether the current
         ROC-AUC of 0.7429 is near the data's intrinsic ceiling or
         whether better hyperparameters can push it meaningfully higher.

Designed to drop into 04_model_training.ipynb as a new cell after Step 5
or as a standalone script.  Uses the same train/val/test split and
LightGBM Dataset objects from the notebook.

Author: Hermes Agent
Date:   2026-06-01
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
import json, time, warnings, os
warnings.filterwarnings("ignore", category=UserWarning)


# ═══════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

N_TRIALS        = 50          # Optuna trials (50 gives good coverage without excess cost)
N_SPLITS        = 5           # 5-fold stratified CV on the training set
TIMEOUT_MINUTES = None        # Set to e.g. 120 for a hard wall-clock limit
RANDOM_SEED     = 42
OUTPUT_DIR      = Path(__file__).resolve().parent.parent / "outputs"  # adjust for your tree
STUDY_DB         = str(OUTPUT_DIR / "optuna_study.db")  # SQLite for resume capability

# ── These should already exist from the notebook ──
# X_train, X_val, y_train, y_val, X_test, y_test
# train_data, val_data   (LightGBM Dataset objects)
# CAT_COLS, spw          (categorical cols, scale_pos_weight)


# ═══════════════════════════════════════════════════════════════════
# 1. HYPERPARAMETER SEARCH SPACE
# ═══════════════════════════════════════════════════════════════════
#
# DESIGN RATIONALE:
# ─────────────────
# We have a moderately-imbalanced binary classification (~17.6% positive)
# with 50k samples and 33 features (5 categorical).  LightGBM's tree-growing
# strategy is controlled by three interacting families:
#
#   A. STRUCTURE   (num_leaves, min_child_samples, max_depth)
#      - Controls model capacity / overfitting risk.
#      - num_leaves is the primary knob; max_depth is a hard cap.
#      - min_child_samples prevents splitting on noise (anti-overfitting).
#
#   B. REGULARISATION (reg_alpha, reg_lambda, min_split_gain)
#      - L1 (alpha) and L2 (lambda) weight penalties.
#      - min_split_gain is the tree-split equivalent of a p-value threshold.
#
#   C. SUBSAMPLING (feature_fraction, bagging_fraction, bagging_freq)
#      - Row and column subsampling for stochastic regularisation.
#      - Also impacts speed: smaller fractions = faster iterations.
#
#   D. IMBALANCE (scale_pos_weight)
#      - We already compute this from the data; Optuna can tune a multiplier
#        around the computed value to see if adjusting the weight helps.
#
# The ranges are based on:
#   - LightGBM official tuning guide
#   - Empirical best-practice for imbalanced binary tasks with n=30k-50k
#   - Conservative upper bounds to avoid massive trees on 6-core machine
#
# For a 50-trial budget we use TPESampler (Tree-structured Parzen Estimator)
# which converges faster than random search by modelling good vs bad trials.

def suggest_params(trial: optuna.Trial, base_spw: float) -> dict:
    """Suggest a complete LightGBM parameter set for one trial.

    Parameters
    ----------
    trial    : optuna.Trial
    base_spw : float
        scale_pos_weight computed from the training split (neg/pos ratio).
    """

    # ── A. Structure ──────────────────────────────────────────
    num_leaves = trial.suggest_int("num_leaves", 15, 127)
    # ^ 15-127: below 15 = too simple; above 127 = rarely needed for 33 features.
    #   Default 31.  Wider range lets TPE explore complexity.

    min_child_samples = trial.suggest_int("min_child_samples", 10, 200, log=True)
    # ^ 10-200 (log-uniform): smaller -> splits easier -> more complex trees.
    #   Default 20.  log=True because the effect is multiplicative.

    max_depth = trial.suggest_int("max_depth", 3, 12)
    # ^ 3-12: soft cap on tree depth.  -1 (unlimited) is dangerous with
    #   high num_leaves; a hard cap prevents runaway trees.  Default -1.

    # ── B. Regularisation ─────────────────────────────────────
    reg_alpha = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True)
    # ^ 1e-4 - 10.0: L1 regularisation.  log-uniform because optimal values
    #   often cluster near 0.01-1.0.  Default 0.0.

    reg_lambda = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True)
    # ^ Same range as alpha.  L2 regularisation.  Default 0.0.

    min_split_gain = trial.suggest_float("min_split_gain", 0.0, 1.0)
    # ^ 0.0-1.0: minimum loss reduction to split.  0 = no threshold.
    #   Default 0.0.  Values >0.5 act as strong anti-overfitting.

    # ── C. Subsampling ────────────────────────────────────────
    feature_fraction = trial.suggest_float("feature_fraction", 0.5, 0.95)
    # ^ 0.5-0.95: fraction of features used per tree.  Below 0.5 = too
    #   aggressive; above 0.95 = almost no subsampling.  Default 1.0.

    bagging_fraction = trial.suggest_float("bagging_fraction", 0.5, 0.95)
    # ^ 0.5-0.95: fraction of rows used per iteration.  Default 1.0.

    bagging_freq = trial.suggest_int("bagging_freq", 1, 10)
    # ^ 1-10: bag every N iterations.  freq=1 with fraction<1 = full
    #   bagging.  freq>1 = subsample less often (faster but less regularised).
    #   Default 0 (off).

    # ── D. Imbalance ──────────────────────────────────────────
    spw_multiplier = trial.suggest_float("spw_multiplier", 0.5, 2.0)
    # ^ 0.5-2.0: multiplier around the computed scale_pos_weight.
    #   0.5 = less weight on positives (fewer false positives, more recall at
    #         cost of lower precision in the positive class).
    #   2.0 = stronger weight (aggressively penalise missing positives).
    #   This explores the sensitivity frontier.

    # ── Fixed params (not tuned) ─────────────────────────────
    params = {
        "objective":          "binary",
        "metric":             "auc",          # AUC for early stopping
        "boosting_type":      "gbdt",
        "learning_rate":      0.02,           # slightly higher for faster trials
        "num_leaves":         num_leaves,
        "min_child_samples":  min_child_samples,
        "max_depth":          max_depth,
        "reg_alpha":          reg_alpha,
        "reg_lambda":         reg_lambda,
        "min_split_gain":     min_split_gain,
        "feature_fraction":   feature_fraction,
        "bagging_fraction":   bagging_fraction,
        "bagging_freq":       bagging_freq,
        "scale_pos_weight":   base_spw * spw_multiplier,
        "verbose":            -1,
        "n_jobs":             -1,
        "seed":               RANDOM_SEED,
        "deterministic":      True,
    }

    return params


# ═══════════════════════════════════════════════════════════════════
# 2. OBJECTIVE FUNCTION
# ═══════════════════════════════════════════════════════════════════

def objective(trial: optuna.Trial,
              X_train: pd.DataFrame,
              y_train: pd.Series,
              cat_cols: list,
              base_spw: float,
              n_splits: int = 5) -> float:
    """Optuna objective: mean validation AUC across stratified K-fold CV.

    Uses 5-fold CV on the *training* set.  This is the gold-standard approach:
    the Optuna search never touches the hold-out val or test sets.  After the
    study completes, we train the best params on full train set and evaluate
    once on val and test.

    Returns
    -------
    float
        Mean ROC-AUC across folds.  Optuna maximises this.
    """

    params = suggest_params(trial, base_spw)

    # ── 5-fold stratified CV ──────────────────────────────────
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    auc_scores = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        # Rebuild datasets for each fold (must convert categoricals)
        for col in cat_cols:
            X_tr[col] = X_tr[col].astype("category")
            X_va[col] = X_va[col].astype("category")

        dtrain = lgb.Dataset(X_tr, label=y_tr,
                             categorical_feature=cat_cols,
                             free_raw_data=False)
        dvalid = lgb.Dataset(X_va, label=y_va,
                             categorical_feature=cat_cols,
                             reference=dtrain,
                             free_raw_data=False)

        # ── Train ─────────────────────────────────────────────
        model = lgb.train(
            params,
            train_set=dtrain,
            valid_sets=[dvalid],
            valid_names=["val"],
            num_boost_round=2000,
            callbacks=[
                lgb.early_stopping(50, first_metric_only=True, verbose=False),
                # ^ shorter patience (50 vs 100) = faster trials; we want
                #   relative comparison, not final accuracy per trial.
            ],
        )

        # ── Evaluate ──────────────────────────────────────────
        y_pred = model.predict(X_va, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_va, y_pred)
        auc_scores.append(auc)

        # ── Optuna pruning (optional) ─────────────────────────
        # Report intermediate value after each fold; Optuna can prune
        # unpromising trials early.
        trial.report(np.mean(auc_scores), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # ── Clean up ──────────────────────────────────────────
        del dtrain, dvalid, model

    mean_auc = np.mean(auc_scores)
    std_auc  = np.std(auc_scores)

    # Store fold-level detail as user attribute
    trial.set_user_attr("cv_auc_mean", float(mean_auc))
    trial.set_user_attr("cv_auc_std",  float(std_auc))
    trial.set_user_attr("cv_auc_folds", [float(x) for x in auc_scores])
    trial.set_user_attr("params", params)

    return mean_auc


# ═══════════════════════════════════════════════════════════════════
# 3. RUN THE STUDY
# ═══════════════════════════════════════════════════════════════════

def run_optuna_search(X_train, y_train, cat_cols, base_spw,
                      n_trials=N_TRIALS, n_splits=N_SPLITS,
                      timeout_minutes=TIMEOUT_MINUTES):
    """Run the Optuna hyperparameter search.

    Parameters
    ----------
    X_train, y_train : training data from notebook Step 3
    cat_cols         : list of categorical column names
    base_spw         : scale_pos_weight from notebook Step 4
    n_trials         : number of Optuna trials (default 50)
    n_splits         : CV folds (default 5)
    timeout_minutes  : optional wall-clock timeout

    Returns
    -------
    optuna.Study
        The completed study object.
    """

    # ── Create study with SQLite storage for resume capability ─
    storage = f"sqlite:///{STUDY_DB}"
    study_name = "lgbm_hp_search_v1"

    # TPE sampler with multivariate=True for better interaction modelling
    sampler = TPESampler(seed=RANDOM_SEED, multivariate=True)

    # Median pruner: kill trials in the bottom half after each fold
    pruner = MedianPruner(
        n_startup_trials=5,      # don't prune first 5 trials
        n_warmup_steps=2,        # wait for 2 folds before pruning
        interval_steps=1,        # check every fold
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,     # resume if crashed
    )

    # ── Objective wrapper ─────────────────────────────────────
    def _obj(trial):
        return objective(trial, X_train, y_train, cat_cols, base_spw, n_splits)

    # ── Run ───────────────────────────────────────────────────
    t_start = time.time()

    study.optimize(
        _obj,
        n_trials=n_trials,
        timeout=None if timeout_minutes is None else timeout_minutes * 60,
        show_progress_bar=True,
        n_jobs=1,  # thread-safety: use 1 with LightGBM's internal parallelism
    )

    t_elapsed = time.time() - t_start

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  OPTUNA HYPERPARAMETER SEARCH - RESULTS")
    print("=" * 70)
    print(f"  Trials completed:     {len(study.trials)}")
    print(f"  Pruned trials:        {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)}")
    print(f"  Elapsed time:         {t_elapsed/60:.1f} min ({t_elapsed:.0f} sec)")
    print(f"  Best CV AUC:          {study.best_value:.4f}")
    print(f"  Best trial number:    {study.best_trial.number}")
    print("-" * 70)
    print("  Best hyperparameters:")
    for k, v in sorted(study.best_params.items()):
        print(f"    {k:<25s}  {v}")
    print("-" * 70)
    print(f"  CV fold AUCs (best trial): {study.best_trial.user_attrs.get('cv_auc_folds', 'N/A')}")
    print(f"  CV AUC std:                 {study.best_trial.user_attrs.get('cv_auc_std', 'N/A'):.4f}")
    print("=" * 70)

    return study


# ═══════════════════════════════════════════════════════════════════
# 4. RETRAIN WITH BEST PARAMS & EVALUATE
# ═══════════════════════════════════════════════════════════════════

def retrain_and_evaluate(study, X_train, y_train, X_val, y_val, X_test, y_test,
                         cat_cols, base_spw):
    """Train final model with best hyperparameters on full train set,
    evaluate on val and test.

    Returns
    -------
    dict
        Final evaluation metrics.
    """

    best_params = study.best_params.copy()

    # Reconstruct full params
    full_params = {
        "objective":        "binary",
        "metric":           "auc",
        "boosting_type":    "gbdt",
        "learning_rate":    0.01,          # lower LR for final model
        "num_leaves":       best_params["num_leaves"],
        "min_child_samples": best_params["min_child_samples"],
        "max_depth":        best_params.get("max_depth", -1),
        "reg_alpha":        best_params["reg_alpha"],
        "reg_lambda":       best_params["reg_lambda"],
        "min_split_gain":   best_params["min_split_gain"],
        "feature_fraction": best_params["feature_fraction"],
        "bagging_fraction": best_params["bagging_fraction"],
        "bagging_freq":     best_params["bagging_freq"],
        "scale_pos_weight": base_spw * best_params["spw_multiplier"],
        "verbose":          -1,
        "n_jobs":           -1,
        "seed":             RANDOM_SEED,
        "deterministic":    True,
    }

    # Build fresh datasets
    train_data = lgb.Dataset(X_train, label=y_train,
                             categorical_feature=cat_cols,
                             free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val,
                           categorical_feature=cat_cols,
                           reference=train_data,
                           free_raw_data=False)

    model = lgb.train(
        full_params,
        train_set=train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(100, first_metric_only=True, verbose=True),
            lgb.log_evaluation(50),
        ],
    )

    # ── Test evaluation ───────────────────────────────────────
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    auc  = roc_auc_score(y_test, y_pred)
    prauc = average_precision_score(y_test, y_pred)

    # Business metric: recall at top 20%
    test_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
    test_df = test_df.sort_values("y_pred", ascending=False).reset_index(drop=True)
    top20 = test_df.head(int(len(test_df) * 0.20))
    recall_top20 = top20["y_true"].sum() / y_test.sum()
    precision_top20 = top20["y_true"].mean()

    results = {
        "roc_auc":              round(auc, 4),
        "pr_auc":               round(prauc, 4),
        "recall_top20pct":      round(recall_top20, 4),
        "precision_top20pct":   round(precision_top20, 4),
        "best_iteration":       model.best_iteration,
        "best_cv_auc":          round(study.best_value, 4),
        "best_trial":           study.best_trial.number,
        "best_params":          best_params,
        "full_params":          {k: v for k, v in full_params.items()
                                 if k not in ("verbose", "n_jobs", "seed")},
    }

    print("\n" + "=" * 70)
    print("  FINAL MODEL - TEST SET EVALUATION")
    print("=" * 70)
    print(f"  Best CV AUC (5-fold):     {results['best_cv_auc']:.4f}")
    print(f"  Test ROC-AUC:             {results['roc_auc']:.4f}")
    print(f"  Test PR-AUC:              {results['pr_auc']:.4f}")
    print(f"  Test Recall@top20%:       {results['recall_top20pct']:.4f}")
    print(f"  Test Precision@top20%:    {results['precision_top20pct']:.4f}")
    print(f"  Best iteration:           {results['best_iteration']}")
    print("=" * 70)

    # ── Save ──────────────────────────────────────────────────
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "optuna_best_params.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved best params to {out_dir / 'optuna_best_params.json'}")

    return results


# ═══════════════════════════════════════════════════════════════════
# 5. COST ESTIMATE
# ═══════════════════════════════════════════════════════════════════
#
# Platform: v530s (Intel i5-9400 @ 2.90 GHz, 6 cores, 15 GB RAM)
# Data:     30k train x 33 features, 5-fold CV
# Task:     50 Optuna trials
#
# PER-TRIAL BREAKDOWN:
#   - 5 folds x ~1 LightGBM train per fold
#   - Each train: ~150-300 boosting rounds (early stopping with patience=50)
#   - ~1.5-3 seconds per training run on 6 cores with 30k samples
#   - Total per trial: ~5 x 2.5s = ~12.5 seconds
#   - Plus overhead (data prep, Optuna bookkeeping): ~3s
#   - ~15-16 seconds per trial
#
# TOTAL ESTIMATE (50 trials):
#   - 50 x 16s = ~800 seconds ~ 13 minutes
#   - Pruning (MedianPruner) may cut ~20-30% of trials early
#   - Realistic: 10-15 minutes wall-clock
#
# WITH HIGHER LEARNING RATE (0.02 used during search):
#   - Fewer boosting rounds per fold
#   - Per-trial time drops to ~8-10 seconds
#   - Total: ~7-9 minutes
#
# MEMORY: ~2-4 GB peak (LightGBM datasets + Optuna storage)
#          Comfortably within the 15 GB available.
#
# RECOMMENDATION: Run with learning_rate=0.02 during search (as coded above),
# then retrain final model with learning_rate=0.01 for best accuracy.
#
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# 6. NOTEBOOK INTEGRATION SNIPPET
# ═══════════════════════════════════════════════════════════════════
#
# Paste the cells below into 04_model_training.ipynb AFTER Step 5
# (after the LightGBM Dataset objects are created) and BEFORE Step 6.
#
# --- Cell A: Install/import ---
# !pip install optuna  # run once
# from optuna_hp_search import suggest_params, objective, run_optuna_search, retrain_and_evaluate
#
# --- Cell B: Run search ---
# study = run_optuna_search(
#     X_train=X_train, y_train=y_train,
#     cat_cols=CAT_COLS, base_spw=spw,
#     n_trials=50, n_splits=5,
# )
#
# --- Cell C: Retrain & evaluate ---
# final_results = retrain_and_evaluate(
#     study,
#     X_train=X_train, y_train=y_train,
#     X_val=X_val, y_val=y_val,
#     X_test=X_test, y_test=y_test,
#     cat_cols=CAT_COLS, base_spw=spw,
# )
#
# --- Cell D: Compare with baseline ---
# baseline = json.load(open(outputs_dir / "eval_metrics.json"))
# print(f"Baseline AUC:  {baseline['roc_auc']:.4f}")
# print(f"Optuna   AUC:  {final_results['roc_auc']:.4f}")
# print(f"Delta:         {final_results['roc_auc'] - baseline['roc_auc']:+.4f}")
# if final_results['roc_auc'] <= baseline['roc_auc'] + 0.005:
#     print("-> AUC CEILING SUPPORTED: hyperparameter search did not meaningfully improve AUC.")
# else:
#     print("-> AUC CEILING REFUTED: better hyperparameters found a higher AUC.")
#
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# 7. SENSITIVITY ANALYSIS (optional, run after main study)
# ═══════════════════════════════════════════════════════════════════

def sensitivity_sweep(best_params, base_spw, X_train, y_train, X_val, y_val,
                      cat_cols):
    """One-factor-at-a-time sweep around the best params.
    Varies each parameter +-20% to show which are most sensitive.
    """

    key_params = ["num_leaves", "min_child_samples", "reg_alpha",
                  "reg_lambda", "feature_fraction", "bagging_fraction"]

    print("\nSensitivity sweep (+-20% around best params):")
    print("-" * 65)
    print(f"{'Parameter':<25s} {'-20% AUC':>10s} {'Best AUC':>10s} {'+20% AUC':>10s} {'Sensitivity':>10s}")
    print("-" * 65)

    sensitivities = {}
    for param in key_params:
        best_val = best_params[param]

        aucs = []
        for factor in [0.8, 1.0, 1.2]:
            test_params = {
                "objective": "binary", "metric": "auc",
                "learning_rate": 0.02,
                "num_leaves": best_params["num_leaves"],
                "min_child_samples": best_params["min_child_samples"],
                "max_depth": best_params.get("max_depth", -1),
                "reg_alpha": best_params["reg_alpha"],
                "reg_lambda": best_params["reg_lambda"],
                "min_split_gain": best_params["min_split_gain"],
                "feature_fraction": best_params["feature_fraction"],
                "bagging_fraction": best_params["bagging_fraction"],
                "bagging_freq": best_params["bagging_freq"],
                "scale_pos_weight": base_spw * best_params["spw_multiplier"],
                "verbose": -1, "n_jobs": -1, "seed": 42, "deterministic": True,
            }

            if isinstance(best_val, float):
                test_params[param] = best_val * factor
            else:
                test_params[param] = int(round(best_val * factor))

            dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols,
                                free_raw_data=False)
            dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols,
                              reference=dtrain, free_raw_data=False)

            model = lgb.train(test_params, dtrain,
                            valid_sets=[dval], valid_names=["val"],
                            num_boost_round=1000,
                            callbacks=[lgb.early_stopping(30, verbose=False)])

            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            aucs.append(roc_auc_score(y_val, y_pred))

        sensitivity = max(abs(aucs[0] - aucs[1]), abs(aucs[2] - aucs[1]))
        sensitivities[param] = sensitivity

        print(f"{param:<25s} {aucs[0]:>10.4f} {aucs[1]:>10.4f} {aucs[2]:>10.4f} {sensitivity:>10.4f}")

    print("-" * 65)
    print("Parameters ranked by sensitivity:")
    for param, sens in sorted(sensitivities.items(), key=lambda x: -x[1]):
        print(f"  {param:<25s} {sens:.4f}")

    return sensitivities


# ═══════════════════════════════════════════════════════════════════
# 8. HYPERPARAMETER IMPORTANCE (Optuna built-in)
# ═══════════════════════════════════════════════════════════════════

def plot_importance(study: optuna.Study):
    """Print Optuna hyperparameter importance (how much each param
    contributed to the objective value).  Uses Optuna's built-in
    fANOVA importance."""
    try:
        importance = optuna.importance.get_param_importances(study)
        print("\nHyperparameter importance (Optuna fANOVA):")
        print("-" * 50)
        for param, imp in sorted(importance.items(), key=lambda x: -x[1]):
            bar = "#" * int(imp * 40)
            print(f"  {param:<25s}  {imp:.3f}  {bar}")
        print("-" * 50)
    except Exception as e:
        print(f"Could not compute importance: {e}")


# ═══════════════════════════════════════════════════════════════════
# END
# ═══════════════════════════════════════════════════════════════════
