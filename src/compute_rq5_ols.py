#!/usr/bin/env python3
"""Cluster-robust OLS for RQ5: SDP-native signals vs intrinsic features."""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJ = Path("/Users/moss/Lab/projects/sdp-research-program/papers/paper_e_da_compatibility/v9")
RUNS = PROJ / "runs" / "v15_full_nested_cv"

# ── Load inputs ──────────────────────────────────────────────────────────────
signals    = pd.read_csv(RUNS / "rq5_signals.csv")
family_sum = pd.read_csv(RUNS / "family_summary.csv")

# family from signals is lowercase AEEEM; family_sum has uppercase
# signals cols: dataset, family, a1_dts, a2_chc, a3_ccp, a4_dle, n_samples, n_features, imbalance
# family_sum cols: family, dataset, n_outer_folds, mean_auc_at_0_5, ... (selected candidate metrics)

# ── Compute outer-realized delta AUC from per_dataset CSVs ───────────────────
# per_dataset/{ds}.csv has baseline_auc_at_0_5, delta_auc_at_0_5 per fold
# We aggregate by taking the mean of per-fold delta values (leakage-safe)
delta_rows = []
for ds_row in family_sum.itertuples():
    ds = ds_row.dataset
    csv_path = RUNS / "per_dataset" / f"{ds}.csv"
    if csv_path.exists():
        df_fold = pd.read_csv(csv_path)
        delta_rows.append({
            "dataset": ds,
            "baseline_auc_at_0_5":   df_fold["baseline_auc_at_0_5"].mean(),
            "delta_auc_at_0_5":      df_fold["delta_auc_at_0_5"].mean(),
            "baseline_f1_at_0_5":    df_fold["baseline_f1_at_0_5"].mean(),
            "delta_f1_at_0_5":       df_fold["delta_f1_at_0_5"].mean(),
            "baseline_g_at_0_5":     df_fold["baseline_g_measure_at_0_5"].mean(),
            "delta_g_at_0_5":       df_fold["delta_g_measure_at_0_5"].mean(),
            "baseline_mcc_at_0_5":   df_fold["baseline_mcc_at_0_5"].mean(),
            "delta_mcc_at_0_5":     df_fold["delta_mcc_at_0_5"].mean(),
            # youden
            "baseline_auc_youden":    df_fold["baseline_auc_youden"].mean(),
            "delta_auc_youden":      df_fold["delta_auc_youden"].mean(),
            "baseline_f1_youden":     df_fold["baseline_f1_youden"].mean(),
            "delta_f1_youden":       df_fold["delta_f1_youden"].mean(),
            "baseline_g_youden":     df_fold["baseline_g_measure_youden"].mean(),
            "delta_g_youden":       df_fold["delta_g_measure_youden"].mean(),
            "baseline_mcc_youden":  df_fold["baseline_mcc_youden"].mean(),
            "delta_mcc_youden":     df_fold["delta_mcc_youden"].mean(),
        })
    else:
        delta_rows.append({"dataset": ds})

delta_df = pd.DataFrame(delta_rows)

# ── Merge ─────────────────────────────────────────────────────────────────────
# signals has family_y (uppercase from family_from_key), family_sum has family_x (uppercase from CSV)
merged = family_sum.merge(signals, on="dataset", how="inner")
merged = merged.merge(delta_df, on="dataset", how="left")
# merged now has: family_x (from family_summary), family_y (from signals), dataset, mean_*, signals, deltas

# family_cluster
family_id_map = {'aeeem': 0, 'jira': 1, 'nasa': 2, 'promise': 3, 'relink': 4, 'softlab': 5}
merged["family_cluster"] = merged["family_x"].str.lower().map(family_id_map)
print(f"Merged: {len(merged)} rows. Columns: {merged.columns.tolist()}")
print(merged[["dataset","family_x","delta_auc_at_0_5","delta_f1_at_0_5"]].head(5).to_string())

# ── Signal correlation matrix ─────────────────────────────────────────────────
signal_cols = ["a1_dts", "a2_chc", "a3_ccp", "a4_dle"]
sig_corr = merged[signal_cols].corr()
sig_corr.to_csv(RUNS / "rq5_signal_correlations.csv")
print("\nSignal correlation matrix:\n", sig_corr.round(4).to_string())

# ── Cluster-robust OLS ───────────────────────────────────────────────────────
import statsmodels.api as sm

outcomes = {
    "delta_auc_at_0_5":  merged["delta_auc_at_0_5"].values,
    "delta_f1_at_0_5":   merged["delta_f1_at_0_5"].values,
    "delta_g_at_0_5":    merged["delta_g_at_0_5"].values,
    "delta_mcc_at_0_5":  merged["delta_mcc_at_0_5"].values,
}

X_specs = {
    "null":      [],
    "signals":   signal_cols,
    "intrinsic": ["n_samples", "n_features", "imbalance"],
    "both":      signal_cols + ["n_samples", "n_features", "imbalance"],
}

all_results = []
for outcome_name, y_arr in outcomes.items():
    y_clean = pd.Series(y_arr).replace([np.inf, -np.inf], np.nan).dropna()
    valid_idx = y_clean.index
    clusters = merged.loc[valid_idx, "family_cluster"].values.astype(int)

    for model_name, X_cols in X_specs.items():
        if X_cols:
            X = sm.add_constant(merged.loc[valid_idx, X_cols])
        else:
            X = sm.add_constant(pd.DataFrame(index=merged.loc[valid_idx].index))
        y_series = y_clean.values
        try:
            mod = sm.OLS(y_series, X.values)
            res = mod.fit(cov_type='cluster', cov_kwds={'groups': clusters})
            # Build param dict keyed by const / variable names
            param_keys = list(res.params.keys()) if hasattr(res.params, 'keys') else []
            # X columns after add_constant (first is 'const')
            all_cols = (['const'] if X_cols else ['const']) + list(X_cols)

            def _get(key, arr):
                try:
                    if isinstance(arr, np.ndarray):
                        idx = all_cols.index(key) if key in all_cols else -1
                        return float(arr[idx]) if idx >= 0 else np.nan
                except Exception:
                    pass
                return np.nan

            row = {
                "outcome": outcome_name,
                "model": model_name,
                "n_obs": int(res.nobs),
                "r2_adj": float(res.rsquared_adj),
                "aic": float(res.aic),
                "bic": float(res.bic),
                "f_stat": float(res.fvalue) if res.fvalue not in [None, np.nan] else np.nan,
                "f_pvalue": float(res.f_pvalue) if res.f_pvalue not in [None, np.nan] else np.nan,
            }
            for c in X_cols:
                idx = X_cols.index(c) + 1  # +1 because const at index 0
                row[f"coef_{c}"] = float(res.params[idx]) if idx < len(res.params) else np.nan
                row[f"se_{c}"]   = float(res.bse[idx])    if idx < len(res.bse)    else np.nan
                row[f"p_{c}"]    = float(res.pvalues[idx]) if idx < len(res.pvalues) else np.nan
            all_results.append(row)
            print(f"  {outcome_name}/{model_name}: n={res.nobs}, R2_adj={res.rsquared_adj:.4f}, "
                  f"AIC={res.aic:.2f}, BIC={res.bic:.2f}")
        except Exception as e:
            print(f"  [WARN] {outcome_name}/{model_name}: {e}", file=sys.stderr)

results_df = pd.DataFrame(all_results)
results_df.to_csv(RUNS / "rq5_ols_results.csv", index=False)
print(f"\nSaved OLS results: {len(results_df)} rows")

# ── Save merged CSV ────────────────────────────────────────────────────────────
merged.to_csv(RUNS / "rq5_merged.csv", index=False)
print("Saved rq5_merged.csv")
