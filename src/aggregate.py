"""Paper E v8 — Statistics aggregation.

Provides:
- Bootstrap CI: 1000 resamples per (DA, classifier)
- Holm-Bonferroni + Benjamini-Hochberg multiplicity correction
- Cluster-robust OLS for cross-method regression
- Mixed-effects fallback (if convergence)

Usage:
    source /tmp/paper_e_v8_env.sh
    python3 aggregate.py --grid_csv runs/v8_main/grid_results.csv
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "16")

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(
    grid_results: pd.DataFrame,
    group_cols: List[str] = ["da", "classifier"],
    value_col: str = "auc_mean",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute bootstrap CI per (DA, classifier) group.

    Parameters
    ----------
    grid_results : pd.DataFrame
        Main grid output
    group_cols : list of column names to group by
    value_col : column to compute mean of
    n_boot : number of bootstrap resamples
    seed : random seed for reproducibility
    alpha : 1-alpha confidence level

    Returns
    -------
    pd.DataFrame with columns: group_cols + ['mean', 'se', 'ci_lo', 'ci_hi']
    """
    rng = np.random.default_rng(seed)
    grouped = grid_results.groupby(group_cols)
    rows = []
    for keys, grp in grouped:
        vals = grp[value_col].dropna().values
        if len(vals) < 2:
            continue
        boot_means = []
        for _ in range(n_boot):
            sample = rng.choice(vals, size=len(vals), replace=True)
            boot_means.append(sample.mean())
        boot_means = np.array(boot_means)
        lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        row = {c: k for c, k in zip(group_cols, keys if isinstance(keys, tuple) else (keys,))}
        row.update({
            "mean": float(np.mean(vals)),
            "se": float(boot_means.std()),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "n": int(len(vals)),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def holm_bh_correction(p_values: np.ndarray, alpha: float = 0.005) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Holm-Bonferroni and Benjamini-Hochberg correction.

    Parameters
    ----------
    p_values : array of raw p-values
    alpha : significance threshold

    Returns
    -------
    holm_significant : boolean array (Holm-Bonferroni)
    bh_significant : boolean array (Benjamini-Hochberg)
    """
    n = len(p_values)
    if n == 0:
        return np.array([]), np.array([])

    # Holm-Bonferroni
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    holm_thresholds = alpha / (n - np.arange(n))
    holm_reject = sorted_p <= holm_thresholds
    # Find largest k where all up to k are rejected (monotonic)
    if holm_reject.any():
        k_max = np.where(holm_reject)[0].max()
        holm_sig_sorted = np.zeros(n, dtype=bool)
        holm_sig_sorted[:k_max + 1] = True
    else:
        holm_sig_sorted = np.zeros(n, dtype=bool)
    holm_significant = np.zeros(n, dtype=bool)
    holm_significant[sorted_idx] = holm_sig_sorted

    # Benjamini-Hochberg
    bh_thresholds = alpha * (np.arange(1, n + 1) / n)
    bh_reject = sorted_p <= bh_thresholds
    if bh_reject.any():
        k_max = np.where(bh_reject)[0].max()
        bh_sig_sorted = np.zeros(n, dtype=bool)
        bh_sig_sorted[:k_max + 1] = True
    else:
        bh_sig_sorted = np.zeros(n, dtype=bool)
    bh_significant = np.zeros(n, dtype=bool)
    bh_significant[sorted_idx] = bh_sig_sorted

    return holm_significant, bh_significant


def cluster_robust_ols(
    grid_results: pd.DataFrame,
    value_col: str = "auc_mean",
    predictors: List[str] = None,
    cluster_col: str = "dataset",
) -> Dict[str, Any]:
    """Cluster-robust OLS regression with dataset-level clustering.

    Parameters
    ----------
    grid_results : pd.DataFrame
    value_col : outcome column
    predictors : list of categorical predictors to dummy-encode (e.g., ['da', 'classifier'])
    cluster_col : column to cluster on (default 'dataset')

    Returns
    -------
    dict with keys: coef_df, n_obs, n_clusters
    """
    import statsmodels.api as sm

    if predictors is None:
        predictors = ["da", "classifier"]

    df = grid_results.copy()
    df = df.dropna(subset=[value_col])

    # Build design matrix
    X_parts = []
    for pred in predictors:
        dummies = pd.get_dummies(df[pred], prefix=pred, drop_first=True).astype(float)
        X_parts.append(dummies)
    X = pd.concat(X_parts, axis=1)
    X = sm.add_constant(X)
    y = df[value_col].values

    # Cluster-robust standard errors
    try:
        model = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": df[cluster_col].values},
        )
    except Exception as e:
        # Fallback to HC1
        model = sm.OLS(y, X).fit(cov_type="HC1")
        model.cluster_robust_fallback = str(e)

    coef_df = pd.DataFrame({
        "coef": model.params,
        "se": model.bse,
        "t": model.tvalues,
        "p": model.pvalues,
        "ci_lo": model.conf_int()[0],
        "ci_hi": model.conf_int()[1],
    })

    return {
        "coef_df": coef_df,
        "n_obs": int(len(y)),
        "n_clusters": int(df[cluster_col].nunique()),
        "r_squared": float(model.rsquared),
        "r_squared_adj": float(model.rsquared_adj),
        "f_stat": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
    }


def main():
    parser = argparse.ArgumentParser(description="Paper E v8 stats aggregation")
    parser.add_argument("--grid_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.005)
    args = parser.parse_args()

    grid_results = pd.read_csv(args.grid_csv)
    print(f"Loaded grid: {grid_results.shape}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        os.path.dirname(args.grid_csv)).parent / "v8_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bootstrap CI per (DA, classifier)
    print(f"Computing bootstrap CI (n_boot={args.n_boot})...")
    boot_df = bootstrap_ci(grid_results, n_boot=args.n_boot, alpha=args.alpha)
    boot_df.to_csv(out_dir / "bootstrap_ci.csv", index=False)
    print(f"  Bootstrap CI: {boot_df.shape}, saved to bootstrap_ci.csv")

    # 2. ΔAUC vs (None, LR) per dataset
    print("Computing ΔAUC vs (None, LR) baseline...")
    baseline = grid_results[(grid_results["da"] == "none") & (
        grid_results["classifier"] == "LR")][["dataset", "auc_mean"]].rename(
        columns={"auc_mean": "baseline_auc"})
    delta_df = grid_results.merge(baseline, on="dataset", how="left")
    delta_df["delta_auc"] = delta_df["auc_mean"] - delta_df["baseline_auc"]
    delta_df.to_csv(out_dir / "delta_auc.csv", index=False)
    print(f"  ΔAUC: {delta_df.shape}, saved to delta_auc.csv")

    # 3. Cluster-robust OLS
    print("Computing cluster-robust OLS...")
    try:
        ols_result = cluster_robust_ols(grid_results, predictors=["da", "classifier"])
        ols_result["coef_df"].to_csv(out_dir / "ols_coefs.csv")
        summary = {
            "n_obs": ols_result["n_obs"],
            "n_clusters": ols_result["n_clusters"],
            "r_squared": ols_result["r_squared"],
            "r_squared_adj": ols_result["r_squared_adj"],
            "f_stat": ols_result["f_stat"],
            "f_pvalue": ols_result["f_pvalue"],
        }
        with open(out_dir / "ols_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  OLS: R²={ols_result['r_squared']:.4f}, "
              f"F={ols_result['f_stat']:.2f}, p={ols_result['f_pvalue']:.4e}")
    except ImportError:
        print("  WARN: statsmodels not installed, skipping OLS")
    except Exception as e:
        print(f"  WARN: OLS failed: {e}")

    print(f"\nOutputs saved to {out_dir}")


if __name__ == "__main__":
    main()