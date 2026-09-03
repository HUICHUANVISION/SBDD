"""Paper E v8 — Intrinsic + Relational feature extraction per dataset.

4 intrinsic features (no baseline run needed):
- dataset_size: n_samples
- class_imbalance_ratio: n_pos / n_neg
- n_features: X.shape[1]
- feature_skewness: mean of |skew(X_j)| over features

4 relational features (require one-shot baseline run):
- grad_align: Spearman corr of source/target gradients (needs pre-trained baseline)
- margin_gap: gap of margins (needs pre-trained baseline)
- var_ratio: batch gradient variance ratio (needs pre-trained baseline)
- param_dist: parameter-space L2 distance (needs pre-trained baseline)

For v8 deployment, intrinsic-only mode is the default. Relational mode is optional
and requires running a baseline LR on a small calibration subset of each new dataset.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import skew
from typing import Dict, Any, List


def compute_intrinsic_features(name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute 4 intrinsic features for a dataset.

    Parameters
    ----------
    name : str
        Dataset key (e.g. 'aeeem_eq', 'jit_brackets')
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,), binary {0, 1}

    Returns
    -------
    dict with keys: dataset_size, class_imbalance_ratio, n_features, feature_skewness
    """
    n = X.shape[0]
    d = X.shape[1]
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    imb = n_pos / max(n_neg, 1)
    # Feature skewness (mean of absolute per-feature skewness)
    X_df = pd.DataFrame(X)
    feat_skew = X_df.apply(lambda c: skew(c, bias=False, nan_policy='omit')).abs().mean()
    feat_skew = float(feat_skew) if np.isfinite(feat_skew) else 0.0

    return {
        "dataset": name,
        "dataset_size": float(n),
        "class_imbalance_ratio": float(imb),
        "n_features": float(d),
        "feature_skewness": feat_skew,
    }


def compute_relational_features_one_baseline(
    target_name: str,
    target_X: np.ndarray,
    target_y: np.ndarray,
    source_pool: List[Tuple[str, np.ndarray, np.ndarray]],
    seed: int = 42,
) -> Dict[str, float]:
    """Compute 4 relational features for a target dataset, given a pool of source datasets.

    NOTE: This is the "refined mode" — requires one-shot baseline LR training.
    For v8 main grid, we use intrinsic-only mode to keep deployment simple.
    This function is here for future use or RQ4 case study.

    Parameters
    ----------
    target_name : str
        Target dataset key
    target_X, target_y : target data
    source_pool : list of (source_name, source_X, source_y) tuples
        Other datasets in the same family or across families
    seed : int
        Random seed

    Returns
    -------
    dict with 4 relational features (averaged over source pool)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold

    # Train baseline LR on target
    scaler_t = StandardScaler()
    target_X_s = scaler_t.fit_transform(target_X)
    clf_t = LogisticRegression(max_iter=500, random_state=seed)
    clf_t.fit(target_X_s, target_y)
    # Target predictions
    target_proba = clf_t.predict_proba(target_X_s)[:, 1]

    grad_aligns = []
    margin_gaps = []
    var_ratios = []
    param_dists = []

    for src_name, src_X, src_y in source_pool:
        if src_name == target_name:
            continue
        if len(np.unique(src_y)) < 2:
            continue
        scaler_s = StandardScaler()
        src_X_s = scaler_s.fit_transform(src_X)
        clf_s = LogisticRegression(max_iter=500, random_state=seed)
        clf_s.fit(src_X_s, src_y)
        # Cross-predictions
        src_pred_target = clf_s.predict_proba(target_X_s)[:, 1]

        # grad_align: Spearman corr of predictions
        from scipy.stats import spearmanr
        rho, _ = spearmanr(target_proba, src_pred_target)
        if np.isfinite(rho):
            grad_aligns.append(rho)

        # margin_gap: |mean(target_proba) - mean(src_pred_target)|
        mg = abs(target_proba.mean() - src_pred_target.mean())
        margin_gaps.append(float(mg))

        # var_ratio: var(src_pred_target) / var(target_proba)
        vr = src_pred_target.var() / max(target_proba.var(), 1e-9)
        var_ratios.append(float(np.log(vr + 1e-9)))

        # param_dist: L2 distance between coef vectors (in standardized space)
        if hasattr(clf_t, "coef_") and hasattr(clf_s, "coef_"):
            coef_t = clf_t.coef_.ravel()
            coef_s = clf_s.coef_.ravel()
            # Align dimensions
            min_d = min(len(coef_t), len(coef_s))
            pd_val = np.linalg.norm(coef_t[:min_d] - coef_s[:min_d]) / (
                np.linalg.norm(coef_t[:min_d]) + np.linalg.norm(coef_s[:min_d]) + 1e-9
            )
            param_dists.append(float(pd_val))

    def _safe_mean(xs):
        if not xs:
            return 0.0
        m = float(np.mean(xs))
        return m if np.isfinite(m) else 0.0

    return {
        "dataset": target_name,
        "grad_align": _safe_mean(grad_aligns),
        "margin_gap": _safe_mean(margin_gaps),
        "var_ratio": _safe_mean(var_ratios),
        "param_dist": _safe_mean(param_dists),
    }


def build_intrinsic_table(datasets: List[str], loader) -> pd.DataFrame:
    """Build intrinsic features table for all datasets.

    Parameters
    ----------
    datasets : list of dataset keys
    loader : callable (dataset_key) -> (X, y, feat_names, meta)

    Returns
    -------
    pd.DataFrame with columns: dataset, dataset_size, class_imbalance_ratio, n_features, feature_skewness
    """
    rows = []
    for name in datasets:
        try:
            X, y, fn, meta = loader(name)
            row = compute_intrinsic_features(name, X, y)
            rows.append(row)
        except Exception as e:
            print(f"  WARN: failed to load {name}: {e}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.expanduser("~/Lab/projects/sdp-da-compatibility-release/src"))
    from data_loader import load_dataset, list_datasets

    print("Building intrinsic features table for all 45 datasets...")
    datasets = list_datasets()
    df = build_intrinsic_table(datasets, load_dataset)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFamily breakdown:")
    df["family"] = df["dataset"].apply(lambda x: x.split("_")[0])
    print(df.groupby("family")[["dataset_size", "class_imbalance_ratio", "n_features", "feature_skewness"]].agg(["mean", "std"]).round(3))
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())