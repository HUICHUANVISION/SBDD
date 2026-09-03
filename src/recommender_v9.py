"""Paper E v9 — Signal-aware Recommender System.

Uses SDP-native signals (A1-DTS, A2-CHC, A3-CCP, A4-DLE) to recommend
best (DA, specialist) pair per dataset.

Architecture:
- Input features (per dataset):
    * Dataset intrinsic: n_samples, n_features, class_imb_ratio
    * A1 DTS: dts_hal, dts_log, dts_mix, dts_type (one-hot)
    * A2 CHC: chc_nf, chc_ns, chc_la, chc_ld, chc_lt, chc_ndev, chc_age, chc_nuc, chc_nd, chc_exp, chc_rexp, chc_sexp
    * A3 CCP: ccp_halstead_V, ccp_cyclomatic_proxy, ccp_loc_proxy, ccp_gini_dispersion, ccp_coupling, ccp_avg_unique_per_feature
    * A4 DLE: dle_h_y, dle_h_x_mean, dle_h_x_std, dle_h_y_given_x, dle_info_gain, dle_defect_concentration
- Output: ΔAUC vs (None, LR) baseline for each (DA, specialist) pair
- Training: leave-one-dataset-out (LODO) CV across 35 static datasets
- Loss: LambdaRank (LightGBM) for ranking, OR MSE for regression on ΔAUC

Two modes:
  - ranker: predicts top-k best (DA, specialist) pair per dataset
  - regressor: predicts exact ΔAUC for each (DA, specialist) pair per dataset
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import warnings
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    print("WARN: lightgbm not installed, recommender will not work.")


# ---------------------------------------------------------------------------
# Feature definitions (signals A1-A4)
# ---------------------------------------------------------------------------
INTRINSIC_FEATURES = ["n_samples", "n_features", "class_imb_ratio"]

# A1 Defect Type Signature
A1_FEATURES = ["dts_hal", "dts_log", "dts_mix"]  # dts_type is categorical, handle separately

# A2 Churn-Complexity Heuristic (Kamei 2013)
A2_FEATURES = [
    "chc_nf", "chc_ns", "chc_la", "chc_ld", "chc_lt", "chc_ndev",
    "chc_age", "chc_nuc", "chc_nd", "chc_exp", "chc_rexp", "chc_sexp",
]

# A3 Code-Complexity Proxy
A3_FEATURES = [
    "ccp_halstead_V", "ccp_cyclomatic_proxy", "ccp_loc_proxy",
    "ccp_gini_dispersion", "ccp_coupling", "ccp_avg_unique_per_feature",
]

# A4 Data-Label Entropy
A4_FEATURES = [
    "dle_h_y", "dle_h_x_mean", "dle_h_x_std", "dle_h_y_given_x",
    "dle_info_gain", "dle_defect_concentration",
]

SIGNAL_FEATURES = A1_FEATURES + A2_FEATURES + A3_FEATURES + A4_FEATURES

ALL_FEATURES = INTRINSIC_FEATURES + SIGNAL_FEATURES


def median_impute_signals(signals_table: pd.DataFrame) -> pd.DataFrame:
    """Fill sparse numeric signal gaps with their column medians."""
    signals_table = signals_table.copy()
    for col in ALL_FEATURES:
        signals_table[col] = pd.to_numeric(signals_table[col], errors="coerce")
        signals_table[col] = signals_table[col].fillna(signals_table[col].median())
    return signals_table


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def build_recommender_training_data(
    grid_results: pd.DataFrame,
    signals_table: pd.DataFrame,
    static_only: bool = True,
) -> pd.DataFrame:
    """Build per-(dataset, DA, specialist) training instances.

    Each instance has:
    - 3 intrinsic features
    - 27 signal features (A1-A4)
    - 8 DA one-hot dummies
    - 4 specialist one-hot dummies
    - target: ΔAUC vs (None, LR_baseline) baseline on that dataset

    Parameters
    ----------
    grid_results : pd.DataFrame
        Output of run_v9 specialist grid. Must have columns: dataset, da, classifier, seed, auc_mean
    signals_table : pd.DataFrame
        signals_table.csv with 27 A1-A4 signal columns per dataset
    static_only : bool
        If True, exclude JIT datasets (which have different signal distributions)

    Returns
    -------
    pd.DataFrame with columns: dataset, da, classifier + features + target (delta_auc)
    """
    # Filter to static only if requested
    if static_only:
        jit_datasets = {r["dataset"] for r in
                        [{"dataset": d} for d in
                         ["jit_brackets", "jit_broadleaf", "jit_camel", "jit_fabric8",
                          "jit_jgroups", "jit_neutron", "jit_npm", "jit_spring",
                          "jit_tomcat"]]}
        grid_results = grid_results[~grid_results["dataset"].isin(jit_datasets)].copy()

    # Average over seeds
    grp = grid_results.groupby(["dataset", "da", "classifier"]).agg(
        auc_mean=("auc_mean", "mean"),
    ).reset_index()

    # Compute ΔAUC vs (None, LR) baseline per dataset
    baseline = grp[(grp["da"] == "none") & (grp["classifier"] == "CodeBERT")][["dataset", "auc_mean"]]
    baseline = baseline.rename(columns={"auc_mean": "baseline_auc"})

    merged = grp.merge(baseline, on="dataset", how="left")
    merged["delta_auc"] = merged["auc_mean"] - merged["baseline_auc"]

    # Merge signals
    sig_features = ["dataset"] + ALL_FEATURES
    signals_table = median_impute_signals(signals_table)
    merged = merged.merge(signals_table[sig_features], on="dataset", how="left")

    # Add DA one-hot
    das = sorted(merged["da"].unique())
    for da in das:
        merged[f"da_{da}"] = (merged["da"] == da).astype(int)

    # Add specialist one-hot
    specialists = sorted(merged["classifier"].unique())
    for sp in specialists:
        merged[f"sp_{sp}"] = (merged["classifier"] == sp).astype(int)

    return merged


# ---------------------------------------------------------------------------
# LODO recommender training
# ---------------------------------------------------------------------------
def train_lodo_recommender(
    training_data: pd.DataFrame,
    target_col: str = "delta_auc",
    model_type: str = "regressor",  # "regressor" or "ranker"
    n_top_da: int = 3,  # For ranker: number of top DAs to optimize
    num_leaves: int = 15,
    learning_rate: float = 0.05,
    n_estimators: int = 200,
    min_child_samples: int = 20,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Train LODO recommender with leave-one-dataset-out CV.

    Returns:
        - per_fold_results: dict with per-fold predictions and metrics
        - test_predictions_df: long-format DataFrame with predictions per (dataset, DA, specialist)
    """
    if lgb is None:
        raise RuntimeError("lightgbm not installed")

    datasets = sorted(training_data["dataset"].unique())
    feature_cols = [c for c in training_data.columns if c.startswith(("da_", "sp_")) or c in ALL_FEATURES]

    print(f"LODO CV: {len(datasets)} datasets, {len(feature_cols)} features, {len(training_data)} instances")

    all_predictions = []
    fold_metrics = []

    for held_out in datasets:
        train = training_data[training_data["dataset"] != held_out].copy()
        test = training_data[training_data["dataset"] == held_out].copy()

        if len(test) == 0:
            continue

        X_train = train[feature_cols].fillna(0)
        y_train = train[target_col].values
        X_test = test[feature_cols].fillna(0)

        if model_type == "ranker":
            # Group by dataset for ranking
            train_groups = train.groupby("dataset").size().values
            train["rank_label"] = train.groupby("dataset")[target_col].transform(
                lambda x: pd.qcut(x, q=min(n_top_da, len(x)), labels=False, duplicates="drop")
            ).fillna(0).astype(int)
            y_train_rank = train["rank_label"].values

            model = lgb.LGBMRanker(
                num_leaves=num_leaves, learning_rate=learning_rate,
                n_estimators=n_estimators, min_child_samples=min_child_samples,
                objective="lambdarank", metric="ndcg",
                random_state=42, verbose=-1,
            )
            model.fit(X_train, y_train_rank, group=train_groups)
            test["predicted_delta"] = model.predict(X_test)
        else:
            # Regressor
            model = lgb.LGBMRegressor(
                num_leaves=num_leaves, learning_rate=learning_rate,
                n_estimators=n_estimators, min_child_samples=min_child_samples,
                random_state=42, verbose=-1,
            )
            model.fit(X_train, y_train)
            test["predicted_delta"] = model.predict(X_test)

        # Compute top-1 recommendation accuracy
        top_pred = test.loc[test["predicted_delta"].idxmax()]
        actual_top = test.loc[test[target_col].idxmax()]

        fold_metrics.append({
            "held_out": held_out,
            "top_pred_pair": f"({top_pred['da']}, {top_pred['classifier']})",
            "top_pred_delta": top_pred["predicted_delta"],
            "actual_top_pair": f"({actual_top['da']}, {actual_top['classifier']})",
            "actual_top_delta": actual_top[target_col],
            "n_test_instances": len(test),
        })
        all_predictions.append(test)

    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    metrics_df = pd.DataFrame(fold_metrics)

    # Summary stats
    if not metrics_df.empty:
        # Top-1 recommendation accuracy (predicted pair == actual best pair)
        top1_acc = (metrics_df["top_pred_pair"] == metrics_df["actual_top_pair"]).mean()
        # Mean regret (how much AUC we lose by picking predicted top instead of actual top)
        mean_regret = (metrics_df["actual_top_delta"] - metrics_df["top_pred_delta"]).mean()
        print(f"Top-1 recommendation accuracy: {top1_acc:.3f}")
        print(f"Mean regret (actual_top_delta - top_pred_delta): {mean_regret:+.4f}")

    return {"metrics": metrics_df, "predictions": predictions_df}, predictions_df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="V9.5 Signal-aware recommender")
    parser.add_argument("--grid", type=str,
                        default="/Users/moss/Lab/projects/sdp-research-program/papers/paper_e_da_compatibility/v9/runs/v9_specialist_full/grid.csv")
    parser.add_argument("--signals", type=str,
                        default="/Users/moss/Lab/projects/sdp-research-program/papers/paper_e_da_compatibility/v9/data/signals_table.csv")
    parser.add_argument("--out", type=str,
                        default="/Users/moss/Lab/projects/sdp-research-program/papers/paper_e_da_compatibility/v9/runs/v9_recommender/predictions.csv")
    parser.add_argument("--model_type", default="regressor", choices=["regressor", "ranker"])
    parser.add_argument("--mode", default="legacy", choices=["legacy", "signal_lodo"],
                        help="signal_lodo runs the preregistered A/B/C analysis")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    if args.mode == "signal_lodo":
        from lodo_signal_v9 import run_all
        run_all(args.grid, args.signals, args.out_dir)
        return

    print(f"Loading grid from {args.grid}")
    grid = pd.read_csv(args.grid)
    print(f"  Grid shape: {grid.shape}, OK: {(grid['status']=='ok').sum()}")

    print(f"Loading signals from {args.signals}")
    signals = pd.read_csv(args.signals)
    print(f"  Signals shape: {signals.shape}")

    print("\nBuilding training data...")
    train_data = build_recommender_training_data(grid, signals, static_only=True)
    print(f"  Training data shape: {train_data.shape}")

    Path_out = os.path.dirname(args.out)
    if Path_out:
        os.makedirs(Path_out, exist_ok=True)

    print(f"\nTraining LODO recommender (model_type={args.model_type})...")
    results, predictions = train_lodo_recommender(train_data, model_type=args.model_type)

    # Save outputs
    if not predictions.empty:
        predictions.to_csv(args.out, index=False)
        print(f"\nPredictions saved to {args.out}")

    if not results["metrics"].empty:
        metrics_path = args.out.replace(".csv", "_metrics.csv")
        results["metrics"].to_csv(metrics_path, index=False)
        print(f"Metrics saved to {metrics_path}")

        # Per-DA recommendation analysis
        print("\n=== Per-DA analysis ===")
        pred_analysis = predictions.groupby(["dataset", "da"]).agg(
            actual_max_delta=("delta_auc", "max"),
            predicted_max_delta=("predicted_delta", "max"),
        ).reset_index()
        # For each (dataset, DA), is predicted_max_delta >= actual_max_delta?
        pred_analysis["prediction_correct"] = (
            pred_analysis["predicted_max_delta"] >= pred_analysis["actual_max_delta"] * 0.9
        )
        print(f"Per-(dataset, DA) prediction accuracy: {pred_analysis['prediction_correct'].mean():.3f}")


if __name__ == "__main__":
    main()