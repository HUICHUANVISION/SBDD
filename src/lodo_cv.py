"""Paper E v8 — Leave-one-dataset-out (LODO) cross-validation for recommender.

For each held-out dataset D_i:
1. Train grid on D_j ≠ D_i (using grid_results.csv filtered)
2. Compute ΔAUC targets from grid (per DA, classifier)
3. Train recommender on the 44 training datasets
4. Predict top-k (DA, classifier) for D_i
5. Evaluate: did we recommend a pair that actually performs well on D_i?

Outputs:
- lodo_results.csv: per-fold top-k predictions + actual AUCs
- metrics: top-1, top-3, top-5 hit rate, NDCG@5, mean ΔAUC uplift

Usage:
    source /tmp/paper_e_v8_env.sh
    cd ~/Lab/projects/sdp-research-program/papers/paper_e_da_compatibility/v8/src
    python3 lodo_cv.py --grid_csv runs/v8_main/grid_results.csv
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "16")

import argparse
import time
import json
import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from intrinsic_features import compute_intrinsic_features, build_intrinsic_table
from recommender import (
    build_recommender_training_data, train_ranker, predict_topk, FEATURE_NAMES,
)
from data_loader import list_datasets


def lodo_cv(
    grid_results: pd.DataFrame,
    intrinsic_table: pd.DataFrame,
    k_list: List[int] = [1, 3, 5],
    baseline_methods: List[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run LODO CV for the recommender.

    Parameters
    ----------
    grid_results : pd.DataFrame
        Output of run_v8 main grid.
    intrinsic_table : pd.DataFrame
        Intrinsic features for all datasets.
    k_list : list of k values to evaluate
    baseline_methods : list of baseline names to compare

    Returns
    -------
    lodo_results : pd.DataFrame
        Per-fold results
    metrics : dict of aggregate metrics
    """
    if baseline_methods is None:
        baseline_methods = ["random", "always_best_single", "default", "automl_lite"]

    datasets = sorted(grid_results["dataset"].unique())
    n_datasets = len(datasets)
    print(f"LODO CV over {n_datasets} datasets")

    # Compute ΔAUC targets (already in build_recommender_training_data)
    train_data = build_recommender_training_data(grid_results, intrinsic_table)
    das = sorted(train_data["da"].unique())
    classifiers = sorted(train_data["classifier"].unique())

    # Per-fold results
    fold_results = []
    for held_out in datasets:
        t0 = time.time()
        # Train data: all other datasets
        train_fold = train_data[train_data["dataset"] != held_out].copy()
        if len(train_fold) < 50:
            print(f"  WARN: held_out={held_out}, train_fold has only {len(train_fold)} rows, skipping")
            continue

        # Test data: held-out dataset, need actual AUC for evaluation
        test_fold = train_data[train_data["dataset"] == held_out].copy()
        if len(test_fold) == 0:
            print(f"  WARN: held_out={held_out} not in test data, skipping")
            continue

        # Train recommender
        try:
            model = train_ranker(train_fold)
        except Exception as e:
            print(f"  WARN: train_ranker failed for {held_out}: {e}")
            continue

        # Predict top-k for held-out
        target_intrinsic = intrinsic_table[intrinsic_table["dataset"] == held_out].iloc[0].to_dict()
        target_intrinsic = {k: target_intrinsic[k] for k in FEATURE_NAMES}
        topk_max = max(k_list)
        top_pairs = predict_topk(model, target_intrinsic, das, classifiers, k=topk_max)

        # Compute baselines
        baseline_topk = _compute_baselines(train_fold, test_fold, topk_max, baseline_methods, held_out=held_out)

        # Evaluate
        # For each (DA, classifier) pair in test_fold, get actual AUC
        test_actual = test_fold.set_index(["da", "classifier"])["auc_mean"].to_dict()
        # Oracle: best AUC in test_fold
        oracle_pair = max(test_actual.items(), key=lambda x: x[1])
        oracle_auc = oracle_pair[1]
        oracle_pair_str = f"{oracle_pair[0][0]}+{oracle_pair[0][1]}"

        # Compute hit rates for recommender and baselines
        def _hit_at_k(top_pairs_list, k, test_actual):
            top_k_set = set((da, clf) for da, clf, _ in top_pairs_list[:k])
            # Hit if any of top-k is in the top-1 of test_actual
            best_test_pair = max(test_actual.items(), key=lambda x: x[1])[0]
            return int(best_test_pair in top_k_set)

        rec_metrics = {}
        for k in k_list:
            rec_metrics[f"rec_top{k}_hit"] = _hit_at_k(top_pairs, k, test_actual)

        # NDCG@5
        rec_metrics["rec_ndcg5"] = _ndcg_at_k(top_pairs, 5, test_actual)

        # Mean ΔAUC of top-1 recommendation vs default (None, LR)
        default_auc = test_actual.get(("none", "LR"), oracle_auc)
        top1_da, top1_clf, _ = top_pairs[0]
        top1_auc = test_actual.get((top1_da, top1_clf), default_auc)
        rec_metrics["rec_top1_auc"] = top1_auc
        rec_metrics["rec_delta_auc"] = top1_auc - default_auc
        rec_metrics["oracle_auc"] = oracle_auc
        rec_metrics["oracle_pair"] = oracle_pair_str
        rec_metrics["default_auc"] = default_auc
        rec_metrics["rec_top5_pairs"] = ";".join([f"{da}+{clf}" for da, clf, _ in top_pairs[:5]])

        # Baselines
        for bname, bpairs in baseline_topk.items():
            for k in k_list:
                rec_metrics[f"{bname}_top{k}_hit"] = _hit_at_k(bpairs, k, test_actual)
            rec_metrics[f"{bname}_ndcg5"] = _ndcg_at_k(bpairs, 5, test_actual)
            rec_metrics[f"{bname}_top1_auc"] = test_actual.get(
                (bpairs[0][0], bpairs[0][1]), default_auc)
            rec_metrics[f"{bname}_top5_pairs"] = ";".join([f"{da}+{clf}" for da, clf, _ in bpairs[:5]])

        rec_metrics["dataset"] = held_out
        rec_metrics["runtime_sec"] = time.time() - t0
        fold_results.append(rec_metrics)
        print(f"  {held_out:25s} rec_top3_hit={rec_metrics.get('rec_top3_hit', '?')} "
              f"delta_auc={rec_metrics['rec_delta_auc']:+.4f}  ({rec_metrics['runtime_sec']:.1f}s)")

    # Aggregate
    df = pd.DataFrame(fold_results)
    metrics = {}
    for col in df.columns:
        if col.endswith("_hit") or col.endswith("_auc") or col.endswith("_ndcg5") or col == "rec_delta_auc":
            metrics[f"{col}_mean"] = float(df[col].mean())
            metrics[f"{col}_std"] = float(df[col].std())
    metrics["n_folds"] = len(df)

    return df, metrics


def _compute_baselines(train_fold, test_fold, k, baseline_methods, held_out=None):
    """Compute baseline top-k pairs from training data."""
    out = {}
    das = sorted(test_fold["da"].unique())
    classifiers = sorted(test_fold["classifier"].unique())
    rng = np.random.default_rng(42)

    if "random" in baseline_methods:
        # Random pairs (deterministic seed)
        pairs = []
        for _ in range(k):
            d = das[rng.integers(0, len(das))]
            c = classifiers[rng.integers(0, len(classifiers))]
            pairs.append((d, c, 0.0))
        out["random"] = pairs

    if "always_best_single" in baseline_methods:
        # Best (DA, clf) by mean AUC over training datasets
        grp = train_fold.groupby(["da", "classifier"])["delta_auc"].mean().reset_index()
        grp = grp.sort_values("delta_auc", ascending=False)
        out["always_best"] = [(r["da"], r["classifier"], 0.0) for _, r in grp.head(k).iterrows()]

    if "default" in baseline_methods:
        # Always recommend (None, LR)
        out["default"] = [("none", "LR", 0.0)] * k

    if "automl_lite" in baseline_methods:
        # Budgeted AutoML-lite baseline: sample a small fixed search budget of
        # candidate (DA, clf) pairs, then choose the best training-mean AUC among
        # that subset. This avoids collapsing to always_best_single while still
        # representing a cheap, non-adaptive random search baseline.
        grp = train_fold.groupby(["da", "classifier"])["auc_mean"].mean().reset_index()
        all_pairs = list(grp[["da", "classifier"]].itertuples(index=False, name=None))
        budget = min(20, len(all_pairs))
        seed = 4242 + (int(hashlib.md5(str(held_out).encode("utf-8")).hexdigest()[:8], 16) % 100000 if held_out is not None else 0)
        local_rng = np.random.default_rng(seed)
        idx = local_rng.choice(len(all_pairs), size=budget, replace=False)
        sampled = set(all_pairs[i] for i in idx)
        sampled_grp = grp[grp.apply(lambda r: (r["da"], r["classifier"]) in sampled, axis=1)]
        sampled_grp = sampled_grp.sort_values("auc_mean", ascending=False)
        out["automl_lite"] = [(r["da"], r["classifier"], 0.0) for _, r in sampled_grp.head(k).iterrows()]

    return out


def _ndcg_at_k(top_pairs_list, k, test_actual):
    """Compute NDCG@k given a list of (da, clf, score) and actual AUCs."""
    if not top_pairs_list or len(test_actual) == 0:
        return 0.0
    # DCG
    dcg = 0.0
    for i, (da, clf, _) in enumerate(top_pairs_list[:k]):
        rel = test_actual.get((da, clf), 0.5)
        # Discount: log2(i+2)
        dcg += (rel - 0.5) / np.log2(i + 2)
    # Ideal DCG: rank by actual AUC
    actual_sorted = sorted(test_actual.items(), key=lambda x: -x[1])[:k]
    idcg = 0.0
    for i, (_, auc) in enumerate(actual_sorted):
        idcg += (auc - 0.5) / np.log2(i + 2)
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Paper E v8 LODO CV")
    parser.add_argument("--grid_csv", type=str, required=True,
                        help="Path to grid_results.csv from run_v8.py")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--max_folds", type=int, default=None,
                        help="Limit number of LODO folds (for debugging)")
    args = parser.parse_args()

    grid_results = pd.read_csv(args.grid_csv)
    print(f"Loaded grid results: {grid_results.shape}")

    # Build intrinsic features table for ALL datasets first (max_folds only affects LODO loop)
    print("Building intrinsic features table...")
    all_datasets = sorted(grid_results["dataset"].unique())
    rows = []
    from data_loader import load_dataset
    for d in all_datasets:
        try:
            X, y, _, _ = load_dataset(d)
            rows.append(compute_intrinsic_features(d, X, y))
        except Exception as e:
            print(f"  WARN: failed to load {d}: {e}")
    intrinsic_table = pd.DataFrame(rows)
    print(f"Intrinsic table: {intrinsic_table.shape}")
    if args.max_folds:
        print(f"Limiting LODO to first {args.max_folds} folds")

    # Run LODO CV
    lodo_df, metrics = lodo_cv(grid_results, intrinsic_table)

    # Save outputs
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        os.path.dirname(args.grid_csv)).parent / "v8_recommender"
    out_dir.mkdir(parents=True, exist_ok=True)
    lodo_df.to_csv(out_dir / "lodo_results.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n=== LODO CV Metrics ===")
    for k, v in metrics.items():
        if k.endswith("_mean"):
            print(f"  {k:30s} = {v:.4f}")
    print(f"\nOutputs saved to {out_dir}")


if __name__ == "__main__":
    main()