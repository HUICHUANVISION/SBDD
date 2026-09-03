"""Paper E v8 — Recommender system.

LightGBM LambdaRank ranker that learns to predict the top-k (DA, classifier)
pairs for a new SDP dataset, given its intrinsic features.

Architecture:
- Input: 4 intrinsic features (dataset_size, imb_ratio, n_features, feature_skewness)
- Output: ranking of 9 DAs × 12 classifiers = 108 (DA, classifier) pairs
- Training: leave-one-dataset-out (LODO) CV
- Loss: LambdaRank (optimizes NDCG directly)

Hyperparameters tuned for small training set (per-fold ≈ 540 instances):
- num_leaves=15 (small, prevent overfitting)
- learning_rate=0.05
- n_estimators=200
- min_child_samples=20
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "16")

import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    print("WARN: lightgbm not installed, recommender will not work.")


FEATURE_NAMES = [
    "dataset_size", "class_imbalance_ratio", "n_features", "feature_skewness",
]

DA_ONEHOT_PREFIX = "da_"
CLF_ONEHOT_PREFIX = "clf_"


def build_recommender_training_data(
    grid_results: pd.DataFrame,
    intrinsic_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-(dataset, DA, classifier) training instances.

    Each instance has:
    - 4 intrinsic features (from dataset)
    - one-hot DA (9 dummies)
    - one-hot classifier (10-13 dummies)
    - target: ΔAUC vs (None, LR) baseline on that dataset

    This allows the recommender to learn:
    "given dataset X, which (DA, classifier) pair is best?"

    Parameters
    ----------
    grid_results : pd.DataFrame
        Output of run_v8 main grid, with columns: dataset, da, classifier, seed, auc_mean, ...
        Should be averaged over seeds first.
    intrinsic_table : pd.DataFrame
        Output of build_intrinsic_table, with columns: dataset, dataset_size, ... feature_skewness

    Returns
    -------
    pd.DataFrame with columns: dataset, da, classifier + 4 intrinsic + DA one-hots + clf one-hots + delta_auc
    """
    # Average over seeds
    grp = grid_results.groupby(["dataset", "da", "classifier"]).agg(
        auc_mean=("auc_mean", "mean"),
    ).reset_index()

    # Compute ΔAUC vs (None, LR) baseline per dataset
    baseline = grp[(grp["da"] == "none") & (grp["classifier"] == "LR")][["dataset", "auc_mean"]]
    baseline = baseline.rename(columns={"auc_mean": "baseline_auc"})
    grp = grp.merge(baseline, on="dataset", how="left")
    grp["delta_auc"] = grp["auc_mean"] - grp["baseline_auc"]

    # Merge intrinsic features
    df = grp.merge(intrinsic_table, on="dataset", how="left")

    # One-hot encode DA and classifier (drop first to avoid collinearity)
    df = pd.get_dummies(df, columns=["da"], prefix=DA_ONEHOT_PREFIX, drop_first=False)
    df = pd.get_dummies(df, columns=["classifier"], prefix=CLF_ONEHOT_PREFIX, drop_first=False)

    # Keep original categorical columns for downstream lookup
    df["da"] = grp["da"].values
    df["classifier"] = grp["classifier"].values

    return df


def train_ranker(
    train_df: pd.DataFrame,
    params: Dict[str, Any] = None,
    num_boost_round: int = 200,
) -> Any:
    """Train a LightGBM ranker/regressor on intrinsic features → ΔAUC.

    Uses regression objective (predict continuous ΔAUC, then rank by predicted value).
    This is simpler than LambdaRank and avoids label_gain constraints.

    Parameters
    ----------
    train_df : pd.DataFrame
        Output of build_recommender_training_data. Must contain
        4 intrinsic features and a 'group' column.
    params : dict
        LightGBM parameters. If None, use defaults.
    num_boost_round : int

    Returns
    -------
    lgb.Booster model
    """
    if lgb is None:
        raise ImportError("lightgbm required for recommender")

    if params is None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 15,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_jobs": 1,
            "random_state": 42,
        }

    # Build feature matrix: intrinsic + DA one-hot + classifier one-hot
    feat_cols = list(FEATURE_NAMES)
    da_cols = sorted([c for c in train_df.columns if c.startswith(DA_ONEHOT_PREFIX)])
    clf_cols = sorted([c for c in train_df.columns if c.startswith(CLF_ONEHOT_PREFIX)])
    feat_cols = feat_cols + da_cols + clf_cols

    X_train = train_df[feat_cols].values
    y_train = train_df["delta_auc"].values

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feat_cols)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        callbacks=[lgb.log_evaluation(period=0)],
    )

    return model


def predict_topk(
    model: Any,
    target_intrinsic: Dict[str, float],
    da_list: List[str],
    classifier_list: List[str],
    k: int = 5,
    train_da_cols: List[str] = None,
    train_clf_cols: List[str] = None,
) -> List[Tuple[str, str, float]]:
    """Predict top-k (DA, classifier) pairs for a new dataset.

    Parameters
    ----------
    model : trained LightGBM ranker
    target_intrinsic : dict with keys: dataset_size, imb_ratio, n_features, feature_skewness
    da_list : list of DA names (e.g., 9 DAs)
    classifier_list : list of classifier names (e.g., 12 classifiers)
    k : int, number of top pairs to return
    train_da_cols : list of DA column names from training (e.g., ['da_smote', 'da_adasyn', ...])
    train_clf_cols : list of classifier column names from training

    Returns
    -------
    list of (da, classifier, score) tuples, ranked descending
    """
    if train_da_cols is None:
        train_da_cols = [f"{DA_ONEHOT_PREFIX}{d}" for d in da_list]
    if train_clf_cols is None:
        train_clf_cols = [f"{CLF_ONEHOT_PREFIX}{c}" for c in classifier_list]

    # Build feature matrix for all (DA, classifier) pairs
    n_pairs = len(da_list) * len(classifier_list)
    feat_names = list(FEATURE_NAMES) + list(train_da_cols) + list(train_clf_cols)
    X = np.zeros((n_pairs, len(feat_names)))

    # Fill intrinsic features (same for all pairs of this dataset)
    for i, fn in enumerate(FEATURE_NAMES):
        X[:, i] = target_intrinsic[fn]

    # Fill DA + classifier one-hot for each pair
    pairs = []
    idx = 0
    for da in da_list:
        for clf in classifier_list:
            # Set DA one-hot
            da_col = f"{DA_ONEHOT_PREFIX}{da}"
            if da_col in train_da_cols:
                da_idx = len(FEATURE_NAMES) + train_da_cols.index(da_col)
                X[idx, da_idx] = 1.0
            # Set classifier one-hot
            clf_col = f"{CLF_ONEHOT_PREFIX}{clf}"
            if clf_col in train_clf_cols:
                clf_idx = len(FEATURE_NAMES) + len(train_da_cols) + train_clf_cols.index(clf_col)
                X[idx, clf_idx] = 1.0
            pairs.append((da, clf, 0.0))  # placeholder score
            idx += 1

    # Predict (regression: continuous ΔAUC; higher = better)
    scores = model.predict(X)
    pairs = [(da, clf, float(s)) for (da, clf, _), s in zip(pairs, scores)]

    # Sort by score descending
    pairs.sort(key=lambda x: -x[2])
    return pairs[:k]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Recommender smoke test (synthetic data)")
    rng = np.random.default_rng(0)
    n_datasets = 5
    das = ["none", "smote", "adasyn", "borderline", "peters_filter",
           "burak_filter", "cost_sensitive", "mutation", "kco"]
    classifiers = ["LR", "RF", "LGBM", "MLP", "LR_CS", "RF_CS", "LGBM_CS",
                   "MLP_CS", "CORAL_LR", "CORAL_RF"]
    n_pairs = len(das) * len(classifiers)

    rows = []
    for d in range(n_datasets):
        for i in range(n_pairs):
            # True delta_auc depends on (DA, classifier) pair
            da_idx = i // len(classifiers)
            clf_idx = i % len(classifiers)
            # Synthetic signal: LR + smote is best; mutation is worst
            true_delta = (
                -0.05 * (da_idx == 8)  # mutation is bad
                + 0.03 * (da_idx == 1)  # smote is good
                + 0.02 * (clf_idx < 4)  # generic clf slight edge
                + 0.05 * rng.standard_normal()  # noise
            )
            rows.append({
                "dataset": f"ds_{d}",
                "da": das[da_idx],
                "classifier": classifiers[clf_idx],
                "auc_mean": 0.7 + true_delta,
                "delta_auc": true_delta,
                "dataset_size": float(500 + 100 * d),
                "class_imbalance_ratio": float(0.3 + 0.1 * d),
                "n_features": float(60.0),
                "feature_skewness": float(2.0 + 0.5 * d),
            })
    raw_df = pd.DataFrame(rows)

    # Build training data (this one-hot encodes DA + classifier)
    intrinsic_cols = ["dataset", "dataset_size", "class_imbalance_ratio",
                       "n_features", "feature_skewness"]
    intrinsic_df = raw_df[intrinsic_cols].drop_duplicates().reset_index(drop=True)
    train_df = build_recommender_training_data(raw_df, intrinsic_df)
    print(f"Training data: {train_df.shape}, features: {len(FEATURE_NAMES)} intrinsic + "
          f"{len([c for c in train_df.columns if c.startswith(DA_ONEHOT_PREFIX)])} DA + "
          f"{len([c for c in train_df.columns if c.startswith(CLF_ONEHOT_PREFIX)])} clf")

    model = train_ranker(train_df, num_boost_round=100)
    print(f"Model trained. Top-5 predictions for new dataset:")
    target = {"dataset_size": 700, "class_imbalance_ratio": 0.4,
              "n_features": 65.0, "feature_skewness": 2.5}
    top5 = predict_topk(model, target, das, classifiers, k=5)
    for da, clf, score in top5:
        print(f"  {da:20s} {clf:10s} score={score:+.4f}")