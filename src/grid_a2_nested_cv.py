#!/usr/bin/env python3
"""Grid A2: leakage-safe nested CV for frozen specialist representations."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import KMeansSMOTE

ROOT = Path("/Users/moss/Lab/projects/sdp-research-program/papers/paper_e_da_compatibility/v9")
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from da_methods import DA_METHODS, apply_da
from data_loader import load_dataset
from specialists import SPECIALIST_REGISTRY, batch_contexts

OUT_DIR = ROOT / "runs" / "grid_a2"
NOTE_DIR = ROOT / "notes" / "grid_a2"
CACHE_DIR = OUT_DIR / "embedding_cache"
OUTER_SEED = 42
CANDIDATE_SEEDS = (42, 2027, 1729)
DATASETS = {
    "SOFTLAB": ["softlab_ar1", "softlab_ar3", "softlab_ar4", "softlab_ar5", "softlab_ar6"],
    "ReLink": ["relink_apache", "relink_safe", "relink_zxing"],
}
SPECIALISTS = tuple(SPECIALIST_REGISTRY)
DA_CANDIDATES = (
    "adasyn", "borderline_smote", "ctgan", "kmeans_smote", "mixup",
    "noise", "smote_tomek", "smote_enn", "none",
)


def family_of(dataset: str) -> str:
    return next(f for f, ds in DATASETS.items() if dataset in ds)


def embedding_path(dataset: str, specialist: str) -> Path:
    return CACHE_DIR / f"{dataset}__{specialist}.npz"


def build_embedding_cache() -> None:
    """Encode every original row once per frozen transformer, never per CV fold."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pending = [(d, s) for ds in DATASETS.values() for d in ds for s in SPECIALISTS
               if not embedding_path(d, s).exists()]
    if not pending:
        print("Embedding cache complete; reusing all cached representations.", flush=True)
        return

    # Model-major order keeps only one transformer active and maximizes cache reuse.
    for specialist in SPECIALISTS:
        specialist_pending = [d for ds in DATASETS.values() for d in ds
                              if not embedding_path(d, specialist).exists()]
        if not specialist_pending:
            continue
        cls = SPECIALIST_REGISTRY[specialist]["cls"]
        model = cls(device="cpu", head_type="lr", seed=OUTER_SEED)
        for dataset in specialist_pending:
            X, y, feature_names, _ = load_dataset(dataset)
            contexts, valid = batch_contexts(
                X, np.zeros(len(y), dtype=int), feature_names, use_values=True
            )
            chunks = []
            for start in range(0, len(contexts), 64):
                chunks.append(model._extract_embeddings(contexts[start:start + 64]))
            embeddings = np.vstack(chunks).astype(np.float32)
            np.savez_compressed(embedding_path(dataset, specialist), embeddings=embeddings,
                                valid=valid, y=y)
            print(f"cached {dataset} {specialist}: {embeddings.shape}", flush=True)
        # Cached class instance owns the same model; clear only after all datasets.
        from specialists import BaseSpecialistClassifier
        BaseSpecialistClassifier.clear_cache()


def safe_apply_da(X: np.ndarray, y: np.ndarray, method: str, seed: int):
    """Apply all registered DAs; CTGAN works in a bounded PCA space for stability."""
    minority = int(np.bincount(y).min())
    neighbors = min(5, max(1, minority - 1))
    if method == "kmeans_smote":
        return KMeansSMOTE(
            random_state=seed, k_neighbors=neighbors,
            cluster_balance_threshold=0.0,
        ).fit_resample(X, y)
    if method == "smote_enn":
        # SMOTEENN constructs SMOTE lazily; the default neighbor count is valid
        # for all nested folds in the eight selected datasets.
        return SMOTEENN(random_state=seed).fit_resample(X, y)
    if method != "ctgan" or X.shape[1] <= 32:
        return apply_da(X, y, method, seed=seed)
    n_components = min(32, X.shape[1], len(X) - 1)
    pca = PCA(n_components=n_components, random_state=seed)
    Z = pca.fit_transform(X).astype(np.float32)
    Z_aug, y_aug = apply_da(Z, y, method, seed=seed)
    return pca.inverse_transform(Z_aug).astype(np.float32), y_aug


def fit_predict_auc(X_train, y_train, X_test, y_test, da: str, seed: int) -> float:
    Xa, ya = safe_apply_da(X_train, y_train, da, seed)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed, n_jobs=1),
    )
    clf.fit(Xa, ya)
    return float(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))


def baseline_auc(X_train, y_train, X_test, y_test, seed: int) -> float:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed, n_jobs=1),
    )
    clf.fit(X_train, y_train)
    return float(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))


def run_dataset(dataset: str):
    X_raw, y, _, _ = load_dataset(dataset)
    embeddings = {
        s: np.load(embedding_path(dataset, s))["embeddings"].astype(np.float32)
        for s in SPECIALISTS
    }
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    rows, selections = [], []

    for outer_fold, (inner_idx, test_idx) in enumerate(outer.split(X_raw, y), start=1):
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=OUTER_SEED)
        raw_inner = X_raw[inner_idx]
        y_inner = y[inner_idx]
        split_indices = list(inner_cv.split(raw_inner, y_inner))
        baseline_inner = np.mean([
            baseline_auc(raw_inner[tr], y_inner[tr], raw_inner[va], y_inner[va],
                         OUTER_SEED + inner_fold)
            for inner_fold, (tr, va) in enumerate(split_indices)
        ])

        best = None
        failures = []
        for specialist in SPECIALISTS:
            E_inner = embeddings[specialist][inner_idx]
            for da in DA_CANDIDATES:
                for seed in CANDIDATE_SEEDS:
                    aucs = []
                    try:
                        for inner_fold, (tr, va) in enumerate(split_indices):
                            aucs.append(fit_predict_auc(
                                E_inner[tr], y_inner[tr], E_inner[va], y_inner[va],
                                da, seed + inner_fold,
                            ))
                    except Exception as exc:
                        failures.append(f"{specialist}/{da}/{seed}: {type(exc).__name__}")
                        continue
                    mean_auc = float(np.mean(aucs))
                    candidate = (mean_auc, specialist, da, seed)
                    if best is None or candidate > best:
                        best = candidate
        if best is None:
            raise RuntimeError(f"No valid candidate for {dataset} fold {outer_fold}: {failures[:5]}")

        inner_auc, specialist, da, seed = best
        inner_dauc = inner_auc - baseline_inner
        E = embeddings[specialist]
        outer_auc = fit_predict_auc(E[inner_idx], y[inner_idx], E[test_idx], y[test_idx], da, seed)
        outer_baseline = baseline_auc(X_raw[inner_idx], y[inner_idx], X_raw[test_idx],
                                      y[test_idx], OUTER_SEED + outer_fold)
        outer_dauc = outer_auc - outer_baseline
        rows.append({
            "dataset": dataset,
            "family": family_of(dataset),
            "inner_oracle_dauc": inner_dauc,
            "outer_realized_dauc": outer_dauc,
            "optimism": inner_dauc - outer_dauc,
            "outer_fold": outer_fold,
        })
        selections.append({
            "dataset": dataset, "outer_fold": outer_fold, "specialist": specialist,
            "da": da, "candidate_seed": seed, "inner_auc": inner_auc,
            "inner_baseline_auc": baseline_inner, "outer_auc": outer_auc,
            "outer_baseline_auc": outer_baseline, "failed_candidates": len(failures),
        })
        print(f"{dataset} fold={outer_fold} selected={specialist}/{da}/{seed} "
              f"inner_dAUC={inner_dauc:+.4f} outer_dAUC={outer_dauc:+.4f}", flush=True)
    return rows, selections


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator):
    draws = rng.choice(values, size=(1000, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize(results: pd.DataFrame, selections: list[dict], runtime: float) -> dict:
    rng = np.random.default_rng(OUTER_SEED)
    families = {}
    for family, group in results.groupby("family", sort=False):
        inner = group.inner_oracle_dauc.to_numpy(float)
        outer = group.outer_realized_dauc.to_numpy(float)
        optimism = group.optimism.to_numpy(float)
        families[family] = {
            "n_datasets": int(group.dataset.nunique()),
            "n_outer_folds": int(len(group)),
            "inner_oracle_dauc_mean": float(inner.mean()),
            "inner_oracle_dauc_ci95": bootstrap_ci(inner, rng),
            "outer_realized_dauc_mean": float(outer.mean()),
            "outer_realized_dauc_ci95": bootstrap_ci(outer, rng),
            "optimism_mean": float(optimism.mean()),
            "optimism_ci95": bootstrap_ci(optimism, rng),
        }
    return {
        "design": {
            "outer_folds": 5, "outer_seed": OUTER_SEED, "inner_folds": 3,
            "candidate_seeds": list(CANDIDATE_SEEDS), "bootstrap_resamples": 1000,
            "bootstrap_seed": OUTER_SEED, "specialists": list(SPECIALISTS),
            "da_methods": list(DA_CANDIDATES),
            "representation": "frozen transformer embeddings cached once per original row; DA applied in embedding space",
            "ctgan_note": "CTGAN uses a fold-fitted <=32-component PCA space, inverse-transformed before LR",
            "limitation": "One outer split seed (42) only.",
        },
        "families": families,
        "selections": selections,
        "runtime_seconds": runtime,
    }


def write_report(summary: dict) -> None:
    fam = summary["families"]
    lines = [
        "# Grid A2 - Oracle Optimism / Nested CV", "",
        "## Design", "",
        "Leakage-safe nested CV used fixed outer 5-fold stratification (seed 42) and inner 3-fold selection over 4 frozen specialist representations, 8 DA methods, and candidate seeds 42/2027/1729. Original-row embeddings were encoded once and cached; augmentation was fitted only on each training fold in embedding space. CTGAN was stabilized with fold-fitted PCA (at most 32 components).", "",
        "## Family results", "",
        "| Family | Inner-oracle DeltaAUC (95% CI) | Outer-realized DeltaAUC (95% CI) | Optimism (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for family in ("SOFTLAB", "ReLink"):
        x = fam[family]
        fmt = lambda m, ci: f"{m:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        lines.append(f"| {family} | {fmt(x['inner_oracle_dauc_mean'], x['inner_oracle_dauc_ci95'])} | "
                     f"{fmt(x['outer_realized_dauc_mean'], x['outer_realized_dauc_ci95'])} | "
                     f"{fmt(x['optimism_mean'], x['optimism_ci95'])} |")
    soft = fam["SOFTLAB"]["outer_realized_dauc_mean"]
    relink = fam["ReLink"]["outer_realized_dauc_mean"]
    lines += ["", "## Verdict", ""]
    if soft >= 0.10:
        lines.append(f"SOFTLAB remains meaningful ({soft:+.3f}, above +0.10); the existence claim survives nested evaluation.")
    elif soft < 0.05:
        lines.append(f"SOFTLAB collapses below +0.05 ({soft:+.3f}); soften the headline to an upper-bound existence claim.")
    else:
        lines.append(f"SOFTLAB remains positive but attenuates ({soft:+.3f}); the claim survives only in a qualified form.")
    lines.append(f"ReLink outer-realized DeltaAUC is {relink:+.3f}; its bootstrap CI is reported even if it crosses zero.")
    lines += ["", "## Limitation", "", "Only one outer split seed (42) was used; uncertainty therefore reflects resampling of realized outer folds, not repeated partition seeds.", ""]
    (NOTE_DIR / "02_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE_DIR.mkdir(parents=True, exist_ok=True)
    build_embedding_cache()
    partial = OUT_DIR / "results.partial.csv"
    if partial.exists():
        prior = pd.read_csv(partial)
        # Preserve only complete datasets so an interrupted dataset is rerun cleanly.
        complete = prior.groupby("dataset").filter(lambda g: len(g) == 5)
        rows = complete.to_dict("records")
        completed = set(complete.dataset)
        print(f"Resuming with complete datasets: {sorted(completed)}", flush=True)
    else:
        rows, completed = [], set()
    selections = []
    for datasets in DATASETS.values():
        for dataset in datasets:
            if dataset in completed:
                continue
            ds_rows, ds_selections = run_dataset(dataset)
            rows.extend(ds_rows)
            selections.extend(ds_selections)
            pd.DataFrame(rows).to_csv(partial, index=False)
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "results.csv", index=False, float_format="%.9f")
    summary = summarize(results, selections, time.time() - t0)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary)
    print(f"Completed Grid A2 in {(time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
