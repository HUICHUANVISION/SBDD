"""Paper E v9 — Grid runner with proper specialist pipeline.

Key design:
  - Standard classifiers (LR/RF/XGB/LGBM/MLP + cost + CORAL): per-fold DA + per-fold training (v8 style)
  - Specialist classifiers (CodeBERT/UniXcoder/GraphCodeBERT/RoBERTa):
      Load model ONCE per cell → extract embeddings for all data → 3-fold CV in embedding space
      This avoids re-loading the transformer for every fold (5-10s saved per fold)

Usage:
  # Smoke (2 datasets × 4 specialists × 9 DAs × 1 seed)
  python3 run_v9.py --specialists_only --smoke --max_cells 80

  # Full specialist grid (35 static datasets × 4 specialists × 9 DAs × 3 seeds)
  python3 run_v9.py --specialists_only --seeds 42,2027,1729

  # Full standard grid (35 static datasets × 12 classifiers × 9 DAs × 5 seeds)
  python3 run_v9.py --standard_only --seeds 42,2027,1729,31415,2187

  # Both
  python3 run_v9.py --both --seeds 42,2027,1729
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import time
import warnings
import argparse
import csv
import gc
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

SRC = Path(__file__).parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from data_loader import load_dataset, list_datasets
# Defer xgboost/lightgbm import — they interfere with torch's CPU thread pool
# causing SIGSEGV during transformer forward pass
# from classifiers import make_classifier, CLASSIFIER_REGISTRY
from da_methods import apply_da, DA_METHODS
from specialists import (
    SPECIALIST_REGISTRY, SpecialistWrapper,
    CodeBERTClassifier, UniXcoderClassifier,
    GraphCodeBERTClassifier, RoBERTaClassifier,
)

# Lazy import helper for standard classifiers
def _get_classifier_api():
    """Lazy import of classifiers module (avoids xgboost/lightgbm loading during specialists)."""
    from classifiers import make_classifier, CLASSIFIER_REGISTRY
    return make_classifier, CLASSIFIER_REGISTRY

DEVICE = ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else
          "mps" if __import__("torch").backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Specialist CV (load once → embed all → CV in embedding space)
# ---------------------------------------------------------------------------
def _run_specialist_cell(
    dataset: str,
    da_method: str,
    specialist_name: str,
    seed: int,
    n_folds: int = 3,
) -> Dict[str, Any]:
    """Run one specialist cell: load model ONCE, embed all, CV in embedding space.

    Key insight: transformer loads once → embeddings extracted for full data →
    LR head trained per fold (fast, <1s per fold).
    """
    import gc as gc_mod
    t0 = time.time()
    info = SPECIALIST_REGISTRY[specialist_name]
    spec_cls = info["cls"]

    try:
        X, y, feat_names, meta = load_dataset(dataset)
    except Exception as e:
        return _skip(dataset, da_method, specialist_name, seed, f"load: {e}")

    n, d = X.shape
    n_pos = int(y.sum())
    n_neg = n - n_pos
    imb_ratio = n_pos / max(n_neg, 1)

    # Apply DA once to full data
    try:
        X_aug, y_aug = apply_da(X, y, da_method, seed=seed)
    except Exception as e:
        return _skip(dataset, da_method, specialist_name, seed, f"da: {e}")

    # Subsample augmented data to max 500 for embedding extraction speed
    MAX_SPECIALIST_TRAIN = 500
    if len(X_aug) > MAX_SPECIALIST_TRAIN:
        idx = np.random.RandomState(seed).choice(len(X_aug), MAX_SPECIALIST_TRAIN, replace=False)
        X_aug_sub = X_aug[idx]
        y_aug_sub = y_aug[idx]
        print(f"  Subsampled {len(X_aug)} → {len(X_aug_sub)} for embedding extraction")
    else:
        X_aug_sub = X_aug
        y_aug_sub = y_aug

    # Load specialist model ONCE → fit it once → extract embeddings for ALL data
    try:
        specialist = SpecialistWrapper(spec_cls, device=DEVICE, head_type="lr", seed=seed, max_train=MAX_SPECIALIST_TRAIN)
        specialist.fit(X_aug_sub, y_aug_sub, feature_names=feat_names)
        # Extract embeddings for the subsampled augmented dataset ONCE
        all_embeddings, all_valid_mask = specialist.extract_embeddings(X_aug_sub, feat_names)
        # Release transformer from MPS memory immediately — embeddings are now in CPU numpy
        specialist.cleanup()
    except Exception as e:
        return _skip(dataset, da_method, specialist_name, seed, f"specialist_fit/embed: {e}")

    # CV in embedding space: only LR head changes per fold (transformer frozen)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs, f1s, mccs, bas = [], [], [], []

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(all_embeddings, y_aug_sub)):
        try:
            # Get pre-extracted embeddings
            emb_tr = all_embeddings[tr_idx]
            emb_te = all_embeddings[te_idx]
            y_tr = y_aug_sub[tr_idx]
            y_te = y_aug_sub[te_idx]

            # Fit scaler + LR head on train-fold embeddings only
            scaler_fold = StandardScaler()
            emb_tr_s = scaler_fold.fit_transform(emb_tr)
            emb_te_s = scaler_fold.transform(emb_te)

            lr_head = LogisticRegression(max_iter=1000, random_state=seed + fold_i, n_jobs=1)
            lr_head.fit(emb_tr_s, y_tr)
            proba = lr_head.predict_proba(emb_te_s)
            pred = (proba[:, 1] >= 0.5).astype(int)

            auc = roc_auc_score(y_te, proba[:, 1])
            f1 = f1_score(y_te, pred, zero_division=0)
            mcc = matthews_corrcoef(y_te, pred)
            ba = balanced_accuracy_score(y_te, pred)
            aucs.append(auc); f1s.append(f1); mccs.append(mcc); bas.append(ba)
        except Exception as e:
            pass

    if not aucs:
        return _skip(dataset, da_method, specialist_name, seed, "all_folds_failed")

    runtime = time.time() - t0
    return {
        "dataset": dataset, "da": da_method, "classifier": specialist_name,
        "seed": seed, "n_samples": n, "n_pos": n_pos,
        "imb_ratio": round(imb_ratio, 4),
        "auc_mean": round(float(np.mean(aucs)), 6),
        "auc_std": round(float(np.std(aucs)), 6),
        "f1_mean": round(float(np.mean(f1s)), 6),
        "f1_std": round(float(np.std(f1s)), 6),
        "mcc_mean": round(float(np.mean(mccs)), 6),
        "mcc_std": round(float(np.std(mccs)), 6),
        "ba_mean": round(float(np.mean(bas)), 6),
        "ba_std": round(float(np.std(bas)), 6),
        "n_folds_ok": len(aucs),
        "status": "ok", "skip_reason": "", "runtime_seconds": round(runtime, 4),
    }


# ---------------------------------------------------------------------------
# Standard classifier cell (v8 style: per-fold DA + per-fold training)
# ---------------------------------------------------------------------------
def _run_standard_cell(
    dataset: str,
    da_method: str,
    classifier: str,
    seed: int,
    n_folds: int = 3,
) -> Dict[str, Any]:
    """Run one standard classifier cell (LR/RF/XGB/LGBM/MLP + cost + CORAL)."""
    t0 = time.time()

    try:
        X, y, feat_names, meta = load_dataset(dataset)
    except Exception as e:
        return _skip(dataset, da_method, classifier, seed, f"load: {e}")

    n, d = X.shape
    n_pos = int(y.sum())
    n_neg = n - n_pos
    imb_ratio = n_pos / max(n_neg, 1)

    # Lazy import classifiers (xgboost/lightgbm)
    make_classifier, _ = _get_classifier_api()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs, f1s, mccs, bas = [], [], [], []

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        try:
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            X_tr_aug, y_tr_aug = apply_da(X_tr, y_tr, da_method, seed=seed + fold_i)
            clf = make_classifier(classifier, seed=seed + fold_i)
            clf.fit(X_tr_aug, y_tr_aug)
            proba = clf.predict_proba(X_te)
            pred = (proba[:, 1] >= 0.5).astype(int)
            auc = roc_auc_score(y_te, proba[:, 1])
            f1 = f1_score(y_te, pred, zero_division=0)
            mcc = matthews_corrcoef(y_te, pred)
            ba = balanced_accuracy_score(y_te, pred)
            aucs.append(auc); f1s.append(f1); mccs.append(mcc); bas.append(ba)
        except Exception:
            pass

    if not aucs:
        return _skip(dataset, da_method, classifier, seed, "all_folds_failed")

    runtime = time.time() - t0
    return {
        "dataset": dataset, "da": da_method, "classifier": classifier,
        "seed": seed, "n_samples": n, "n_pos": n_pos,
        "imb_ratio": round(imb_ratio, 4),
        "auc_mean": round(float(np.mean(aucs)), 6),
        "auc_std": round(float(np.std(aucs)), 6),
        "f1_mean": round(float(np.mean(f1s)), 6),
        "f1_std": round(float(np.std(f1s)), 6),
        "mcc_mean": round(float(np.mean(mccs)), 6),
        "mcc_std": round(float(np.std(mccs)), 6),
        "ba_mean": round(float(np.mean(bas)), 6),
        "ba_std": round(float(np.std(bas)), 6),
        "n_folds_ok": len(aucs),
        "status": "ok", "skip_reason": "", "runtime_seconds": round(runtime, 4),
    }


def _skip(dataset, da, clf, seed, reason):
    return {
        "dataset": dataset, "da": da, "classifier": clf, "seed": seed,
        "n_samples": 0, "n_pos": 0, "imb_ratio": 0,
        "auc_mean": 0, "auc_std": 0, "f1_mean": 0, "f1_std": 0,
        "mcc_mean": 0, "mcc_std": 0, "ba_mean": 0, "ba_std": 0,
        "n_folds_ok": 0, "status": "skipped", "skip_reason": reason, "runtime_seconds": 0,
    }


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------
def build_cells(args) -> List[tuple]:
    """Build (runner_fn, cell_kwargs) tuples."""
    datasets = list_datasets("static")
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    das = list(DA_METHODS)

    specialist_names = list(SPECIALIST_REGISTRY.keys())

    # Lazy-load classifiers only when standard classifiers are needed
    standard_names = []
    if args.mode in ("standard_only", "both"):
        _, CLASSIFIER_REGISTRY = _get_classifier_api()
        standard_names = list(CLASSIFIER_REGISTRY.keys())

    cells = []

    if args.mode in ("specialists_only", "both"):
        seeds = ([42] if args.smoke else [int(s) for s in args.seeds.split(",")])
        for d in datasets:
            for da in das:
                for c in specialist_names:
                    for s in seeds:
                        cells.append((_run_specialist_cell, {"dataset": d, "da_method": da,
                                                           "specialist_name": c, "seed": s,
                                                           "n_folds": args.n_folds}))

    if args.mode in ("standard_only", "both"):
        seeds = ([42, 2027] if args.smoke else [int(s) for s in args.seeds.split(",")])
        for d in datasets:
            for da in das:
                for c in standard_names:
                    for s in seeds:
                        cells.append((_run_standard_cell, {"dataset": d, "da_method": da,
                                                           "classifier": c, "seed": s,
                                                           "n_folds": args.n_folds}))

    if args.max_cells:
        cells = cells[: args.max_cells]
    return cells


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Paper E v9 grid runner")
    parser.add_argument("--mode", default="specialists_only",
                        choices=["specialists_only", "standard_only", "both"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="42,2027,1729")
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--max_cells", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cells = build_cells(args)
    print(f"[v9] mode={args.mode} | cells={len(cells)} | device={DEVICE}")

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = ROOT / "runs" / f"v9_{args.mode}" / "grid_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[v9] out={out_path}")

    fieldnames = ["dataset", "da", "classifier", "seed", "n_samples", "n_pos",
                  "imb_ratio", "auc_mean", "auc_std", "f1_mean", "f1_std",
                  "mcc_mean", "mcc_std", "ba_mean", "ba_std", "n_folds_ok",
                  "status", "skip_reason", "runtime_seconds"]

    existing_ok = set()
    if out_path.exists() and out_path.stat().st_size:
        with open(out_path, newline="") as existing_file:
            for row in csv.DictReader(existing_file):
                if row.get("status") == "ok":
                    existing_ok.add((row["dataset"], row["da"], row["classifier"], str(row["seed"])))

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    fout = open(out_path, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        fout.flush()

    t0 = time.time()
    done = skipped = already_done = 0

    import torch as _torch
    # Import clear_cache for specialist runs
    if args.mode in ("specialists_only", "both"):
        from specialists import BaseSpecialistClassifier

    cells_per_flush = 36  # Clear model cache every dataset (4 specialists × 9 DAs × 1 seed)
    for fn, kwargs in cells:
        cell_key = (kwargs["dataset"], kwargs["da_method"],
                    kwargs.get("classifier", "") or kwargs.get("specialist_name", ""), str(kwargs["seed"]))
        if cell_key in existing_ok:
            already_done += 1
            continue
        try:
            result = fn(**kwargs)
        except Exception as e:
            result = _skip(kwargs.get("dataset", ""), kwargs.get("da_method", ""),
                           kwargs.get("classifier", "") or kwargs.get("specialist_name", ""),
                           kwargs.get("seed", 0), f"crash: {type(e).__name__}: {e}")
        writer.writerow(result)
        # Memory cleanup (skipped for specialist-only mode on CPU — no MPS to clear)
        if DEVICE == "mps":
            try:
                gc.collect()
                _torch.mps.empty_cache()
            except Exception:
                pass
        # Periodic model-cache clear for specialist runs (force full MPS memory reset)
        if DEVICE == "mps" and args.mode in ("specialists_only", "both") and done > 0 and done % cells_per_flush == 0:
            print(f"  [cache flush at cell {done}]")
            BaseSpecialistClassifier.clear_cache()
            gc.collect()
            try:
                _torch.mps.empty_cache()
            except Exception:
                pass
        done += 1
        if result["status"] == "skipped":
            skipped += 1
        if done % 10 == 0:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1)
            eta = (len(cells) - already_done - done) / max(rate, 0.01)
            print(f"  [{done}/{len(cells)-already_done}] {rate:.2f} cells/s  ETA:{eta/60:.1f}min  skip:{skipped}")

    fout.close()
    elapsed = time.time() - t0
    print(f"\n=== Done: {done} cells | {skipped} skipped | {elapsed:.1f}s ({done/max(elapsed,1):.2f} cells/s) ===")
    print(f"Output: {out_path}")

    report_path = out_path.parent / f"resume_report_{time.strftime('%Y%m%d_%H%M%S')}.md"
    report_path.write_text(
        "# V9 Grid Resume Report\n\n"
        f"- Total cells attempted: **{len(cells)}**\n"
        f"- Already-done skipped: **{already_done}**\n"
        f"- New cells run: **{done}**\n"
        f"- New cells skipped: **{skipped}**\n"
        f"- Runtime seconds: **{elapsed:.1f}**\n"
        f"- Output: `{out_path}`\n"
    )
    print(f"Resume report: {report_path}")


if __name__ == "__main__":
    main()
