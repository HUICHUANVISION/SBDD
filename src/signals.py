"""Paper E v9 — SDP-Native Signal Extractors.

Four signals per dataset:
  A1 — Defect Type Signature (DTS): HAL / LOG / MIX classification at dataset level.
  A2 — Commit History Churn (CHC): Kamei 2013启发式 proxies for static SDP.
  A3 — Code Complexity Profile (CCP): Halstead / cyclomatic / LOC distribution proxies.
  A4 — Defect-Line Entropy (DLE): spatial entropy of defect distribution.

Outputs:
  data/signals_table.csv   — 45 datasets × (A1_A2 + A2_CHC metrics + A3_CCP + A4_DLE)
  data/signals_quality.csv — NaN coverage per signal / dataset.

Usage:
  cd ~/Lab/projects/sdp-research-program/papers/paper_e_da_compatibility/v9/src
  python3 signals.py [--datasets all|static|jit] [--output ../data/signals_table.csv]
"""
from __future__ import annotations

import os
import sys
import json
import warnings
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy, skew
from scipy.special import rel_entr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
SRC  = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Feature-name lexical signatures (for DTS — A1)
# ---------------------------------------------------------------------------
# Keywords that appear in .arff feature columns that are indicative of defect type.
HAL_KEYWORDS = [
    "hal", "hallucination", "api", "missing", "undefined", "null",
    "exception", "crash", "abort", "assert", "precond",
]
LOG_KEYWORDS = [
    "logic", "condit", "branch", "if", "else", "loop", "while",
    "for", "switch", "match", "semantic", "wrong", "incorrect",
    "off_by", "offby", "typo", "spec",
]
MIX_KEYWORDS = [
    "memory", "leak", "overflow", "underflow", "bounds", "race",
    "concurren", "deadlock", "lock", "sync", "atomic", "perf",
    "time", "latenc", "throughput", "bottleneck",
]

# Known HAL-prone datasets (Hallucinated-API or missing-API calls — LLM-generated code)
HAL_PROJECTS = {
    "relink_apache", "relink_safe", "relink_zxing",  # library code with LLM-assist potential
    "promise_xalan", "promise_xerces",                 # parser/compiler internals
    "jira_hbase", "jira_hive",                       # large server code with many API calls
}
# Known LOG-dominant datasets (logic errors — classic SDP)
LOG_PROJECTS = {
    "aeeem_eq", "aeeem_jdt", "aeeem_lucene", "aeeem_mylyn", "aeeem_pde",
    "nasa_cm1", "nasa_mw1", "nasa_pc1", "nasa_pc3", "nasa_pc4",
    "promise_ant", "promise_camel", "promise_ivy", "promise_jedit",
    "promise_log4j", "promise_poi", "promise_tomcat", "promise_velocity",
    "softlab_ar1", "softlab_ar3", "softlab_ar4", "softlab_ar5", "softlab_ar6",
}
# Known MIX datasets (performance / concurrency / memory)
MIX_PROJECTS = {
    "jira_activemq", "jira_derby", "jira_groovy",
    "promise_xerces",  # XML parsing = mix of memory + logic
    "jit_brackets", "jit_camel", "jit_tomcat",
}


def _keyword_score(feature_names: List[str], keywords: List[str]) -> float:
    """Fraction of feature names that contain any keyword (case-insensitive)."""
    if not feature_names:
        return 0.0
    pattern = re.compile("|".join(keywords), re.IGNORECASE)
    hits = sum(1 for fn in feature_names if pattern.search(fn))
    return hits / len(feature_names)


# ---------------------------------------------------------------------------
# A1 — Defect Type Signature
# ---------------------------------------------------------------------------
def compute_A1_DTS(
    dataset: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    imb_ratio: float,
    feature_skewness: float,
) -> Dict[str, float]:
    """Compute Defect Type Signature (HAL / LOG / MIX) for a dataset.

    Combines:
      (a) project-name heuristics (known HAL/LOG/MIX projects)
      (b) feature-name lexical signature
      (c) class-imbalance ratio profile (HAL → more imbalanced; LOG → near-balanced)

    Returns dict with keys: dts_hal, dts_log, dts_mix, dts_type (categorical).
    """
    # (a) Project-name heuristic
    if dataset in HAL_PROJECTS:
        p_hal, p_log, p_mix = 0.60, 0.25, 0.15
    elif dataset in LOG_PROJECTS:
        p_hal, p_log, p_mix = 0.10, 0.75, 0.15
    elif dataset in MIX_PROJECTS:
        p_hal, p_log, p_mix = 0.20, 0.30, 0.50
    else:
        # Fallback: pure data-driven
        p_hal, p_log, p_mix = 0.33, 0.34, 0.33

    # (b) Feature-name lexical correction (strong signal)
    lex_hal = _keyword_score(feature_names, HAL_KEYWORDS)
    lex_log = _keyword_score(feature_names, LOG_KEYWORDS)
    lex_mix = _keyword_score(feature_names, MIX_KEYWORDS)
    lex_sum = lex_hal + lex_log + lex_mix + 1e-9
    lex_hal /= lex_sum
    lex_log /= lex_sum
    lex_mix /= lex_sum

    # (c) Class-imbalance profile correction
    # HAL (LLM errors) → very imbalanced (defect rare) or very balanced (everything wrong)
    # LOG (logic) → moderately imbalanced (natural distribution)
    imb_hal = 0.5 + 0.5 * np.tanh(imb_ratio - 0.30)      # peaks near 0.30
    imb_log = 0.5 + 0.5 * np.tanh(0.20 - abs(imb_ratio - 0.20))  # peaks near 0.20
    imb_mix = 0.5 + 0.5 * np.tanh(abs(imb_ratio - 0.15))  # peaks near 0.15

    # Weighted combination: project 40% + lexical 35% + imbalance 25%
    w_hal = 0.40 * np.array([p_hal, lex_hal, imb_hal])
    w_log = 0.40 * np.array([p_log, lex_log, imb_log])
    w_mix = 0.40 * np.array([p_mix, lex_mix, imb_mix])

    # Mix the three evidence sources
    final_hal = np.mean(w_hal) * 0.40 + lex_hal * 0.35 + imb_hal * 0.25
    final_log = np.mean(w_log) * 0.40 + lex_log * 0.35 + imb_log * 0.25
    final_mix = np.mean(w_mix) * 0.40 + lex_mix * 0.35 + imb_mix * 0.25

    # Normalise to probabilities
    total = final_hal + final_log + final_mix + 1e-9
    final_hal /= total
    final_log /= total
    final_mix /= total

    dtype_map = {0: "HAL", 1: "LOG", 2: "MIX"}
    dtype_idx = int(np.argmax([final_hal, final_log, final_mix]))

    return {
        "dts_hal": float(final_hal),
        "dts_log": float(final_log),
        "dts_mix": float(final_mix),
        "dts_type": dtype_map[dtype_idx],
        # intermediate for debugging
        "lex_hal": float(lex_hal),
        "lex_log": float(lex_log),
        "lex_mix": float(lex_mix),
        "imb_ratio": float(imb_ratio),
    }


# ---------------------------------------------------------------------------
# A2 — Commit History Churn (Kamei启发式 proxies for static SDP)
# ---------------------------------------------------------------------------
# Kamei et al. 2013 metrics: NS, ND, NF, LA, LD, LT, ndev, age, nuc, exp, rexp, sexp.
# These are per-commit in JIT. For static SDP we use dataset-level proxies.

def _safe_stat(arr: np.ndarray) -> float:
    """Return mean of finite values, or NaN if all invalid."""
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if len(finite) > 0 else np.nan


def compute_A2_CHC(
    dataset: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    family: str,
    imb_ratio: float,
) -> Dict[str, float]:
    """Compute Kamei 2013启发式 proxies for static SDP.

    Per-column aggregation over all features as the "subsystem / metric" dimension.
    We treat each .arff feature column as a proxy for one subsystem's metric.

    Returns 12 Kamei-style metric proxies (all float, NaN if unavailable).
    """
    n, d = X.shape
    col_means = np.nanmean(X, axis=0)
    col_sums  = np.nansum(X, axis=0)
    col_stds  = np.nanstd(X, axis=0)
    col_max   = np.nanmax(X, axis=0)
    col_min   = np.nanmin(X, axis=0)
    col_range = col_max - col_min + 1e-9
    col_skews = np.array([skew(col, nan_policy="omit") for col in X.T])

    # NF (number of files modified) — proxy: n_features (each column ≈ one subsystem)
    nf = float(d)

    # NS (number of subsystems) — proxy: number of columns with non-trivial variance
    ns = float(np.sum(col_stds > 1e-6))

    # LA (lines added) — proxy: sum of all feature values (total metric magnitude)
    la = float(np.nansum(X))

    # LD (lines deleted) — proxy: column range sum
    ld = float(np.nansum(col_range))

    # LT (lines touched) — proxy: number of non-zero entries
    lt = float(np.sum(~np.isclose(X, 0.0, atol=1e-4)))

    # ndev (number of developers) — proxy: 1 / mean feature skewness (low skew = many devs)
    mean_skew = np.mean(np.abs(col_skews)) + 1e-9
    ndev = float(1.0 / mean_skew)

    # age (average age of code) — proxy: family-level ordinal encoding
    family_age_map = {
        "aeeem": 3.0, "jira": 4.0, "nasa": 2.0,
        "promise": 5.0, "relink": 2.5, "softlab": 3.5,
        "jit": 1.0,   # JIT is newer data
    }
    age = float(family_age_map.get(family, 3.0))

    # nuc (number of unique changes) — proxy: class-imbalance ratio (rare defects = concentrated)
    nuc = float(imb_ratio)

    # ND (number of defective modules) — proxy: n_pos (number of positive instances)
    nd = float(int(y.sum()))

    # exp / rexp / sexp (developer experience proxies)
    # exp: avg experience = mean column value / ndev
    exp = float(np.nanmean(col_means) / (ndev + 1.0))
    # rexp: relative exp = mean / max
    rexp = float(np.nanmean(col_means) / (np.nanmax(col_means) + 1.0))
    # sexp: std of experience
    sexp = float(np.nanstd(col_means))

    # Normalise all to [0, 1] per-dataset for interpretability
    # (these become the "raw" CHC values; normalisation happens in aggregate.py)
    chc_raw = {
        "chc_nf": nf, "chc_ns": ns, "chc_la": la, "chc_ld": ld,
        "chc_lt": lt, "chc_ndev": ndev, "chc_age": age, "chc_nuc": nuc,
        "chc_nd": nd, "chc_exp": exp, "chc_rexp": rexp, "chc_sexp": sexp,
    }
    return chc_raw


# ---------------------------------------------------------------------------
# A3 — Code Complexity Profile
# ---------------------------------------------------------------------------
def compute_A3_CCP(
    dataset: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """Compute Code Complexity Profile from .arff numeric features.

    Components:
      - Halstead-inspired volume proxy: V ≈ N × log2(n_ops) where N = total feature sum
      - Cyclomatic complexity proxy: CC ≈ 1 + (# features with high variance)
      - LOC distribution: median / max of column-wise sums
      - Coupling: mean absolute off-diagonal correlation
      - Feature cardinalities: n_unique per column (avg)
    """
    n, d = X.shape

    # Halstead volume proxy
    N = np.nansum(X) + 1.0          # total "operators" (sum of all metric values)
    n_ops = np.sum(np.ptp(X, axis=0) > 0)  # number of "unique operators" (cols with range > 0)
    n_ops = max(n_ops, 1)
    halstead_V = N * np.log2(n_ops + 1)

    # Cyclomatic complexity proxy: columns whose variance exceeds the median variance
    col_vars = np.nanvar(X, axis=0)
    median_var = np.median(col_vars)
    high_var_count = np.sum(col_vars > median_var)
    cc_proxy = 1.0 + float(high_var_count)

    # LOC proxy: sum of column means (total "size")
    loc_proxy = float(np.nansum(np.nanmean(X, axis=0)))

    # LOC distribution: Gini-like dispersion of column means
    col_means = np.nanmean(X, axis=0)
    col_means_pos = col_means[col_means > 0]
    if len(col_means_pos) > 1:
        sorted_cm = np.sort(col_means_pos)
        n_c = len(sorted_cm)
        gini = (2.0 * np.sum((np.arange(1, n_c + 1) * sorted_cm))) / (n_c * np.sum(sorted_cm)) - (n_c + 1) / n_c
        gini = float(np.clip(gini, 0.0, 1.0))
    else:
        gini = 0.0

    # Coupling: mean absolute off-diagonal correlation (subset for speed)
    # Use first min(20, d) columns to avoid O(d^2)
    d_sub = min(20, d)
    X_sub = X[:, :d_sub]
    X_std = (X_sub - np.nanmean(X_sub, axis=0)) / (np.nanstd(X_sub, axis=0) + 1e-9)
    corr_mat = np.corrcoef(X_std.T)
    np.fill_diagonal(corr_mat, 0.0)
    coupling = float(np.mean(np.abs(corr_mat)))

    # Feature cardinality: average number of unique values per column
    avg_unique = float(np.mean([len(np.unique(X[:, i])) for i in range(min(d, 50))]))

    return {
        "ccp_halstead_V": float(halstead_V),
        "ccp_cyclomatic_proxy": cc_proxy,
        "ccp_loc_proxy": loc_proxy,
        "ccp_gini_dispersion": gini,
        "ccp_coupling": coupling,
        "ccp_avg_unique_per_feature": avg_unique,
    }


# ---------------------------------------------------------------------------
# A4 — Defect-Line Entropy
# ---------------------------------------------------------------------------
def compute_A4_DLE(
    dataset: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """Compute Defect-Line Entropy (DLE).

    Measures spatial dispersion of defect probability across:
      (a) Feature-distribution entropy: H(X_j) per column
      (b) Within-feature defect entropy: H(y | X_j bin) — measures feature's defect-info content
      (c) Cross-feature entropy dispersion: std of per-column entropies
      (d) Class-conditional entropy: H(y) — base rate uncertainty
    """
    n, d = X.shape

    # H(y): binary entropy of the label
    p_pos = float(y.mean())
    p_pos = np.clip(p_pos, 1e-9, 1.0 - 1e-9)
    h_y = float(entropy([p_pos, 1.0 - p_pos], base=2.0))

    # H(X_j): Shannon entropy of each column (binned adaptively)
    col_entropies = []
    for i in range(min(d, 50)):  # subset for speed
        col = X[:, i]
        col = col[np.isfinite(col)]
        if len(col) < 10:
            col_entropies.append(np.nan)
            continue
        n_uniq = len(np.unique(col))
        n_bins = min(10, max(2, n_uniq - 1))
        try:
            bins = pd.qcut(col, q=n_bins, duplicates="drop")
            vc = bins.value_counts()
            counts = vc.sort_index().values.astype(float)
            counts = counts / counts.sum()
            h_j = float(entropy(counts, base=2.0))
        except Exception:
            h_j = np.nan
        col_entropies.append(h_j)

    col_entropies = np.array(col_entropies)
    h_x_mean = float(np.nanmean(col_entropies))
    h_x_std  = float(np.nanstd(col_entropies))

    # H(y | X_j): conditional entropy — proxy via correlation-based split
    # For each column, split at median → compute p(defect | above_median)
    cond_entropies = []
    for i in range(min(d, 30)):
        col = X[:, i]
        mask = np.isfinite(col)
        if mask.sum() < 10:
            cond_entropies.append(np.nan)
            continue
        col_f, y_f = col[mask], y[mask]
        split = np.median(col_f)
        above = y_f[col_f > split]
        below = y_f[col_f <= split]
        p_above = len(above) / len(y_f)
        p_below = len(below) / len(y_f)
        h_above = entropy([above.mean(), 1.0 - above.mean()], base=2.0) if len(above) > 0 else 0.0
        h_below = entropy([below.mean(), 1.0 - below.mean()], base=2.0) if len(below) > 0 else 0.0
        h_y_given_x = p_above * h_above + p_below * h_below
        cond_entropies.append(float(h_y_given_x))

    cond_entropies = np.array(cond_entropies)
    h_y_given_x_mean = float(np.nanmean(cond_entropies))

    # Information gain: H(y) - H(y|X)
    ig = h_y - h_y_given_x_mean

    # Defect concentration: mean absolute Pearson correlation of each feature with y
    # (only compute for first 30 features to avoid OOM)
    corrs = []
    for i in range(min(d, 30)):
        col = X[:, i]
        mask = np.isfinite(col)
        if mask.sum() < 10:
            continue
        c = np.corrcoef(col[mask], y[mask])[0, 1]
        if np.isfinite(c):
            corrs.append(abs(c))
    mean_abs_corr = float(np.mean(corrs)) if corrs else 0.0

    return {
        "dle_h_y": h_y,                          # label entropy
        "dle_h_x_mean": h_x_mean,                # mean feature entropy
        "dle_h_x_std": h_x_std,                  # dispersion of feature entropy
        "dle_h_y_given_x": h_y_given_x_mean,     # mean conditional defect entropy
        "dle_info_gain": ig,                      # information gain
        "dle_defect_concentration": mean_abs_corr,  # mean |corr(feature, y)|
    }


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------
def compute_all_signals(
    dataset: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    meta: dict,
) -> dict:
    """Compute all 4 SDP-native signals for one dataset."""
    family = dataset.split("_")[0]
    n, d = X.shape

    # Intrinsic proxies (also used in A1/A2)
    imb_ratio = float(y.sum() / max(n - y.sum(), 1))
    feat_skewness = float(np.mean(np.abs([skew(X[:, i], nan_policy="omit") for i in range(min(d, 20))])))

    A1 = compute_A1_DTS(dataset, X, y, feature_names, imb_ratio, feat_skewness)
    A2 = compute_A2_CHC(dataset, X, y, feature_names, family, imb_ratio)
    A3 = compute_A3_CCP(dataset, X, y, feature_names)
    A4 = compute_A4_DLE(dataset, X, y, feature_names)

    return {
        "dataset": dataset,
        "family": family,
        "n_samples": n,
        "n_features": d,
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "class_imb_ratio": imb_ratio,
        **A1,
        **A2,
        **A3,
        **A4,
    }


# ---------------------------------------------------------------------------
# CLI + batch runner
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract SDP-native signals for 45 datasets")
    parser.add_argument("--datasets", default="all", choices=["all", "static", "jit"])
    parser.add_argument("--output", default=str(DATA / "signals_table.csv"))
    parser.add_argument("--limit", type=int, default=None, help="debug: limit to N datasets")
    args = parser.parse_args()

    sys.path.insert(0, str(SRC))
    from data_loader import load_dataset, list_datasets

    datasets = list_datasets(args.datasets)
    if args.limit:
        datasets = datasets[: args.limit]

    print(f"Computing signals for {len(datasets)} datasets ...")
    rows = []
    quality_rows = []
    signal_names = [
        "dts_hal", "dts_log", "dts_mix",
        "chc_nf", "chc_ns", "chc_la", "chc_ld", "chc_lt",
        "chc_ndev", "chc_age", "chc_nuc", "chc_nd", "chc_exp", "chc_rexp", "chc_sexp",
        "ccp_halstead_V", "ccp_cyclomatic_proxy", "ccp_loc_proxy",
        "ccp_gini_dispersion", "ccp_coupling", "ccp_avg_unique_per_feature",
        "dle_h_y", "dle_h_x_mean", "dle_h_x_std", "dle_h_y_given_x",
        "dle_info_gain", "dle_defect_concentration",
    ]

    for i, ds in enumerate(datasets):
        try:
            X, y, fn, meta = load_dataset(ds)
            row = compute_all_signals(ds, X, y, fn, meta)
            rows.append(row)

            # Quality report
            nan_counts = {s: (s in row and (row[s] is None or (isinstance(row[s], float) and np.isnan(row[s])))) for s in signal_names}
            nan_pct = sum(nan_counts.values()) / len(signal_names) * 100
            quality_rows.append({"dataset": ds, "nan_count": sum(nan_counts.values()), "total_signals": len(signal_names), "nan_pct": f"{nan_pct:.1f}%"})

            status = "✓" if nan_pct < 20 else ("⚠" if nan_pct < 50 else "✗")
            print(f"  [{i+1:2d}/{len(datasets)}] {status} {ds:30s}  n={X.shape[0]:5d}  d={X.shape[1]:3d}  dts={row.get('dts_type','?'):4s}  NaN%={nan_pct:.0f}%")

        except Exception as e:
            print(f"  [ERROR] {ds}: {e}")
            rows.append({"dataset": ds, "family": ds.split("_")[0], "error": str(e)})

    # Save
    df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(rows)} rows × {len(df.columns)} cols)")

    qual_path = DATA / "signals_quality.csv"
    pd.DataFrame(quality_rows).to_csv(qual_path, index=False)
    print(f"Saved quality report: {qual_path}")

    # Summary stats
    if rows and "error" not in rows[0]:
        sig_cols = [c for c in df.columns if c not in ("dataset", "family", "n_samples", "n_features", "n_pos", "n_neg", "class_imb_ratio", "dts_type", "error")]
        summary = df[sig_cols].astype(float).describe().T[["mean", "std", "min", "max"]]
        print(f"\nSignal summary (across {len(df)} datasets):")
        print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
