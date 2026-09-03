"""Data loader for Paper E.

Loads 35 static SDP .arff datasets + 10 JIT SDP .csv datasets from ~/Lab/datasets/.
Returns standardized (X, y, feature_names, dataset_meta) for each dataset.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.io import arff

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Dataset catalog (path + label column heuristic)
# ---------------------------------------------------------------------------
STATIC_SDP_ROOT = Path("/Users/moss/Lab/datasets/static sdp")
JIT_SDP_ROOT = Path("/Users/moss/Lab/datasets/jit-sdp-data")


# (dataset_key, filepath, label_col_hint, group)
# label_col_hint: 'last' = use last column, or explicit column name
STATIC_DATASETS: List[Tuple[str, Path, str]] = [
    # AEEEM
    ("aeeem_eq", STATIC_SDP_ROOT / "AEEEM" / "EQ.arff", "class"),
    ("aeeem_jdt", STATIC_SDP_ROOT / "AEEEM" / "JDT.arff", "class"),
    ("aeeem_lucene", STATIC_SDP_ROOT / "AEEEM" / "Lucene.arff", "class"),
    ("aeeem_mylyn", STATIC_SDP_ROOT / "AEEEM" / "Mylyn.arff", "class"),
    ("aeeem_pde", STATIC_SDP_ROOT / "AEEEM" / "PDE.arff", "class"),
    # JIRA
    ("jira_activemq", STATIC_SDP_ROOT / "JIRA" / "activemq-5.0.0.arff", "last"),
    ("jira_derby", STATIC_SDP_ROOT / "JIRA" / "derby-10.5.1.1.arff", "last"),
    ("jira_groovy", STATIC_SDP_ROOT / "JIRA" / "groovy-1_6_BETA_1.arff", "last"),
    ("jira_hbase", STATIC_SDP_ROOT / "JIRA" / "hbase-0.94.0.arff", "last"),
    ("jira_hive", STATIC_SDP_ROOT / "JIRA" / "hive-0.9.0.arff", "last"),
    ("jira_jruby", STATIC_SDP_ROOT / "JIRA" / "jruby-1.1.arff", "last"),
    ("jira_wicket", STATIC_SDP_ROOT / "JIRA" / "wicket-1.3.0-beta2.arff", "last"),
    # NASA
    ("nasa_cm1", STATIC_SDP_ROOT / "NASA" / "CM1.arff", "last"),
    ("nasa_mw1", STATIC_SDP_ROOT / "NASA" / "MW1.arff", "last"),
    ("nasa_pc1", STATIC_SDP_ROOT / "NASA" / "PC1.arff", "last"),
    ("nasa_pc3", STATIC_SDP_ROOT / "NASA" / "PC3.arff", "last"),
    ("nasa_pc4", STATIC_SDP_ROOT / "NASA" / "PC4.arff", "last"),
    # PROMISE
    ("promise_ant", STATIC_SDP_ROOT / "PROMISE" / "ant-1.7.arff", "defects"),
    ("promise_camel", STATIC_SDP_ROOT / "PROMISE" / "camel-1.4.arff", "defects"),
    ("promise_ivy", STATIC_SDP_ROOT / "PROMISE" / "ivy-2.0.arff", "defects"),
    ("promise_jedit", STATIC_SDP_ROOT / "PROMISE" / "jedit-4.0.arff", "defects"),
    ("promise_log4j", STATIC_SDP_ROOT / "PROMISE" / "log4j-1.0.arff", "defects"),
    ("promise_poi", STATIC_SDP_ROOT / "PROMISE" / "poi-2.0.arff", "defects"),
    ("promise_tomcat", STATIC_SDP_ROOT / "PROMISE" / "tomcat.arff", "defects"),
    ("promise_velocity", STATIC_SDP_ROOT / "PROMISE" / "velocity-1.6.arff", "defects"),
    ("promise_xalan", STATIC_SDP_ROOT / "PROMISE" / "xalan-2.4.arff", "defects"),
    ("promise_xerces", STATIC_SDP_ROOT / "PROMISE" / "xerces-1.3.arff", "defects"),
    # ReLink
    ("relink_apache", STATIC_SDP_ROOT / "ReLink" / "Apache.arff", "last"),
    ("relink_safe", STATIC_SDP_ROOT / "ReLink" / "Safe.arff", "last"),
    ("relink_zxing", STATIC_SDP_ROOT / "ReLink" / "Zxing.arff", "last"),
    # SOFTLAB
    ("softlab_ar1", STATIC_SDP_ROOT / "SOFTLAB" / "ar1.arff", "last"),
    ("softlab_ar3", STATIC_SDP_ROOT / "SOFTLAB" / "ar3.arff", "last"),
    ("softlab_ar4", STATIC_SDP_ROOT / "SOFTLAB" / "ar4.arff", "last"),
    ("softlab_ar5", STATIC_SDP_ROOT / "SOFTLAB" / "ar5.arff", "last"),
    ("softlab_ar6", STATIC_SDP_ROOT / "SOFTLAB" / "ar6.arff", "last"),
]

# JIT datasets — commit-level, label is typically "defective" or "bug"
JIT_DATASETS: List[Tuple[str, Path, List[str]]] = [
    # (key, path, candidate_label_cols)
    ("jit_brackets", JIT_SDP_ROOT / "brackets.csv", ["defective", "bug", "label", "is_bug", "defects"]),
    ("jit_broadleaf", JIT_SDP_ROOT / "broadleaf.csv", ["defective", "bug", "label", "is_bug", "defects"]),
    ("jit_camel", JIT_SDP_ROOT / "camel.csv", ["defective", "bug", "label", "is_bug", "defects"]),
    ("jit_fabric8", JIT_SDP_ROOT / "fabric8.csv", ["defective", "bug", "label", "is_bug", "defects"]),
    ("jit_jgroups", JIT_SDP_ROOT / "jgroups.csv", ["defective", "bug", "label", "is_bug", "defects"]),
    ("jit_neutron", JIT_SDP_ROOT / "neutron.csv", ["defective", "bug", "label", "is_bug", "defects"]),
    ("jit_nova", JIT_SDP_ROOT / "nova.csv", ["defective", "bug", "label", "is_bug", "defects"]),
    ("jit_npm", JIT_SDP_ROOT / "npm.csv", ["defective", "bug", "label", "is_bug", "defects"]),
    ("jit_spring_integration", JIT_SDP_ROOT / "spring-integration.csv", ["defective", "bug", "label", "is_bug", "defects"]),
    ("jit_tomcat", JIT_SDP_ROOT / "tomcat.csv", ["defective", "bug", "label", "is_bug", "defects"]),
]


# ---------------------------------------------------------------------------
# Label semantics
# ---------------------------------------------------------------------------
_DEFECT_POSITIVE_TOKENS = {
    "1", "bug", "buggy", "defect", "defective", "faulty", "true", "y", "yes",
}
_DEFECT_NEGATIVE_TOKENS = {
    "0", "clean", "false", "n", "no", "non-buggy", "non-defective",
    "nonbuggy", "nondefective",
}


def _binary_defect_labels(y_raw: pd.Series) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Map raw SDP labels to 1=defective and 0=clean, or fail closed."""
    if pd.api.types.is_numeric_dtype(y_raw):
        numeric = pd.to_numeric(y_raw, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("label column contains non-finite numeric values")
        y = (numeric > 0).astype(int)
        rule = "numeric_gt_zero"
        positive_raw = sorted({float(v) for v in numeric if v > 0})
        negative_raw = sorted({float(v) for v in numeric if v <= 0})
    else:
        normalized = y_raw.map(
            lambda value: str(value.decode() if isinstance(value, bytes) else value)
            .strip()
            .lower()
        )
        observed = set(normalized.unique())
        unknown = observed - _DEFECT_POSITIVE_TOKENS - _DEFECT_NEGATIVE_TOKENS
        if unknown:
            # Numeric strings are accepted under the same >0 defect-count rule.
            numeric = pd.to_numeric(normalized, errors="coerce")
            if numeric.isna().any():
                raise ValueError(
                    "unknown defect label token(s): " + ", ".join(sorted(unknown))
                )
            y = (numeric.to_numpy(dtype=float) > 0).astype(int)
            rule = "numeric_string_gt_zero"
            positive_raw = sorted({str(v) for v in normalized[numeric > 0]})
            negative_raw = sorted({str(v) for v in normalized[numeric <= 0]})
        else:
            y = normalized.isin(_DEFECT_POSITIVE_TOKENS).astype(int).to_numpy()
            rule = "explicit_defect_tokens"
            positive_raw = sorted(observed & _DEFECT_POSITIVE_TOKENS)
            negative_raw = sorted(observed & _DEFECT_NEGATIVE_TOKENS)

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            "expected both clean and defective classes after semantic mapping; "
            f"got {sorted(np.unique(y).tolist())}"
        )
    return y, {
        "label_rule": rule,
        "positive_class": "defective",
        "positive_raw_values": positive_raw,
        "negative_raw_values": negative_raw,
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_arff(path: Path, label_hint: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """Load .arff file, return (X, y, feature_names, meta)."""
    with open(path, "r", errors="ignore") as f:
        data, meta = arff.loadarff(f)
    df = pd.DataFrame(data)
    # Decode bytes
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda v: v.decode() if isinstance(v, bytes) else v)

    # Pick label column
    if label_hint == "last":
        label_col = df.columns[-1]
    elif label_hint in df.columns:
        label_col = label_hint
    else:
        # Fall back: pick a column with 2 unique values
        for col in df.columns:
            if df[col].nunique() == 2:
                label_col = col
                break
        else:
            label_col = df.columns[-1]

    y_raw = df[label_col]
    feat_cols = [c for c in df.columns if c != label_col]
    X = df[feat_cols]

    # Convert categorical to numeric
    X = pd.get_dummies(X, drop_first=True).astype(np.float32)
    # Map labels explicitly to 1=defective, 0=clean. Never use lexical ordering.
    y, label_meta = _binary_defect_labels(y_raw)
    return X.values.astype(np.float32), y, list(X.columns), {
        "label_col": label_col,
        "n_features": X.shape[1],
        **label_meta,
    }


def _load_jit_csv(path: Path, candidate_label_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """Load JIT .csv file, identify label column, return (X, y, ...)."""
    # Some files have wide rows; bump csv field size
    import csv
    csv.field_size_limit(10**7)
    df = pd.read_csv(path, low_memory=False)
    # Find label col
    label_col = None
    for c in candidate_label_cols:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        # Fall back: pick 2-unique-value column
        for c in df.columns:
            if df[c].nunique() == 2:
                label_col = c
                break
    if label_col is None:
        raise ValueError(f"{path.name}: no label column found among {candidate_label_cols}")

    y_raw = df[label_col]
    feat_cols = [c for c in df.columns if c != label_col]
    X = df[feat_cols]

    # Drop non-numeric columns (commit hash, author, etc.)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols].fillna(0).astype(np.float32)

    y, label_meta = _binary_defect_labels(y_raw)
    return X.values.astype(np.float32), y, list(X.columns), {
        "label_col": label_col,
        "n_features": X.shape[1],
        **label_meta,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """Load a dataset by key. See STATIC_DATASETS and JIT_DATASETS for valid keys."""
    for key, path, label_hint in STATIC_DATASETS:
        if key == name:
            return _load_arff(path, label_hint)
    for key, path, cand in JIT_DATASETS:
        if key == name:
            return _load_jit_csv(path, cand)
    raise KeyError(f"Dataset {name!r} not found")


def list_datasets(group: str = "all") -> List[str]:
    """List available dataset keys. group: 'all' | 'static' | 'jit'."""
    if group == "static":
        return [k for k, _, _ in STATIC_DATASETS]
    if group == "jit":
        return [k for k, _, _ in JIT_DATASETS]
    return [k for k, _, _ in STATIC_DATASETS] + [k for k, _, _ in JIT_DATASETS]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Static: {len(STATIC_DATASETS)} datasets")
    print(f"JIT:    {len(JIT_DATASETS)} datasets")
    print(f"Total:  {len(list_datasets())} datasets")
    print()
    print("Smoke test (first 3 of each group):")
    for k in list_datasets("static")[:3] + list_datasets("jit")[:3]:
        try:
            X, y, fn, meta = load_dataset(k)
            n_pos = int(y.sum())
            n_neg = int((1 - y).sum())
            imb = n_pos / max(n_neg, 1)
            print(f"  {k:30s} X:{X.shape}  y:0/1={n_neg}/{n_pos}  imb={imb:.3f}  meta={meta}")
        except Exception as e:
            print(f"  {k:30s} ERROR: {e}")
