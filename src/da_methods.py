"""Unified Data Augmentation interface for Paper E.

7 DA methods + 1 baseline, all returning (X_aug, y_aug) with consistent
random state handling. This module is the single source of truth for
"what does it mean to apply DA method M to dataset D" in the paper.

Author: prime_minister
Date: 2026-06-12
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Any, Optional

# Optional deps — fail gracefully if missing
try:
    from imblearn.over_sampling import (
        SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler,
    )
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
DA_METHODS = [
    "none",              # baseline: identity
    "smote",             # Chawla 2002
    "adasyn",            # He 2008
    "borderline_smote",  # Han 2005
    "smote_tomek",       # Batista 2003 (hybrid over+under)
    "ctgan",             # Xu 2019
    "mixup",             # Zhang 2018 (linear interpolation)
    "noise",             # Gaussian feature noise injection
]


# ---------------------------------------------------------------------------
# Individual methods
# ---------------------------------------------------------------------------
def _da_none(X, y, seed=42, **kw):
    return X.copy(), y.copy()


def _da_smote(X, y, seed=42, sampling_strategy="auto", k_neighbors=5, **kw):
    if not IMBLEARN_AVAILABLE:
        raise RuntimeError("imblearn not installed")
    sm = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=min(k_neighbors, max(1, min(np.bincount(y)) - 1)),
        random_state=seed,
    )
    return sm.fit_resample(X, y)


def _da_adasyn(X, y, seed=42, sampling_strategy="auto", n_neighbors=5, **kw):
    if not IMBLEARN_AVAILABLE:
        raise RuntimeError("imblearn not installed")
    ad = ADASYN(
        sampling_strategy=sampling_strategy,
        n_neighbors=min(n_neighbors, max(1, min(np.bincount(y)) - 1)),
        random_state=seed,
    )
    return ad.fit_resample(X, y)


def _da_borderline_smote(X, y, seed=42, sampling_strategy="auto", k_neighbors=5,
                          m_neighbors=10, **kw):
    if not IMBLEARN_AVAILABLE:
        raise RuntimeError("imblearn not installed")
    bs = BorderlineSMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=min(k_neighbors, max(1, min(np.bincount(y)) - 1)),
        m_neighbors=min(m_neighbors, max(1, len(X) - 1)),
        random_state=seed,
    )
    return bs.fit_resample(X, y)


def _da_smote_tomek(X, y, seed=42, **kw):
    if not IMBLEARN_AVAILABLE:
        raise RuntimeError("imblearn not installed")
    st = SMOTETomek(random_state=seed)
    return st.fit_resample(X, y)


def _da_ctgan(X, y, seed=42, n_synth_ratio=1.0, epochs=10, **kw):
    """CTGAN: train on (X, y), generate synthetic minority samples to balance.

    n_synth_ratio: how many synthetic samples to generate per real minority sample.
    epochs: CTGAN training epochs (default 10, sufficient for tabular SDP data).
    """
    if not SDV_AVAILABLE:
        raise RuntimeError("sdv not installed; pip install sdv")
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y)
    if len(counts) < 2:
        return X, y  # degenerate
    minority_label = int(np.argmin(counts))
    n_synth = int(counts[minority_label] * n_synth_ratio)
    if n_synth < 1:
        return X, y

    # Build DataFrame (CTGAN needs column names)
    n_feat = X.shape[1]
    cols = [f"f{i}" for i in range(n_feat)]
    import pandas as pd
    df_minority = pd.DataFrame(X[y == minority_label], columns=cols)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_minority)
    synth = CTGANSynthesizer(
        metadata=metadata, epochs=epochs,
        verbose=False, cuda=False,
    )
    synth.fit(df_minority)

    samples = synth.sample(num_rows=n_synth)
    # Match column order
    X_syn = samples[cols].values
    y_syn = np.full(len(X_syn), minority_label, dtype=int)
    return np.concatenate([X, X_syn.astype(np.float32)], axis=0), \
           np.concatenate([y, y_syn], axis=0)


def _da_mixup(X, y, seed=42, alpha=0.4, n_synth_ratio=1.0, **kw):
    """Mixup: linear interpolation between pairs of minority-class samples.
    Generate `n_synth_ratio * n_minority` synthetic samples.
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y)
    if len(counts) < 2:
        return X, y
    minority_label = int(np.argmin(counts))
    n_minority = counts[minority_label]
    n_synth = int(n_minority * n_synth_ratio)
    if n_synth < 1:
        return X, y

    minority_idx = np.where(y == minority_label)[0]
    pairs = rng.choice(minority_idx, size=(n_synth, 2), replace=True)
    lam = rng.beta(alpha, alpha, size=n_synth).reshape(-1, 1).astype(np.float32)
    X_syn = lam * X[pairs[:, 0]] + (1 - lam) * X[pairs[:, 1]]
    y_syn = np.full(n_synth, minority_label, dtype=int)
    return np.concatenate([X, X_syn], axis=0), np.concatenate([y, y_syn], axis=0)


def _da_noise(X, y, seed=42, sigma=0.1, n_synth_ratio=1.0, **kw):
    """Gaussian feature noise: add noise to copies of minority samples."""
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y)
    if len(counts) < 2:
        return X, y
    minority_label = int(np.argmin(counts))
    n_minority = counts[minority_label]
    n_synth = int(n_minority * n_synth_ratio)
    if n_synth < 1:
        return X, y

    minority_idx = np.where(y == minority_label)[0]
    sample_idx = rng.choice(minority_idx, size=n_synth, replace=True)
    noise = rng.normal(0, sigma, size=(n_synth, X.shape[1])).astype(np.float32)
    X_syn = X[sample_idx] + noise
    y_syn = np.full(n_synth, minority_label, dtype=int)
    return np.concatenate([X, X_syn], axis=0), np.concatenate([y, y_syn], axis=0)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
_DISPATCH = {
    "none": _da_none,
    "smote": _da_smote,
    "adasyn": _da_adasyn,
    "borderline_smote": _da_borderline_smote,
    "smote_tomek": _da_smote_tomek,
    "ctgan": _da_ctgan,
    "mixup": _da_mixup,
    "noise": _da_noise,
}


def apply_da(X: np.ndarray, y: np.ndarray, method: str, seed: int = 42, **kwargs
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Apply DA method to (X, y), return (X_aug, y_aug).

    Parameters
    ----------
    X : (n, d) feature array
    y : (n,) label array (binary, 0/1)
    method : one of DA_METHODS
    seed : random seed
    **kwargs : method-specific parameters (sampling_strategy, alpha, sigma, etc.)
    """
    method = method.lower()
    if method not in _DISPATCH:
        raise ValueError(f"Unknown DA method: {method}. Valid: {DA_METHODS}")
    X_out, y_out = _DISPATCH[method](X, y, seed=seed, **kwargs)
    return np.asarray(X_out, dtype=np.float32), np.asarray(y_out, dtype=int)


def is_available(method: str) -> Tuple[bool, str]:
    """Check if a DA method's dependencies are installed."""
    method = method.lower()
    if method in ("none", "mixup", "noise"):
        return True, "pure numpy"
    if method in ("smote", "adasyn", "borderline_smote", "smote_tomek"):
        if IMBLEARN_AVAILABLE:
            return True, "imblearn"
        return False, "imblearn not installed"
    if method == "ctgan":
        if SDV_AVAILABLE:
            return True, "sdv"
        return False, "sdv not installed"
    return False, "unknown method"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("DA Methods availability:")
    for m in DA_METHODS:
        ok, note = is_available(m)
        print(f"  {m:20s} {'OK' if ok else 'MISSING':8s} ({note})")

    # Quick smoke test
    rng = np.random.RandomState(0)
    X = rng.randn(200, 10).astype(np.float32)
    y = np.array([0] * 180 + [1] * 20, dtype=int)
    rng.shuffle(y)
    X = X[np.argsort(y)]  # group by class
    y = y[np.argsort(y)]

    print("\nSmoke test (n=200, imbalanced 180/20):")
    for m in DA_METHODS:
        if not is_available(m)[0]:
            print(f"  {m:20s} SKIP (not available)")
            continue
        try:
            Xa, ya = apply_da(X, y, m, seed=42)
            n_pos = int(ya.sum())
            print(f"  {m:20s} X:{X.shape}->{Xa.shape}  y_pos:{int(y.sum())}->{n_pos}")
        except Exception as e:
            print(f"  {m:20s} ERROR: {e}")
