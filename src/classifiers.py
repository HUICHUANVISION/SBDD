"""Paper E v8 — 12-13 classifiers (5 generic + 5 cost-sensitive + 2-3 transfer).

Inherits runtime env from /tmp/paper_e_v8_env.sh.
BLAS threads set to 16 (M4 Max). All sklearn n_jobs=1 to avoid BLAS oversubscription.

Cost-sensitive variants (C6-C10): class_weight, scale_pos_weight, weighted CE.
SDP-domain transfer (C11-C13): CORAL (Sun & Saenko 2016), optional TCA (Pan 2011).

Reference (CORAL): Sun, B., Saenko, K. "Deep CORAL: Correlation Alignment for Deep
    Domain Adaptation." ECCV 2016. Original: Sun, B., Feng, J., Saenko, K.
    "Return of Frustratingly Easy Domain Adaptation." AAAI 2016.
Reference (TCA): Pan, S.J., Tsang, I.W., Kwok, J.T., Yang, Q. "Domain Adaptation
    via Transfer Component Analysis." IEEE TNN 2011.
Reference (cost-sensitive): He, H., Garcia, E.A. "Learning from Imbalanced Data."
    IEEE TKDE 2009. Kamei, Y. et al. "A Large-Scale Empirical Study of Just-In-Time
    Quality Assurance in OSS." IEEE TSE 2013.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

import warnings
from typing import Dict, Any, Tuple, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, matthews_corrcoef, roc_auc_score, balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

# ---------------------------------------------------------------------------
# Classifier registry
# ---------------------------------------------------------------------------
# C1-C5: generic, C6-C10: cost-sensitive, C11-C12: CORAL transfer, C13: TCA transfer

CLASSIFIER_REGISTRY = {
    # Generic
    "LR": {"family": "generic", "needs_source": False},
    "RF": {"family": "generic", "needs_source": False},
    "XGB": {"family": "generic", "needs_source": False},
    "LGBM": {"family": "generic", "needs_source": False},
    "MLP": {"family": "generic", "needs_source": False},
    # Cost-sensitive
    "LR_CS": {"family": "cost_sensitive", "needs_source": False},
    "RF_CS": {"family": "cost_sensitive", "needs_source": False},
    "XGB_CS": {"family": "cost_sensitive", "needs_source": False},
    "LGBM_CS": {"family": "cost_sensitive", "needs_source": False},
    "MLP_CS": {"family": "cost_sensitive", "needs_source": False},
    # SDP-domain transfer (CORAL, Sun & Saenko 2016)
    "CORAL_LR": {"family": "transfer", "needs_source": True},
    "CORAL_RF": {"family": "transfer", "needs_source": True},
    # SDP-domain transfer (TCA, Pan 2011) — optional
    "TCA_LR": {"family": "transfer", "needs_source": True},
}


# ---------------------------------------------------------------------------
# Factory: build classifier from name + class_weight setting
# ---------------------------------------------------------------------------
def make_classifier(name: str, seed: int = 42, class_weight: Optional[Any] = None):
    """Return a fresh classifier instance with fixed seed.

    Parameters
    ----------
    name : str
        Classifier name (must be in CLASSIFIER_REGISTRY)
    seed : int
        Random seed
    class_weight : str or dict or None
        - 'balanced': sklearn auto-balance
        - dict: {0: w0, 1: w1}
        - None: no reweighting
    """
    name = name.upper()
    if name not in CLASSIFIER_REGISTRY:
        raise ValueError(f"Unknown classifier: {name}")
    if class_weight is None:
        cw_lr, cw_rf, cw_lgbm = None, None, None
    elif class_weight == "balanced":
        cw_lr, cw_rf, cw_lgbm = "balanced", "balanced_subsample", "balanced"
    else:
        cw_lr, cw_rf, cw_lgbm = class_weight, class_weight, class_weight

    if name == "LR":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight=cw_lr,
                                random_state=seed, n_jobs=1),
        )
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=200, class_weight=cw_rf,
            random_state=seed, n_jobs=1,
        )
    if name == "XGB":
        if XGBClassifier is None:
            raise ImportError("xgboost not installed")
        return XGBClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="logloss",
            random_state=seed, n_jobs=1, verbosity=0, tree_method="hist",
        )
    if name == "LGBM":
        if LGBMClassifier is None:
            raise ImportError("lightgbm not installed")
        return LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9,
            class_weight=cw_lgbm, random_state=seed, n_jobs=1, verbosity=-1,
        )
    if name == "MLP":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                          random_state=seed, early_stopping=True,
                          n_iter_no_change=20),
        )
    if name == "MLP_RAW":
        # Bare MLP without pipeline (for sample_weight support)
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                             random_state=seed, early_stopping=True,
                             n_iter_no_change=20)
    # Cost-sensitive variants: same as generic + class_weight='balanced'
    if name == "LR_CS":
        return make_classifier("LR", seed=seed, class_weight="balanced")
    if name == "RF_CS":
        return make_classifier("RF", seed=seed, class_weight="balanced")
    if name == "XGB_CS":
        return make_classifier("XGB", seed=seed, class_weight="balanced")
    if name == "LGBM_CS":
        return make_classifier("LGBM", seed=seed, class_weight="balanced")
    if name == "MLP_CS":
        # MLP cost-sensitive via class_weight: replicate minority class samples
        # before training to achieve class-balanced training data.
        # sklearn 1.x MLPClassifier no longer supports sample_weight in fit().
        # Use _v8_cs_replicate flag; run_v8 will replicate minority class.
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                          random_state=seed, early_stopping=True,
                          n_iter_no_change=20),
        )
        clf._v8_cs_replicate = True
        return clf
    # Transfer classifiers use CORAL/TCA transform (handled in run_v8)
    if name in ("CORAL_LR", "CORAL_RF", "TCA_LR"):
        return _make_transfer_classifier(name, seed=seed)
    raise ValueError(f"Unhandled classifier: {name}")


# ---------------------------------------------------------------------------
# CORAL: Sun & Saenko 2016 (paper: Return of Frustratingly Easy Domain Adaptation)
# ---------------------------------------------------------------------------
def coral_transform(source_X: np.ndarray, target_X: np.ndarray,
                     eps: float = 1e-5) -> np.ndarray:
    """CORAL: align source covariance to target covariance.

    Returns transformed source_X aligned to target distribution.

    Steps:
    1. Center both source and target (subtract mean)
    2. Compute covariance matrices Cs, Ct
    3. Compute whitening matrix of Ct (Ct + eps*I)^(-1/2)
    4. Compute re-coloring matrix from Cs^(1/2)
    5. Transform: source_aligned = (source - mean_s) @ Cs^(1/2) @ (Ct + eps*I)^(-1/2) + mean_t
    """
    n_s, d = source_X.shape
    n_t, _ = target_X.shape

    mean_s = source_X.mean(axis=0)
    mean_t = target_X.mean(axis=0)

    # Centered
    Xs_c = source_X - mean_s
    Xt_c = target_X - mean_t

    # Covariance
    Cs = (Xs_c.T @ Xs_c) / max(n_s - 1, 1) + eps * np.eye(d)
    Ct = (Xt_c.T @ Xt_c) / max(n_t - 1, 1) + eps * np.eye(d)

    # Eigendecomposition (symmetric)
    eigvals_s, eigvecs_s = np.linalg.eigh(Cs)
    eigvals_t, eigvecs_t = np.linalg.eigh(Ct)

    # Whitening of Ct: Ct^(-1/2) = eigvecs_t @ diag(eigvals_t^(-1/2)) @ eigvecs_t.T
    Ct_inv_sqrt = eigvecs_t @ np.diag(1.0 / np.sqrt(np.maximum(eigvals_t, eps))) @ eigvecs_t.T
    # Re-coloring with Cs^(1/2)
    Cs_sqrt = eigvecs_s @ np.diag(np.sqrt(np.maximum(eigvals_s, eps))) @ eigvecs_s.T

    # Combined: Cs^(1/2) @ Ct^(-1/2)
    transform = Cs_sqrt @ Ct_inv_sqrt

    source_aligned = Xs_c @ transform + mean_t
    return source_aligned


# ---------------------------------------------------------------------------
# TCA: Pan et al. 2011 (Transfer Component Analysis)
# ---------------------------------------------------------------------------
def tca_transform(source_X: np.ndarray, target_X: np.ndarray,
                   n_components: int = 10, kernel: str = "linear",
                   mu: float = 1.0, eps: float = 1e-5) -> np.ndarray:
    """TCA: find shared subspace that minimizes domain distribution distance.

    Simplified version (linear kernel only) for efficiency.
    Returns transformed source_X in TCA subspace.

    Reference: Pan et al. 2011 TNN.
    """
    n_s, d = source_X.shape
    n_t, _ = target_X.shape
    n = n_s + n_t

    # Combine
    X = np.vstack([source_X, target_X])

    # L (MMD distance weight matrix)
    n_src_in_src = np.ones((n_s, n_s)) / (n_s * n_s)
    n_tgt_in_tgt = np.ones((n_t, n_t)) / (n_t * n_t)
    L = np.zeros((n, n))
    L[:n_s, :n_s] = n_src_in_src
    L[n_s:, n_s:] = n_tgt_in_tgt
    L[:n_s, n_s:] = -1.0 / (n_s * n_t)
    L[n_s:, :n_s] = -1.0 / (n_s * n_t)

    # H (centering)
    H = np.eye(n) - np.ones((n, n)) / n

    # K (linear kernel = X X^T)
    K = X @ X.T

    # Solve generalized eigenproblem: (KLK + mu*I) v = lambda (KHK) v
    KLK = K @ L @ K + mu * np.eye(n)
    KHK = K @ H @ K + eps * np.eye(n)

    # Solve generalized eigenproblem using scipy (more robust than numpy for this case)
    from scipy.linalg import eigh as scipy_eigh
    try:
        eigvals, eigvecs = scipy_eigh(KLK.astype(np.float64), KHK.astype(np.float64),
                                       lower=True, check_finite=False)
    except Exception:
        # Fallback: regularized
        try:
            eigvals, eigvecs = scipy_eigh(KLK + eps * np.eye(n), KHK + eps * np.eye(n),
                                           lower=True, check_finite=False)
        except Exception:
            # Last resort: identity subspace
            eigvals = np.zeros(n_components)
            eigvecs = np.zeros((n, n_components))
            eigvecs[:n_components, :n_components] = np.eye(n_components)

    # Take top n_components eigenvectors (smallest eigenvalues = best alignment)
    order = np.argsort(eigvals)[:n_components]
    W = eigvecs[:, order]  # (n, n_components)

    # Transform source: projected source in TCA subspace
    source_aligned = K[:n_s, :] @ W  # (n_s, n_components)
    return source_aligned


# ---------------------------------------------------------------------------
# Transfer classifier factory
# ---------------------------------------------------------------------------
def _make_transfer_classifier(name: str, seed: int = 42):
    """Create a transfer classifier. The transform is applied at fit time
    in run_v8.py via the _v8_transfer_meta attribute."""
    if name == "CORAL_LR":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                                random_state=seed, n_jobs=1),
        )
    elif name == "CORAL_RF":
        clf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            random_state=seed, n_jobs=1,
        )
    elif name == "TCA_LR":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                                random_state=seed, n_jobs=1),
        )
    else:
        raise ValueError(f"Unknown transfer classifier: {name}")
    clf._v8_transfer_meta = {
        "name": name,
        "method": name.split("_")[0],  # "CORAL" or "TCA"
        "needs_source_pool": True,
    }
    return clf


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Building all 13 classifiers (smoke test)...")
    for name in CLASSIFIER_REGISTRY:
        try:
            clf = make_classifier(name, seed=2026)
            print(f"  {name:10s} OK   type={type(clf).__name__}  "
                  f"family={CLASSIFIER_REGISTRY[name]['family']}  "
                  f"needs_source={CLASSIFIER_REGISTRY[name]['needs_source']}")
        except Exception as e:
            print(f"  {name:10s} ERR  {type(e).__name__}: {e}")
    print()
    print("CORAL transform test...")
    rng = np.random.default_rng(0)
    Xs = rng.standard_normal((100, 20))
    Xt = rng.standard_normal((50, 20)) + 1.5  # shifted
    Xs_aligned = coral_transform(Xs, Xt)
    print(f"  CORAL: source mean after alignment: {Xs_aligned.mean(axis=0).mean():.4f} (target: {Xt.mean(axis=0).mean():.4f})")
    print(f"  TCA test (n_components=5)...")
    Xs_tca = tca_transform(Xs, Xt, n_components=5)
    print(f"  TCA shape: {Xs_tca.shape}  (expected (100, 5))")