"""
Honest evaluation utilities — additive module, no edits to existing symbols.

Purpose: make the site/dataset confound visible and report metrics at the unit
the TransformEEG paper actually uses (per-split distribution, subject-level
aggregation), instead of segment-level balanced accuracy on a pooled corpus.

Three things live here:
  1. site_prior_null      — balanced accuracy reachable with ZERO neural info,
                            using only "which dataset is this from -> predict that
                            dataset's majority class". This is the real null model
                            for the COMBINED N-LNSO protocol (not 0.50).
  2. subject_level_metrics — aggregate per-segment scores to one prediction per
                            subject before scoring. This is the clinically
                            meaningful unit and removes segment-count domination.
  3. fold_summary / bootstrap_ci — report median + IQR across folds (paper's unit)
                            and bootstrap confidence intervals.

A "sample" everywhere below is the project's standard tuple:
    (segment_tensor, label:int, subject_key:str)  where subject_key == "ds_id/sub-XX".
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score


def _ds_of(subject_key: str) -> str:
    """'ds004584/sub-12' -> 'ds004584'."""
    return subject_key.split("/", 1)[0]


# ── 1. Site-prior null ────────────────────────────────────────────────────────

def site_prior_null(samples):
    """
    Balanced accuracy of a classifier that sees ONLY the dataset of origin and
    predicts that dataset's majority class. Uses no EEG signal at all.

    Returns dict with segment-level and subject-level balanced accuracy, plus the
    per-dataset majority decision used. If this number is >= a model's reported
    balanced accuracy on the same pool, the model's score cannot be attributed to
    pathology detection.
    """
    labels = np.array([s[1] for s in samples])
    ds = np.array([_ds_of(s[2]) for s in samples])
    subj = np.array([s[2] for s in samples])

    # Per-dataset majority class over segments.
    majority = {}
    for d in np.unique(ds):
        m = ds == d
        majority[d] = int(labels[m].mean() >= 0.5)  # 1 if PD-majority else 0

    seg_pred = np.array([majority[d] for d in ds])

    # Segment level.
    seg_ba = balanced_accuracy_score(labels, seg_pred)

    # Subject level: one row per subject (label is constant within subject).
    subj_keys = np.unique(subj)
    subj_true, subj_pred = [], []
    for sk in subj_keys:
        m = subj == sk
        subj_true.append(int(labels[m][0]))
        subj_pred.append(majority[_ds_of(sk)])
    subj_true = np.array(subj_true)
    subj_pred = np.array(subj_pred)
    subj_ba = balanced_accuracy_score(subj_true, subj_pred)

    return {
        "segment_balanced_accuracy": float(seg_ba),
        "subject_balanced_accuracy": float(subj_ba),
        "per_dataset_majority": {d: ("PD" if v == 1 else "HC") for d, v in majority.items()},
        "n_segments": int(len(labels)),
        "n_subjects": int(len(subj_keys)),
    }


# ── 2. Subject-level metrics ──────────────────────────────────────────────────

def subject_level_metrics(scores, labels, subjects, threshold=0.5):
    """
    Aggregate per-segment probabilities to one prediction per subject (soft vote:
    mean probability over the subject's segments), then score at the subject level.

    scores   : array-like of per-segment P(PD) in [0,1]
    labels   : array-like of per-segment true labels (constant within subject)
    subjects : array-like of per-segment subject keys
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    subjects = np.asarray(subjects)

    subj_keys = np.unique(subjects)
    subj_true, subj_prob = [], []
    for sk in subj_keys:
        m = subjects == sk
        subj_true.append(int(labels[m][0]))
        subj_prob.append(float(scores[m].mean()))
    subj_true = np.array(subj_true)
    subj_prob = np.array(subj_prob)
    subj_pred = (subj_prob > threshold).astype(int)

    out = {
        "balanced_accuracy": float(balanced_accuracy_score(subj_true, subj_pred)),
        "sensitivity": float(recall_score(subj_true, subj_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(subj_true, subj_pred, pos_label=0, zero_division=0)),
        "n_subjects": int(len(subj_keys)),
        "n_pd": int((subj_true == 1).sum()),
        "n_hc": int((subj_true == 0).sum()),
    }
    # ROC-AUC needs both classes present.
    if out["n_pd"] > 0 and out["n_hc"] > 0:
        from sklearn.metrics import roc_auc_score
        try:
            out["roc_auc"] = float(roc_auc_score(subj_true, subj_prob))
        except Exception:
            pass
    return out


def segment_level_metrics(scores, labels, threshold=0.5):
    """Segment-level metrics (what the current pipeline reports), kept for comparison."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pred = (scores > threshold).astype(int)
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "sensitivity": float(recall_score(labels, pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(labels, pred, pos_label=0, zero_division=0)),
        "n_segments": int(len(labels)),
    }
    if len(np.unique(labels)) == 2:
        from sklearn.metrics import roc_auc_score
        try:
            out["roc_auc"] = float(roc_auc_score(labels, scores))
        except Exception:
            pass
    return out


# ── 3. Distribution across folds / bootstrap CI ───────────────────────────────

def fold_summary(values):
    """Mean / median / IQR / [1,99] range across folds — the paper reports median + IQR."""
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std()),
        "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
        "q25": float(np.percentile(v, 25)),
        "q75": float(np.percentile(v, 75)),
        "p01": float(np.percentile(v, 1)),
        "p99": float(np.percentile(v, 99)),
        "n_folds": int(len(v)),
    }


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    """Bootstrap CI for the mean of per-fold (or per-subject) values."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)])
    return {
        "mean": float(v.mean()),
        "ci_low": float(np.percentile(boots, 100 * alpha / 2)),
        "ci_high": float(np.percentile(boots, 100 * (1 - alpha / 2))),
        "alpha": alpha,
        "n_boot": n_boot,
    }
