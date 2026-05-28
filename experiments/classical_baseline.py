"""Classical sanity baseline — band-power features + logistic regression.

Purpose: Decisive signal test. If this hits ~70%+ balanced accuracy, the EEG
signal is real and the bug is in the deep model/training. If it also sits at
~56%, the problem is upstream: labels, preprocessing, or data loading.

CPU-only. Runs in ~5 minutes. No GPU, no heavy memory pressure.

Evaluation: same N-LNSO subject-held-out CV as TransformEEG baseline.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.finetune import LabeledEEGDataset  # reuse existing dataset loader

DATASET_IDS  = ["ds002778", "ds003490", "ds004584"]
PROCESSED_DIR = REPO_ROOT / "data" / "processed_unified"
RESULTS_DIR   = REPO_ROOT / "results" / "classical"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Band definitions (Hz) — delta, theta, alpha, beta, low-gamma
BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
FS    = 250   # assumed sample rate after preprocessing


def bandpower(segment: np.ndarray, fs: int, low: float, high: float) -> np.ndarray:
    """Mean power in [low, high] Hz via Welch PSD — shape (n_channels,)."""
    from scipy.signal import welch
    _, pxx = welch(segment, fs=fs, nperseg=min(256, segment.shape[-1]), axis=-1)
    freqs  = np.fft.rfftfreq(min(256, segment.shape[-1]), 1.0 / fs)
    mask   = (freqs >= low) & (freqs < high)
    return pxx[:, mask].mean(axis=-1)  # (n_channels,)


def extract_features(segment: np.ndarray) -> np.ndarray:
    """Extract band-power features from one EEG segment.

    segment: (n_channels, n_times)
    returns: flat feature vector (n_channels * n_bands,)
    """
    feats = []
    for low, high in BANDS:
        feats.append(bandpower(segment, FS, low, high))
    return np.concatenate(feats)  # (n_channels * n_bands,)


def run_dataset(ds_id: str) -> dict:
    labels_csv = PROCESSED_DIR / ds_id / "labels.csv"
    data_dir   = PROCESSED_DIR / ds_id
    if not labels_csv.exists():
        print(f"  [{ds_id}] SKIPPED — no labels.csv in {data_dir}")
        return {}

    ds = LabeledEEGDataset(str(data_dir), str(labels_csv))
    print(f"  [{ds_id}] {len(ds)} segments loaded")

    # Group segments by subject
    from collections import defaultdict
    subj_map: dict[str, list[int]] = defaultdict(list)
    for idx, (_, _, subj) in enumerate(ds.samples):
        subj_map[subj].append(idx)
    subjects = sorted(subj_map.keys())
    print(f"  [{ds_id}] {len(subjects)} subjects")

    # Pre-extract features for all segments
    print(f"  [{ds_id}] Extracting band-power features...", flush=True)
    all_feats  = []
    all_labels = []
    all_subjs  = []
    for idx in range(len(ds)):
        seg, label, subj = ds.samples[idx]
        if isinstance(seg, __import__("torch").Tensor):
            seg = seg.numpy()
        feats = extract_features(seg.astype(np.float32))
        all_feats.append(feats)
        all_labels.append(int(label))
        all_subjs.append(subj)

    X = np.array(all_feats)
    y = np.array(all_labels)
    subjs = np.array(all_subjs)

    # N-LNSO cross-validation (leave-one-subject-out)
    bal_accs = []
    for held_out in subjects:
        train_mask = subjs != held_out
        test_mask  = subjs == held_out
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        if len(set(y[train_mask])) < 2:
            continue   # skip if training fold lacks both classes

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test  = scaler.transform(X[test_mask])

        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
        clf.fit(X_train, y[train_mask])
        preds = clf.predict(X_test)
        bal_accs.append(balanced_accuracy_score(y[test_mask], preds))

    if not bal_accs:
        return {}

    result = {
        "dataset":          ds_id,
        "n_subjects":       len(subjects),
        "n_segments":       len(ds),
        "mean_bal_acc":     float(np.mean(bal_accs)),
        "std_bal_acc":      float(np.std(bal_accs)),
        "n_folds":          len(bal_accs),
    }
    print(f"  [{ds_id}] bal_acc = {result['mean_bal_acc']:.4f} ± {result['std_bal_acc']:.4f}")
    return result


def main() -> None:
    print("=" * 55)
    print("Classical Sanity Baseline — Band-Power + LogReg")
    print("=" * 55)

    results = {}
    for ds_id in DATASET_IDS:
        r = run_dataset(ds_id)
        if r:
            results[ds_id] = r

    agg = np.mean([r["mean_bal_acc"] for r in results.values()]) if results else 0.0
    results["aggregate"] = {"mean_bal_acc": float(agg)}

    print(f"\nAggregate bal_acc: {agg:.4f}")
    if agg >= 0.70:
        print("✅ Signal confirmed — bug is in deep model/training, not preprocessing")
    elif agg >= 0.60:
        print("⚠️  Weak signal — partial signal or preprocessing issues")
    else:
        print("❌ No signal — preprocessing or label problem upstream")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"classical_baseline_{ts}.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved: {out}")
    print("All done")


if __name__ == "__main__":
    main()
