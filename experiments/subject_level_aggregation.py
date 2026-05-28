"""
Subject-level aggregation re-analysis.

Re-runs the per-dataset cross-validation but evaluates at SUBJECT level
via majority vote over all segments for each subject, rather than
segment-level metrics.

Clinically this is the correct metric: you want to know if a patient
is PD or HC, not whether each 4-second window is correct.

No new training required — re-uses the same fold structure and
recomputes metrics with subject-level aggregation on top.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score
import json
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import build_encoder, EEGClassifier
from src.finetune import LabeledEEGDataset, train_epoch
from baseline import (
    DEVICE, EPOCHS, BATCH_SIZE, LR, N_OUTER,
    PD_DATASET_IDS, load_all_datasets, FlatDataset
)

PROCESSED_DIR = "data/processed_unified"


def eval_with_subject_aggregation(model, test_samples, device):
    """Segment-level inference → majority vote per subject."""
    ds = FlatDataset(test_samples)
    loader = DataLoader(ds, batch_size=BATCH_SIZE)

    model.eval()
    all_preds, all_labels, all_subjects = [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            logits = model(x).squeeze(-1).cpu()
            preds = (torch.sigmoid(logits) > 0.5).long().numpy()
            labels = y.numpy()
            # recover subject IDs from samples
            batch_start = i * BATCH_SIZE
            batch_subjects = [test_samples[min(batch_start + j, len(test_samples)-1)][2]
                              for j in range(len(preds))]
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_subjects.extend(batch_subjects)

    # Majority vote per subject
    subj_votes = defaultdict(list)
    subj_true = {}
    for pred, label, subj in zip(all_preds, all_labels, all_subjects):
        subj_votes[subj].append(pred)
        subj_true[subj] = label

    subj_pred_agg = {}
    for subj, votes in subj_votes.items():
        subj_pred_agg[subj] = int(np.mean(votes) > 0.5)

    subjects = list(subj_pred_agg.keys())
    y_pred = np.array([subj_pred_agg[s] for s in subjects])
    y_true = np.array([subj_true[s] for s in subjects])

    seg_metrics = {
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "sensitivity": recall_score(all_labels, all_preds, pos_label=1, zero_division=0),
        "specificity": recall_score(all_labels, all_preds, pos_label=0, zero_division=0),
    }

    if len(set(y_true)) < 2:
        subj_metrics = seg_metrics  # fallback if test fold is single class
    else:
        subj_metrics = {
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        }

    return seg_metrics, subj_metrics, len(subjects)


def run_subject_level_analysis():
    print(f"\nDevice: {DEVICE} | Epochs: {EPOCHS} | Subject-level aggregation analysis\n")
    datasets = load_all_datasets(PROCESSED_DIR)

    all_seg_results = {}
    all_subj_results = {}

    for ds_id in PD_DATASET_IDS:
        ds = datasets[ds_id]
        n_channels = min(sample[0].shape[0] for sample in ds.samples)
        subjects = np.unique([s[2] for s in ds.samples])

        pd_subjects = [s for s in subjects if any(x[2] == s and x[1] == 1 for x in ds.samples)]
        hc_subjects = [s for s in subjects if s not in pd_subjects]
        np.random.seed(42)
        np.random.shuffle(pd_subjects)
        np.random.shuffle(hc_subjects)

        print(f"\n--- {ds_id} ({len(subjects)} subjects, Chan={n_channels}) ---")

        fold_seg, fold_subj = [], []

        for fold in range(N_OUTER):
            pd_test = pd_subjects[fold::N_OUTER]
            hc_test = hc_subjects[fold::N_OUTER]
            test_subjects = set(pd_test) | set(hc_test)

            test_samples  = [s for s in ds.samples if s[2] in test_subjects]
            train_samples = [s for s in ds.samples if s[2] not in test_subjects]

            test_labels = [s[1] for s in test_samples]
            if len(set(test_labels)) < 2:
                continue

            # Train model (same as baseline)
            train_ds = FlatDataset(train_samples, n_channels=n_channels)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

            n_pd = sum(1 for s in train_samples if s[1] == 1)
            n_hc = sum(1 for s in train_samples if s[1] == 0)
            pos_weight = torch.tensor([n_hc / max(n_pd, 1)], device=DEVICE)

            encoder = build_encoder(Chan=n_channels)
            model = EEGClassifier(encoder, nb_classes=2).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            for _ in range(EPOCHS):
                train_epoch(model, train_loader, optimizer, criterion, DEVICE)

            seg_m, subj_m, n_subj = eval_with_subject_aggregation(model, test_samples, DEVICE)
            fold_seg.append(seg_m)
            fold_subj.append(subj_m)

            print(f"  Fold {fold+1} ({n_subj} test subjects): "
                  f"seg_bal={seg_m['balanced_accuracy']:.3f} | "
                  f"subj_bal={subj_m['balanced_accuracy']:.3f}")

        if fold_seg:
            mean_seg  = {k: np.mean([m[k] for m in fold_seg])  for k in fold_seg[0]}
            mean_subj = {k: np.mean([m[k] for m in fold_subj]) for k in fold_subj[0]}
            all_seg_results[ds_id]  = mean_seg
            all_subj_results[ds_id] = mean_subj
            print(f"  Mean segment:  bal_acc={mean_seg['balanced_accuracy']:.3f}  "
                  f"sens={mean_seg['sensitivity']:.3f}  spec={mean_seg['specificity']:.3f}")
            print(f"  Mean subject:  bal_acc={mean_subj['balanced_accuracy']:.3f}  "
                  f"sens={mean_subj['sensitivity']:.3f}  spec={mean_subj['specificity']:.3f}")

    if all_seg_results:
        agg_seg  = {k: np.mean([r[k] for r in all_seg_results.values()])  for k in list(all_seg_results.values())[0]}
        agg_subj = {k: np.mean([r[k] for r in all_subj_results.values()]) for k in list(all_subj_results.values())[0]}

        print(f"\n{'='*60}")
        print(f"  FINAL — Segment-level (current):  bal_acc={agg_seg['balanced_accuracy']:.4f}")
        print(f"  FINAL — Subject-level (majority):  bal_acc={agg_subj['balanced_accuracy']:.4f}")
        print(f"  Delta: {agg_subj['balanced_accuracy'] - agg_seg['balanced_accuracy']:+.4f}")
        print(f"{'='*60}\n")

        results = {
            "segment_level": {**all_seg_results, "aggregate": agg_seg},
            "subject_level":  {**all_subj_results, "aggregate": agg_subj},
        }

        out_dir = Path("results/subject_aggregation")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"subject_aggregation_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")
        return results


if __name__ == "__main__":
    run_subject_level_analysis()
