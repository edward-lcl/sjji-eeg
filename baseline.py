"""
Supervised-only baseline — two evaluation modes:
1. COMBINED: All 4 datasets merged, N-LNSO CV (replicates TransformEEG protocol)
2. CROSS-DATASET: Train on 3 datasets, test on held-out dataset (our novel contribution)

Usage:
    python baseline.py              # runs both modes
    python baseline.py --mode combined
    python baseline.py --mode cross
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score, recall_score

from src.model import build_encoder, EEGClassifier
from src.finetune import LabeledEEGDataset, train_epoch, eval_epoch, compute_metrics
from src.evaluate import print_results, save_results, TRANSFORM_EEG_BASELINE
import yaml

DATASET_IDS = ["ds004148", "ds002778", "ds003490", "ds004584"]

EPOCHS = 30        # enough to converge; keep runtime reasonable
BATCH_SIZE = 64
LR = 1e-3
N_OUTER = 5        # 5-fold for speed; paper uses 10
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_all_datasets(processed_dir: str):
    datasets = {}
    for ds_id in DATASET_IDS:
        labels_csv = Path(processed_dir) / ds_id / "labels.csv"
        data_dir = Path(processed_dir) / ds_id
        if not labels_csv.exists():
            print(f"  skipping {ds_id} — no labels.csv")
            continue
        ds = LabeledEEGDataset(str(data_dir), str(labels_csv))
        print(f"  {ds_id}: {len(ds)} segments, PD={sum(1 for s in ds.samples if s[1]==1)}, HC={sum(1 for s in ds.samples if s[1]==0)}")
        datasets[ds_id] = ds
    return datasets


def run_fold(train_idx, test_idx, full_dataset, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    """Train one fold, return metrics."""
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=batch_size)

    encoder = build_encoder()
    model = EEGClassifier(encoder, nb_classes=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, DEVICE)

    preds, labels = eval_epoch(model, test_loader, DEVICE)
    return compute_metrics(preds, labels)


def mode_combined(datasets):
    """N-LNSO CV on all datasets merged — replicates TransformEEG protocol."""
    print("\n" + "="*60)
    print("MODE 1: COMBINED (replicates TransformEEG)")
    print("="*60)

    # Only include datasets that have both classes
    mixed_datasets = {k: v for k, v in datasets.items()
                      if sum(1 for s in v.samples if s[1]==1) > 0
                      and sum(1 for s in v.samples if s[1]==0) > 0}

    # For HC-only dataset (ds004148), pair with PD from others
    # Build combined dataset with subject IDs as group labels
    all_samples = []
    all_subjects = []
    subject_counter = 0
    subject_map = {}

    for ds_id, ds in datasets.items():
        for seg, label, subj_id in ds.samples:
            key = f"{ds_id}_{subj_id}"
            if key not in subject_map:
                subject_map[key] = subject_counter
                subject_counter += 1
            all_samples.append((seg, label, subject_map[key]))

    # Build flat arrays
    segs = [s[0] for s in all_samples]
    labels = np.array([s[1] for s in all_samples])
    groups = np.array([s[2] for s in all_samples])
    indices = np.arange(len(all_samples))

    # Check class balance
    n_pd = labels.sum()
    n_hc = len(labels) - n_pd
    print(f"Combined: {len(labels)} segments, PD={n_pd}, HC={n_hc}, subjects={subject_counter}")

    # Simple combined dataset
    class FlatDataset(torch.utils.data.Dataset):
        def __init__(self, segs, labels):
            self.segs = segs
            self.labels = labels
        def __len__(self): return len(self.segs)
        def __getitem__(self, i): return self.segs[i], self.labels[i]

    flat_ds = FlatDataset(segs, labels)

    # N-LNSO: leave groups of subjects out
    unique_subjects = np.unique(groups)
    np.random.seed(42)
    fold_subjects = np.array_split(unique_subjects, N_OUTER)

    all_metrics = []
    for fold_i, test_subjs in enumerate(fold_subjects):
        test_mask = np.isin(groups, test_subjs)
        train_idx = indices[~test_mask]
        test_idx = indices[test_mask]

        # Skip if test set has only one class
        if len(np.unique(labels[test_idx])) < 2:
            print(f"  Fold {fold_i+1}: skipped (single class in test)")
            continue

        metrics = run_fold(train_idx, test_idx, flat_ds)
        all_metrics.append(metrics)
        print(f"  Fold {fold_i+1}: bal_acc={metrics['balanced_accuracy']:.3f} "
              f"sens={metrics['sensitivity']:.3f} spec={metrics['specificity']:.3f}")

    if all_metrics:
        agg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        print_results(agg, "COMBINED — aggregate")
        print(f"TransformEEG paper reported: {TRANSFORM_EEG_BASELINE['balanced_accuracy']:.4f}")
        return agg
    return {}


def mode_cross_dataset(datasets):
    """Leave-one-dataset-out: train on 3, test on 1. 4 rotations."""
    print("\n" + "="*60)
    print("MODE 2: CROSS-DATASET (our novel contribution)")
    print("="*60)

    results = {}
    ds_ids = list(datasets.keys())

    for held_out_id in ds_ids:
        train_ids = [d for d in ds_ids if d != held_out_id]
        train_datasets = [datasets[d] for d in train_ids]
        test_dataset = datasets[held_out_id]

        # Check test set has both classes
        test_labels = [s[1] for s in test_dataset.samples]
        if len(set(test_labels)) < 2:
            print(f"\n  Hold-out {held_out_id}: single class — skip")
            continue

        print(f"\n  Train on: {train_ids} → Test on: {held_out_id}")

        # Combine train datasets
        all_train = []
        for ds in train_datasets:
            all_train.extend(ds.samples)

        train_segs = [s[0] for s in all_train]
        train_labels = [s[1] for s in all_train]

        class FlatDS(torch.utils.data.Dataset):
            def __init__(self, segs, labels):
                self.segs = segs; self.labels = labels
            def __len__(self): return len(self.segs)
            def __getitem__(self, i): return self.segs[i], self.labels[i]

        train_ds = FlatDS(train_segs, train_labels)
        test_ds = FlatDS([s[0] for s in test_dataset.samples], [s[1] for s in test_dataset.samples])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        encoder = build_encoder()
        model = EEGClassifier(encoder, nb_classes=2).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(EPOCHS):
            loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            if (epoch + 1) % 10 == 0:
                print(f"    epoch {epoch+1}/{EPOCHS} loss={loss:.4f}")

        preds, labels_arr = eval_epoch(model, test_loader, DEVICE)
        metrics = compute_metrics(preds, labels_arr)
        results[held_out_id] = metrics
        print(f"  → bal_acc={metrics['balanced_accuracy']:.3f} "
              f"sens={metrics['sensitivity']:.3f} spec={metrics['specificity']:.3f}")

    if results:
        agg = {k: np.mean([r[k] for r in results.values()]) for k in list(results.values())[0]}
        print_results(agg, "CROSS-DATASET — aggregate")
        return results
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["combined", "cross", "both"], default="both")
    args = parser.parse_args()

    print(f"\nDevice: {DEVICE}")
    print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, Folds: {N_OUTER}\n")

    print("Loading datasets...")
    datasets = load_all_datasets("data/processed")

    all_results = {}

    if args.mode in ("combined", "both"):
        r = mode_combined(datasets)
        all_results["combined"] = r

    if args.mode in ("cross", "both"):
        r = mode_cross_dataset(datasets)
        all_results["cross_dataset"] = r

    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    save_results(all_results, "results/baseline", "supervised_baseline")
    print("\nDone. Results saved to results/baseline/")


if __name__ == "__main__":
    main()
