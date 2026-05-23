"""
Supervised-only baseline — two evaluation modes:

1. PER-DATASET N-LNSO: For each PD dataset, train on it + ds004148 (HC),
   evaluate with N-LNSO CV. Matches TransformEEG's actual protocol.

2. CROSS-DATASET: Train on 3 datasets, test on held-out dataset.
   Our novel contribution.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import balanced_accuracy_score, recall_score

from src.model import build_encoder, EEGClassifier
from src.finetune import LabeledEEGDataset, train_epoch, eval_epoch, compute_metrics
from src.evaluate import print_results, save_results, TRANSFORM_EEG_BASELINE

DATASET_IDS = ["ds004148", "ds002778", "ds003490", "ds004584"]
PD_DATASET_IDS = ["ds002778", "ds003490", "ds004584"]  # datasets with PD subjects

EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
N_OUTER = 5
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_all_datasets(processed_dir: str):
    datasets = {}
    for ds_id in DATASET_IDS:
        labels_csv = Path(processed_dir) / ds_id / "labels.csv"
        data_dir = Path(processed_dir) / ds_id
        if not labels_csv.exists():
            continue
        ds = LabeledEEGDataset(str(data_dir), str(labels_csv))
        n_pd = sum(1 for s in ds.samples if s[1] == 1)
        n_hc = sum(1 for s in ds.samples if s[1] == 0)
        print(f"  {ds_id}: {len(ds)} segs  PD={n_pd}  HC={n_hc}")
        datasets[ds_id] = ds
    return datasets


class FlatDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i][0], self.samples[i][1]
    def subject_ids(self): return np.array([s[2] for s in self.samples])


def train_eval(train_samples, test_samples):
    train_ds = FlatDataset(train_samples)
    test_ds = FlatDataset(test_samples)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    encoder = build_encoder()
    model = EEGClassifier(encoder, nb_classes=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(EPOCHS):
        train_epoch(model, train_loader, optimizer, criterion, DEVICE)

    preds, labels = eval_epoch(model, test_loader, DEVICE)
    return compute_metrics(preds, labels)


def mode_per_dataset(datasets):
    """N-LNSO CV on each PD dataset paired with HC from ds004148."""
    print("\n" + "="*60)
    print("MODE 1: PER-DATASET N-LNSO (TransformEEG protocol)")
    print("="*60)

    hc_samples = datasets["ds004148"].samples  # HC-only pool

    all_results = {}
    for ds_id in PD_DATASET_IDS:
        ds = datasets[ds_id]
        subjects = np.unique([s[2] for s in ds.samples])
        n_pd = sum(1 for s in ds.samples if s[1] == 1)
        n_hc_ds = sum(1 for s in ds.samples if s[1] == 0)
        print(f"\n--- {ds_id} ({len(subjects)} subjects, PD={n_pd}, HC={n_hc_ds}) ---")

        # Stratified subject folds
        pd_subjects = [s for s in subjects if any(x[2] == s and x[1] == 1 for x in ds.samples)]
        hc_subjects = [s for s in subjects if s not in pd_subjects]
        np.random.seed(42)
        np.random.shuffle(pd_subjects)
        np.random.shuffle(hc_subjects)

        fold_metrics = []
        for fold in range(N_OUTER):
            # Stratified: pick ~1/N_OUTER from each class as test subjects
            pd_test = pd_subjects[fold::N_OUTER]
            hc_test = hc_subjects[fold::N_OUTER]
            test_subjects = set(pd_test) | set(hc_test)

            test_samples = [s for s in ds.samples if s[2] in test_subjects]
            train_ds_samples = [s for s in ds.samples if s[2] not in test_subjects]

            # Augment train with HC from ds004148
            train_samples = train_ds_samples + hc_samples

            test_labels = [s[1] for s in test_samples]
            if len(set(test_labels)) < 2:
                continue

            metrics = train_eval(train_samples, test_samples)
            fold_metrics.append(metrics)
            print(f"  Fold {fold+1}: bal_acc={metrics['balanced_accuracy']:.3f}  "
                  f"sens={metrics['sensitivity']:.3f}  spec={metrics['specificity']:.3f}")

        if fold_metrics:
            agg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            all_results[ds_id] = agg
            print(f"  Mean: bal_acc={agg['balanced_accuracy']:.3f}  "
                  f"sens={agg['sensitivity']:.3f}  spec={agg['specificity']:.3f}")

    if all_results:
        overall = {k: np.mean([r[k] for r in all_results.values()]) for k in list(all_results.values())[0]}
        print_results(overall, "PER-DATASET — overall mean")
        print(f"TransformEEG paper reported: {TRANSFORM_EEG_BASELINE['balanced_accuracy']:.4f}")
        all_results["aggregate"] = overall
    return all_results


def mode_cross_dataset(datasets):
    """Leave-one-dataset-out generalization test."""
    print("\n" + "="*60)
    print("MODE 2: CROSS-DATASET (novel contribution)")
    print("="*60)

    results = {}
    for held_out_id in PD_DATASET_IDS:
        train_ids = [d for d in DATASET_IDS if d != held_out_id]
        test_ds = datasets[held_out_id]

        # Combine all training datasets
        train_samples = []
        for tid in train_ids:
            train_samples.extend(datasets[tid].samples)

        test_labels = [s[1] for s in test_ds.samples]
        if len(set(test_labels)) < 2:
            print(f"\n  Hold-out {held_out_id}: single class, skip")
            continue

        print(f"\n  Train: {train_ids} → Test: {held_out_id}")
        print(f"  Train: {len(train_samples)} segs | Test: {len(test_ds.samples)} segs")

        train_ds_flat = FlatDataset(train_samples)
        test_ds_flat = FlatDataset(test_ds.samples)
        train_loader = DataLoader(train_ds_flat, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds_flat, batch_size=BATCH_SIZE)

        encoder = build_encoder()
        model = EEGClassifier(encoder, nb_classes=2).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(EPOCHS):
            loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            if (epoch + 1) % 10 == 0:
                print(f"    epoch {epoch+1}/{EPOCHS}  loss={loss:.4f}")

        preds, labels_arr = eval_epoch(model, test_loader, DEVICE)
        metrics = compute_metrics(preds, labels_arr)
        results[held_out_id] = metrics
        print(f"  → bal_acc={metrics['balanced_accuracy']:.3f}  "
              f"sens={metrics['sensitivity']:.3f}  spec={metrics['specificity']:.3f}")

    if results:
        agg = {k: np.mean([r[k] for r in results.values()]) for k in list(results.values())[0]}
        print_results(agg, "CROSS-DATASET — mean")
        results["aggregate"] = agg
    return results


if __name__ == "__main__":
    print(f"\nDevice: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Folds: {N_OUTER}\n")
    print("Loading datasets...")
    datasets = load_all_datasets("data/processed")

    all_results = {}
    all_results["per_dataset"] = mode_per_dataset(datasets)
    all_results["cross_dataset"] = mode_cross_dataset(datasets)

    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    save_results(all_results, "results/baseline", "supervised_baseline")
    print("\nAll done. Results saved.")
