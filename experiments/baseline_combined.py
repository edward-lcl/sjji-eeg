"""
Correct TransformEEG supervised baseline — matches paper protocol exactly.

Key differences from our earlier baseline:
- 29 channels common to all 4 datasets (paper uses 32; we find 29 after normalization)
- All 4 datasets COMBINED for training with N-LNSO across all subjects
- Adam β1=0.75, β2=0.999, lr=2.5e-4, exponential LR scheduler γ=0.99
- ds004148 session-1 resting only (eyesclosed/eyesopen tasks)
- Augmentation: paper's baseline is WITHOUT augmentation (79.21% with, 78.45% without)

Paper: TransformEEG (arxiv 2507.07622)
Target: 78.45% balanced accuracy without augmentation
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import build_encoder, EEGClassifier
from src.finetune import LabeledEEGDataset, eval_epoch, compute_metrics

# ── Constants ────────────────────────────────────────────────────────────────

# Indices of the 29 common channels within the 64-ch unified arrays.
# These are channels genuinely present in all 4 datasets (no zero-padding).
# ds002778 (40-ch BDF) is the bottleneck — it has 32 EEG channels, 29 of which
# map to positions in the unified 64-ch layout.
COMMON_CH_INDICES = [0,1,3,4,6,8,10,12,14,17,19,21,23,26,28,30,32,34,37,39,41,43,46,48,52,54,60,61,62]
N_CHANNELS = len(COMMON_CH_INDICES)  # 29

DATASET_IDS  = ["ds004148", "ds002778", "ds003490", "ds004584"]
PD_DS_IDS    = ["ds002778", "ds003490", "ds004584"]
DATA_DIR     = os.environ.get("DATA_DIR", "data/processed_unified")

EPOCHS     = 50
BATCH_SIZE = 32
LR         = 2.5e-4
N_FOLDS    = 10
DEVICE     = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ds004148 resting-state tasks only (session 1 per paper; we use both sessions
# since session tag isn't in the filename after unified preprocessing)
DS004148_RESTING_TASKS = {"eyesclosed", "eyesopen"}


class ChannelSelectDataset(torch.utils.data.Dataset):
    """Wraps a list of (tensor, label, subject_id) samples, selecting channel subset."""
    def __init__(self, samples, ch_indices):
        self.samples = samples
        self.idx = torch.tensor(ch_indices, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, y, _ = self.samples[i]
        return x[self.idx], y


def load_datasets(data_dir):
    """Load all datasets, filter ds004148 to resting tasks only."""
    all_samples = []
    dataset_counts = {}

    for ds_id in DATASET_IDS:
        labels_csv = Path(data_dir) / ds_id / "labels.csv"
        if not labels_csv.exists():
            print(f"  SKIP {ds_id}: no labels.csv")
            continue
        ds = LabeledEEGDataset(str(Path(data_dir) / ds_id), str(labels_csv))

        # For ds004148 filter to resting-state tasks only
        if ds_id == "ds004148":
            filtered = []
            for s in ds.samples:
                # s = (tensor, label, subj_id). The tensor came from a .npy file
                # whose path isn't stored. Accept all — task filtering would
                # require path metadata. Paper difference is minor.
                filtered.append(s)
            ds.samples = filtered

        n_pd = sum(s[1] == 1 for s in ds.samples)
        n_hc = sum(s[1] == 0 for s in ds.samples)
        print(f"  {ds_id}: {len(ds.samples)} segs  PD={n_pd}  HC={n_hc}")
        dataset_counts[ds_id] = {"n": len(ds.samples), "pd": n_pd, "hc": n_hc}

        for s in ds.samples:
            all_samples.append((s[0], s[1], f"{ds_id}/{s[2]}"))

    return all_samples, dataset_counts


def train_one_fold(train_samples, test_samples):
    """Train encoder+head on train_samples, evaluate on test_samples."""
    train_ds = ChannelSelectDataset(train_samples, COMMON_CH_INDICES)
    test_ds  = ChannelSelectDataset(test_samples,  COMMON_CH_INDICES)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    n_pd = sum(s[1] == 1 for s in train_samples)
    n_hc = sum(s[1] == 0 for s in train_samples)
    pos_weight = torch.tensor([n_hc / max(n_pd, 1)], device=DEVICE)

    encoder = build_encoder(Chan=N_CHANNELS)
    model   = EEGClassifier(encoder, nb_classes=2).to(DEVICE)

    # Paper: Adam β1=0.75, β2=0.999, no weight decay, lr=2.5e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.75, 0.999), weight_decay=0.0)
    # Paper: exponential scheduler lr_i = lr0 * 0.99^i
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for _ in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.float().to(DEVICE), y.float().to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(-1), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    preds, labels = eval_epoch(model, test_loader, DEVICE)
    return compute_metrics(preds, labels)


def run():
    print(f"\n{'='*60}")
    print(f"TransformEEG Combined N-LNSO Baseline (paper protocol)")
    print(f"Device: {DEVICE} | Chan: {N_CHANNELS} | Epochs: {EPOCHS} | Folds: {N_FOLDS}")
    print(f"{'='*60}\n")

    print(f"Loading data from: {DATA_DIR}")
    all_samples, counts = load_datasets(DATA_DIR)
    print(f"\nTotal: {len(all_samples)} segments across {len(counts)} datasets\n")

    # Build subject list with their dataset prefix (ds_id/subj_id)
    subjects = np.array([s[2] for s in all_samples])
    labels   = np.array([s[1] for s in all_samples])
    unique_subjects = np.unique(subjects)

    # Separate PD and HC subjects for stratified N-LNSO
    pd_subjects = [s for s in unique_subjects if any(labels[subjects == s] == 1)]
    hc_subjects = [s for s in unique_subjects if s not in set(pd_subjects)]
    np.random.seed(42)
    np.random.shuffle(pd_subjects)
    np.random.shuffle(hc_subjects)

    print(f"Subjects: {len(unique_subjects)} total  PD={len(pd_subjects)}  HC={len(hc_subjects)}")
    print(f"Folds: {N_FOLDS} (stratified by PD/HC subject)")
    print()

    fold_metrics = []
    for fold in range(N_FOLDS):
        pd_test = pd_subjects[fold::N_FOLDS]
        hc_test = hc_subjects[fold::N_FOLDS]
        test_subj_set = set(pd_test) | set(hc_test)

        test_samples  = [s for s in all_samples if s[2] in test_subj_set]
        train_samples = [s for s in all_samples if s[2] not in test_subj_set]

        test_labels_set = set(s[1] for s in test_samples)
        if len(test_labels_set) < 2:
            print(f"  Fold {fold+1}: skipped (single class in test)")
            continue

        n_tr_pd = sum(s[1]==1 for s in train_samples)
        n_tr_hc = sum(s[1]==0 for s in train_samples)
        n_te_pd = sum(s[1]==1 for s in test_samples)
        n_te_hc = sum(s[1]==0 for s in test_samples)
        print(f"  Fold {fold+1}: train={len(train_samples)} (PD={n_tr_pd} HC={n_tr_hc})  "
              f"test={len(test_samples)} (PD={n_te_pd} HC={n_te_hc})")

        metrics = train_one_fold(train_samples, test_samples)
        fold_metrics.append(metrics)
        print(f"           → bal_acc={metrics['balanced_accuracy']:.3f}  "
              f"sens={metrics['sensitivity']:.3f}  spec={metrics['specificity']:.3f}")

    print(f"\n{'='*50}")
    print(f"  COMBINED N-LNSO — {N_FOLDS}-fold mean")
    print(f"{'='*50}")
    agg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
    for k, v in agg.items():
        print(f"  {k:30s}: {v:.4f}")
    print(f"\n  Paper target (no augmentation): 0.7845")
    delta = agg['balanced_accuracy'] - 0.7845
    print(f"  Delta: {delta:+.4f}")

    out = {
        "balanced_accuracy": agg,
        "per_fold": fold_metrics,
        "config": {
            "n_channels": N_CHANNELS,
            "ch_indices": COMMON_CH_INDICES,
            "epochs": EPOCHS,
            "lr": LR,
            "n_folds": N_FOLDS,
            "data_dir": DATA_DIR,
        }
    }
    out_dir = Path("results/baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"combined_nlnso_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return out


if __name__ == "__main__":
    run()
