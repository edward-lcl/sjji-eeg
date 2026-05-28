"""
Dataset fingerprinting — class-balanced version.

Same as dataset_fingerprint.py but uses WeightedRandomSampler so each
dataset class gets equal representation during training. Rules out the
ds003490 majority-class collapse from the first run.

If balanced_accuracy drops vs the unbalanced run → datasets genuinely
share EEG characteristics (covariate shift, not site artifacts).
If balanced_accuracy rises sharply → real artifacts exist, we just
trained poorly the first time.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from src.model import build_encoder
from src.finetune import LabeledEEGDataset
import json
from datetime import datetime

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
N_CHANNELS = 64
DATASET_IDS = ["ds002778", "ds003490", "ds004584"]
DATASET_TO_IDX = {ds: i for i, ds in enumerate(DATASET_IDS)}


def run_fingerprint_balanced(processed_dir: str = "data/processed_unified"):
    print(f"\nDevice: {DEVICE} | Epochs: {EPOCHS} | Task: predict dataset origin (BALANCED)\n")

    all_samples = []
    for ds_id in DATASET_IDS:
        ds_dir = Path(processed_dir) / ds_id
        labels_csv = ds_dir / "labels.csv"
        if not labels_csv.exists():
            continue
        raw_ds = LabeledEEGDataset(str(ds_dir), str(labels_csv))
        label = DATASET_TO_IDX[ds_id]
        for seg, _, subj_id in raw_ds.samples:
            all_samples.append((seg, label, subj_id, ds_id))
        print(f"  {ds_id}: {len(raw_ds)} segments → class {label}")

    print(f"\nTotal: {len(all_samples)} | Classes: {DATASET_IDS}")

    np.random.seed(42)
    idx = np.random.permutation(len(all_samples))
    split = int(0.8 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]

    class FlatDS(Dataset):
        def __init__(self, samples, indices):
            self.items = [samples[i] for i in indices]
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            seg, label, _, _ = self.items[i]
            if seg.shape[0] > N_CHANNELS:
                seg = seg[:N_CHANNELS]
            return seg, label

    train_ds = FlatDS(all_samples, train_idx)
    test_ds  = FlatDS(all_samples, test_idx)

    # Weighted sampler: each dataset class gets equal expected draws per batch
    train_labels = [all_samples[i][1] for i in train_idx]
    class_counts = np.bincount(train_labels, minlength=3)
    class_weights = 1.0 / class_counts.astype(float)
    sample_weights = torch.tensor([class_weights[l] for l in train_labels])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    print(f"\nClass counts (train): {dict(zip(DATASET_IDS, class_counts))}")
    print(f"Class weights: {dict(zip(DATASET_IDS, class_weights.round(4)))}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    encoder = build_encoder(Chan=N_CHANNELS)
    feat_dim = encoder.feat_dim
    model = nn.Sequential(
        encoder,
        nn.Linear(feat_dim, feat_dim // 2),
        nn.LeakyReLU(),
        nn.Linear(feat_dim // 2, 3),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  epoch {epoch}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x.to(DEVICE)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc     = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    cm      = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*50}")
    print(f"  BALANCED FINGERPRINTING RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}  (unbalanced run was 0.579)")
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print(f"  Labels: {DATASET_IDS}")
    for i, row in enumerate(cm):
        print(f"    {DATASET_IDS[i]:12s}: {row}")

    if bal_acc < 0.55:
        print(f"\n  → DROPS vs unbalanced: datasets share EEG signal (covariate shift story holds)")
    elif bal_acc > 0.75:
        print(f"\n  → RISES sharply: real site artifacts present, first run had majority-class bias")
    else:
        print(f"\n  → SIMILAR: mild artifacts but not dominant")

    results = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "unbalanced_run_balanced_accuracy": 0.579,
        "confusion_matrix": cm.tolist(),
        "dataset_labels": DATASET_IDS,
        "class_counts_train": class_counts.tolist(),
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "epochs": EPOCHS,
        "n_channels": N_CHANNELS,
        "sampler": "WeightedRandomSampler",
    }

    out_dir = Path("results/fingerprint")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"dataset_fingerprint_balanced_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    run_fingerprint_balanced("data/processed_unified")
