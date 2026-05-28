"""
Dataset fingerprinting experiment.

Trains a classifier to predict WHICH DATASET a segment came from
(ds002778 vs ds003490 vs ds004584), not PD vs HC.

If accuracy is high (~90%+), it proves that site/device artifacts dominate
the EEG signal enough that the model can identify recording origin from
raw signal alone — which explains why cross-dataset generalization fails.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import json
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import build_encoder
from src.finetune import LabeledEEGDataset, train_epoch, eval_epoch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
N_CHANNELS = 64  # unified


DATASET_IDS = ["ds002778", "ds003490", "ds004584"]
DATASET_TO_IDX = {ds: i for i, ds in enumerate(DATASET_IDS)}


class DatasetLabeledEEG(Dataset):
    """Loads EEG segments labeled by dataset origin, not PD/HC."""

    def __init__(self, data_dir: str, dataset_label: int, labels_csv: str):
        self.samples = []
        ds = LabeledEEGDataset(data_dir, labels_csv)
        for seg, _, subj_id in ds.samples:
            self.samples.append((seg, dataset_label, subj_id))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i][0], self.samples[i][1]


def run_fingerprint(processed_dir: str = "data/processed_unified"):
    print(f"\nDevice: {DEVICE} | Epochs: {EPOCHS} | Task: predict dataset origin\n")

    all_samples = []
    for ds_id in DATASET_IDS:
        ds_dir = Path(processed_dir) / ds_id
        labels_csv = ds_dir / "labels.csv"
        if not labels_csv.exists():
            print(f"  {ds_id}: no labels.csv, skip")
            continue
        raw_ds = LabeledEEGDataset(str(ds_dir), str(labels_csv))
        label = DATASET_TO_IDX[ds_id]
        n = 0
        for seg, _, subj_id in raw_ds.samples:
            all_samples.append((seg, label, subj_id, ds_id))
            n += 1
        print(f"  {ds_id}: {n} segments → class {label}")

    print(f"\nTotal segments: {len(all_samples)}")
    print(f"Classes: {DATASET_IDS}\n")

    # --- Random 80/20 split (segment-level, no subject leakage needed here
    #     since we WANT to know if raw signal alone encodes dataset identity) ---
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
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    print(f"Train: {len(train_ds)} segs | Test: {len(test_ds)} segs\n")

    # 3-class classifier
    encoder = build_encoder(Chan=N_CHANNELS)
    feat_dim = encoder.feat_dim
    model = nn.Sequential(
        encoder,
        nn.Linear(feat_dim, feat_dim // 2),
        nn.LeakyReLU(),
        nn.Linear(feat_dim // 2, 3),   # 3 datasets
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
            avg = total_loss / len(train_loader)
            print(f"  epoch {epoch}/{EPOCHS}  loss={avg:.4f}")

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(DEVICE))
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc     = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    cm      = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*50}")
    print(f"  DATASET FINGERPRINTING RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print(f"  Labels: {DATASET_IDS}")
    for i, row in enumerate(cm):
        print(f"    {DATASET_IDS[i]:12s}: {row}")

    print(f"\n  Interpretation:")
    if bal_acc > 0.90:
        print(f"  *** STRONG fingerprint ({bal_acc:.2f}): dataset artifacts dominate signal.")
        print(f"  *** This explains cross-dataset collapse: model learns site identity, not PD.")
    elif bal_acc > 0.70:
        print(f"  ** MODERATE fingerprint ({bal_acc:.2f}): some dataset-specific artifacts present.")
    else:
        print(f"  * WEAK fingerprint ({bal_acc:.2f}): datasets may share common signal structure.")

    results = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "confusion_matrix": cm.tolist(),
        "dataset_labels": DATASET_IDS,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "epochs": EPOCHS,
        "n_channels": N_CHANNELS,
    }

    out_dir = Path("results/fingerprint")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"dataset_fingerprint_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    run_fingerprint("data/processed_unified")
