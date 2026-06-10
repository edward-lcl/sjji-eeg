"""
29-ch SSL pretrain + combined N-LNSO probe.

Correct-protocol SSL experiment:
- Same 29 channels as the supervised baseline (COMMON_CH_INDICES)
- VICReg pretraining on OpenNeuro data (labels ignored — pure self-supervised)
- Combined N-LNSO linear probe evaluation on all 4 datasets
- Direct comparison to the 89.1% supervised median from experiments/baseline_combined.py

This is the minimum viable SSL experiment under the correct protocol.
Scope is small (~18k OpenNeuro segments for pretraining) — the purpose is to
verify whether any SSL signal exists before investing in full-scale TUH pretrain.

Usage:
  python experiments/ssl_29ch_local.py              # pretrain + probe
  python experiments/ssl_29ch_local.py --probe-only  # skip pretrain, probe only
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import build_encoder, EEGClassifier
from src.pretrain import eeg_augment_batch, vicreg_loss
from src.finetune import LabeledEEGDataset, eval_epoch, compute_metrics

# ── Config ───────────────────────────────────────────────────────────────────

COMMON_CH_INDICES = [0,1,3,4,6,8,10,12,14,17,19,21,23,26,28,30,32,34,37,39,41,43,46,48,52,54,60,61,62]
N_CHANNELS = len(COMMON_CH_INDICES)  # 29

DATASET_IDS = ["ds004148", "ds002778", "ds003490", "ds004584"]
PD_DS_IDS   = ["ds002778", "ds003490", "ds004584"]
DATA_DIR    = os.environ.get("DATA_DIR", "data/processed_unified")

PRETRAIN_EPOCHS = 100
PRETRAIN_LR     = 2.5e-4
PRETRAIN_BATCH  = 64
PRETRAIN_PATIENCE = 20

PROBE_EPOCHS = 30
PROBE_LR     = 1e-3
PROBE_BATCH  = 32

N_FOLDS = 10
DEVICE  = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

ENCODER_SAVE = "results/ssl/pretrained_encoder_29ch_opennero.pt"


# ── Datasets ─────────────────────────────────────────────────────────────────

class UnlabeledChannelSelectDataset(Dataset):
    """
    Load all .npy files under data_dir (shape: N_segs × C × T per file),
    apply channel index selection to yield (29, T) segments.
    """
    def __init__(self, data_dir, ch_indices):
        self.ch = torch.tensor(ch_indices, dtype=torch.long)
        self.files = []
        self.lengths = []
        self._cumlen = None

        for p in sorted(Path(data_dir).glob("**/*.npy")):
            arr = np.load(str(p), mmap_mode="r")
            if arr.ndim == 3:  # (N_segs, C, T)
                self.files.append(p)
                self.lengths.append(arr.shape[0])
            # skip 2D files (individual segments — shouldn't happen in unified)

        self._cumlen = np.cumsum([0] + self.lengths)
        total = int(self._cumlen[-1])
        print(f"[pretrain-data] {len(self.files)} files, {total} segments, ch={len(ch_indices)}")

    def __len__(self):
        return int(self._cumlen[-1])

    def __getitem__(self, idx):
        fi = int(np.searchsorted(self._cumlen[1:], idx, side="right"))
        li = idx - int(self._cumlen[fi])
        if not hasattr(self, "_cache"):
            self._cache = {}
        if fi not in self._cache:
            if len(self._cache) >= 8:
                self._cache.pop(next(iter(self._cache)))
            self._cache[fi] = np.load(str(self.files[fi]), mmap_mode="r")
        x = torch.from_numpy(self._cache[fi][li].copy())  # (C, T)
        return x[self.ch]  # (29, T)


class ChannelSelectLabeledDataset(Dataset):
    """Wraps LabeledEEGDataset samples, selecting 29 channels."""
    def __init__(self, samples, ch_indices):
        self.samples = samples
        self.idx = torch.tensor(ch_indices, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, y, _ = self.samples[i]
        return x[self.idx], y


# ── Pretrain ─────────────────────────────────────────────────────────────────

def pretrain_vicreg_opennero(encoder, data_dir, output_path):
    """VICReg pretrain on OpenNeuro 29-ch data."""
    print(f"\n{'='*60}")
    print(f"Phase 1: VICReg pretrain — 29-ch OpenNeuro")
    print(f"  Epochs: {PRETRAIN_EPOCHS}  Batch: {PRETRAIN_BATCH}  LR: {PRETRAIN_LR}")
    print(f"{'='*60}")

    # Only use the 4 OpenNeuro datasets (not TUH — wrong channel ordering)
    pretrain_dirs = [str(Path(data_dir) / ds) for ds in DATASET_IDS if (Path(data_dir) / ds).exists()]
    print(f"  Pretrain dirs: {[d.split('/')[-1] for d in pretrain_dirs]}")

    all_files = []
    all_lengths = []
    for d in pretrain_dirs:
        for p in sorted(Path(d).glob("**/*.npy")):
            arr = np.load(str(p), mmap_mode="r")
            if arr.ndim == 3:
                all_files.append(p)
                all_lengths.append(arr.shape[0])
    total_segs = sum(all_lengths)
    print(f"  {len(all_files)} files, {total_segs} segments")

    ch_tensor = torch.tensor(COMMON_CH_INDICES, dtype=torch.long)

    class _DS(Dataset):
        def __init__(self, files, lengths):
            self.files = files
            self._cumlen = np.cumsum([0] + lengths)

        def __len__(self):
            return int(self._cumlen[-1])

        def __getitem__(self, idx):
            fi = int(np.searchsorted(self._cumlen[1:], idx, side="right"))
            li = idx - int(self._cumlen[fi])
            if not hasattr(self, "_cache"):
                self._cache = {}
            if fi not in self._cache:
                if len(self._cache) >= 8:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[fi] = np.load(str(self.files[fi]), mmap_mode="r")
            x = torch.from_numpy(self._cache[fi][li].copy())
            return x[ch_tensor]

    dataset = _DS(all_files, all_lengths)
    loader = DataLoader(dataset, batch_size=PRETRAIN_BATCH, shuffle=True, drop_last=True)

    feat_dim = encoder.feat_dim
    projector = nn.Sequential(
        nn.Linear(feat_dim, feat_dim),
        nn.ReLU(),
        nn.Linear(feat_dim, 128),
    ).to(DEVICE)

    encoder = encoder.to(DEVICE)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=PRETRAIN_LR,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS, eta_min=PRETRAIN_LR * 0.01)

    best_loss = float("inf")
    patience_counter = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        encoder.train()
        projector.train()
        total = 0.0
        n_batches = 0

        for x in loader:
            x = x.float().to(DEVICE)
            x1 = eeg_augment_batch(x)
            x2 = eeg_augment_batch(x)
            z1 = projector(encoder(x1))
            z2 = projector(encoder(x2))
            loss = vicreg_loss(z1, z2)

            if torch.isnan(loss) or torch.isinf(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total / max(n_batches, 1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{PRETRAIN_EPOCHS}  loss={avg_loss:.4f}  {'*' if avg_loss < best_loss else ''}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), output_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PRETRAIN_PATIENCE:
                print(f"  Early stop at epoch {epoch} (patience={PRETRAIN_PATIENCE})")
                break

    print(f"  Best loss: {best_loss:.4f}  Saved to {output_path}")


# ── Probe ─────────────────────────────────────────────────────────────────────

def load_all_labeled(data_dir):
    all_samples = []
    for ds_id in DATASET_IDS:
        labels_csv = Path(data_dir) / ds_id / "labels.csv"
        if not labels_csv.exists():
            continue
        ds = LabeledEEGDataset(str(Path(data_dir) / ds_id), str(labels_csv))
        n_pd = sum(s[1] == 1 for s in ds.samples)
        n_hc = sum(s[1] == 0 for s in ds.samples)
        print(f"  {ds_id}: {len(ds.samples)} segs  PD={n_pd}  HC={n_hc}")
        for s in ds.samples:
            all_samples.append((s[0], s[1], f"{ds_id}/{s[2]}"))
    return all_samples


def linear_probe_fold(encoder, train_samples, test_samples):
    """Frozen encoder + linear head, combined N-LNSO style."""
    train_ds = ChannelSelectLabeledDataset(train_samples, COMMON_CH_INDICES)
    test_ds  = ChannelSelectLabeledDataset(test_samples,  COMMON_CH_INDICES)
    train_loader = DataLoader(train_ds, batch_size=PROBE_BATCH, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=PROBE_BATCH)

    n_pd = sum(s[1] == 1 for s in train_samples)
    n_hc = sum(s[1] == 0 for s in train_samples)
    pos_weight = torch.tensor([n_hc / max(n_pd, 1)], device=DEVICE)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    head = nn.Linear(encoder.feat_dim, 1).to(DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=PROBE_LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for _ in range(PROBE_EPOCHS):
        head.train()
        for x, y in train_loader:
            x, y = x.float().to(DEVICE), y.float().to(DEVICE)
            with torch.no_grad():
                feats = encoder(x)
            logits = head(feats).squeeze(-1)
            optimizer.zero_grad()
            criterion(logits, y).backward()
            optimizer.step()

    head.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.float().to(DEVICE)
            feats = encoder(x)
            preds = (torch.sigmoid(head(feats).squeeze(-1)) > 0.5).long().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y.tolist())

    for p in encoder.parameters():
        p.requires_grad = True

    return compute_metrics(all_preds, all_labels)


def run_combined_probe(encoder, data_dir):
    print(f"\n{'='*60}")
    print(f"Phase 2: Combined N-LNSO linear probe — 29-ch SSL encoder")
    print(f"  Probe epochs: {PROBE_EPOCHS}  Folds: {N_FOLDS}")
    print(f"{'='*60}\n")

    print("Loading labeled data:")
    all_samples = load_all_labeled(data_dir)
    print(f"\n  Total: {len(all_samples)} segments\n")

    subjects = np.array([s[2] for s in all_samples])
    labels   = np.array([s[1] for s in all_samples])
    unique_subjects = np.unique(subjects)

    pd_subjects = [s for s in unique_subjects if any(labels[subjects == s] == 1)]
    hc_subjects = [s for s in unique_subjects if s not in set(pd_subjects)]
    np.random.seed(42)
    np.random.shuffle(pd_subjects)
    np.random.shuffle(hc_subjects)

    print(f"  Subjects: {len(unique_subjects)} total  PD={len(pd_subjects)}  HC={len(hc_subjects)}\n")

    fold_metrics = []
    for fold in range(N_FOLDS):
        pd_test  = pd_subjects[fold::N_FOLDS]
        hc_test  = hc_subjects[fold::N_FOLDS]
        test_set = set(pd_test) | set(hc_test)

        test_s  = [s for s in all_samples if s[2] in test_set]
        train_s = [s for s in all_samples if s[2] not in test_set]

        if len(set(s[1] for s in test_s)) < 2:
            continue

        n_tr_pd = sum(s[1]==1 for s in train_s)
        n_te_pd = sum(s[1]==1 for s in test_s)
        n_te_hc = sum(s[1]==0 for s in test_s)

        metrics = linear_probe_fold(encoder, train_s, test_s)
        fold_metrics.append(metrics)
        print(f"  Fold {fold+1:2d}: bal_acc={metrics['balanced_accuracy']:.3f}  "
              f"sens={metrics['sensitivity']:.3f}  spec={metrics['specificity']:.3f}  "
              f"(test: PD={n_te_pd} HC={n_te_hc})")

    print(f"\n{'='*50}")
    print(f"  SSL 29-ch combined N-LNSO — {N_FOLDS}-fold mean")
    print(f"{'='*50}")
    agg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
    for k, v in agg.items():
        print(f"  {k:30s}: {v:.4f}")

    fold_vals = [m['balanced_accuracy'] for m in fold_metrics]
    print(f"\n  Median:  {np.median(fold_vals):.4f}")
    print(f"  Std:     {np.std(fold_vals):.4f}")
    print(f"\n  Supervised baseline (29-ch): mean=0.882  median=0.891")
    delta_mean   = agg['balanced_accuracy'] - 0.8819
    delta_median = np.median(fold_vals) - 0.8910
    print(f"  SSL delta (mean):   {delta_mean:+.4f}")
    print(f"  SSL delta (median): {delta_median:+.4f}")

    return agg, fold_metrics


def run(probe_only=False):
    encoder = build_encoder(Chan=N_CHANNELS)

    if probe_only or Path(ENCODER_SAVE).exists():
        if Path(ENCODER_SAVE).exists():
            print(f"Loading encoder from {ENCODER_SAVE}")
            encoder.load_state_dict(torch.load(ENCODER_SAVE, map_location="cpu"))
        else:
            print(f"ERROR: --probe-only but no encoder at {ENCODER_SAVE}")
            return
    else:
        pretrain_vicreg_opennero(encoder, DATA_DIR, ENCODER_SAVE)
        encoder.load_state_dict(torch.load(ENCODER_SAVE, map_location="cpu"))

    encoder = encoder.to(DEVICE)
    agg, fold_metrics = run_combined_probe(encoder, DATA_DIR)

    out = {
        "mean": agg,
        "per_fold": fold_metrics,
        "supervised_baseline": {"mean": 0.8819, "median": 0.8910},
        "config": {
            "n_channels": N_CHANNELS,
            "ch_indices": COMMON_CH_INDICES,
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "probe_epochs": PROBE_EPOCHS,
            "encoder_path": ENCODER_SAVE,
        },
    }
    out_dir = Path("results/ssl")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ssl_29ch_opennero_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-only", action="store_true", help="Skip pretrain, load existing encoder")
    args = parser.parse_args()
    run(probe_only=args.probe_only)
