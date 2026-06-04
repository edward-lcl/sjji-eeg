"""
SSL pilot experiment — SimCLR pretraining + linear probe.

Phase 1: Pretrain TransformEEG encoder with SimCLR on ALL available EEG
         (ds004148 HC-only + labeled segments from ds002778/003490/004584,
          labels ignored — pure self-supervised).

Phase 2: For each labeled dataset, freeze encoder and train a linear
         classifier head with the same N-LNSO cross-validation protocol
         as the supervised baseline.

Phase 3: Leave-one-dataset-out cross-dataset evaluation with frozen encoder
         + linear head trained on train split.

Key question: does SSL pretraining improve cross-dataset balanced accuracy
above the supervised baseline's ~0.503?
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import balanced_accuracy_score, recall_score
import json
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import build_encoder, EEGClassifier
from src.pretrain import pretrain_simclr, UnlabeledEEGDataset
from src.finetune import LabeledEEGDataset, train_epoch, eval_epoch, compute_metrics
from baseline import DEVICE, BATCH_SIZE, LR, N_OUTER, PD_DATASET_IDS, load_all_datasets, FlatDataset

# --- Config ---
PRETRAIN_EPOCHS   = 100
PRETRAIN_LR       = 2.5e-4
PRETRAIN_BATCH    = 128   # safe on A10G 24GB with grad checkpointing
MANIFEST_FILE     = "manifest.json"  # sub400k prefix has its own manifest.json with 281 shards
FINETUNE_EPOCHS   = 30   # shorter than supervised: encoder already has structure
N_CHANNELS        = 64
# SageMaker mounts packed channel first, fall back to unpacked for local dev
# Sub400k packed shards used for Phase 1 pretraining (fast, File mode)
PROCESSED_UNIFIED = (
    os.environ.get("SM_CHANNEL_PROCESSED_UNIFIED_SUB400K")
    or os.environ.get("SM_CHANNEL_PROCESSED_UNIFIED_PACKED")
    or os.environ.get("SM_CHANNEL_PROCESSED_UNIFIED")
    or "data/processed_unified"
)
# Full processed_unified used for Phase 2/3 (has labels.csv per dataset, FastFile)
LABELED_BASE = (
    os.environ.get("SM_CHANNEL_PROCESSED_UNIFIED")
    or "data/processed_unified"
)

ENCODER_PATH      = os.path.join(os.environ.get("SM_MODEL_DIR", "results/ssl"), "pretrained_encoder.pt")


def get_n_channels(dataset):
    return min(sample[0].shape[0] for sample in dataset.samples)


def linear_probe_train_eval(encoder, train_samples, test_samples, n_channels):
    """Freeze encoder, train + eval linear head only."""
    train_ds = FlatDataset(train_samples, n_channels=n_channels)
    test_ds  = FlatDataset(test_samples,  n_channels=n_channels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    n_pd = sum(1 for s in train_samples if s[1] == 1)
    n_hc = sum(1 for s in train_samples if s[1] == 0)
    pos_weight = torch.tensor([n_hc / max(n_pd, 1)], device=DEVICE)

    model = EEGClassifier(encoder, nb_classes=2).to(DEVICE)
    # Freeze encoder — only train the head
    for param in model.encoder.parameters():
        param.requires_grad = False
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(head_params, lr=LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for _ in range(FINETUNE_EPOCHS):
        train_epoch(model, train_loader, optimizer, criterion, DEVICE)

    preds, labels = eval_epoch(model, test_loader, DEVICE)
    # Unfreeze for next fold (encoder gets re-used)
    for param in model.encoder.parameters():
        param.requires_grad = True
    return compute_metrics(preds, labels)


def run_ssl_pilot():
    print(f"\n{'='*60}")
    print(f"SSL PILOT — SimCLR + Linear Probe")
    print(f"Device: {DEVICE} | Pretrain epochs: {PRETRAIN_EPOCHS} | Finetune epochs: {FINETUNE_EPOCHS}")
    print(f"{'='*60}\n")

    results_dir = Path(os.environ.get("SM_MODEL_DIR", "results/ssl"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Pretrain ──────────────────────────────────────────
    print("PHASE 1: SimCLR pretraining on all unlabeled EEG")
    print(f"  Data: {PROCESSED_UNIFIED} (all datasets, labels ignored)")

    encoder = build_encoder(Chan=N_CHANNELS)

    if Path(ENCODER_PATH).exists():
        print(f"  Found existing encoder at {ENCODER_PATH}, loading...")
        encoder.load_state_dict(torch.load(ENCODER_PATH, map_location="cpu"))
        print("  Skipping pretraining. Delete the file to retrain.")
    else:
        pretrain_simclr(
            encoder=encoder,
            data_dir=PROCESSED_UNIFIED,
            output_path=ENCODER_PATH,
            epochs=PRETRAIN_EPOCHS,
            batch_size=PRETRAIN_BATCH,
            lr=PRETRAIN_LR,
            patience=20,
            device=DEVICE,
            n_channels=N_CHANNELS,
            manifest_name=MANIFEST_FILE,
        )
        print(f"  Encoder saved to {ENCODER_PATH}")

    encoder = encoder.to(DEVICE)

    # ── Phase 2: Per-dataset linear probe ─────────────────────────
    print("\nPHASE 2: Per-dataset linear probe (N-LNSO, frozen encoder)")
    datasets = load_all_datasets(LABELED_BASE)
    per_dataset_results = {}

    for ds_id in PD_DATASET_IDS:
        ds = datasets[ds_id]
        n_channels = get_n_channels(ds)
        subjects = np.unique([s[2] for s in ds.samples])

        pd_subjects = [s for s in subjects if any(x[2] == s and x[1] == 1 for x in ds.samples)]
        hc_subjects = [s for s in subjects if s not in pd_subjects]
        np.random.seed(42)
        np.random.shuffle(pd_subjects)
        np.random.shuffle(hc_subjects)

        n_pd = sum(1 for s in ds.samples if s[1] == 1)
        n_hc = sum(1 for s in ds.samples if s[1] == 0)
        print(f"\n  --- {ds_id} ({len(subjects)} subjects, PD={n_pd}, HC={n_hc}, Chan={n_channels}) ---")

        fold_metrics = []
        for fold in range(N_OUTER):
            pd_test = pd_subjects[fold::N_OUTER]
            hc_test = hc_subjects[fold::N_OUTER]
            test_subjects = set(pd_test) | set(hc_test)
            test_samples  = [s for s in ds.samples if s[2] in test_subjects]
            train_samples = [s for s in ds.samples if s[2] not in test_subjects]
            if len(set(s[1] for s in test_samples)) < 2:
                continue
            # Use the pretrained encoder (shared), train only head
            metrics = linear_probe_train_eval(encoder, train_samples, test_samples, n_channels)
            fold_metrics.append(metrics)
            print(f"    Fold {fold+1}: bal_acc={metrics['balanced_accuracy']:.3f}  "
                  f"sens={metrics['sensitivity']:.3f}  spec={metrics['specificity']:.3f}")

        if fold_metrics:
            agg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
            per_dataset_results[ds_id] = agg
            print(f"    Mean: bal_acc={agg['balanced_accuracy']:.3f}  "
                  f"sens={agg['sensitivity']:.3f}  spec={agg['specificity']:.3f}")

    if per_dataset_results:
        overall = {k: np.mean([r[k] for r in per_dataset_results.values()]) for k in list(per_dataset_results.values())[0]}
        per_dataset_results["aggregate"] = overall
        print(f"\n  SSL per-dataset aggregate:   bal_acc={overall['balanced_accuracy']:.4f}")
        print(f"  Supervised baseline was:      bal_acc=0.5376")

    # ── Phase 3: Cross-dataset linear probe ───────────────────────
    print("\nPHASE 3: Cross-dataset linear probe (frozen encoder)")
    cross_results = {}

    for held_out_id in PD_DATASET_IDS:
        train_ids = [d for d in list(datasets.keys()) if d != held_out_id]
        test_ds   = datasets[held_out_id]
        n_channels = get_n_channels(test_ds)

        train_samples = []
        for tid in train_ids:
            if tid in datasets and get_n_channels(datasets[tid]) == n_channels:
                train_samples.extend(datasets[tid].samples)

        if not train_samples:
            continue

        test_labels = [s[1] for s in test_ds.samples]
        if len(set(test_labels)) < 2:
            continue

        print(f"\n  Train: {[t for t in train_ids if t in datasets]} → Test: {held_out_id}")
        metrics = linear_probe_train_eval(encoder, train_samples, test_ds.samples, n_channels)
        cross_results[held_out_id] = metrics
        print(f"    → bal_acc={metrics['balanced_accuracy']:.3f}  "
              f"sens={metrics['sensitivity']:.3f}  spec={metrics['specificity']:.3f}")

    if cross_results:
        agg = {k: np.mean([r[k] for r in cross_results.values()]) for k in list(cross_results.values())[0]}
        cross_results["aggregate"] = agg
        print(f"\n  SSL cross-dataset aggregate:  bal_acc={agg['balanced_accuracy']:.4f}")
        print(f"  Supervised baseline was:       bal_acc=0.5026")
        delta = agg['balanced_accuracy'] - 0.5026
        print(f"  Delta vs supervised:          {delta:+.4f}")
        if delta > 0.02:
            print(f"  ✅ SSL improves cross-dataset generalization")
        elif delta > -0.02:
            print(f"  ⚠️  SSL roughly equal to supervised baseline")
        else:
            print(f"  ❌ SSL does not improve generalization — domain gap deeper than SSL can fix")

    all_results = {
        "per_dataset": per_dataset_results,
        "cross_dataset": cross_results,
        "config": {
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "n_channels": N_CHANNELS,
            "encoder_path": ENCODER_PATH,
        }
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"ssl_pilot_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out_path}")
    return all_results


if __name__ == "__main__":
    run_ssl_pilot()
