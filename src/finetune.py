"""
Supervised fine-tuning on labeled Parkinson's EEG datasets.
Uses Nested-Leave-N-Subjects-Out (N-LNSO) cross-validation,
matching the evaluation protocol from the original TransformEEG paper.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score
from src.model import EEGClassifier


class LabeledEEGDataset(Dataset):
    """
    Loads processed .npy segments from BIDS-structured directory tree.
    Matches subjects via labels.csv (subject_id, label).
    """

    def __init__(self, data_dir: str, labels_csv: str):
        import pandas as pd
        self.meta = pd.read_csv(labels_csv).set_index("subject_id")
        self.data_dir = Path(data_dir)
        self.samples = []   # (segment_tensor, label, subject_id)

        for npy_path in sorted(self.data_dir.glob("**/*.npy")):
            # Extract subject ID from path (first sub-XX component)
            parts = npy_path.parts
            subj_id = next((p for p in parts if p.startswith("sub-")), None)
            if subj_id is None or subj_id not in self.meta.index:
                continue
            label = int(self.meta.loc[subj_id, "label"])
            segs = np.load(npy_path)  # [N, C, T]
            for seg in segs:
                self.samples.append((torch.from_numpy(seg), label, subj_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, subj = self.samples[idx]
        return x, y

    def subject_ids(self):
        return np.array([s[2] for s in self.samples])


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y_f = y.float().to(device)
        optimizer.zero_grad()
        logits = model(x).squeeze(-1)  # [B]
        loss = criterion(logits, y_f)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x).squeeze(-1).cpu()
        preds = (torch.sigmoid(logits) > 0.5).long().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    return np.array(all_preds), np.array(all_labels)


def compute_metrics(preds, labels, scores=None):
    """Core metrics. If scores (e.g. logits) are provided, also computes roc_auc and avg_precision (per ISSUES.md)."""
    m = {
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "sensitivity": recall_score(labels, preds, pos_label=1, zero_division=0),
        "specificity": recall_score(labels, preds, pos_label=0, zero_division=0),
        "precision": precision_score(labels, preds, pos_label=1, zero_division=0),
    }
    if scores is not None:
        try:
            m["roc_auc"] = roc_auc_score(labels, scores)
            m["avg_precision"] = average_precision_score(labels, scores)
        except Exception:
            pass  # e.g. degenerate fold
    return m


def run_lnso_cv(
    classifier,
    dataset: LabeledEEGDataset,
    n_outer: int = 10,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "auto",
):
    """10-outer N-LNSO cross-validation matching TransformEEG eval protocol."""
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    subject_ids = dataset.subject_ids()
    unique_subjects = np.unique(subject_ids)
    indices = np.arange(len(dataset))

    all_metrics = []
    logo = LeaveOneGroupOut()

    fold = 0
    for train_subj_idx, test_subj_idx in logo.split(unique_subjects, groups=unique_subjects):
        if fold >= n_outer:
            break

        test_subjects = set(unique_subjects[test_subj_idx])
        train_subjects = set(unique_subjects[train_subj_idx])

        train_idx = [i for i in indices if dataset.samples[i][2] in train_subjects]
        test_idx = [i for i in indices if dataset.samples[i][2] in test_subjects]

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size)

        # Re-init classifier head each fold, keep pretrained encoder
        model = EEGClassifier(classifier.encoder, nb_classes=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
        train_labels = [dataset.samples[i][1] for i in train_idx]
        n_pd = sum(1 for l in train_labels if l == 1)
        n_hc = sum(1 for l in train_labels if l == 0)
        pos_weight = torch.tensor([n_hc / max(n_pd, 1)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)

        preds, labels = eval_epoch(model, test_loader, device)
        metrics = compute_metrics(preds, labels)
        all_metrics.append(metrics)
        fold += 1
        print(f"Fold {fold}: balanced_acc={metrics['balanced_accuracy']:.4f} "
              f"sens={metrics['sensitivity']:.4f} spec={metrics['specificity']:.4f}")

    summary = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    summary_std = {f"{k}_std": np.std([m[k] for m in all_metrics]) for k in all_metrics[0]}
    return {**summary, **summary_std}
