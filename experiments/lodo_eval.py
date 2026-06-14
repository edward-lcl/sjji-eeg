"""
Leave-One-Dataset-Out (LODO) evaluation — the honest cross-site protocol.

The combined N-LNSO protocol pools all 4 datasets and holds out *subjects*, which
leaves the dataset/site shortcut fully available (every site appears in both train
and test). LODO holds out an entire *dataset*: train on 3, test on the 4th, whose
acquisition signature was never seen. This is the protocol that actually matches
the project's thesis ("models trained on one site fail on another") and the one in
which the site-prior shortcut is destroyed.

Held-out folds are the 3 datasets that contain BOTH classes (ds002778, ds003490,
ds004584). ds004148 is HC-only, so it can only ever be a training contributor, not
a held-out test set (its test balanced accuracy would be undefined).

Two modes:
  --mode supervised   train encoder+head end-to-end per fold (paper hyperparams)
  --mode probe        freeze a pretrained SSL encoder, train a linear head
                      (features are cached once per fold, so this is fast)

Everything is reported at BOTH segment and subject level, alongside the site-prior
null and the chance line (0.50), via src/honest_eval.

Usage:
  python experiments/lodo_eval.py --mode supervised
  python experiments/lodo_eval.py --mode probe --encoder results/ssl/pretrained_encoder_29ch_opennero.pt
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
from src.preprocess import common_ch_indices
from src.finetune import LabeledEEGDataset
from src.honest_eval import (
    site_prior_null, subject_level_metrics, segment_level_metrics,
    fold_summary, subject_scores, calibration_report,
)

# ── Config (matches baseline_combined.py / paper supervised) ───────────────────

# Common channels (SJJI_CH_SET env: 29 default = OpenNeuro-only, 19 = TUH∩OpenNeuro)
COMMON_CH_INDICES = common_ch_indices()
N_CHANNELS = len(COMMON_CH_INDICES)

DATASET_IDS = ["ds004148", "ds002778", "ds003490", "ds004584"]
HELDOUT_IDS = ["ds002778", "ds003490", "ds004584"]   # both-class datasets only
DATA_DIR    = os.environ.get("DATA_DIR", "data/processed_unified")

SUP_EPOCHS, SUP_LR, SUP_BATCH = 50, 2.5e-4, 32
PROBE_EPOCHS, PROBE_LR, PROBE_BATCH = 30, 1e-3, 256
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class _ChSel(Dataset):
    def __init__(self, samples, ch_indices):
        self.samples = samples
        self.idx = torch.tensor(ch_indices, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, y, _ = self.samples[i]
        return x[self.idx], y


def load_all(data_dir):
    """Return {ds_id: [(seg, label, 'ds/sub'), ...]} plus a flat list."""
    by_ds, flat = {}, []
    for ds_id in DATASET_IDS:
        labels_csv = Path(data_dir) / ds_id / "labels.csv"
        if not labels_csv.exists():
            print(f"  SKIP {ds_id}: no labels.csv"); continue
        ds = LabeledEEGDataset(str(Path(data_dir) / ds_id), str(labels_csv))
        samples = [(s[0], s[1], f"{ds_id}/{s[2]}") for s in ds.samples]
        by_ds[ds_id] = samples
        flat.extend(samples)
        n_pd = sum(s[1] == 1 for s in samples)
        print(f"  {ds_id}: {len(samples)} segs  PD={n_pd}  HC={len(samples)-n_pd}")
    return by_ds, flat


# ── Per-segment scoring helpers ───────────────────────────────────────────────

@torch.no_grad()
def _score_supervised(model, samples):
    loader = DataLoader(_ChSel(samples, COMMON_CH_INDICES), batch_size=128)
    model.eval()
    scores = []
    for x, _ in loader:
        logits = model(x.float().to(DEVICE)).squeeze(-1)
        scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.array(scores)


@torch.no_grad()
def _extract_features(encoder, samples):
    loader = DataLoader(_ChSel(samples, COMMON_CH_INDICES), batch_size=128)
    encoder.eval()
    feats, labels = [], []
    for x, y in loader:
        f = encoder(x.float().to(DEVICE)).cpu().numpy()
        feats.append(f)
        labels.extend(y.numpy().tolist())
    return np.concatenate(feats, 0), np.array(labels)


def _fold_report(name, scores, samples, train_scores=None, train_samples=None):
    labels = np.array([s[1] for s in samples])
    subs = np.array([s[2] for s in samples])
    seg = segment_level_metrics(scores, labels)
    sub = subject_level_metrics(scores, labels, subs)

    cal = None
    if train_scores is not None and train_samples is not None:
        te_s, te_l = subject_scores(scores, labels, subs)
        tr_labels = np.array([s[1] for s in train_samples])
        tr_subs = np.array([s[2] for s in train_samples])
        tr_s, tr_l = subject_scores(train_scores, tr_labels, tr_subs)
        cal = calibration_report(te_s, te_l, tr_s, tr_l)

    line = (f"    [{name}] segment={seg['balanced_accuracy']:.3f} | "
            f"subject(0.5)={sub['balanced_accuracy']:.3f} "
            f"(sens={sub['sensitivity']:.3f} spec={sub['specificity']:.3f})")
    if cal:
        line += (f" | AUC={cal.get('roc_auc') or float('nan'):.3f}"
                 f"  ba@[train={cal.get('train_transferred', float('nan')):.3f}"
                 f" prev={cal.get('prevalence_matched', float('nan')):.3f}"
                 f" oracle={cal.get('oracle_youden', float('nan')):.3f}]")
    print(line)
    return {"segment": seg, "subject": sub, "calibration": cal}


# ── Supervised LODO ───────────────────────────────────────────────────────────

def train_supervised(train_samples):
    train_loader = DataLoader(_ChSel(train_samples, COMMON_CH_INDICES),
                              batch_size=SUP_BATCH, shuffle=True, drop_last=True)
    n_pd = sum(s[1] == 1 for s in train_samples)
    n_hc = sum(s[1] == 0 for s in train_samples)
    pos_weight = torch.tensor([n_hc / max(n_pd, 1)], device=DEVICE)

    encoder = build_encoder(Chan=N_CHANNELS)
    model = EEGClassifier(encoder, nb_classes=2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=SUP_LR, betas=(0.75, 0.999), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for ep in range(SUP_EPOCHS):
        for x, y in train_loader:
            x, y = x.float().to(DEVICE), y.float().to(DEVICE)
            opt.zero_grad()
            crit(model(x).squeeze(-1), y).backward()
            opt.step()
        sched.step()
    return model


# ── Probe LODO (frozen encoder + linear head on cached features) ───────────────

def train_probe(encoder, train_samples):
    Xtr, ytr = _extract_features(encoder, train_samples)
    n_pd = int((ytr == 1).sum()); n_hc = int((ytr == 0).sum())
    pos_weight = torch.tensor([n_hc / max(n_pd, 1)], device=DEVICE)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)
    head = nn.Linear(Xtr.shape[1], 1).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=PROBE_LR)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n = Xtr.shape[0]
    for ep in range(PROBE_EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, PROBE_BATCH):
            b = perm[i:i + PROBE_BATCH]
            opt.zero_grad()
            crit(head(Xtr_t[b]).squeeze(-1), ytr_t[b]).backward()
            opt.step()
    return head


@torch.no_grad()
def _score_probe(encoder, head, samples):
    Xte, _ = _extract_features(encoder, samples)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
    return torch.sigmoid(head(Xte_t).squeeze(-1)).cpu().numpy()


# ── Driver ────────────────────────────────────────────────────────────────────

def run(mode, encoder_path=None):
    print(f"\n{'='*64}")
    print(f"LODO evaluation — mode={mode} | Chan={N_CHANNELS} | device={DEVICE}")
    print(f"{'='*64}\n")

    by_ds, flat = load_all(DATA_DIR)

    # Diagnostic: site-prior null on the FULL combined pool (the threat model for
    # the combined N-LNSO headline number).
    null = site_prior_null(flat)
    print(f"\n  SITE-PRIOR NULL on combined pool (zero EEG info):")
    print(f"    segment bal_acc = {null['segment_balanced_accuracy']:.3f}")
    print(f"    subject bal_acc = {null['subject_balanced_accuracy']:.3f}")
    print(f"    per-dataset majority: {null['per_dataset_majority']}\n")

    folds = {}
    for held in HELDOUT_IDS:
        print(f"  ── Hold out {held} ─────────────────────────────")
        test_samples = by_ds[held]
        train_samples = [s for d, lst in by_ds.items() if d != held for s in lst]
        n_tr_pd = sum(s[1] == 1 for s in train_samples)
        print(f"    train={len(train_samples)} (PD={n_tr_pd} HC={len(train_samples)-n_tr_pd}) "
              f"from {[d for d in by_ds if d != held]}")
        print(f"    test ={len(test_samples)} ({held})")

        if mode == "supervised":
            model = train_supervised(train_samples)
            scores = _score_supervised(model, test_samples)
            train_scores = _score_supervised(model, train_samples)
        else:
            encoder = build_encoder(Chan=N_CHANNELS)
            encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
            encoder = encoder.to(DEVICE)
            head = train_probe(encoder, train_samples)
            scores = _score_probe(encoder, head, test_samples)
            train_scores = _score_probe(encoder, head, train_samples)

        folds[held] = _fold_report(held, scores, test_samples, train_scores, train_samples)
        del scores, train_scores

    # Macro averages across held-out datasets.
    seg_ba = [folds[d]["segment"]["balanced_accuracy"] for d in folds]
    sub_ba = [folds[d]["subject"]["balanced_accuracy"] for d in folds]
    print(f"\n{'='*64}")
    print(f"  LODO macro-average across {len(folds)} held-out datasets:")
    print(f"    segment bal_acc: mean={np.mean(seg_ba):.3f}  median={np.median(seg_ba):.3f}")
    print(f"    subject bal_acc: mean={np.mean(sub_ba):.3f}  median={np.median(sub_ba):.3f}")
    print(f"    chance = 0.500   |   site-prior null (combined) = "
          f"{null['subject_balanced_accuracy']:.3f} subj / {null['segment_balanced_accuracy']:.3f} seg")

    cal_macro = {}
    for k in ["fixed_0.5", "train_transferred", "prevalence_matched", "oracle_youden", "roc_auc"]:
        vals = [folds[d]["calibration"].get(k) for d in folds if folds[d].get("calibration")]
        vals = [v for v in vals if v is not None]
        if vals:
            cal_macro[k] = float(np.mean(vals))
    if cal_macro:
        print(f"\n    CALIBRATION (subject bal_acc, macro across held-out sites):")
        print(f"      fixed 0.5         = {cal_macro.get('fixed_0.5', float('nan')):.3f}  (the collapse)")
        print(f"      train-transferred = {cal_macro.get('train_transferred', float('nan')):.3f}  (honest, deployable)")
        print(f"      prevalence-matched= {cal_macro.get('prevalence_matched', float('nan')):.3f}  (realistic clinical)")
        print(f"      oracle (ceiling)  = {cal_macro.get('oracle_youden', float('nan')):.3f}  (= what the AUC implies)")
        print(f"      mean ROC-AUC      = {cal_macro.get('roc_auc', float('nan')):.3f}  (threshold-independent)")
    print(f"{'='*64}\n")

    out = {
        "mode": mode,
        "encoder_path": encoder_path,
        "site_prior_null": null,
        "per_heldout": folds,
        "macro": {
            "segment": fold_summary(seg_ba),
            "subject": fold_summary(sub_ba),
            "calibration": cal_macro,
        },
        "config": {"n_channels": N_CHANNELS, "heldout_ids": HELDOUT_IDS,
                   "sup_epochs": SUP_EPOCHS, "probe_epochs": PROBE_EPOCHS, "data_dir": DATA_DIR},
    }
    out_dir = Path("results/lodo"); out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"lodo_{mode}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {out_path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["supervised", "probe"], default="supervised")
    ap.add_argument("--encoder", default=f"results/ssl/pretrained_encoder_{N_CHANNELS}ch_opennero.pt")
    args = ap.parse_args()
    run(args.mode, encoder_path=args.encoder if args.mode == "probe" else None)
