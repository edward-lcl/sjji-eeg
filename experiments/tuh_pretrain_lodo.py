"""
Local small-scale TUH × LODO proof-of-concept.

Pretrain a 19-ch VICReg encoder on TUH ONLY (disjoint from OpenNeuro by
construction — reads only the TUH unified dir), then LODO-probe on OpenNeuro via
experiments/lodo_eval.py. This is the real TUH×LODO experiment in miniature
(local TUH recordings) to validate the pipeline end-to-end and get an early
directional signal before the full-scale SageMaker run.

Channels are forced to the 19-ch TUH∩OpenNeuro montage (verified 19/19 alive in
TUH), so the encoder never trains a depthwise filter on a dead channel.

Usage:
  # 1. ensure TUH is ingested unified:  process_dataset_dir(..., unified=True)
  python experiments/tuh_pretrain_lodo.py
  # 2. evaluate (note SJJI_CH_SET=19 so the probe builds a 19-ch encoder):
  SJJI_CH_SET=19 python experiments/lodo_eval.py --mode probe \
      --encoder results/ssl/pretrained_encoder_19ch_tuh.pt
"""

import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import build_encoder
from src.preprocess import common_ch_indices
from src.pretrain import eeg_augment_batch, vicreg_loss

CH = common_ch_indices(19)            # force the 19-ch TUH∩OpenNeuro montage
N_CHANNELS = len(CH)
TUH_DIR = os.environ.get("TUH_DIR", "data/processed_tuh_unified")
ENCODER_SAVE = "results/ssl/pretrained_encoder_19ch_tuh.pt"
EPOCHS, LR, BATCH, PATIENCE = 100, 2.5e-4, 64, 20
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class TUHDataset(Dataset):
    """TUH unified (64-ch) .npy segments, channel-selected to the 19-ch montage."""
    def __init__(self, data_dir, ch):
        self.ch = torch.tensor(ch, dtype=torch.long)
        self.files, lengths = [], []
        for p in sorted(glob.glob(f"{data_dir}/**/*.npy", recursive=True)):
            a = np.load(p, mmap_mode="r")
            if a.ndim == 3 and a.shape[1] >= 64:    # must be unified, not native
                self.files.append(p); lengths.append(a.shape[0])
        self._cum = np.cumsum([0] + lengths)
        print(f"[tuh] {len(self.files)} unified files, {int(self._cum[-1])} segments, ch={len(ch)}")

    def __len__(self):
        return int(self._cum[-1])

    def __getitem__(self, idx):
        fi = int(np.searchsorted(self._cum[1:], idx, side="right"))
        li = idx - int(self._cum[fi])
        if not hasattr(self, "_cache"):
            self._cache = {}
        if fi not in self._cache:
            if len(self._cache) >= 8:
                self._cache.pop(next(iter(self._cache)))
            self._cache[fi] = np.load(self.files[fi], mmap_mode="r")
        x = torch.from_numpy(self._cache[fi][li].copy())
        return x[self.ch]


def run():
    print(f"\n{'='*60}\nTUH-only VICReg pretrain — {N_CHANNELS}ch | device={DEVICE}\n{'='*60}")
    ds = TUHDataset(TUH_DIR, CH)
    if len(ds) == 0:
        print(f"No unified TUH segments in {TUH_DIR}. Run process_dataset_dir(unified=True) first.")
        return

    # Sanity: confirm the 19 channels actually carry signal (no dead-channel pretraining)
    probe = np.stack([ds[i].numpy() for i in range(min(64, len(ds)))])  # [n,19,T]
    alive = int((np.abs(probe).mean(axis=(0, 2)) > 1e-6).sum())
    print(f"[tuh] channel sanity: {alive}/{N_CHANNELS} channels alive in sample")

    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)
    enc = build_encoder(Chan=N_CHANNELS).to(DEVICE)
    proj = nn.Sequential(nn.Linear(enc.feat_dim, enc.feat_dim), nn.ReLU(),
                         nn.Linear(enc.feat_dim, 128)).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)
    Path(ENCODER_SAVE).parent.mkdir(parents=True, exist_ok=True)

    best, bad = float("inf"), 0
    for ep in range(1, EPOCHS + 1):
        enc.train(); proj.train()
        tot, nb = 0.0, 0
        for x in loader:
            x = x.float().to(DEVICE)
            z1 = proj(enc(eeg_augment_batch(x)))
            z2 = proj(enc(eeg_augment_batch(x)))
            loss = vicreg_loss(z1, z2)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        avg = tot / max(nb, 1)
        if ep % 10 == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{EPOCHS}  loss={avg:.4f}  {'*' if avg < best else ''}")
        if avg < best:
            best = avg; torch.save(enc.state_dict(), ENCODER_SAVE); bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  early stop at epoch {ep}"); break
    print(f"  best loss {best:.4f}  ->  {ENCODER_SAVE}")


if __name__ == "__main__":
    run()
