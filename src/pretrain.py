"""
SimCLR self-supervised pretraining on unlabeled EEG data via SelfEEG.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class UnlabeledEEGDataset(Dataset):
    """Loads preprocessed .npy segment files from a directory tree."""

    def __init__(self, data_dir: str, n_channels: int = None):
        self.n_channels = n_channels
        self.files = []
        self.offsets = []
        self.lengths = []

        for npy_path in sorted(Path(data_dir).glob("**/*.npy")):
            arr = np.load(npy_path, mmap_mode="r")
            self.files.append(npy_path)
            self.offsets.append(len(self.offsets) and self.offsets[-1] + self.lengths[-1] or 0)
            self.lengths.append(len(arr))

        self._cumlen = np.cumsum([0] + self.lengths)

    def __len__(self):
        return int(self._cumlen[-1])

    def __getitem__(self, idx: int):
        file_idx = np.searchsorted(self._cumlen[1:], idx, side="right")
        local_idx = idx - int(self._cumlen[file_idx])
        arr = np.load(self.files[file_idx], mmap_mode="r")
        x = torch.from_numpy(arr[local_idx].copy())
        if self.n_channels is not None:
            C = x.shape[0]
            if C > self.n_channels:
                x = x[:self.n_channels]
            elif C < self.n_channels:
                x = torch.nn.functional.pad(x, (0, 0, 0, self.n_channels - C))
        return x


def eeg_augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Vectorized batch augmentation — runs on whatever device x lives on."""
    B, C, T = x.shape
    device = x.device

    # Random crop: zero out outside [start, start+crop_len) per sample
    crop_fracs = torch.empty(B, device=device).uniform_(0.7, 0.9)
    crop_lens = (crop_fracs * T).long().clamp(min=1, max=T)
    max_starts = (T - crop_lens).clamp(min=0)
    starts = (torch.rand(B, device=device) * (max_starts.float() + 1)).long().clamp(max=max_starts)
    t_idx = torch.arange(T, device=device).unsqueeze(0)          # [1, T]
    in_crop = (t_idx >= starts.unsqueeze(1)) & (t_idx < (starts + crop_lens).unsqueeze(1))  # [B, T]
    x = x * in_crop.unsqueeze(1).float()

    # Gaussian noise
    x = x + 0.05 * torch.randn_like(x)

    # Channel dropout
    drop_mask = torch.rand(B, device=device) < 0.3
    if drop_mask.any():
        drop_ch = torch.randint(0, C, (B,), device=device)
        ch_mask = torch.ones(B, C, 1, device=device)
        ch_mask[torch.where(drop_mask)[0], drop_ch[drop_mask]] = 0.0
        x = x * ch_mask

    return x


def pretrain_simclr(
    encoder: nn.Module,
    data_dir: str,
    output_path: str,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 2.5e-4,
    temperature: float = 0.5,
    patience: int = 30,
    device: str = "auto",
    n_channels: int = None,
):
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Pretraining on {device}")
    encoder = encoder.to(device)

    # Projection head for SimCLR
    feat_dim = encoder.feat_dim
    projector = nn.Sequential(
        nn.Linear(feat_dim, feat_dim),
        nn.ReLU(),
        nn.Linear(feat_dim, 128),
    ).to(device)

    use_cuda = device.startswith("cuda")
    n_workers = 4 if use_cuda else 0

    dataset = UnlabeledEEGDataset(data_dir, n_channels=n_channels)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=use_cuda,
        persistent_workers=n_workers > 0,
        prefetch_factor=2 if n_workers > 0 else None,
    )

    params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.75, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temp: float) -> torch.Tensor:
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)  # [2N, D]
        sim = torch.mm(z, z.T) / temp
        N = z1.shape[0]
        labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)]).to(device)
        mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, float("-inf"))
        return nn.functional.cross_entropy(sim, labels)

    best_loss = float("inf")
    no_improve = 0
    start_epoch = 1

    ckpt_dir = Path(os.environ.get("SM_HP_CHECKPOINT_DIR", "/opt/ml/checkpoints"))
    ckpt_path = ckpt_dir / "simclr_checkpoint.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        encoder.load_state_dict(ckpt["encoder"])
        projector.load_state_dict(ckpt["projector"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["best_loss"]
        no_improve = ckpt["no_improve"]
        print(f"Resumed from checkpoint at epoch {ckpt['epoch']} (loss={best_loss:.4f})")

    for epoch in range(start_epoch, epochs + 1):
        encoder.train()
        projector.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x = batch.float().to(device, non_blocking=use_cuda)

            with torch.amp.autocast("cuda", enabled=use_cuda):
                x1 = eeg_augment_batch(x)
                x2 = eeg_augment_batch(x)
                z1 = projector(encoder(x1))
                z2 = projector(encoder(x2))
                loss = nt_xent_loss(z1, z2, temperature)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step()
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            no_improve = 0
            torch.save(encoder.state_dict(), output_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 5 == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch, "encoder": encoder.state_dict(),
                "projector": projector.state_dict(), "optimizer": optimizer.state_dict(),
                "best_loss": best_loss, "no_improve": no_improve,
            }, ckpt_path)

    print(f"Pretraining done. Best loss: {best_loss:.4f}. Weights saved to {output_path}")
