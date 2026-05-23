"""
SimCLR self-supervised pretraining on unlabeled EEG data via SelfEEG.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class UnlabeledEEGDataset(Dataset):
    """Loads preprocessed .npy segment files from a directory tree."""

    def __init__(self, data_dir: str):
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
        return torch.from_numpy(arr[local_idx].copy())


def eeg_augment(x: torch.Tensor) -> torch.Tensor:
    """Time-domain augmentations for SimCLR views."""
    # Random crop + pad
    T = x.shape[-1]
    crop_len = int(T * np.random.uniform(0.7, 0.9))
    start = np.random.randint(0, T - crop_len)
    x = x[:, start:start + crop_len]
    x = torch.nn.functional.pad(x, (0, T - crop_len))

    # Gaussian noise
    x = x + 0.05 * torch.randn_like(x)

    # Channel dropout
    if np.random.rand() < 0.3:
        drop_ch = np.random.randint(0, x.shape[0])
        x[drop_ch] = 0.0

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

    dataset = UnlabeledEEGDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.75, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

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

    for epoch in range(1, epochs + 1):
        encoder.train()
        projector.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x = batch.to(device)
            x1 = torch.stack([eeg_augment(s) for s in x])
            x2 = torch.stack([eeg_augment(s) for s in x])

            z1 = projector(encoder(x1))
            z2 = projector(encoder(x2))
            loss = nt_xent_loss(z1, z2, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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

    print(f"Pretraining done. Best loss: {best_loss:.4f}. Weights saved to {output_path}")
