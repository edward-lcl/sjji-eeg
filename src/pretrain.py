"""
SimCLR self-supervised pretraining on unlabeled EEG data.
"""

import json
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
try:
    import boto3 as _boto3
except ImportError:
    _boto3 = None


MANIFEST_FILE = "manifest.json"
_EPOCH_CKPT = "simclr_checkpoint.pt"
_MID_CKPT = "simclr_mid_epoch.pt"


def _s3_upload_best(local_path, best_loss, epoch):
    """Upload best encoder to S3 immediately so a job failure can't wipe it."""
    if _boto3 is None:
        return
    bucket = os.environ.get("S3_BUCKET")
    job_name = os.environ.get("TRAINING_JOB_NAME") or os.environ.get("SM_JOB_NAME") or os.environ.get("SAGEMAKER_JOB_NAME", "local")
    if not bucket:
        return
    key = f"model-artifacts/{job_name}/pretrained_encoder_best.pt"
    try:
        _boto3.client("s3").upload_file(str(local_path), bucket, key)
        print(f"[s3] Best encoder uploaded → s3://{bucket}/{key} (epoch={epoch} loss={best_loss:.4f})")
    except Exception as e:
        print(f"[s3] Upload failed (non-fatal): {e}")
_CKPT_EVERY = 500  # save mid-epoch checkpoint every N batches


class UnlabeledEEGDataset(Dataset):
    """
    Loads preprocessed .npy segment files from a directory tree.

    Fast path: reads manifest.json (file → segment count) instead of opening
    every file. Pass manifest_name to use a subsample manifest.
    """

    def __init__(self, data_dir: str, n_channels: int = None, manifest_name: str = None):
        self.n_channels = n_channels
        self.files = []
        self.lengths = []

        manifest_path = Path(data_dir) / (manifest_name or MANIFEST_FILE)
        if manifest_path.exists():
            print(f"[dataset] Loading manifest from {manifest_path}")
            with open(manifest_path) as f:
                manifest = json.load(f)
            for rel_path, length in manifest.items():
                self.files.append(Path(data_dir) / rel_path)
                self.lengths.append(length)
            print(f"[dataset] {len(self.files):,} files  {sum(self.lengths):,} segments")
        else:
            print(f"[dataset] No manifest — scanning {data_dir} (run scripts/build_manifest.py first!)")
            for npy_path in sorted(Path(data_dir).glob("**/*.npy")):
                arr = np.load(str(npy_path), mmap_mode="r")
                self.files.append(npy_path)
                self.lengths.append(len(arr))
            print(f"[dataset] {len(self.files):,} files  {sum(self.lengths):,} segments")
            try:
                rel = {str(p.relative_to(data_dir)): l for p, l in zip(self.files, self.lengths)}
                manifest_path.write_text(json.dumps(rel, indent=2))
                print(f"[dataset] Manifest saved to {manifest_path}")
            except Exception as e:
                print(f"[dataset] Could not save manifest: {e}")

        self._cumlen = np.cumsum([0] + self.lengths)

    def __len__(self):
        return int(self._cumlen[-1])

    def __getitem__(self, idx: int):
        file_idx = int(np.searchsorted(self._cumlen[1:], idx, side="right"))
        local_idx = idx - int(self._cumlen[file_idx])
        # Per-worker LRU file cache — avoids re-opening the same mmap on consecutive accesses.
        # With FileGroupedSampler, most accesses hit the same 1-2 files per batch.
        if not hasattr(self, "_cache"):
            self._cache = {}
        if file_idx not in self._cache:
            if len(self._cache) >= 8:
                self._cache.pop(next(iter(self._cache)))
            self._cache[file_idx] = np.load(str(self.files[file_idx]), mmap_mode="r")
        arr = self._cache[file_idx]
        x = torch.from_numpy(arr[local_idx].copy())
        if self.n_channels is not None:
            C = x.shape[0]
            if C > self.n_channels:
                x = x[:self.n_channels]
            elif C < self.n_channels:
                x = torch.nn.functional.pad(x, (0, 0, 0, self.n_channels - C))
        return x


class FileGroupedSampler(Sampler):
    """
    Yields sample indices grouped by source file.

    Files are shuffled each epoch; within-file order is also shuffled.
    Compared to pure random shuffle, this cuts per-batch S3/disk seeks by ~100x
    because consecutive batches read from the same file rather than 256 random ones.
    Set self.epoch before each epoch to get a different shuffle order.
    """

    def __init__(self, dataset: UnlabeledEEGDataset, epoch: int = 0):
        self._dataset = dataset
        self.epoch = epoch

    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        n_files = len(self._dataset.files)
        file_order = torch.randperm(n_files, generator=g).tolist()
        for file_idx in file_order:
            start = int(self._dataset._cumlen[file_idx])
            end = int(self._dataset._cumlen[file_idx + 1])
            n = end - start
            within = (torch.randperm(n, generator=g) + start).tolist()
            yield from within


def eeg_augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Vectorized batch augmentation — runs on whatever device x lives on."""
    B, C, T = x.shape
    device = x.device

    # Random crop: zero out outside [start, start+crop_len) per sample
    crop_fracs = torch.empty(B, device=device).uniform_(0.7, 0.9)
    crop_lens = (crop_fracs * T).long().clamp(min=1, max=T)
    max_starts = (T - crop_lens).clamp(min=0)
    starts = (torch.rand(B, device=device) * (max_starts.float() + 1)).long().clamp(max=max_starts)
    t_idx = torch.arange(T, device=device).unsqueeze(0)
    in_crop = (t_idx >= starts.unsqueeze(1)) & (t_idx < (starts + crop_lens).unsqueeze(1))
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


def vicreg_loss(z1: torch.Tensor, z2: torch.Tensor,
                lam: float = 25.0, mu: float = 25.0, nu: float = 1.0) -> torch.Tensor:
    """
    VICReg loss (Bardes et al. 2022).
    lam=invariance, mu=variance, nu=covariance.
    Works well at small batch sizes (no large negative pool needed).
    """
    N, D = z1.shape

    # Invariance
    inv = nn.functional.mse_loss(z1, z2)

    # Variance — push std above 1 per dimension
    z1 = z1 - z1.mean(0)
    z2 = z2 - z2.mean(0)
    std1 = torch.sqrt(z1.var(0) + 1e-4)
    std2 = torch.sqrt(z2.var(0) + 1e-4)
    var = (torch.mean(nn.functional.relu(1 - std1)) +
           torch.mean(nn.functional.relu(1 - std2))) / 2

    # Covariance — decorrelate dimensions
    cov1 = (z1.T @ z1) / (N - 1)
    cov2 = (z2.T @ z2) / (N - 1)
    cov = (off_diagonal(cov1).pow(2).sum() +
           off_diagonal(cov2).pow(2).sum()) / D

    return lam * inv + mu * var + nu * cov


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


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
    manifest_name: str = None,
):
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Pretraining on {device}")
    encoder = encoder.to(device)

    feat_dim = encoder.feat_dim
    projector = nn.Sequential(
        nn.Linear(feat_dim, feat_dim),
        nn.ReLU(),
        nn.Linear(feat_dim, 128),
    ).to(device)

    use_cuda = device.startswith("cuda")
    n_workers = 8 if use_cuda else 0

    dataset = UnlabeledEEGDataset(data_dir, n_channels=n_channels, manifest_name=manifest_name)
    sampler = FileGroupedSampler(dataset, epoch=0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=n_workers,
        pin_memory=use_cuda,
        persistent_workers=n_workers > 0,
        prefetch_factor=2 if n_workers > 0 else None,
    )

    params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    print(f"Using VICReg loss (no large-batch negatives required)")

    best_loss = float("inf")
    best_encoder_state = None  # track best in memory so output_path is always writable
    no_improve = 0
    start_epoch = 1

    ckpt_dir = Path(os.environ.get("SM_HP_CHECKPOINT_DIR", "/opt/ml/checkpoints"))
    ckpt_path = ckpt_dir / _EPOCH_CKPT
    mid_ckpt_path = ckpt_dir / _MID_CKPT

    # Restore from epoch-level checkpoint (saved every 5 epochs)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        encoder.load_state_dict(ckpt["encoder"])
        projector.load_state_dict(ckpt["projector"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["best_loss"]
        no_improve = ckpt["no_improve"]
        best_encoder_state = {k: v.clone() for k, v in encoder.state_dict().items()}
        print(f"[ckpt] Resumed epoch checkpoint at epoch {ckpt['epoch']} (loss={best_loss:.4f})")

    # Restore from mid-epoch checkpoint if it belongs to start_epoch
    mid_start_batch = 0
    if mid_ckpt_path.exists():
        mid_ckpt = torch.load(mid_ckpt_path, map_location="cpu")
        if mid_ckpt.get("epoch") == start_epoch:
            encoder.load_state_dict(mid_ckpt["encoder"])
            projector.load_state_dict(mid_ckpt["projector"])
            optimizer.load_state_dict(mid_ckpt["optimizer"])
            mid_start_batch = mid_ckpt["batch_idx"] + 1
            best_loss = mid_ckpt.get("best_loss", best_loss)
            no_improve = mid_ckpt.get("no_improve", no_improve)
            print(f"[ckpt] Resumed mid-epoch checkpoint at epoch {start_epoch} batch {mid_start_batch}")

    for epoch in range(start_epoch, epochs + 1):
        encoder.train()
        projector.train()
        total_loss = 0.0
        sampler.epoch = epoch  # re-shuffle files each epoch

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)):
            # Skip batches already processed (mid-epoch resume)
            if epoch == start_epoch and batch_idx < mid_start_batch:
                continue

            x = batch.float().to(device, non_blocking=use_cuda)

            with torch.amp.autocast("cuda", enabled=use_cuda):
                x1 = eeg_augment_batch(x)
                x2 = eeg_augment_batch(x)
                z1 = projector(encoder(x1))
                z2 = projector(encoder(x2))
                loss = vicreg_loss(z1, z2)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if (batch_idx + 1) % _CKPT_EVERY == 0:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "encoder": encoder.state_dict(),
                    "projector": projector.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "no_improve": no_improve,
                }, mid_ckpt_path)

        # After each epoch, clear mid-epoch checkpoint (it's stale now)
        n_batches = len(loader) - (mid_start_batch if epoch == start_epoch else 0)
        if mid_ckpt_path.exists():
            mid_ckpt_path.unlink()
        mid_start_batch = 0  # reset for next epoch

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            no_improve = 0
            best_encoder_state = {k: v.clone() for k, v in encoder.state_dict().items()}
            torch.save(best_encoder_state, output_path)
            _s3_upload_best(output_path, best_loss, epoch)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 2 == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "projector": projector.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss,
                "no_improve": no_improve,
            }, ckpt_path)

    # Always ensure output_path exists — if we resumed and didn't improve,
    # best_encoder_state holds the checkpoint weights (last known best)
    if best_encoder_state is not None and not Path(output_path).exists():
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_encoder_state, output_path)
        print(f"[ckpt] No improvement this run — wrote checkpoint weights to {output_path}")
    print(f"Pretraining done. Best loss: {best_loss:.4f}. Weights → {output_path}")
