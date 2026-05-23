"""
EEG preprocessing pipeline for BIDS/OpenNeuro datasets.
Aligns datasets to a common channel set, filters, and segments.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional


TARGET_SFREQ = 250          # Hz — paper uses 250Hz
TARGET_CHANNELS = 61        # channel count matching TransformEEG
EPOCH_DURATION = 16.0       # seconds per window (paper: 16s, 25% overlap)
EPOCH_OVERLAP = 0.25        # 25% overlap
BANDPASS = (1.0, 45.0)      # Hz — paper uses 1-45Hz bandpass


def load_edf(path: str):
    """Load a single EDF file using MNE."""
    import mne
    mne.set_log_level("WARNING")
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    return raw


def preprocess_raw(raw, target_sfreq: int = TARGET_SFREQ, bandpass=BANDPASS):
    """Filter, resample, and pick EEG channels."""
    import mne
    raw.filter(bandpass[0], bandpass[1], fir_window="hamming", verbose=False)
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, verbose=False)
    raw.pick_types(eeg=True, verbose=False)
    return raw


def align_channels(raw, target_n: int = TARGET_CHANNELS):
    """Select or pad to target channel count."""
    n = len(raw.ch_names)
    if n >= target_n:
        raw.pick(raw.ch_names[:target_n])
    else:
        # Repeat channels cyclically to reach target (rough but functional)
        data, times = raw.get_data(return_times=True)
        pad = np.tile(data, (target_n // n + 1, 1))[:target_n]
        import mne
        info = mne.create_info(
            ch_names=[f"EEG{i:03d}" for i in range(target_n)],
            sfreq=raw.info["sfreq"],
            ch_types="eeg",
        )
        raw = mne.io.RawArray(pad, info, verbose=False)
    return raw


def segment(raw, duration: float = EPOCH_DURATION, overlap: float = EPOCH_OVERLAP) -> np.ndarray:
    """Split continuous recording into overlapping windows. Returns [N, C, T]."""
    sfreq = raw.info["sfreq"]
    n_samples = int(duration * sfreq)
    step = int(n_samples * (1 - overlap))
    data = raw.get_data()  # [C, total_samples]
    total = data.shape[1]
    starts = range(0, total - n_samples + 1, step)
    segments = np.stack([data[:, s:s + n_samples] for s in starts])
    return segments.astype(np.float32)


def zscore(segments: np.ndarray) -> np.ndarray:
    """Per-channel z-score normalization across time."""
    mean = segments.mean(axis=-1, keepdims=True)
    std = segments.std(axis=-1, keepdims=True) + 1e-8
    return (segments - mean) / std


def process_edf_file(edf_path: str, output_path: Optional[str] = None) -> np.ndarray:
    """Full pipeline for one EDF file → normalized segments [N, C, T]."""
    raw = load_edf(edf_path)
    raw = preprocess_raw(raw)
    raw = align_channels(raw)
    segs = segment(raw)
    segs = zscore(segs)
    if output_path:
        np.save(output_path, segs)
    return segs


def process_dataset_dir(input_dir: str, output_dir: str, pattern: str = "**/*.edf"):
    """Batch process all EDF files in a BIDS dataset directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob(pattern))
    print(f"Found {len(files)} EDF files in {input_dir}")

    for edf_path in files:
        rel = edf_path.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            continue
        try:
            process_edf_file(str(edf_path), str(out_path))
            print(f"  processed {rel}")
        except Exception as e:
            print(f"  SKIP {rel}: {e}")
