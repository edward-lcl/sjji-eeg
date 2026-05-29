"""
EEG preprocessing pipeline for BIDS/OpenNeuro datasets.
Aligns datasets to a common channel set, filters, and segments.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional


TARGET_SFREQ = 250          # Hz — paper uses 250Hz
TARGET_CHANNELS = None       # None = keep native channel count per dataset (TransformEEG uses per-dataset tokenization)
EPOCH_DURATION = 16.0       # seconds per window (paper: 16s, 25% overlap)
EPOCH_OVERLAP = 0.25        # 25% overlap
BANDPASS = (1.0, 45.0)      # Hz — paper uses 1-45Hz bandpass


def load_eeg(path: str):
    """Load an EEG file (EDF, BDF, SET/FDT) using MNE."""
    import mne
    mne.set_log_level("WARNING")
    path = str(path)
    if path.endswith(".bdf"):
        raw = mne.io.read_raw_bdf(path, preload=True, verbose=False)
    elif path.endswith(".set"):
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    elif path.endswith(".edf"):
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    elif path.endswith(".vhdr"):
        raw = mne.io.read_raw_brainvision(path, preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported format: {path}")
    return raw


def load_edf(path: str):  # kept for backwards compat
    return load_eeg(path)


def preprocess_raw(raw, target_sfreq: int = TARGET_SFREQ, bandpass=BANDPASS):
    """Filter, resample, and pick EEG channels."""
    import mne
    raw.filter(bandpass[0], bandpass[1], fir_window="hamming", verbose=False)
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, verbose=False)
    raw.pick_types(eeg=True, verbose=False)
    return raw


def align_channels(raw, target_n: int = None):
    """Optionally truncate to target_n channels. If target_n is None, keep all channels.
    Never pads — cyclic padding corrupts depthwise conv tokenizers."""
    if target_n is not None and len(raw.ch_names) > target_n:
        raw.pick(raw.ch_names[:target_n])
    return raw


# Standard 64-ch 10-20 subset used for unified cross-dataset encoder
# Chosen as the intersection of standard 10-20 positions present across all 4 datasets
UNIFIED_64_CHANNELS = [
    'Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2',
    'Iz',
]


def _normalize_ch(name: str) -> str:
    """Strip common EDF prefixes/suffixes so 'EEG FP1-LE' matches 'FP1'."""
    n = name.upper()
    for prefix in ('EEG ', 'ECG ', 'EOG ', 'EMG '):
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    n = n.split('-')[0].split('_')[0].strip()
    return n


def interpolate_to_unified(raw, target_channels=None):
    """Map EEG to a standard 64-ch 10-20 montage for cross-dataset compatibility.
    Available channels are selected; missing channels are zero-padded.
    Uses zero-padding instead of MNE spherical interpolation to avoid NaN/inf
    from recordings whose montage doesn't map to standard 10-20 positions (e.g. TUH tcp_le)."""
    import mne
    if target_channels is None:
        target_channels = UNIFIED_64_CHANNELS

    # Set standard montage so channel positions are known
    montage = mne.channels.make_standard_montage('standard_1020')
    try:
        raw.set_montage(montage, match_case=False, on_missing='ignore', verbose=False)
    except Exception:
        pass

    norm_map = {_normalize_ch(ch): ch for ch in raw.ch_names}
    n_samples = raw.n_times
    sfreq = raw.info['sfreq']

    # Build output array: select available channels, zero-pad missing ones
    rows = []
    for ch in target_channels:
        if ch.upper() in norm_map:
            idx = raw.ch_names.index(norm_map[ch.upper()])
            rows.append(raw.get_data(picks=[idx])[0])
        else:
            rows.append(np.zeros(n_samples, dtype=np.float32))

    data = np.array(rows, dtype=np.float32)  # [64, n_samples]

    # Rebuild a minimal Raw object so the rest of the pipeline (segment, zscore) works
    info = mne.create_info(list(target_channels), sfreq=sfreq, ch_types='eeg')
    return mne.io.RawArray(data, info, verbose=False)


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


def process_eeg_file(eeg_path: str, output_path: Optional[str] = None,
                     unified: bool = False) -> np.ndarray:
    """Full pipeline for one EEG file → normalized segments [N, C, T].
    unified=True: interpolate to standard 64-ch montage for cross-dataset use.
    unified=False (default): keep native channel count for per-dataset training.
    """
    raw = load_eeg(eeg_path)
    raw = preprocess_raw(raw)
    if unified:
        raw = interpolate_to_unified(raw)
    else:
        raw = align_channels(raw)  # no-op unless target_n explicitly set
    segs = segment(raw)
    segs = zscore(segs)
    if output_path:
        np.save(output_path, segs)
    return segs


def process_edf_file(edf_path: str, output_path: Optional[str] = None) -> np.ndarray:
    return process_eeg_file(edf_path, output_path)


def process_dataset_dir(input_dir: str, output_dir: str, pattern: str = None,
                        unified: bool = False):
    """Batch process all EEG files (BDF, SET, EDF, VHDR) in a BIDS dataset directory.
    unified=True: interpolate to 64-ch standard montage (for SSL/cross-dataset).
    unified=False: keep native channel count (for per-dataset supervised baseline).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for ext in ["**/*.bdf", "**/*.set", "**/*.edf", "**/*.vhdr"]:
        files.extend(input_dir.glob(ext))
    print(f"Found {len(files)} EEG files in {input_dir}")

    n_ok = n_skip = n_exists = 0
    for eeg_path in files:
        rel = eeg_path.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            n_exists += 1
            continue
        try:
            result = process_eeg_file(str(eeg_path), str(out_path), unified=unified)
            if result is None or (hasattr(result, 'shape') and result.shape[0] == 0):
                n_skip += 1
                print(f"  SKIP {rel}: produced 0 segments")
            else:
                n_ok += 1
                print(f"  processed {rel}")
        except Exception as e:
            n_skip += 1
            print(f"  SKIP {rel}: {e}")

    print(f"  summary: {n_ok} processed, {n_skip} skipped, {n_exists} already existed")
    if files and n_ok == 0 and n_exists == 0:
        raise RuntimeError(
            f"process_dataset_dir: 0/{len(files)} files produced output in {input_dir}. "
            f"All {n_skip} skipped — likely a channel name or format mismatch."
        )
