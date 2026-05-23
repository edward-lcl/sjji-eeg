"""
Utilities: BIDS metadata parsing, labels CSV generation, dataset inspection.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path


DATASET_LABELS = {
    # ds004148: ALL subjects are healthy controls (used as HC arm in TransformEEG)
    # subject IDs encode nothing — all are HC (label=0)
    "ds004148": {"group_col": None, "pd_value": None, "hc_value": None, "all_hc": True},
    # ds002778: subject_id prefix encodes group (sub-hc* = HC, sub-pd* = PD)
    "ds002778": {"group_col": "participant_id", "pd_prefix": "sub-pd", "hc_prefix": "sub-hc"},
    # ds003490: Group column uses CTL for controls (not HC)
    "ds003490": {"group_col": "Group", "pd_value": "PD", "hc_value": "CTL"},
    # ds004584: GROUP column; HC subjects labeled as 'Control'
    "ds004584": {"group_col": "GROUP", "pd_value": "PD", "hc_value": "Control"},
}


def build_labels_csv(dataset_dir: str, dataset_id: str, output_path: str) -> pd.DataFrame:
    """
    Parse BIDS participants.tsv to extract subject IDs and PD/HC labels.
    Saves a labels.csv with columns: subject_id, label (0=HC, 1=PD).
    """
    dataset_dir = Path(dataset_dir)
    tsv_path = dataset_dir / "participants.tsv"

    if not tsv_path.exists():
        raise FileNotFoundError(f"participants.tsv not found in {dataset_dir}")

    participants = pd.read_csv(tsv_path, sep="\t")
    print(f"\n{dataset_id} participants columns: {list(participants.columns)}")
    print(participants.head())

    cfg = DATASET_LABELS.get(dataset_id)
    if cfg is None:
        print(f"WARNING: No label config for {dataset_id}. Inspect participants.tsv manually.")
        participants.to_csv(output_path, index=False)
        return participants

    # ds004148: all subjects are healthy controls
    if cfg.get("all_hc"):
        participants["label"] = 0
        out = participants[["participant_id", "label"]].rename(columns={"participant_id": "subject_id"})
        out.to_csv(output_path, index=False)
        print(f"Saved {len(out)} subjects to {output_path}  (all HC, PD=0, HC={len(out)})")
        return out

    group_col = cfg["group_col"]

    # ds002778: label encoded in subject ID prefix
    if "pd_prefix" in cfg:
        def map_by_prefix(pid):
            pid = str(pid).strip()
            if pid.startswith(cfg["pd_prefix"]):
                return 1
            elif pid.startswith(cfg["hc_prefix"]):
                return 0
            return -1
        participants["label"] = participants["participant_id"].apply(map_by_prefix)
    else:
        if group_col not in participants.columns:
            match = [c for c in participants.columns if c.lower() == group_col.lower()]
            if match:
                group_col = match[0]
            else:
                print(f"WARNING: Column '{group_col}' not found. Columns: {list(participants.columns)}")
                participants.to_csv(output_path, index=False)
                return participants

        def map_label(val):
            val = str(val).strip()
            if val.upper() == cfg["pd_value"].upper():
                return 1
            elif val.upper() == cfg["hc_value"].upper():
                return 0
            return -1

        participants["label"] = participants[group_col].apply(map_label)

    unknown = participants[participants["label"] == -1]
    if len(unknown) > 0:
        print(f"WARNING: {len(unknown)} subjects with unknown label")

    out = participants[["participant_id", "label"]].rename(columns={"participant_id": "subject_id"})
    out = out[out["label"] >= 0]
    out.to_csv(output_path, index=False)
    print(f"Saved {len(out)} subjects to {output_path}  (PD={sum(out.label==1)}, HC={sum(out.label==0)})")
    return out


def build_all_labels(raw_dir: str, output_base_dir: str):
    """Run labels CSV generation for all 4 PD datasets."""
    raw_dir = Path(raw_dir)
    for ds_id in DATASET_LABELS:
        ds_path = raw_dir / ds_id
        if not ds_path.exists():
            print(f"Skipping {ds_id} — not downloaded yet")
            continue
        out_dir = Path(output_base_dir) / ds_id
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            build_labels_csv(str(ds_path), ds_id, str(out_dir / "labels.csv"))
        except Exception as e:
            print(f"ERROR on {ds_id}: {e}")


def inspect_dataset(dataset_dir: str, n_subjects: int = 3):
    """Print a quick summary of EDF files in a BIDS dataset."""
    dataset_dir = Path(dataset_dir)
    edf_files = list(dataset_dir.glob("**/*.edf"))
    print(f"\nDataset: {dataset_dir.name}")
    print(f"  EDF files: {len(edf_files)}")

    if not edf_files:
        print("  No EDF files found yet.")
        return

    import mne
    mne.set_log_level("WARNING")
    for edf in edf_files[:n_subjects]:
        try:
            raw = mne.io.read_raw_edf(str(edf), preload=False, verbose=False)
            info = raw.info
            duration = raw.times[-1]
            print(f"  {edf.name}: {len(info['ch_names'])} ch, "
                  f"{info['sfreq']}Hz, {duration:.1f}s")
        except Exception as e:
            print(f"  {edf.name}: ERROR — {e}")
