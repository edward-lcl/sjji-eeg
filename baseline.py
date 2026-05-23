"""
Supervised-only baseline (no SSL pretraining) — replicates TransformEEG protocol.
Run this first to establish comparison numbers before SSL pretraining.

Usage:
    python baseline.py
"""

import torch
import numpy as np
from pathlib import Path
from src.model import build_encoder, EEGClassifier
from src.finetune import LabeledEEGDataset, run_lnso_cv
from src.evaluate import print_results, save_results, TRANSFORM_EEG_BASELINE
import yaml


def run_baseline():
    cfg = yaml.safe_load(open("configs/finetune.yaml"))
    t = cfg["training"]

    print("=" * 60)
    print("SUPERVISED BASELINE (no SSL pretraining)")
    print("Replicating TransformEEG evaluation protocol")
    print("=" * 60)

    all_results = {}
    for ds_cfg in cfg["datasets"]:
        ds_id = ds_cfg["id"]
        labels_csv = ds_cfg["labels_csv"]
        if not Path(labels_csv).exists():
            print(f"Skipping {ds_id} — not ready yet")
            continue

        print(f"\n--- {ds_id} ---")
        dataset = LabeledEEGDataset(ds_cfg["data_dir"], labels_csv)
        print(f"  {len(dataset)} segments, {len(np.unique(dataset.subject_ids()))} subjects")

        encoder = build_encoder()  # random init — no pretraining
        classifier = EEGClassifier(encoder, n_classes=2)

        results = run_lnso_cv(
            classifier=classifier,
            dataset=dataset,
            n_outer=t["n_outer_folds"],
            epochs=t["epochs"],
            batch_size=t["batch_size"],
            lr=t["lr"],
            device=t.get("device", "auto"),
        )
        all_results[ds_id] = results
        print_results(results, label=f"{ds_id} baseline")

    if all_results:
        agg = {
            k: np.mean([r[k] for r in all_results.values() if k in r])
            for k in ["balanced_accuracy", "sensitivity", "specificity"]
        }
        print_results(agg, label="AGGREGATE BASELINE")
        print(f"TransformEEG paper reported: {TRANSFORM_EEG_BASELINE['balanced_accuracy']:.4f}")
        save_results({**all_results, "aggregate": agg}, "results/baseline", "supervised_baseline")


if __name__ == "__main__":
    run_baseline()
