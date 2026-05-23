"""
Main entry point: preprocess → pretrain → finetune → evaluate.
Run stages individually or end-to-end.

Usage:
    python train.py preprocess
    python train.py pretrain
    python train.py finetune
    python train.py all
"""

import argparse
import yaml
import torch
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def stage_preprocess(args):
    from src.preprocess import process_dataset_dir
    from src.utils import build_all_labels

    raw_dir = "data/raw"
    processed_dir = "data/processed"

    datasets = ["ds004148", "ds002778", "ds003490", "ds004584"]
    for ds in datasets:
        src = Path(raw_dir) / ds
        dst = Path(processed_dir) / ds
        if not src.exists():
            print(f"Skipping {ds} — not downloaded")
            continue
        print(f"\nPreprocessing {ds}...")
        process_dataset_dir(str(src), str(dst))

    print("\nBuilding label CSVs...")
    build_all_labels(raw_dir, processed_dir)


def stage_pretrain(args):
    from src.model import build_encoder
    from src.pretrain import pretrain_simclr

    cfg = load_config("configs/pretrain.yaml")
    encoder = build_encoder(
        n_channels=cfg["model"]["n_channels"],
        feat_dim=cfg["model"]["feat_dim"],
    )

    pretrain_simclr(
        encoder=encoder,
        data_dir=cfg["data_dir"],
        output_path=cfg["output_path"],
        **cfg["training"],
    )


def stage_finetune(args):
    from src.model import build_encoder, EEGClassifier
    from src.finetune import LabeledEEGDataset, run_lnso_cv
    from src.evaluate import print_results, save_results, TRANSFORM_EEG_BASELINE
    import json

    cfg = load_config("configs/finetune.yaml")
    t = cfg["training"]
    device = t.get("device", "auto")

    encoder = build_encoder()
    pretrained_path = cfg.get("pretrained_encoder")
    if pretrained_path and Path(pretrained_path).exists():
        encoder.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
        print(f"Loaded pretrained encoder from {pretrained_path}")
    else:
        print("WARNING: No pretrained encoder found — fine-tuning from scratch (supervised baseline)")

    all_results = {}
    for ds_cfg in cfg["datasets"]:
        ds_id = ds_cfg["id"]
        labels_csv = ds_cfg["labels_csv"]
        if not Path(labels_csv).exists():
            print(f"Skipping {ds_id} — labels.csv not found")
            continue

        print(f"\nFine-tuning on {ds_id}...")
        dataset = LabeledEEGDataset(ds_cfg["data_dir"], labels_csv)
        classifier = EEGClassifier(encoder, n_classes=2)

        results = run_lnso_cv(
            classifier=classifier,
            dataset=dataset,
            n_outer=t["n_outer_folds"],
            epochs=t["epochs"],
            batch_size=t["batch_size"],
            lr=t["lr"],
            device=device,
        )
        all_results[ds_id] = results
        print_results(results, label=ds_id)

    # Aggregate across all datasets
    import numpy as np
    if all_results:
        agg = {
            k: np.mean([r[k] for r in all_results.values() if k in r])
            for k in ["balanced_accuracy", "sensitivity", "specificity"]
        }
        print_results(agg, label="AGGREGATE (all datasets)")
        print(f"\nTransformEEG baseline: {TRANSFORM_EEG_BASELINE['balanced_accuracy']:.4f}")
        delta = agg["balanced_accuracy"] - TRANSFORM_EEG_BASELINE["balanced_accuracy"]
        print(f"Delta vs baseline: {delta:+.4f}")

        save_results({**all_results, "aggregate": agg}, cfg["results_dir"], "finetune")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preprocess", "pretrain", "finetune", "all"])
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)

    if args.stage in ("preprocess", "all"):
        stage_preprocess(args)
    if args.stage in ("pretrain", "all"):
        stage_pretrain(args)
    if args.stage in ("finetune", "all"):
        stage_finetune(args)


if __name__ == "__main__":
    main()
