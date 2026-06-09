"""
Reprocess the 4 OpenNeuro datasets with native (per-dataset) channel counts.
Output: data/processed/<ds_id>/ with .npy files in same BIDS structure.

This is the preprocessing for the per-dataset supervised baseline that matches
the TransformEEG paper's protocol — each dataset keeps its own channel count,
no padding. The encoder is then instantiated with Chan=N per dataset.

Usage:
    python scripts/reprocess_native.py
    python scripts/reprocess_native.py --input-dir data/raw --output-dir data/processed
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocess import process_dataset_dir


DATASETS = ["ds002778", "ds003490", "ds004148", "ds004584"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir",  default="data/raw",       help="Root directory with raw BIDS datasets")
    p.add_argument("--output-dir", default="data/processed", help="Root directory for native-channel .npy output")
    p.add_argument("--datasets", nargs="+", default=DATASETS, help="Dataset IDs to process")
    args = p.parse_args()

    input_root  = Path(args.input_dir)
    output_root = Path(args.output_dir)

    for ds_id in args.datasets:
        input_dir  = input_root  / ds_id
        output_dir = output_root / ds_id
        if not input_dir.exists():
            print(f"  SKIP {ds_id}: {input_dir} not found")
            continue
        print(f"\n{'='*60}")
        print(f"Processing {ds_id} (native channels, unified=False)")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")
        print(f"{'='*60}")
        process_dataset_dir(str(input_dir), str(output_dir), unified=False)

        # Copy labels.csv from processed_unified if it exists there and isn't in output_dir yet
        labels_src = Path("data/processed_unified") / ds_id / "labels.csv"
        labels_dst = output_dir / "labels.csv"
        if labels_src.exists() and not labels_dst.exists():
            shutil.copy2(labels_src, labels_dst)
            print(f"  Copied labels.csv from {labels_src}")
        elif not labels_dst.exists():
            print(f"  WARNING: no labels.csv found for {ds_id} — baseline will skip this dataset")

    print("\nAll done. Native-channel data ready in", output_root)


if __name__ == "__main__":
    main()
