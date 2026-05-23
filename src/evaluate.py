"""
Evaluation utilities and results reporting.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime


def print_results(results: dict, label: str = ""):
    print(f"\n{'='*50}")
    if label:
        print(f"  {label}")
    print(f"{'='*50}")
    for k, v in results.items():
        print(f"  {k:30s}: {v:.4f}")
    print()


def save_results(results: dict, output_dir: str, name: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"{name}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
    return str(out_path)


# TransformEEG paper baseline for direct comparison
TRANSFORM_EEG_BASELINE = {
    "balanced_accuracy": 0.7845,
    "sensitivity": None,   # not reported in paper
    "specificity": None,
}
