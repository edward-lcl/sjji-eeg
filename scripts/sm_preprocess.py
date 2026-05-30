#!/usr/bin/env python3
"""
SageMaker preprocessing — reads TUH EDF files directly from S3 via boto3,
preprocesses to .npy segments, uploads results back to S3.

Resume-safe: checks which .npy files already exist in S3 and skips them,
so repeated runs after failures only process the remaining files.

Env vars:
  S3_BUCKET       — bucket with raw data and output
  S3_RAW_PREFIX   — prefix for raw EDFs
  S3_OUT_PREFIX   — prefix for output .npy files
  SM_NUM_CPUS     — set by SageMaker
"""

import os
import sys
import json
import time
import tempfile
import multiprocessing as mp
from pathlib import Path

import boto3
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocess import process_eeg_file

BUCKET     = os.environ.get("S3_BUCKET",     "sagemaker-us-east-2-506145782110")
RAW_PREFIX = os.environ.get("S3_RAW_PREFIX", "data/raw/tuh_eeg/v2.0.1/edf")
OUT_PREFIX = os.environ.get("S3_OUT_PREFIX", "data/processed_unified/tuh_eeg")
N_WORKERS  = min(int(os.environ.get("SM_NUM_CPUS", mp.cpu_count())), 8)
MODEL_DIR  = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))


def list_keys(s3, bucket, prefix, suffix):
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(suffix):
                keys.append(obj["Key"])
    return keys


def edf_to_out_key(s3_key, raw_prefix, out_prefix):
    rel = s3_key[len(raw_prefix):].lstrip("/")
    return f"{out_prefix}/{Path(rel).with_suffix('.npy')}"


def process_one(args):
    s3_key, bucket, raw_prefix, out_prefix = args
    s3 = boto3.client("s3")
    try:
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            s3.download_file(bucket, s3_key, tmp.name)
            segs = process_eeg_file(tmp.name, unified=True)
        Path(tmp.name).unlink(missing_ok=True)

        if segs is None or segs.shape[0] == 0:
            return s3_key, 0, None

        out_key = edf_to_out_key(s3_key, raw_prefix, out_prefix)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            np.save(tmp.name, segs)
            s3.upload_file(tmp.name, bucket, out_key)
        Path(tmp.name).unlink(missing_ok=True)

        return s3_key, segs.shape[0], None
    except Exception as e:
        return s3_key, 0, str(e)[:120]


def main():
    print("TUH EEG S3-native Preprocessing (resume-safe)")
    print(f"  Bucket:  {BUCKET}")
    print(f"  Raw:     {RAW_PREFIX}")
    print(f"  Out:     {OUT_PREFIX}")
    print(f"  Workers: {N_WORKERS}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")

    print("Listing EDF files...")
    all_keys = list_keys(s3, BUCKET, RAW_PREFIX, ".edf")
    print(f"  Found {len(all_keys):,} EDF files")

    print("Checking existing outputs (resume check)...")
    existing = set(list_keys(s3, BUCKET, OUT_PREFIX, ".npy"))
    print(f"  Already done: {len(existing):,} files")

    # Filter to only files whose output doesn't exist yet
    pending = [k for k in all_keys
               if edf_to_out_key(k, RAW_PREFIX, OUT_PREFIX) not in existing]
    print(f"  Remaining:    {len(pending):,} files to process")

    if not pending:
        print("All files already processed.")
        return

    args = [(k, BUCKET, RAW_PREFIX, OUT_PREFIX) for k in pending]
    start = time.time()
    total_segs = skipped = errors = 0

    with mp.Pool(N_WORKERS) as pool:
        for i, (key, n_segs, err) in enumerate(pool.imap_unordered(process_one, args, chunksize=4)):
            if err:
                errors += 1
            elif n_segs == 0:
                skipped += 1
            else:
                total_segs += n_segs

            if (i + 1) % 500 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                eta = (len(pending) - i - 1) / rate
                print(f"  [{i+1}/{len(pending)}] {total_segs:,} segs | "
                      f"{rate:.1f} files/s | ETA {eta/60:.0f}m | "
                      f"skip={skipped} err={errors}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f}m — {total_segs:,} segments, "
          f"{skipped:,} skipped, {errors:,} errors")

    manifest = {
        "total_edfs": len(all_keys), "already_done": len(existing),
        "processed_this_run": len(pending) - skipped - errors,
        "total_segments": total_segs,
        "skipped": skipped, "errors": errors, "elapsed_min": elapsed / 60,
    }
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Manifest written.")


if __name__ == "__main__":
    main()
