"""
SageMaker job: pack many small .npy segment files into large shards.

Reads processed_unified from S3 (FastFile), writes packed shards back to S3.
Target: SEGMENTS_PER_SHARD segments per output shard → ~8x fewer files.

Each shard is a single .npy of shape (N, C, T) where N <= SEGMENTS_PER_SHARD.
Shards are written to data/processed_unified_packed/<dataset>/<shard_XXXXX.npy>.

Resume-safe: skips shard keys that already exist in S3.

Env vars (SageMaker sets SM_* automatically):
  S3_BUCKET               — bucket name
  SM_CHANNEL_PROCESSED_UNIFIED — mount point for source data (FastFile)
  SM_MODEL_DIR            — where to write the output manifest
"""

import json
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path

import boto3
import numpy as np

BUCKET              = os.environ.get("S3_BUCKET", "sagemaker-us-east-2-506145782110")
SRC_PREFIX          = "data/processed_unified/"  # trailing slash prevents matching processed_unified_packed
DST_PREFIX          = "data/processed_unified_packed"
SEGMENTS_PER_SHARD  = 1024
REGION              = os.environ.get("AWS_REGION", "us-east-2")
N_WORKERS           = min(int(os.environ.get("SM_NUM_CPUS", mp.cpu_count())), 16)
MODEL_DIR           = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

# SageMaker FastFile mount — source files appear here as if local
DATA_DIR = os.environ.get("SM_CHANNEL_PROCESSED_UNIFIED", SRC_PREFIX)


def list_npy_keys(s3, prefix: str) -> list[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".npy") and "manifest" not in obj["Key"]:
                keys.append(obj["Key"])
    return keys


def group_by_dataset(keys: list[str], src_prefix: str) -> dict[str, list[str]]:
    """
    Group S3 keys by shard group key.

    For large datasets (tuh_eeg), split by the second path level (subdirectory
    within the dataset) so all 16 workers get real parallel work instead of one
    worker handling the entire TUH corpus sequentially.

    Small datasets (ds002778 etc.) are kept as a single group — they're tiny.
    """
    # Datasets large enough to warrant sub-grouping
    LARGE_DATASETS = {"tuh_eeg"}

    groups: dict[str, list[str]] = {}
    for key in keys:
        rel = key[len(src_prefix):].lstrip("/")
        parts = rel.split("/")
        dataset = parts[0]
        if dataset in LARGE_DATASETS and len(parts) > 1:
            group_key = f"{dataset}/{parts[1]}"  # e.g. tuh_eeg/022
        else:
            group_key = dataset
        groups.setdefault(group_key, []).append(key)
    return groups


def pack_group(args):
    """Pack one group of source keys into shards, upload to S3."""
    dataset, src_keys, dst_prefix, existing_dst_keys, group_key = args
    s3 = boto3.client("s3", region_name=REGION)
    existing = set(existing_dst_keys)

    total_shards = 0
    total_segments = 0
    errors = 0

    buffer = []
    buffer_shape = None  # (C, T) — flush when shape changes
    shard_idx = 0

    def flush(buf):
        nonlocal shard_idx, total_shards, total_segments
        if not buf:
            return
        arr = np.concatenate(buf, axis=0)  # safe: all same (C, T)
        # Include group_key hash in name to avoid collisions across parallel workers
        import hashlib
        gkey = hashlib.md5(group_key.encode()).hexdigest()[:6]
        dst_key = f"{dst_prefix}/{dataset}/shard_{gkey}_{shard_idx:05d}.npy"
        shard_idx += 1

        if dst_key in existing:
            return  # resume: already done

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            np.save(tmp.name, arr)
            try:
                s3.upload_file(tmp.name, BUCKET, dst_key)
                total_shards += 1
                total_segments += arr.shape[0]
            except Exception as e:
                pass  # counted below
            finally:
                Path(tmp.name).unlink(missing_ok=True)

    for key in src_keys:
        try:
            local_path = os.path.join(DATA_DIR, key[len(SRC_PREFIX):].lstrip("/"))
            segs = np.load(local_path, mmap_mode="r")
            data = segs[:]  # copy out of mmap

            seg_shape = data.shape[1:]  # (C, T)
            if buffer_shape is not None and seg_shape != buffer_shape:
                # Channel/time mismatch — flush current buffer before mixing shapes
                flush(buffer)
                buffer = []
                buffer_shape = None

            buffer.append(data)
            buffer_shape = seg_shape

            if sum(len(b) for b in buffer) >= SEGMENTS_PER_SHARD:
                flush(buffer)
                buffer = []
                buffer_shape = None
        except Exception as e:
            errors += 1

    flush(buffer)  # remainder

    return dataset, total_shards, total_segments, errors


def build_manifest(s3, dst_prefix: str) -> dict:
    keys = list_npy_keys(s3, dst_prefix)
    manifest = {}
    for key in keys:
        rel = key[len(dst_prefix):].lstrip("/")
        manifest[rel] = None  # segment count filled lazily by dataset loader
    return manifest


def main():
    print(f"Packing shards: {SRC_PREFIX} → {DST_PREFIX}")
    print(f"  Target: {SEGMENTS_PER_SHARD} segments/shard")
    print(f"  Workers: {N_WORKERS}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=REGION)

    print("Listing source files...")
    src_keys = list_npy_keys(s3, SRC_PREFIX)
    print(f"  {len(src_keys):,} source files")

    print("Listing existing destination shards (resume check)...")
    dst_keys = list_npy_keys(s3, DST_PREFIX)
    print(f"  {len(dst_keys):,} shards already written")

    groups = group_by_dataset(src_keys, SRC_PREFIX)
    dst_groups = group_by_dataset(dst_keys, DST_PREFIX)

    # Normalize group key to top-level dataset name for output path
    # (tuh_eeg/022 writes shards under tuh_eeg/, not tuh_eeg/022/)
    def dst_dataset(group_key):
        return group_key.split("/")[0]

    args = [
        (dst_dataset(group_key), keys, DST_PREFIX, dst_groups.get(dst_dataset(group_key), []), group_key)
        for group_key, keys in sorted(groups.items())
    ]

    n_tuh = sum(1 for g in groups if g.startswith("tuh_eeg/"))
    print(f"\n{len(args)} worker groups ({n_tuh} TUH subgroups + {len(args)-n_tuh} small datasets)")
    print("Starting packing...\n")

    start = time.time()
    total_shards = total_segments = total_errors = 0

    with mp.Pool(N_WORKERS) as pool:
        for dataset, shards, segs, errors in pool.imap_unordered(pack_group, args):
            total_shards += shards
            total_segments += segs
            total_errors += errors
            print(f"  {dataset}: {shards} shards, {segs:,} segments, {errors} errors")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f}m")
    print(f"  Total shards:   {total_shards:,}")
    print(f"  Total segments: {total_segments:,}")
    print(f"  Errors:         {total_errors}")

    # Write manifest for the packed dataset
    print("\nBuilding packed manifest...")
    packed_keys = list_npy_keys(s3, DST_PREFIX)
    manifest = {}
    for key in packed_keys:
        rel = key[len(DST_PREFIX):].lstrip("/")
        # Read header only to get segment count
        resp = s3.get_object(Bucket=BUCKET, Key=key, Range="bytes=0-511")
        data = resp["Body"].read()
        import ast, struct
        major = data[6]
        if major == 1:
            hl = struct.unpack("<H", data[8:10])[0]
            hdr = data[10:10+hl].decode("latin1")
        else:
            hl = struct.unpack("<I", data[8:12])[0]
            hdr = data[12:12+hl].decode("latin1")
        shape = ast.literal_eval(hdr.strip())["shape"]
        manifest[rel] = shape[0]

    manifest_body = json.dumps(manifest, indent=2).encode()
    manifest_key = f"{DST_PREFIX}/manifest.json"
    s3.put_object(Bucket=BUCKET, Key=manifest_key, Body=manifest_body, ContentType="application/json")
    print(f"Manifest uploaded: {manifest_key} ({len(manifest):,} shards, {sum(manifest.values()):,} segments)")

    summary = {
        "src_files": len(src_keys), "dst_shards": total_shards,
        "total_segments": total_segments, "errors": total_errors,
        "elapsed_min": elapsed / 60,
    }
    (MODEL_DIR / "pack_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
