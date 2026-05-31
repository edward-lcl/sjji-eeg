"""
Build manifest.json for UnlabeledEEGDataset and upload to S3.

Reads only the numpy header (first 512 bytes) of each .npy file via S3 range GET
to extract segment counts — no need to download the full 7.6TB corpus.

Usage:
    python scripts/build_manifest.py
    python scripts/build_manifest.py --prefix data/processed_unified --workers 32
"""

import argparse
import ast
import json
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

BUCKET = os.environ.get("S3_BUCKET", "sagemaker-us-east-2-506145782110")
REGION = os.environ.get("AWS_REGION", "us-east-2")
DEFAULT_PREFIX = "data/processed_unified"
MANIFEST_KEY_SUFFIX = "manifest.json"
HEADER_BYTES = 512


def parse_npy_shape(data: bytes) -> tuple:
    """Return shape tuple from numpy file header bytes."""
    if data[:6] != b"\x93NUMPY":
        raise ValueError("Not a numpy file")
    major = data[6]
    if major == 1:
        header_len = struct.unpack("<H", data[8:10])[0]
        header = data[10 : 10 + header_len].decode("latin1")
    else:
        header_len = struct.unpack("<I", data[8:12])[0]
        header = data[12 : 12 + header_len].decode("latin1")
    d = ast.literal_eval(header.strip())
    return d["shape"]


def get_segment_count(s3, key: str) -> int:
    resp = s3.get_object(Bucket=BUCKET, Key=key, Range=f"bytes=0-{HEADER_BYTES - 1}")
    header_bytes = resp["Body"].read()
    shape = parse_npy_shape(header_bytes)
    return shape[0]


def list_npy_keys(s3, prefix: str) -> list[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".npy"):
                keys.append(obj["Key"])
    return keys


def build_manifest(prefix: str, n_workers: int) -> dict:
    s3 = boto3.client("s3", region_name=REGION)

    print(f"Listing .npy files under s3://{BUCKET}/{prefix}/ ...")
    keys = list_npy_keys(s3, prefix)
    print(f"Found {len(keys):,} files. Reading headers with {n_workers} threads...")

    manifest = {}
    errors = []

    def fetch(key):
        s3_thread = boto3.client("s3", region_name=REGION)
        count = get_segment_count(s3_thread, key)
        rel = key[len(prefix) :].lstrip("/")
        return rel, count

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(fetch, k): k for k in keys}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 1000 == 0:
                print(f"  {done:,}/{len(keys):,} ({100*done/len(keys):.1f}%)")
            try:
                rel, count = fut.result()
                manifest[rel] = count
            except Exception as e:
                errors.append((futures[fut], str(e)))

    if errors:
        print(f"\nWARNING: {len(errors)} files failed:")
        for key, err in errors[:10]:
            print(f"  {key}: {err}")

    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(args.prefix, args.workers)
    total_segments = sum(manifest.values())
    print(f"\nManifest: {len(manifest):,} files, {total_segments:,} segments")

    manifest_key = f"{args.prefix}/{MANIFEST_KEY_SUFFIX}"
    if args.dry_run:
        print(f"[dry-run] Would upload to s3://{BUCKET}/{manifest_key}")
        sample = dict(list(manifest.items())[:3])
        print("Sample:", json.dumps(sample, indent=2))
        return

    s3 = boto3.client("s3", region_name=REGION)
    body = json.dumps(manifest, indent=2).encode()
    s3.put_object(Bucket=BUCKET, Key=manifest_key, Body=body, ContentType="application/json")
    print(f"Uploaded manifest to s3://{BUCKET}/{manifest_key} ({len(body):,} bytes)")


if __name__ == "__main__":
    main()
