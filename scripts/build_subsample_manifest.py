"""
Build a subsampled manifest for SSL pretraining.

Keeps ALL segments from small labeled datasets (ds002778, ds003490, ds004148, ds004584)
and randomly samples TUH shards up to a target total segment count.

Output: data/processed_unified_packed/manifest_400k.json
"""

import json
import random
import argparse
import boto3

BUCKET = "sagemaker-us-east-2-506145782110"
MANIFEST_KEY = "data/processed_unified_packed/manifest.json"
OUT_KEY = "data/processed_unified_packed/manifest_400k.json"
SMALL_DATASETS = {"ds002778", "ds003490", "ds004148", "ds004584"}
TARGET_SEGS = 400_000
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=TARGET_SEGS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name="us-east-2")

    print(f"Loading manifest from s3://{BUCKET}/{MANIFEST_KEY}")
    obj = s3.get_object(Bucket=BUCKET, Key=MANIFEST_KEY)
    manifest = json.loads(obj["Body"].read())
    print(f"  {len(manifest):,} shards, {sum(manifest.values()):,} total segments")

    # Always keep all small labeled dataset shards
    keep = {}
    tuh_shards = {}
    for path, count in manifest.items():
        ds = path.split("/")[0]
        if ds in SMALL_DATASETS:
            keep[path] = count
        else:
            tuh_shards[path] = count

    small_segs = sum(keep.values())
    print(f"  Small datasets: {len(keep)} shards, {small_segs:,} segs (always kept)")

    remaining = args.target - small_segs
    if remaining <= 0:
        print("  Small datasets already exceed target — using all small + 0 TUH")
    else:
        # Randomly sample TUH shards until we hit the target
        rng = random.Random(args.seed)
        tuh_items = list(tuh_shards.items())
        rng.shuffle(tuh_items)
        tuh_kept_segs = 0
        for path, count in tuh_items:
            if tuh_kept_segs >= remaining:
                break
            keep[path] = count
            tuh_kept_segs += count
        print(f"  TUH sampled: {sum(1 for p in keep if p.startswith('tuh'))} shards, {tuh_kept_segs:,} segs")

    total = sum(keep.values())
    print(f"\nSubsample: {len(keep):,} shards, {total:,} segments")
    print(f"  vs full: {sum(manifest.values()):,} segments ({100*total/sum(manifest.values()):.1f}%)")

    out = json.dumps(keep, indent=2)
    s3.put_object(Bucket=BUCKET, Key=OUT_KEY, Body=out.encode())
    print(f"\nUploaded to s3://{BUCKET}/{OUT_KEY}")


if __name__ == "__main__":
    main()
