#!/usr/bin/env python3
"""
TUH EEG preprocess-and-discard pipeline.

Downloads one bucket of raw EDFs from the TUH server, preprocesses them
to .npy segments (unified 64-ch montage), then deletes the raw files.
Accumulates processed segments in data/processed_unified/tuh_eeg/.

Usage:
    # Run buckets 004 through 040
    python scripts/tuh_ingest_pipeline.py --start 4 --end 40

    # Resume after interruption (already-processed buckets are skipped)
    python scripts/tuh_ingest_pipeline.py --start 4 --end 40

    # Process already-downloaded buckets 000-003 without re-downloading
    python scripts/tuh_ingest_pipeline.py --start 0 --end 3 --no-download
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocess import process_dataset_dir

REMOTE        = "nedc-tuh-eeg@www.isip.piconepress.com"
REMOTE_BASE   = "data/tuh_eeg/tuh_eeg/v2.0.1/edf"
SSH_KEY       = Path.home() / ".ssh/tuh_eeg_ed25519"
RAW_STAGING   = Path("data/raw/tuh_eeg/staging")
PROCESSED_OUT = Path("data/processed_unified/tuh_eeg")


def rsync_bucket(bucket_id: int) -> bool:
    bucket = f"{bucket_id:03d}"
    remote_path = f"{REMOTE}:{REMOTE_BASE}/{bucket}/"
    local_path  = str(RAW_STAGING / bucket) + "/"
    Path(local_path).mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync", "-auvxL",
        "-e", f"ssh -i {SSH_KEY} -o IdentitiesOnly=yes -o StrictHostKeyChecking=no",
        remote_path,
        local_path,
    ]
    print(f"  rsync bucket {bucket} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  rsync FAILED (exit {result.returncode}): {result.stderr[:300]}")
        return False
    return True


def smoke_test_bucket(bucket_id: int) -> bool:
    """Run one EDF through the full preprocessing stack before committing to the bucket.
    Returns False if preprocessing produces nothing — catches channel mismatches early."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.preprocess import process_eeg_file
    raw_dir = RAW_STAGING / f"{bucket_id:03d}"
    edfs = list(raw_dir.glob("**/*.edf"))
    if not edfs:
        print(f"  smoke: no EDFs found in staging/{bucket_id:03d}")
        return False
    probe = edfs[0]
    try:
        segs = process_eeg_file(str(probe), unified=False)
        if segs is None or segs.shape[0] == 0:
            print(f"  smoke FAIL: {probe.name} → 0 segments")
            return False
        print(f"  smoke OK: {probe.name} → {segs.shape} (ch={segs.shape[1]})")
        return True
    except Exception as e:
        print(f"  smoke FAIL: {probe.name} → {e}")
        return False


def preprocess_bucket(bucket_id: int) -> int:
    bucket  = f"{bucket_id:03d}"
    raw_dir = RAW_STAGING / bucket
    out_dir = PROCESSED_OUT / bucket
    if not raw_dir.exists():
        print(f"  raw dir missing for bucket {bucket}")
        return 0
    before = len(list(out_dir.glob("**/*.npy"))) if out_dir.exists() else 0
    process_dataset_dir(str(raw_dir), str(out_dir), unified=False)
    after = len(list(out_dir.glob("**/*.npy")))
    return after - before


def delete_raw_bucket(bucket_id: int):
    raw_dir = RAW_STAGING / f"{bucket_id:03d}"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
        print(f"  deleted raw staging/{bucket_id:03d}")


def bucket_is_done(bucket_id: int) -> bool:
    out_dir = PROCESSED_OUT / f"{bucket_id:03d}"
    return out_dir.exists() and any(out_dir.glob("**/*.npy"))


def disk_free_gb() -> float:
    return shutil.disk_usage("/").free / (1024 ** 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",       type=int,   default=4,    help="First bucket (0-indexed, default 4)")
    parser.add_argument("--end",         type=int,   default=40,   help="Last bucket inclusive (default 40)")
    parser.add_argument("--no-download", action="store_true",      help="Skip rsync; preprocess already-staged buckets")
    parser.add_argument("--min-free-gb", type=float, default=20.0, help="Pause if free disk drops below this GB")
    args = parser.parse_args()

    PROCESSED_OUT.mkdir(parents=True, exist_ok=True)
    RAW_STAGING.mkdir(parents=True, exist_ok=True)

    total_new = 0
    skipped   = 0
    failed    = []

    for bucket_id in range(args.start, args.end + 1):
        bucket = f"{bucket_id:03d}"

        if bucket_is_done(bucket_id):
            n = len(list((PROCESSED_OUT / bucket).glob("**/*.npy")))
            print(f"[{bucket}] already done ({n} segments) — skip")
            skipped += 1
            continue

        free_gb = disk_free_gb()
        if free_gb < args.min_free_gb:
            print(f"\n⚠️  {free_gb:.1f}GB free — stopping before bucket {bucket} to protect disk.")
            break

        print(f"\n[{bucket}] disk free: {free_gb:.1f}GB")

        if not args.no_download:
            rsync_ok = False
            for attempt in range(5):
                if rsync_bucket(bucket_id):
                    rsync_ok = True
                    break
                print(f"  rsync attempt {attempt+1}/5 failed, retrying in 60s...")
                time.sleep(60)
            if not rsync_ok:
                failed.append(bucket)
                continue

        # Pre-flight: verify one file preprocesses correctly before committing
        if not smoke_test_bucket(bucket_id):
            print(f"  ⚠️  SMOKE TEST FAILED for bucket {bucket} — skipping delete, marking failed")
            failed.append(bucket)
            continue

        t0     = time.time()
        n_segs = preprocess_bucket(bucket_id)
        total_new += n_segs
        print(f"  {n_segs} new segment files in {time.time() - t0:.0f}s")

        if not args.no_download:
            if n_segs == 0:
                print(f"  ⚠️  ZERO segments from bucket {bucket} — raw data preserved, marking failed")
                failed.append(bucket)
                continue
            delete_raw_bucket(bucket_id)

        print(f"  disk free after: {disk_free_gb():.1f}GB | cumulative new segments: {total_new}")

    print(f"\n{'='*50}")
    print(f"Finished. Buckets processed: {args.end - args.start + 1 - skipped - len(failed)}")
    print(f"Skipped (already done): {skipped}  |  Failed: {failed or 'none'}")
    print(f"Total new segment files: {total_new}")
    print(f"Output: {PROCESSED_OUT.resolve()}")


if __name__ == "__main__":
    main()
