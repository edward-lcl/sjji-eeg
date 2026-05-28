#!/usr/bin/env bash
# Upload EEG datasets to S3 for SageMaker training.
# Run this once — all subsequent training jobs pull from S3.
#
# Usage:
#   ./scripts/upload_to_s3.sh                    # upload everything
#   ./scripts/upload_to_s3.sh --tuh-only         # just TUH raw data
#   ./scripts/upload_to_s3.sh --processed-only   # just processed_unified
#
# For the full TUH corpus (~1.2TB) from NEDC rsync, see the EC2 path below.

set -euo pipefail

BUCKET="${S3_BUCKET:-sagemaker-us-east-2-506145782110}"
REGION="${AWS_REGION:-us-east-2}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"

TUH_ONLY=false
PROCESSED_ONLY=false

for arg in "$@"; do
    case $arg in
        --tuh-only)       TUH_ONLY=true ;;
        --processed-only) PROCESSED_ONLY=true ;;
    esac
done

echo "Bucket:  s3://$BUCKET"
echo "Region:  $REGION"
echo "Source:  $DATA_DIR"
echo ""

upload() {
    local src="$1"
    local dst="$2"
    echo ">>> Uploading: $src"
    echo "          to: s3://$BUCKET/$dst/"
    aws s3 sync "$src" "s3://$BUCKET/$dst/" \
        --region "$REGION" \
        --storage-class STANDARD_IA \
        --exclude "*.pyc" --exclude "__pycache__/*" \
        --no-progress
    echo "    Done: $src"
    echo ""
}

# ------------------------------------------------------------------ #
# Local data (on this Mac)                                            #
# ------------------------------------------------------------------ #

if ! $PROCESSED_ONLY; then
    # TUH — subjects 4-40 (local partial copy)
    if [ -d "$DATA_DIR/raw/tuh_eeg" ]; then
        upload "$DATA_DIR/raw/tuh_eeg" "data/raw/tuh_eeg"
    else
        echo "[skip] No local tuh_eeg found at $DATA_DIR/raw/tuh_eeg"
    fi
fi

if ! $TUH_ONLY; then
    # OpenNeuro labeled PD datasets
    for ds in ds002778 ds003490 ds004148 ds004584; do
        if [ -d "$DATA_DIR/raw/$ds" ]; then
            upload "$DATA_DIR/raw/$ds" "data/raw/$ds"
        else
            echo "[skip] $ds not found locally"
        fi
    done

    # Pre-processed unified segments (saves re-processing time on cloud)
    if [ -d "$DATA_DIR/processed_unified" ]; then
        upload "$DATA_DIR/processed_unified" "data/processed_unified"
    fi
fi

echo "=============================="
echo "Upload complete."
echo ""
echo "Verify with:"
echo "  aws s3 ls s3://$BUCKET/data/ --recursive --human-readable | head -30"
echo ""
echo "Next: python sagemaker_submit.py --job ssl_pilot --dry-run"

# ------------------------------------------------------------------ #
# FULL TUH CORPUS via EC2 (do not run locally — 1.2TB)               #
# ------------------------------------------------------------------ #
# To pull the full TUH corpus directly from NEDC rsync to S3 without
# routing through your Mac:
#
# 1. Launch a c5n.xlarge or m5.xlarge in us-east-2 (same region as S3)
# 2. Attach 1.5TB EBS volume or use instance store
# 3. Install awscli, configure IAM role or creds
# 4. rsync from NEDC:
#      rsync -a --progress nedc-tuh-eeg@www.isip.piconepress.com:/data/eeg/tuh_eeg/ ./tuh_eeg/
# 5. aws s3 sync ./tuh_eeg/ s3://<bucket>/data/raw/tuh_eeg/ --storage-class STANDARD_IA
# 6. Terminate instance — intra-region S3 transfer is free, no egress fees
#
# Estimated time: ~4-6h for 1.2TB at typical EC2 network speeds.
# Estimated cost: ~$0.20/h EC2 + $30 S3 storage (STANDARD_IA) ongoing.
