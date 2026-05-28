#!/usr/bin/env bash
# TUH EEG → S3 pipeline with auto-retry and disk management.
# Pulls subjects in batches, uploads each batch to S3, deletes locally, repeats.
#
# Current state: subjects 000-007 are on disk (and uploading to S3).
# This script handles 008-150.

set -euo pipefail

SSH_KEY="$HOME/.ssh/tuh_eeg_ed25519"
NEDC_USER="nedc-tuh-eeg"
NEDC_HOST="www.isip.piconepress.com"
NEDC_PATH="data/tuh_eeg/tuh_eeg/v2.0.1/edf"
LOCAL_DIR="$HOME/Projects/sjji-eeg/data/raw/tuh_eeg/tuh_eeg/v2.0.1/edf"
S3_DST="s3://sagemaker-us-east-2-506145782110/data/raw/tuh_eeg/v2.0.1/edf"
REGION="us-east-2"
LOG="$HOME/Projects/sjji-eeg/logs/tuh_pipeline.log"
BATCH_SIZE=15       # subjects per batch (~72GB at ~4.8GB/subject)
MAX_RETRIES=20      # per-subject retry attempts before giving up
RETRY_WAIT=60       # seconds between retries on connection drop

mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

rsync_subject() {
    local subj="$1"
    local attempt=0
    while (( attempt < MAX_RETRIES )); do
        log "  rsync subject $subj (attempt $((attempt+1)))"
        rsync -auvxL \
            -e "ssh -i $SSH_KEY -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=5" \
            "${NEDC_USER}@${NEDC_HOST}:${NEDC_PATH}/${subj}/" \
            "${LOCAL_DIR}/${subj}/" \
            2>>"$LOG" && return 0
        attempt=$((attempt+1))
        log "  connection dropped, retrying in ${RETRY_WAIT}s..."
        sleep "$RETRY_WAIT"
    done
    log "ERROR: subject $subj failed after $MAX_RETRIES attempts"
    return 1
}

upload_and_clean() {
    local subjects=("$@")
    log "Uploading ${#subjects[@]} subjects to S3..."
    aws s3 sync "$LOCAL_DIR/" "$S3_DST/" \
        --region "$REGION" \
        --storage-class STANDARD_IA \
        --exclude ".*" \
        --no-progress \
        2>>"$LOG"
    log "Upload done. Cleaning local copies..."
    for subj in "${subjects[@]}"; do
        local size
        size=$(du -sh "${LOCAL_DIR}/${subj}" 2>/dev/null | cut -f1)
        rm -rf "${LOCAL_DIR:?}/${subj}"
        # Leave empty placeholder dir so rsync skips it on resume
        mkdir -p "${LOCAL_DIR}/${subj}"
        log "  deleted local $subj ($size)"
    done
}

# ── Main ──────────────────────────────────────────────────────────────────── #

# Subjects 000-007 already on disk (being uploaded by the existing S3 sync).
# This script handles 008-150.
START=9
END=150

log "=== TUH pipeline started: subjects $(printf '%03d' $START)–$(printf '%03d' $END) ==="
log "Batch size: $BATCH_SIZE subjects | Log: $LOG"

batch=()
for (( i=START; i<=END; i++ )); do
    subj=$(printf '%03d' "$i")

    # Skip if already has real data (resumable)
    local_size=$(du -s "${LOCAL_DIR}/${subj}" 2>/dev/null | cut -f1 || echo 0)
    if (( local_size > 1000 )); then
        log "Subject $subj: already has data locally, adding to current batch"
        batch+=("$subj")
    else
        rsync_subject "$subj"
        batch+=("$subj")
    fi

    # When batch is full, upload and clean
    if (( ${#batch[@]} >= BATCH_SIZE )); then
        log "Batch full (${batch[*]}). Uploading..."
        upload_and_clean "${batch[@]}"
        batch=()
        df -h / | grep -v Filesystem | tee -a "$LOG"
    fi
done

# Final partial batch
if (( ${#batch[@]} > 0 )); then
    log "Final batch (${batch[*]}). Uploading..."
    upload_and_clean "${batch[@]}"
fi

log "=== Pipeline complete. Full corpus in S3. ==="
