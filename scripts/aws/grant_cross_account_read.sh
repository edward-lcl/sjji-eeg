#!/usr/bin/env bash
# Grant the NEW account read access to the OLD account's data bucket.
#
# Run this with creds for the OLD account (the one that owns the data).
#   NEW_ACCOUNT=123456789012 ./scripts/aws/grant_cross_account_read.sh
#
# It MERGES a cross-account read statement into the old bucket's existing
# policy (which already has an "Enforce HTTPS" deny) — it does NOT overwrite.
# The new account's own IAM role still has to grant matching permissions
# (setup_new_account.sh does that); this is only the resource-side half.
set -euo pipefail

OLD_BUCKET="${OLD_BUCKET:-sagemaker-us-east-2-506145782110}"
REGION="${AWS_REGION:-us-east-2}"
: "${NEW_ACCOUNT:?Set NEW_ACCOUNT=<new 12-digit account id>}"

echo "==> Reading existing bucket policy on ${OLD_BUCKET}"
EXISTING_FILE="$(mktemp)"; MERGED_FILE="$(mktemp)"
trap 'rm -f "$EXISTING_FILE" "$MERGED_FILE"' EXIT
# Pass the existing policy via a FILE (not stdin) — `python3 -` already consumes
# stdin for the program, so a here-string would be swallowed and the existing
# statements (e.g. Enforce-HTTPS) silently dropped.
aws s3api get-bucket-policy --bucket "$OLD_BUCKET" --region "$REGION" \
  --query Policy --output text > "$EXISTING_FILE" 2>/dev/null || : > "$EXISTING_FILE"

python3 - "$NEW_ACCOUNT" "$OLD_BUCKET" "$EXISTING_FILE" > "$MERGED_FILE" <<'PY'
import json, sys
new_account, bucket, path = sys.argv[1], sys.argv[2], sys.argv[3]
existing = open(path).read().strip()
policy = json.loads(existing) if existing else {"Version": "2012-10-17", "Statement": []}
policy.setdefault("Statement", [])

sid = "CrossAccountReadSJJI"
policy["Statement"] = [s for s in policy["Statement"] if s.get("Sid") != sid]
policy["Statement"].append({
    "Sid": sid,
    "Effect": "Allow",
    "Principal": {"AWS": f"arn:aws:iam::{new_account}:root"},
    "Action": ["s3:GetObject", "s3:ListBucket", "s3:GetBucketLocation"],
    "Resource": [
        f"arn:aws:s3:::{bucket}",
        f"arn:aws:s3:::{bucket}/data/*",
    ],
})
print(json.dumps(policy))
PY

echo "==> New policy to be applied:"
python3 -m json.tool "$MERGED_FILE"

read -r -p "Apply this policy to ${OLD_BUCKET}? [y/N] " ans
[ "$ans" = "y" ] || { echo "Aborted."; exit 1; }

aws s3api put-bucket-policy --bucket "$OLD_BUCKET" --region "$REGION" \
  --policy "file://${MERGED_FILE}"
echo "==> Done. Account ${NEW_ACCOUNT} can now read s3://${OLD_BUCKET}/data/*"
