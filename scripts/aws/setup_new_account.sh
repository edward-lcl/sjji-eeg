#!/usr/bin/env bash
# Set up the NEW AWS account for SJJI SageMaker runs.
#
# Creates, in the new account:
#   - an S3 bucket for outputs/checkpoints/processed data
#   - a SageMaker execution role with:
#       * AmazonSageMakerFullAccess
#       * read access to the OLD account's data bucket (cross-account)
#       * full access to the new bucket
#
# Run this with a profile that points at the NEW account (admin creds):
#   AWS_PROFILE=sjji-new ./scripts/aws/setup_new_account.sh
#
# Prereq: the OLD bucket must already allow this account — run
#   grant_cross_account_read.sh from the OLD account first (or after; order
#   doesn't matter, both sides must be in place before a job reads old data).
set -euo pipefail

PROFILE="${AWS_PROFILE:-sjji-new}"
REGION="${AWS_REGION:-us-east-2}"
OLD_BUCKET="${OLD_BUCKET:-sagemaker-us-east-2-506145782110}"
ROLE_NAME="${ROLE_NAME:-SJJI-SageMaker-ExecutionRole}"

aws() { command aws --profile "$PROFILE" --region "$REGION" "$@"; }

echo "==> Verifying new-account identity"
NEW_ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
echo "    New account: $NEW_ACCOUNT"
NEW_BUCKET="${NEW_BUCKET:-sagemaker-${REGION}-${NEW_ACCOUNT}}"

echo "==> Creating bucket s3://${NEW_BUCKET} (if absent)"
if aws s3api head-bucket --bucket "$NEW_BUCKET" 2>/dev/null; then
  echo "    exists"
else
  # us-east-1 must NOT pass LocationConstraint; every other region must.
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$NEW_BUCKET"
  else
    aws s3api create-bucket --bucket "$NEW_BUCKET" \
      --create-bucket-configuration "LocationConstraint=${REGION}"
  fi
  aws s3api put-bucket-encryption --bucket "$NEW_BUCKET" \
    --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
  echo "    created"
fi

echo "==> Creating SageMaker execution role ${ROLE_NAME} (if absent)"
TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  echo "    exists"
else
  aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document "$TRUST" \
    --description "SJJI EEG SageMaker training jobs" >/dev/null
  echo "    created"
fi

echo "==> Attaching AmazonSageMakerFullAccess"
aws iam attach-role-policy --role-name "$ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

echo "==> Inline S3 policy (read OLD data bucket + read/write NEW bucket)"
S3_POLICY=$(cat <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadOldDataBucket",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket", "s3:GetBucketLocation"],
      "Resource": [
        "arn:aws:s3:::${OLD_BUCKET}",
        "arn:aws:s3:::${OLD_BUCKET}/data/*"
      ]
    },
    {
      "Sid": "ReadWriteNewBucket",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket", "s3:GetBucketLocation"],
      "Resource": [
        "arn:aws:s3:::${NEW_BUCKET}",
        "arn:aws:s3:::${NEW_BUCKET}/*"
      ]
    }
  ]
}
JSON
)
aws iam put-role-policy --role-name "$ROLE_NAME" \
  --policy-name SJJI-S3-Access --policy-document "$S3_POLICY"

ROLE_ARN="$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text)"

cat <<EOF

================================================================
  New-account setup complete.

  Put these in your .env (see scripts/aws/env.example):
    AWS_REGION=${REGION}
    S3_BUCKET=${NEW_BUCKET}
    SM_ROLE_ARN=${ROLE_ARN}

  Next:
    1. Run grant_cross_account_read.sh from the OLD account so the
       old data bucket trusts account ${NEW_ACCOUNT}.
    2. Confirm g5 quota is approved in the NEW account (Service Quotas
       -> Amazon SageMaker -> "ml.g5.4xlarge for training job usage").
    3. Smoke test cross-account read:
       AWS_PROFILE=${PROFILE} aws s3 ls s3://${OLD_BUCKET}/data/raw/tuh_eeg/ | head
================================================================
EOF
