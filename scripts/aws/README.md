# New AWS account setup — runbook

Migrating SJJI SageMaker compute to a fresh AWS account (the old one ran out of
free credits). **Data stays in the old bucket; compute moves to the new account
via cross-account read.** Same region (`us-east-2`), so there's no data-transfer
cost — only the old account's storage + tiny request charges.

```
  NEW account (compute, fresh credits)        OLD account 506145782110 (data)
  ┌─────────────────────────────┐             ┌──────────────────────────────┐
  │ SageMaker training job       │  reads  →   │ s3://…-506145782110/data/    │
  │ role: SJJI-SageMaker-Exec…   │ ─────────►  │   raw/tuh_eeg  (1.75 TB)     │
  │ bucket: …-<NEW_ACCT> (output)│             │   processed_unified*         │
  └─────────────────────────────┘             └──────────────────────────────┘
```

## Order of operations

### 0. (Do first — it's the long pole) Request GPU quota in the NEW account
New accounts start at **0** GPU instances. The increase can take a day+, so fire
it before anything else. Console → **Service Quotas → Amazon SageMaker** →
request increase for:
- `ml.g5.4xlarge for training job usage` → **1** (pretrain/pilot)
- optionally `ml.r5.4xlarge for processing/training job usage` → **1** (CPU re-ingest)

### 1. Get into the new account locally
Create an admin IAM user (or SSO) in the new account console, then configure a
named profile. Console steps:
- IAM → Users → Create user `sjji-cli` → attach **AdministratorAccess** →
  Security credentials → **Create access key** (CLI use case).
- Locally: `aws configure --profile sjji-new` (paste the key, region `us-east-2`).
- Verify: `aws sts get-caller-identity --profile sjji-new`

### 2. Provision the new account
```bash
AWS_PROFILE=sjji-new ./scripts/aws/setup_new_account.sh
```
Creates the output bucket + `SJJI-SageMaker-ExecutionRole`. Copy the three
printed values into `.env` (template: `scripts/aws/env.example`).

### 3. Open cross-account read on the old data bucket
Run with the OLD account's creds (the default profile is currently the old root):
```bash
NEW_ACCOUNT=<new 12-digit id> ./scripts/aws/grant_cross_account_read.sh
```
Merges a read grant into the old bucket policy (keeps the existing Enforce-HTTPS
statement). Both halves — this grant and the role policy from step 2 — must be in
place before a job can read old data.

### 4. Smoke test the cross-account read
```bash
AWS_PROFILE=sjji-new aws s3 ls s3://sagemaker-us-east-2-506145782110/data/raw/tuh_eeg/ | head
```

### 5. Launch
```bash
set -a; source .env; set +a
python sagemaker_submit.py --job pretrain --dry-run    # sanity-check config
python sagemaker_submit.py --job pretrain              # for real (once quota is live)
```

## Cost note (decide consciously)
The old account holds **>10 TB** (raw 1.75 TB + `processed_unified_packed` 8.2 TB
+ others). With its credits gone that's ~**$250–300/month** of real S3 storage
billed to whatever card is on the old account. Cross-account access keeps that
bill alive. If you want to cut it, the `processed_unified_packed` shards are the
big item and are likely superseded by the upcoming unified re-ingest — delete
them from the old account once the re-ingest lands (separate, deliberate step;
not part of these scripts).

## Why a new account at all
Old account exhausted its free credits; the new one has fresh credits, so GPU
compute runs there for free. Data is left in place to avoid copying 10 TB.
