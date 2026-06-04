"""Submit a SageMaker training job for the SJJI EEG project.

Data lives in S3 — training jobs pull from there, no local data needed.
Spot instances save 60-70%; safe if the training script supports checkpoint resume.

Usage:
    python sagemaker_submit.py --job pretrain              # TUH-scale SSL pretraining
    python sagemaker_submit.py --job finetune              # supervised fine-tune on PD datasets
    python sagemaker_submit.py --job eval                  # cross-dataset evaluation
    python sagemaker_submit.py --job ssl_pilot --spot      # quick pilot on spot

Setup (.env or environment):
    AWS_REGION=us-east-2
    S3_BUCKET=sagemaker-us-east-2-506145782110
    SM_ROLE_ARN=arn:aws:iam::506145782110:role/AmazonSageMaker-ExecutionRole-20260509T231091

Data layout expected in S3:
    s3://<bucket>/data/raw/tuh_eeg/          ← full TUH corpus (~1.2TB)
    s3://<bucket>/data/raw/ds002778/
    s3://<bucket>/data/raw/ds003490/
    s3://<bucket>/data/raw/ds004148/
    s3://<bucket>/data/raw/ds004584/
    s3://<bucket>/data/processed_unified/    ← optional: pre-processed segments
"""
import argparse
import os
import shutil
import tempfile
import time

import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput


# --------------------------------------------------------------------------- #
# Job configs: entry_point, instance, max hours, data channels                #
# --------------------------------------------------------------------------- #
JOB_CONFIGS = {
    "pretrain": {
        "entry_point": "experiments/ssl_pilot.py",
        "instance":    "ml.g5.4xlarge",   # A10G 24GB VRAM
        "max_hours":   30,
        "spot":        True,               # safe: mid-epoch checkpoints enabled
        "data_channels": ["processed_unified_sub400k"],
        "description": "EEG VICReg SSL pretraining (400k subsample, File mode, spot)",
    },
    "pack": {
        "entry_point": "scripts/sm_pack_shards.py",
        "instance":    "ml.r5.4xlarge",   # 16 vCPU 128GB — proven quota; no GPU needed
        "max_hours":   6,
        "spot":        False,
        "data_channels": ["processed_unified"],
        "description": "Pack small .npy files into 1024-seg shards → processed_unified_packed",
    },
    "finetune": {
        "entry_point": "experiments/ssl_pilot.py",  # update when dedicated script exists
        "instance":    "ml.g5.xlarge",
        "max_hours":   6,
        "data_channels": ["labeled_pd", "processed_unified"],
        "description": "Supervised fine-tune on PD datasets (ds002778/3490/4584)",
    },
    "eval": {
        "entry_point": "experiments/ssl_pilot.py",  # update when dedicated script exists
        "instance":    "ml.g5.xlarge",
        "max_hours":   3,
        "data_channels": ["labeled_pd", "processed_unified"],
        "description": "Cross-dataset evaluation (leave-one-dataset-out)",
    },
    "preprocess": {
        "entry_point": "scripts/sm_preprocess.py",
        "instance":    "ml.r5.4xlarge",   # 16 vCPU, 128GB RAM — approved quota, 8 workers safe
        "max_hours":   12,
        "data_channels": [],               # no channel mount — reads S3 directly via boto3
        "description": "Preprocess all 75k TUH EDF files → .npy segments in S3",
    },
    "ssl_pilot": {
        "entry_point": "experiments/ssl_pilot.py",
        "instance":    "ml.g5.xlarge",
        "max_hours":   4,
        "data_channels": ["processed_unified"],
        "description": "Quick SSL pilot (no TUH, use pre-processed segments only)",
    },
}

# S3 data channel prefixes — must match the S3 layout described above
DATA_CHANNEL_S3_PREFIXES = {
    "tuh_eeg":           "data/raw/tuh_eeg",
    "labeled_pd":        "data/raw/ds",        # prefix-matches ds002778/ds003490/ds004148/ds004584 — excludes tuh_eeg
    "processed_unified":        "data/processed_unified",
    "processed_unified_packed": "data/processed_unified_packed",
    "processed_unified_sub400k": "data/processed_unified_sub400k",
}

# Source code whitelist — keeps the upload small (no data, no __pycache__)
SOURCE_WHITELIST = (
    "src",
    "experiments",
    "scripts",
    "configs",
    "baseline.py",
    "train.py",
    "requirements.txt",
    "pyproject.toml",
    "README.md",
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def stage_source_dir(repo_root: str) -> str:
    """Copy only whitelisted paths to a temp dir. Avoids uploading GBs of data."""
    staging = tempfile.mkdtemp(prefix="sm-eeg-source-")
    for name in SOURCE_WHITELIST:
        src = os.path.join(repo_root, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(staging, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".pytest_cache",
                "*.npz", "*.edf", "*.EDF",  # no raw EEG in source upload
            ))
        else:
            shutil.copy2(src, dst)
    return staging


def build_data_inputs(channels: list[str], bucket: str, job_preset: str = "") -> dict:
    """Build SageMaker TrainingInput objects for each requested data channel."""
    inputs = {}
    for ch in channels:
        prefix = DATA_CHANNEL_S3_PREFIXES.get(ch)
        if prefix is None:
            print(f"  [warn] Unknown data channel '{ch}', skipping.")
            continue
        uri = f"s3://{bucket}/{prefix}/"
        inputs[ch] = TrainingInput(
            s3_data=uri,
            s3_data_type="S3Prefix",
            input_mode="FastFile" if (ch in ("tuh_eeg", "processed_unified") or job_preset == "preprocess") else "File",
        )
        print(f"  data[{ch}]: {uri}")
    return inputs


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job", choices=list(JOB_CONFIGS.keys()), default="ssl_pilot",
                   help="Job preset to run (see JOB_CONFIGS above)")
    p.add_argument("--entry-point", default=None,
                   help="Override entry point script (relative to repo root)")
    p.add_argument("--instance", default=None,
                   help="Override instance type")
    p.add_argument("--max-hours", type=int, default=None,
                   help="Override max runtime hours")
    p.add_argument("--spot", action="store_true", default=None,
                   help="Use spot instances (60-70%% cheaper; script must support checkpoint resume)")
    p.add_argument("--no-wait", action="store_true",
                   help="Submit async; poll with describe-training-job")
    p.add_argument("--job-name", default=None,
                   help="Custom job name (default: sjji-eeg-<timestamp>)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print config without submitting")
    args = p.parse_args()

    cfg = JOB_CONFIGS[args.job]
    entry_point   = args.entry_point or cfg["entry_point"]
    instance      = args.instance    or cfg["instance"]
    max_hours     = args.max_hours   or cfg["max_hours"]
    use_spot      = args.spot if args.spot is not None else cfg.get("spot", False)
    data_channels = cfg["data_channels"]

    region   = os.environ.get("AWS_REGION", "us-east-2")
    bucket   = os.environ.get("S3_BUCKET",  "sagemaker-us-east-2-506145782110")
    role_arn = os.environ.get(
        "SM_ROLE_ARN",
        "arn:aws:iam::506145782110:role/service-role/AmazonSageMaker-ExecutionRole-20260509T231091",
    )

    job_name = args.job_name or f"sjji-eeg-{args.job.replace('_', '-')}-{int(time.time())}"

    # PyTorch 2.4 DLC (CUDA 12.4). Works for SimCLR + MNE post-processing.
    # Upgrade: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
    image_uri = (
        f"763104351884.dkr.ecr.{region}.amazonaws.com/"
        f"pytorch-training:2.4.0-gpu-py311-cu124-ubuntu22.04-sagemaker"
    )

    repo_root   = os.path.dirname(os.path.abspath(__file__))
    staging_dir = stage_source_dir(repo_root)

    print(f"\n{'='*60}")
    print(f"  Job:      {job_name}")
    print(f"  Preset:   {args.job} — {cfg['description']}")
    print(f"  Entry:    {entry_point}")
    print(f"  Instance: {instance}{' (spot)' if use_spot else ' (on-demand)'}")
    print(f"  Max run:  {max_hours}h")
    print(f"  Source:   {staging_dir}")
    print(f"  Region:   {region}  Bucket: {bucket}")

    data_inputs = build_data_inputs(data_channels, bucket, job_preset=args.job)

    if args.dry_run:
        print("\n[dry-run] Not submitting.")
        return

    sess = sagemaker.Session(boto_session=boto3.Session(region_name=region))

    metric_definitions = [
        {"Name": "train:loss",  "Regex": r"Epoch \d+: loss=(\S+)"},
        {"Name": "train:epoch", "Regex": r"Epoch (\d+)/"},
    ]

    estimator = Estimator(
        image_uri=image_uri,
        metric_definitions=metric_definitions,
        role=role_arn,
        instance_type=instance,
        instance_count=1,
        volume_size=150 if args.job == "preprocess" else (200 if instance == "ml.g5.xlarge" else 600),   # g5.xlarge local storage cap is 250GB; larger instances get 600GB
        max_run=max_hours * 3600,
        use_spot_instances=use_spot,
        max_wait=(max_hours * 3600 + 14400) if use_spot else None,  # 4h buffer for spot interruptions
        checkpoint_s3_uri=f"s3://{bucket}/checkpoints/{job_name}/" if use_spot else None,
        sagemaker_session=sess,
        output_path=f"s3://{bucket}/runs/{job_name}/output/",
        base_job_name="sjji-eeg",
        entry_point=entry_point,
        source_dir=staging_dir,
        environment={
            "SM_JOB_NAME":   job_name,
            "S3_BUCKET":     bucket,
            "DATA_CHANNELS": ",".join(data_channels),
            # S3 coords for preprocess job (boto3 direct, no channel mount)
            "S3_RAW_PREFIX": "data/raw/tuh_eeg/v2.0.1/edf",
            "S3_OUT_PREFIX": "data/processed_unified/tuh_eeg",
            # Point MNE/cache dirs at instance scratch
            "MNE_DATA":      "/tmp/mne",
            "RESULTS_DIR":   "/opt/ml/model",
        },
        tags=[
            {"Key": "project", "Value": "sjji-eeg"},
            {"Key": "job-preset", "Value": args.job},
        ],
    )

    print(f"{'='*60}\n")
    estimator.fit(
        inputs=data_inputs if data_inputs else None,
        job_name=job_name,
        wait=not args.no_wait,
        logs="All" if not args.no_wait else None,
    )

    if args.no_wait:
        print(f"\nSubmitted async. Poll:")
        print(f"  aws sagemaker describe-training-job --training-job-name {job_name} --region {region}")
    else:
        print(f"\nDone. Pull results:")
        print(f"  aws s3 sync s3://{bucket}/runs/{job_name}/output/ ./outputs/{job_name}/")


if __name__ == "__main__":
    main()
