# SJJI — SSL-Enhanced TransformEEG for Parkinson's Disease Detection

Self-supervised pretraining of a TransformEEG encoder on the TUH EEG corpus (~8M segments), followed by linear probe evaluation on four labeled Parkinson's disease EEG datasets.

## Research Question

Can self-supervised pretraining (VICReg) improve cross-dataset generalizability of a TransformEEG encoder for Parkinson's disease detection?

## Current Status

**Pretraining is running.** VICReg on 400k TUH segments, spot g5.4xlarge (A10G 24GB), ~21 min/epoch. Loss trending down. Next: linear probe evaluation → cross-dataset eval → paper.

| Phase | Status |
|---|---|
| TUH preprocessing (75k EDF → .npy) | ✅ Done |
| Shard packing (8M segs → 5,442 shards) | ✅ Done |
| Supervised baseline (TransformEEG) | ✅ Done — **53.8% balanced accuracy** |
| VICReg SSL pretraining | 🔄 Running |
| Linear probe evaluation | ⏳ Pending |
| Cross-dataset generalization eval | ⏳ Pending |
| Paper | ⏳ Pending |

## Datasets

**Pretraining (unlabeled):**
- TUH-EEG Corpus — Temple University Hospital (~75k recordings, requires access request)

**Evaluation (labeled PD vs HC):**
- [ds004148](https://openneuro.org/datasets/ds004148) — EEG test-retest
- [ds002778](https://openneuro.org/datasets/ds002778) — UC San Diego Parkinson's
- [ds003490](https://openneuro.org/datasets/ds003490) — EEG 3-Stim
- [ds004584](https://openneuro.org/datasets/ds004584) — Parkinson's EEG dataset

## Method

1. **Preprocess** — MNE pipeline: bandpass 0.5–45Hz, resample 128Hz, 4s windows, 64 channels
2. **Pack** — Small .npy files → large shards for efficient S3 training
3. **Pretrain** — VICReg SSL on 400k TUH segments (encoder learns EEG structure, no labels)
4. **Evaluate** — Freeze encoder, train linear probe on labeled PD data (N-LNSO cross-validation)
5. **Cross-dataset eval** — Train on 3 datasets, test on held-out 4th

## Architecture

TransformEEG (Del Pup et al. 2025): depthwise Conv1D tokenizer → 2-layer Transformer encoder → adaptive pooling → 244-dim embeddings.

SSL head: 2-layer MLP projector (244 → 244 → 128), VICReg loss (invariance + variance + covariance terms).

## Infrastructure

All training runs on AWS SageMaker (spot g5.4xlarge). Data lives in S3 (`sagemaker-us-east-2-506145782110`). Submit jobs via:

```bash
source venv/bin/activate
python sagemaker_submit.py --job pretrain   # VICReg pretraining
python sagemaker_submit.py --job ssl_pilot  # full eval pipeline
```

See `skills/sagemaker-ml-training/SKILL.md` for gotchas, quota info, and lessons learned.

## Repo Structure

```
experiments/
  ssl_pilot.py          # main: pretrain → linear probe → cross-dataset eval
  baseline.py           # supervised TransformEEG baseline
scripts/
  sm_pack_shards.py     # pack .npy files into large shards (SageMaker job)
  sm_preprocess.py      # TUH preprocessing (SageMaker job)
  build_subsample_manifest.py  # sample 400k segments from full corpus
src/
  model.py              # TransformEEG encoder + EEGClassifier
  pretrain.py           # VICReg training loop, FileGroupedSampler, checkpointing
  finetune.py           # linear probe training + eval
sagemaker_submit.py     # submit/configure SageMaker training jobs
requirements.txt
```

## Setup (local dev)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Local runs use `data/processed_unified/` by default. SageMaker mounts use `SM_CHANNEL_*` env vars.

## Key Results So Far

| Experiment | Balanced Accuracy |
|---|---|
| Supervised baseline (TransformEEG) | 53.8% |
| SSL linear probe (VICReg) | 🔄 running |
| Cross-dataset SSL | 🔄 running |

## References

- TransformEEG: Del Pup et al. (2025), Neurocomputing — [GitHub](https://github.com/MedMaxLab/TransformEEG)
- VICReg: Bardes et al. (2022), Meta AI — [Paper](https://arxiv.org/abs/2105.04906)
- TUH EEG Corpus: Obeid & Picone (2016)
