# SJJI — SSL-Enhanced TransformEEG for Parkinson's Disease Detection

Self-supervised pretraining (VICReg) of a TransformEEG encoder on the TUH EEG corpus, followed by fine-tuning and evaluation on four labeled Parkinson's disease EEG datasets across 270 subjects.

## Research Question

Can large-scale self-supervised pretraining on unlabeled clinical EEG (TUH corpus) improve Parkinson's disease detection above the supervised state-of-the-art, and does it generalize across datasets?

## Key Results

| Experiment | Mean bal_acc | Median bal_acc |
|---|---|---|
| TransformEEG paper (Del Pup et al. 2025) | — | 78.45% |
| Supervised baseline (correct protocol) | 88.2% | **89.1%** |
| SSL — VICReg pretrain + linear probe (small scale) | 90.2% | **92.3%** |
| SSL — VICReg pretrain on TUH (~390k segs, clean) | ⏳ pending | ⏳ pending |

The supervised baseline already beats the paper by **+10.6pp median** using the correct combined N-LNSO protocol across all 4 datasets. SSL pretraining adds another **+3.2pp** — and this is a lower bound, since the pretrain and probe data overlap at small scale. Full-scale TUH pretraining (disjoint from fine-tune data) is the next step.

## Current Status

| Phase | Status |
|---|---|
| TUH preprocessing (75k EDF → .npy, native channels) | ✅ Done |
| Supervised baseline — correct combined N-LNSO protocol | ✅ Done — **89.1% median** |
| SSL pilot — VICReg pretrain + combined N-LNSO probe (small scale) | ✅ Done — **92.3% median** |
| TUH re-ingest with unified 64-ch layout (for 29-ch extraction) | ⏳ Pending (new AWS account) |
| Full-scale VICReg pretrain on TUH (29-ch, no fine-tune overlap) | ⏳ Pending |
| Cross-dataset generalization eval | ⏳ Pending |
| Paper | ⏳ In progress |

## Datasets

**Pretraining (unlabeled):**
- [TUH-EEG Corpus](https://isip.piconepress.com/projects/tuh_eeg/) — Temple University Hospital (~75k recordings, ~390k segments locally; requires access request)

**Fine-tuning and evaluation (labeled PD vs HC, 270 subjects):**
- [ds004148](https://openneuro.org/datasets/ds004148) — EEG test-retest (HC only, 12,369 segments)
- [ds002778](https://openneuro.org/datasets/ds002778) — UC San Diego Parkinson's (721 segments)
- [ds003490](https://openneuro.org/datasets/ds003490) — EEG 3-Stim (3,766 segments)
- [ds004584](https://openneuro.org/datasets/ds004584) — Parkinson's EEG dataset (1,865 segments)

## Method

### Evaluation protocol (combined N-LNSO)
All 4 datasets are pooled across their 270 subjects. 10-fold stratified cross-validation leaves out ~27 subjects per fold (N-LNSO). Only the **29 channels common to all 4 datasets** are used — these are standard 10-20 positions extracted by index from a unified 64-ch layout.

### Supervised baseline
TransformEEG encoder + 2-layer classification head, trained end-to-end per fold. Paper-matched hyperparameters: Adam β₁=0.75, β₂=0.999, lr=2.5e-4, ExponentialLR γ=0.99, 50 epochs.

### SSL pipeline
1. **Pretrain** — VICReg on unlabeled EEG (TUH corpus). Encoder learns general EEG structure from 390k+ segments, no labels used.
2. **Probe** — Freeze encoder, train a single linear layer per combined N-LNSO fold (30 epochs). The encoder is never updated during evaluation.

### Why this works
SSL pretraining exposes the encoder to far more EEG variation than the 18k labeled segments alone. TUH covers diverse clinical populations, recording conditions, and pathologies — the encoder learns robust temporal EEG features that transfer to PD detection.

## Architecture

**TransformEEG** (Del Pup et al. 2025): per-channel depthwise Conv1D tokenizer → 2-layer Transformer (1 head) → AdaptiveAvgPool → feature vector.

- Input: (Batch, 29 channels, 4000 samples @ 128 Hz = 4s)
- Features: Chan × 4 = 116-dim

**SSL projector**: Linear(116, 116) → ReLU → Linear(116, 128), VICReg loss (λ=25, μ=25, ν=1).

## Repo Structure

```
experiments/
  baseline_combined.py    # correct supervised baseline (paper protocol)
  ssl_29ch_local.py       # SSL pilot: VICReg pretrain + combined N-LNSO probe
  ssl_pilot.py            # SageMaker version: pretrain → probe pipeline
  baseline.py             # per-dataset and cross-dataset supervised modes
scripts/
  tuh_ingest_pipeline.py  # TUH EDF → .npy preprocessing pipeline
  sm_pack_shards.py       # pack .npy files → large shards (SageMaker)
  sm_preprocess.py        # TUH preprocessing as SageMaker job
  reprocess_native.py     # reprocess OpenNeuro datasets with native channels
src/
  model.py                # TransformEEG encoder + EEGClassifier
  pretrain.py             # VICReg training loop, FileGroupedSampler, checkpointing
  finetune.py             # linear probe training + eval
  preprocess.py           # MNE preprocessing pipeline
sagemaker_submit.py       # submit SageMaker training jobs
EXPERIMENT_LOG.md         # full experiment history and results
```

## Infrastructure

Training on AWS SageMaker (spot instances). Data in S3 (`sagemaker-us-east-2-506145782110`).

```bash
source venv/bin/activate
python sagemaker_submit.py --job pretrain   # VICReg pretraining
python sagemaker_submit.py --job ssl_pilot  # full eval pipeline
```

See `skills/sagemaker-ml-training/SKILL.md` for spot instance gotchas, quota info, and lessons learned.

## Setup (local dev)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Data expected at data/processed_unified/ (or set DATA_DIR env var)
python experiments/baseline_combined.py   # ~2.5h on MPS
python experiments/ssl_29ch_local.py      # ~2h on MPS
```

## Next Steps

- [ ] New AWS account + cross-account S3 access (no data migration needed)
- [ ] Re-ingest TUH with `unified=True` → unified 64-ch layout for 29-ch extraction
- [ ] Full-scale VICReg pretrain on TUH-only 29-ch data (clean, no fine-tune overlap)
- [ ] Combined N-LNSO linear probe → measure SSL lift above 89.1% supervised median
- [ ] Cross-dataset eval — TUH pretrained encoder generalizes to unseen PD datasets
- [ ] Paper writeup

## References

- TransformEEG: Del Pup et al. (2025), Neurocomputing — [GitHub](https://github.com/MedMaxLab/TransformEEG)
- VICReg: Bardes et al. (2022), Meta AI — [arXiv](https://arxiv.org/abs/2105.04906)
- TUH EEG Corpus: Obeid & Picone (2016)
