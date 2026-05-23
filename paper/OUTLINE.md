# Paper Outline
## Self-Supervised EEG Pretraining for Cross-Dataset Parkinson's Disease Detection

**Target venue:** IEEE EMBC 2025 / IEEE JBHI / Neurocomputing  
**Status:** In progress — experiments running

---

## Title (working)

> **"Towards Generalizable Parkinson's Disease Detection from EEG: A Self-Supervised Pretraining Approach"**

Alternative:
> **"Self-Supervised EEG Representation Learning for Cross-Dataset Parkinson's Disease Detection"**

---

## Core Narrative (one paragraph)

Parkinson's disease EEG models fail in clinical deployment because they overfit to the specific patient populations and recording setups they were trained on. Supervised models like TransformEEG achieve strong within-dataset accuracy (78.45% balanced accuracy) but degrade significantly when evaluated across datasets. We ask: does self-supervised pretraining on unlabeled EEG data produce representations that generalize better across patient populations? We pretrain a TransformEEG encoder with SimCLR contrastive learning on a large unlabeled EEG corpus, then fine-tune on the same four labeled Parkinson's datasets used in the original TransformEEG paper. Our key metric is cross-dataset generalization — train on one dataset, test on another — which is the clinically relevant measure that prior work has not prioritized.

---

## Abstract (fill after results)

- Problem: EEG-based PD detection models lack cross-dataset generalizability
- Gap: Prior work uses supervised learning on small labeled datasets; SSL has not been systematically applied to PD-specific EEG with cross-dataset eval
- Method: SimCLR pretraining on [TUH-EEG / OpenNeuro unlabeled data] + fine-tuning on 4 PD datasets, N-LNSO cross-validation
- Result: SSL pretraining improves balanced accuracy by [X]% and cross-dataset generalization by [Y]%
- Significance: Demonstrates SSL as a path toward clinically deployable EEG-based PD detection

---

## 1. Introduction

**Paragraph 1 — Hook: the clinical problem**
- PD affects 10M+ people globally; early detection improves outcomes
- EEG is non-invasive, clinically accessible, shows promise for PD biomarker detection
- Problem: models trained at one site/dataset fail when applied elsewhere

**Paragraph 2 — What exists and why it's not enough**
- Deep learning for PD EEG: CNNs, transformers (TransformEEG achieves 78.45% balanced accuracy)
- All rely on supervised learning with scarce labeled data
- Generalization gap: high within-dataset accuracy, unknown cross-dataset performance
- Root cause: small labeled datasets cause overfitting to dataset-specific artifacts

**Paragraph 3 — The SSL opportunity**
- SSL has driven generalization improvements in NLP (BERT), vision (MAE), and recently biomedical signals (DreaMS for mass spectra, LaBraM for EEG)
- Key insight: large unlabeled EEG corpora exist (TUH-EEG: tens of thousands of recordings) — SSL can exploit them
- Nobody has systematically evaluated SSL pretraining for cross-dataset PD detection

**Paragraph 4 — What we do**
- Pretrain TransformEEG encoder with SimCLR on [unlabeled corpus]
- Fine-tune on 4 labeled PD datasets (same as TransformEEG paper — fair comparison)
- Primary eval: cross-dataset generalization (train on A, test on B)
- Secondary eval: within-dataset balanced accuracy vs TransformEEG baseline

**Paragraph 5 — Contributions**
1. First systematic study of SSL pretraining for cross-dataset PD EEG detection
2. Demonstrate [X]% improvement in cross-dataset generalization vs supervised baseline
3. Open-source pipeline: preprocessing + pretraining + evaluation (github link)

---

## 2. Related Work

**2.1 EEG-Based Parkinson's Detection**
- Traditional ML approaches (feature engineering)
- Deep learning: EEGNet, DeepConvNet, transformers
- TransformEEG (Del Pup et al., 2025): channel-specific tokenization, 78.45% balanced accuracy, 4 labeled datasets, N-LNSO evaluation
- Limitation of all: supervised, small datasets, no cross-dataset eval

**2.2 Self-Supervised Learning for EEG**
- Survey (Weng et al., 2024): SSL is effective for general EEG classification
- SelfEEG library (Del Pup et al., 2024): contrastive learning tools for EEG
- LaBraM, BIOT: large-scale EEG foundation models — general purpose
- Gap: none of the above evaluated for Parkinson's with cross-dataset generalization

**2.3 SSL for Biomedical Signals (positioning)**
- DreaMS (Bushuiev et al., 2025): SSL on mass spectra → state-of-the-art fine-tuning performance
- MAMMAL (Shoshan et al., 2024): multi-modal biomedical foundation model
- Pattern: SSL pretraining on abundant unlabeled data → robust fine-tuning with few labels
- We apply this pattern to EEG for the first time in Parkinson's context

---

## 3. Methods

**3.1 Datasets**

*Unlabeled pretraining:*
- [TUH-EEG Corpus (if access granted)] OR OpenNeuro general EEG recordings
- N recordings, duration, channel counts

*Labeled fine-tuning (same as TransformEEG paper):*

| Dataset | Subjects | PD | HC | Sfreq | Channels |
|---------|----------|----|----|-------|----------|
| ds004148 | [N] | [N] | [N] | [X]Hz | [X] |
| ds002778 | [N] | [N] | [N] | [X]Hz | [X] |
| ds003490 | [N] | [N] | [N] | [X]Hz | [X] |
| ds004584 | [N] | [N] | [N] | [X]Hz | [X] |

**3.2 Preprocessing**
- Bandpass filter: 0.5–40 Hz (Hamming window)
- Resample: 256 Hz
- Channel alignment: 61 channels (padding/selection)
- Segmentation: 4-second epochs
- Normalization: per-channel z-score

**3.3 Model Architecture**
- TransformEEG encoder: depthwise convolutional tokenizer + transformer (4 layers, 4 heads, d=244)
- SSL pretraining: SimCLR with projection head (244→244→128)
- Fine-tuning: frozen/unfrozen encoder + 2-class linear head

**3.4 SSL Pretraining (SimCLR)**
- Augmentations: random crop (70–90%), Gaussian noise (σ=0.05), channel dropout (p=0.3)
- Loss: NT-Xent (temperature=0.5)
- Optimizer: Adam (lr=2.5e-4, β=(0.75, 0.999))
- Scheduler: exponential decay (γ=0.99)
- Early stopping: patience=30, min_delta=1e-4
- Max epochs: 300

**3.5 Fine-Tuning**
- Pretrained encoder + classification head
- Adam (lr=1e-3), 50 epochs per fold
- Evaluation: 10-outer N-LNSO cross-validation (matching TransformEEG protocol)

**3.6 Evaluation Protocol**
- *Within-dataset*: N-LNSO CV, metrics: balanced accuracy, sensitivity, specificity
- *Cross-dataset*: train on dataset A, evaluate on dataset B (leave-one-dataset-out)
- Baseline comparison: TransformEEG supervised (78.45% balanced accuracy)

---

## 4. Results

**Table 1: Within-Dataset Performance (Balanced Accuracy ± std)**

| Model | ds004148 | ds002778 | ds003490 | ds004584 | Mean |
|-------|----------|----------|----------|----------|------|
| TransformEEG (reported) | — | — | — | — | 78.45% |
| Supervised baseline (ours) | [X] | [X] | [X] | [X] | [X] |
| SSL pretrained (ours) | [X] | [X] | [X] | [X] | [X] |
| **Δ SSL vs supervised** | [X] | [X] | [X] | [X] | **[X]** |

**Table 2: Cross-Dataset Generalization (Train → Test)**

| Train \ Test | ds004148 | ds002778 | ds003490 | ds004584 |
|--------------|----------|----------|----------|----------|
| Supervised baseline | [X] | [X] | [X] | [X] |
| SSL pretrained | [X] | [X] | [X] | [X] |

**Figure 1: Training pipeline diagram**
- Pretraining phase (SSL on unlabeled) → Fine-tuning phase (labeled PD data)

**Figure 2: t-SNE of encoder representations**
- Supervised baseline vs SSL pretrained — compare clustering of PD vs HC

**Figure 3: Cross-dataset generalization heatmap**
- 4×4 matrix, supervised vs SSL

---

## 5. Discussion

- Did SSL improve within-dataset accuracy? By how much?
- More importantly: did it improve cross-dataset generalization? (the key claim)
- Why it works: SSL forces the encoder to learn general neural signal patterns rather than dataset-specific artifacts
- Limitations: pretraining corpus size, channel alignment tradeoffs, single SSL method (SimCLR)
- Connection to foundation model paradigm: this is the same recipe as DreaMS, BERT, etc. — just for brain signals

---

## 6. Conclusion

- SSL pretraining on unlabeled EEG data improves generalization for Parkinson's detection
- Cross-dataset eval as the right benchmark for clinical deployment
- Open pipeline released for community use
- Future: larger pretraining corpora, other SSL methods (MAE), multi-task fine-tuning

---

## References (key ones)

- Del Pup et al. (2025) — TransformEEG
- Del Pup et al. (2024) — SelfEEG
- Weng et al. (2024) — SSL for EEG survey
- Bushuiev et al. (2025) — DreaMS (Nature Biotechnology)
- Shoshan et al. (2024) — MAMMAL
- Chen et al. (2020) — SimCLR
- OpenNeuro datasets: ds004148, ds002778, ds003490, ds004584
