# RESEARCH.md — SJJI EEG Project Compass

> **Purpose:** Single source of truth for thesis, methodology, current state, open questions, and experimental roadmap.
> Do not drift from this document. Update it when decisions change.

---

## Thesis

**Self-supervised pretraining on unlabeled EEG produces representations that generalize better across patient populations for Parkinson's Disease detection — demonstrated by measurable improvement in cross-dataset balanced accuracy and data-efficiency over a supervised baseline.**

The clinically relevant problem: models trained on one PD EEG dataset fail when applied to a different clinical site, hardware, or patient population. Supervised learning with scarce labeled data causes overfitting to dataset-specific artifacts. SSL pretraining on abundant unlabeled EEG should learn general neural patterns, making the encoder more robust to site shift.

---

## What We're Building On

| Paper | Method | Key Result | Gap We're Closing |
|-------|--------|-----------|-------------------|
| TransformEEG (Del Pup et al., 2025) | Channel-specific tokenization transformer, supervised | 78.45% bal_acc (within-dataset) | No cross-dataset eval, no SSL |
| SelfEEG (Del Pup et al., 2024) | SimCLR + other SSL for EEG | Library, no PD-specific eval | We use this for the pretraining |
| LaBraM (Jiang et al., 2024) | Large-scale SSL EEG foundation model | General EEG tasks | Not PD-specific, no cross-dataset PD eval |
| BIOT (Yang et al., 2023) | Cross-data SSL for biosignals | General | Not PD-specific |

---

## Datasets

### Available Now
| Dataset | Subjects | PD segs | HC segs | Role |
|---------|----------|---------|---------|------|
| ds004148 | 29 HC | 0 | 12,369 | Unlabeled pretraining (HC pool) |
| ds002778 | 31 | 478 | 243 | PD labeled fine-tuning |
| ds003490 | 50 | 2,532 | 1,234 | PD labeled fine-tuning |
| ds004584 | 149 | 1,214 | 651 | PD labeled fine-tuning |

**Total unlabeled pretraining pool: ~18,000 segments**
**Total labeled fine-tuning: ~6,350 segments across 3 PD datasets**

### Acquired (2026-05-26)
- **TUH EEG corpus**: ✅ Access obtained. Subjects 4–40 ingested locally (~21GB, `data/raw/tuh_eeg`). Full corpus is ~1.2TB via NEDC rsync. Plan: upload full corpus to S3 for cloud-scale pretraining.
- **PPMI (Parkinson's Precision Medicine Initiative)**: ✅ LONI portal access granted 2026-05-26. ❌ **No EEG data exists in PPMI** — confirmed by reviewing portal catalog and protocol docs. PPMI collects MRI, DaTSCAN/SPECT, biospecimen, clinical assessments, and wearable sensor data only. The "4th held-out EEG site" framing has been dropped. Access retained for potential future multimodal/longitudinal work. Not relevant to this paper.

### Pending / To Acquire
- **Predict-PD (UK)**: Under-used PD dataset with good clinical metadata. Investigate access.
- **De Novo Parkinson (Spain)**: Under-used. Investigate access.
- **HBN / Cam-CAN**: Large resting-state HC datasets. Good for pretraining scale.
- **NMT corpus (Pakistan) / TUSZ**: Hardware/site shift validation datasets.

### PPMI — Not Used in This Paper
PPMI has no EEG data. The proposed use case ("4th held-out EEG evaluation site") was based on an incorrect assumption. Confirmed 2026-05-27 by navigating the LONI portal: imaging modalities are MRI/fMRI/PET/SPECT/CT/DTI only, no EEG. Access retained for future work (multimodal PD biomarker research, DaTSCAN correlation studies). Remove from all paper framing.

---

## Baseline Results (completed 2026-05-24)

### Supervised baseline, no pretraining
| Eval Mode | bal_acc | sensitivity | specificity |
|-----------|---------|------------|-------------|
| Per-dataset N-LNSO (ds002778) | 52.4% | 38.8% | 66.0% |
| Per-dataset N-LNSO (ds003490) | 54.4% | 13.3% | 95.5% |
| Per-dataset N-LNSO (ds004584) | 49.3% | 17.0% | 81.6% |
| **Per-dataset aggregate** | **52.0%** | 23.0% | 81.0% |
| Cross-dataset aggregate | 51.6% | 4.5% | 98.6% |

**TransformEEG paper reports: 78.45% within-dataset**

### Known Issues Causing the 52% vs 78.45% Gap
See `ISSUES.md` for full diagnosis. Primary causes:

1. **Channel cyclic padding (CRITICAL)**: `src/preprocess.py::align_channels` pads datasets with fewer than 61 channels by repeating channels cyclically. This corrupts channel topology information and injects identical artificial channels — the tokenizer then operates on garbage. TransformEEG's channel-specific tokenizer is designed for the actual channel structure of each recording.

2. **Training hyperparameters**: Baseline uses 30 epochs, N=5 folds, batch 64. Config specifies 50 epochs, N=10 folds, batch 32 for fine-tuning. Not tuned to match the paper.

3. **HC augmentation in N-LNSO**: ALL of ds004148 is added as training HC in every fold, which may not match the TransformEEG protocol exactly. Need to verify against the paper.

4. **Model is correct**: Architecture matches TransformEEG (Conv1DEncoder → TransformerEncoder → pool). Not the issue.

**Implication**: The 52% baseline is a broken baseline, not a meaningful null model. Before SSL pretraining adds any value, we need to reproduce the supervised baseline at or near 78.45%. The SSL story collapses if we can't show a valid baseline to beat.

---

## Experimental Roadmap

### Phase 0: Fix the Baseline (BLOCKING everything else)
**Goal: Reproduce TransformEEG's 78.45% on within-dataset eval**

1. Fix `align_channels` — per-dataset channel selection rather than padding:
   - Each dataset trains and evaluates with its own channel count
   - Build separate `build_encoder(Chan=N)` per dataset
   - OR use channel interpolation/projection to common space (but test this doesn't hurt performance)
2. Match training protocol to paper: 10 N-LNSO folds, 50 epochs, batch 32
3. Verify HC augmentation approach matches the paper
4. Target: within 2-3% of 78.45% before proceeding

**This phase must complete before SSL pretraining starts.**

### Phase 1: SSL Pretraining
**Goal: SimCLR pretraining on unlabeled EEG segments**

- Pretrain on ~18k segments (ds004148 HC + unlabeled segments from PD datasets)
- Augmentation policy sweep (4+ policies) — critical at this data scale:
  - Time shift, amplitude scaling, Gaussian noise, channel dropout, frequency masking
  - Report all; show best is not trivial
- Compare: linear probing vs full fine-tuning after pretraining
- Ablation: frozen vs unfrozen encoder layers
- Ablation: pretrain HC-only → should hurt performance (negative control)

### Phase 2: Cross-Dataset Evaluation
**Goal: Demonstrate SSL encoder generalizes across datasets better than supervised**

- Leave-one-dataset-out: train on 3, test on 1 (already implemented in baseline.py)
- Compare: supervised vs SSL-pretrained encoder on cross-dataset protocol
- Add ROC-AUC and PR-AUC alongside balanced accuracy
- Pretrain on TUH (full corpus via S3) → cross-dataset eval on OpenNeuro PD sets (ds002778, ds003490, ds004584)

### Phase 3: Data-Efficiency Analysis
**Goal: SSL reaches supervised performance with fewer labeled examples**

- Subsample labeled training data: 100%, 50%, 25%, 10% of subjects
- Plot: SSL vs supervised performance vs fraction of labeled data
- Key claim: SSL needs 5-10x fewer labels to reach comparable performance
- This is the most compelling practical argument if achieved

### Phase 4: Mechanistic Validation (if time permits, adds significant impact)
- Frequency band analysis: does SSL encoder activate more strongly on theta/beta bands known to be PD biomarkers?
- Attention map visualization: which channels/timepoints drive predictions?
- Compare SSL vs supervised encoder representations via PCA/UMAP

---

## Novelty Positioning

### What Has Been Done (competitive landscape — will be filled in after meta-analysis)
- LaBraM, BIOT, CBraMod, EEGPT: large-scale SSL EEG, but NOT PD-specific, NOT cross-dataset PD eval
- TransformEEG: SOTA supervised PD detection, but NO SSL, NO cross-dataset eval
- SelfEEG: SSL library for EEG, no PD-specific experiments

### Our Genuine Gap (preliminary — update after meta-analysis completes)
**"First systematic study of SSL pretraining for cross-dataset Parkinson's EEG detection, evaluated on the same labeled benchmark as TransformEEG."**

The combination that hasn't been done:
- SSL pretraining (SimCLR) × TransformEEG architecture × PD-specific datasets × cross-dataset generalization eval

### Narrative Options (ranked by likely impact)
1. **Data-efficiency** (highest impact if it works): SSL reaches supervised performance with far fewer labels. Directly addresses the clinical bottleneck (expert labeling is expensive). This is measurable, clear, and practically meaningful.
2. **Cross-dataset generalization**: SSL improves train-on-A test-on-B performance. Clear eval metric, direct clinical relevance, but more crowded framing.
3. **Mechanistic insight**: SSL encoder learns PD-relevant spectral biomarkers. High impact but highest effort — requires additional analysis work.

### Target Venues
- **IEEE EMBC 2026** (deadline TBD): 4-page limit, needs to be crisp
- **IEEE JBHI**: Full paper, more room for ablations
- **Neurocomputing**: Broader ML venue, lower bar

---

## Key Open Questions (must answer before finalizing methodology)

1. **Can we reproduce 78.45%?** If not, we need to understand why. The paper's code is public (MedMaxLab/transformeeg) — if needed, run their exact preprocessing and compare.

2. **Is 18k segments enough for SimCLR?** No longer the primary concern — TUH full corpus (~1.2TB) is available for S3-backed cloud pretraining. Scale argument is now real.

3. **What is the actual novelty gap?** Meta-analysis in progress. Need to confirm no one has already published SSL + PD + cross-dataset with these same datasets.

4. **Which datasets can we access?** TUH ✅ acquired. PPMI ✅ portal access granted but has no EEG — not used in this paper. Predict-PD, De Novo PD — investigate as potential additional EEG eval sites.

5. **How do we handle channel heterogeneity across datasets?** Options: (a) per-dataset encoder (separate training), (b) channel projection to common space, (c) channel-agnostic tokens (per TransformEEG's original approach). Need to understand exactly what TransformEEG does.

6. **What augmentation policy works for PD-relevant EEG?** Time shift and amplitude scaling are safe. Frequency masking risks masking the theta/beta biomarkers we want to learn. Need to be careful.

7. **What is the right evaluation metric?** Balanced accuracy is noisy on small test sets. ROC-AUC and PR-AUC alongside it. Should report confidence intervals via bootstrap.

---

## Open Issues Tracker

| Issue | Priority | Status |
|-------|----------|--------|
| Channel cyclic padding bug in preprocess.py | CRITICAL | Open |
| Match training protocol to TransformEEG paper (epochs, folds, batch) | HIGH | Open |
| Verify N-LNSO HC augmentation matches paper | HIGH | Open |
| Get TUH EEG access | MEDIUM | ✅ Done (2026-05-26) — full corpus via S3 planned |
| ~~PPMI EEG~~ | ~~HIGH~~ | ❌ Closed — PPMI has no EEG data (confirmed 2026-05-27). Access retained for future work. |
| Upload TUH to S3 | HIGH | Pending — full ~1.2TB, plan to use EC2 + NEDC rsync direct to S3 |
| Investigate Predict-PD and De Novo PD datasets | MEDIUM | Open |
| Complete meta-analysis of competitive landscape | HIGH | In progress (subagent) |
| Run SimCLR pretraining on 18k segments | HIGH | Blocked on Phase 0 |

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-23 | Pivoted from LoRA fine-tuning study to SSL pretraining direction | Original paper had no results; SSL framing is more original and impactful |
| 2026-05-24 | Set cross-dataset generalization as primary eval metric | Clinically relevant, distinguishes from prior work |
| 2026-05-24 | Baseline showed 52% — identified channel padding as root cause | Will fix before SSL runs |
| 2026-05-24 | Consulted Grok for second opinion | Confirmed novelty risk; recommended data-efficiency narrative |

---

## Files

```
sjji-eeg/
├── RESEARCH.md          ← this file (compass)
├── ISSUES.md            ← detailed bug/blocker log
├── baseline.py          ← supervised baseline runner
├── train.py             ← SSL pretraining + fine-tuning runner
├── src/
│   ├── model.py         ← TransformEEG architecture (encoder + classifier)
│   ├── preprocess.py    ← EEG preprocessing pipeline (has channel padding bug)
│   ├── finetune.py      ← N-LNSO CV + fine-tuning utilities
│   ├── pretrain.py      ← SimCLR pretraining utilities
│   ├── evaluate.py      ← metrics + result saving
│   └── utils.py
├── configs/             ← YAML configs for pretraining and fine-tuning
├── data/
│   ├── raw/             ← BIDS-format raw data (not committed to git)
│   └── processed/       ← preprocessed .npy segments + labels.csv
├── results/             ← experiment outputs (not committed to git)
├── paper/
│   ├── OUTLINE.md       ← paper structure
│   ├── original_context/  ← original team's proposal/draft (historical)
│   └── transformeeg_paper.pdf
└── notebooks/           ← exploratory analysis
```
