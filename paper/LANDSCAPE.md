# Research Landscape — SJJI EEG Project

> Generated 2026-05-24 via Grok meta-analysis. Update when new relevant papers appear.

---

## 1. Competitive Landscape

| Paper | Venue/Year | Method | PD Detection | Cross-Dataset Eval | Key Gap |
|-------|-----------|--------|-------------|-------------------|---------|
| TransformEEG (Del Pup et al.) | ~2024/2025 | Channel-specific tokenization + Transformer, supervised | ✅ Yes (ds002778, ds003490, ds004584) | ❌ No — within-dataset only | No SSL, no cross-dataset |
| SelfEEG (Del Pup et al.) | ~2024 | SSL library (SimCLR, etc.) for EEG | ❌ No | ❌ Limited (single-dataset benchmarks) | No PD-specific eval |
| BIOT (Yang et al., 2023) | NeurIPS 2023 | Cross-dataset biosignal SSL (contrastive + reconstruction) | ❌ No (general pathology) | ✅ Yes (multi-source pretraining + transfer) | Not PD-specific, different arch/datasets |
| LaBraM (Jiang et al., 2024) | NeurIPS 2024 | Large-scale masked autoencoding EEG foundation model | ❌ No (general pathology/BCI) | ✅ Yes (TUH + held-out) | Not PD-specific, different arch/datasets |
| EEGPT (Liu et al., ~2024/2025) | Uncertain | EEG pretrained transformer | Uncertain | Uncertain | Need to verify |
| CBraMod | Uncertain | Unknown | Uncertain | Uncertain | Not verifiable — treat as absent |

### Key Finding
**Nobody has applied SSL (specifically SimCLR) to TransformEEG's channel-tokenized architecture on the same four OpenNeuro PD/HC datasets with leave-one-dataset-out cross-dataset evaluation.** BIOT and LaBraM do cross-dataset SSL EEG, but target seizure/general pathology on TUH — not PD, not these datasets, not this architecture.

Cross-dataset PD EEG detection is genuinely sparse in the literature. Most SSL EEG work targets seizure detection or BCI tasks.

---

## 2. Dataset Inventory

### Datasets We Have
| Dataset | Subjects | Condition | Segs (our preprocessing) | Access | Limitations |
|---------|----------|-----------|--------------------------|--------|-------------|
| ds004148 | ~29 | HC only | 12,369 | OpenNeuro (public) | No PD; diverse tasks (music, memory, math, eyes-open/closed) |
| ds002778 | 31 | PD + HC | 721 (478 PD, 243 HC) | OpenNeuro (public) | Small N, heterogeneous setup |
| ds003490 | 50 | PD + HC | 3,766 (2,532 PD, 1,234 HC) | OpenNeuro (public) | Moderate size |
| ds004584 | 149 | PD + HC | 1,865 (1,214 PD, 651 HC) | OpenNeuro (public) | Largest PD dataset we have |

**Total pretraining pool: ~18,000 segments. Total labeled: ~6,350 PD-labeled segments.**

### Pending / To Acquire
| Dataset | Condition | Scale | Access | Notes |
|---------|-----------|-------|--------|-------|
| TUH EEG corpus | General clinical EEG | >10,000 subjects | Restricted (NEDC DUA) | Not PD-enriched; useful for general SSL pretraining diversity. Resubmitting via institutional contact. |
| Predict-PD (UK) | At-risk PD | Modest EEG subset | UK Biobank-linked (restricted) | Not fully public; longitudinal risk-stratification, not diagnostic |
| De Novo Parkinson (Spain) | PD | Unknown | Uncertain | No widely-cited public version confirmed |
| HBN (Healthy Brain Network) | HC (pediatric/adolescent) | Large | OpenNeuro (public) | No PD — useful only for pretraining HC diversity |
| Cam-CAN | HC (adult lifespan) | Large | Public | No PD |
| NMT corpus | Uncertain | Unknown | Unknown | Cannot verify details |
| ~~PPMI EEG subset~~ | — | — | — | ❌ PPMI has no EEG data (confirmed 2026-05-27 via portal). Has MRI/DaTSCAN/biospecimen only. Access retained for future multimodal work, not relevant to this paper. |

### Bottom Line on Datasets
There is no large, curated, public PD resting-state EEG corpus with cross-site standardization. The four OpenNeuro datasets are essentially the state of the art for public PD EEG research — which is simultaneously a limitation and an opportunity (there's a clear gap for better benchmarking).

---

## 3. Genuine Novelty Gap

**The combination that has NOT been published:**

SimCLR pretraining on TransformEEG's channel-tokenized architecture  
× using the same four OpenNeuro PD/HC datasets as TransformEEG  
× evaluated with cross-dataset leave-one-dataset-out  
× with data-efficiency curves (10%, 25%, 50%, 100% labeled data)

This is confirmed novel. BIOT and LaBraM come closest but use different architectures, different datasets (TUH-centric), and don't target PD specifically.

**Additional differentiators that strengthen novelty:**
1. Using SelfEEG (same research group as TransformEEG) makes the SimCLR integration a natural extension — clean narrative
2. Being the first to establish a **cross-dataset PD EEG benchmark** using these exact datasets positions the paper as setting a reproducible standard
3. Data-efficiency curves add practical clinical relevance (labeled EEG data from PD patients is expensive to acquire)

---

## 4. Narrative Ranking

### Rank 1: Option A + D hybrid — "First cross-dataset PD EEG benchmark + SSL improvement"
**"Channel-tokenized transformer with SimCLR pretraining establishes the first reproducible cross-dataset Parkinson's Disease EEG benchmark"**

- Publishable at: IEEE EMBC 2026, IEEE JBHI, Neurocomputing
- **What makes it publishable:** Clear prior baseline (TransformEEG), same datasets, rigorous subject-independent splits, multiple seeds for confidence intervals, open-source pipeline. Frames the work as establishing a missing standard, not just claiming marginal improvement.
- **What gets it rejected:** Marginal gains (<3-5% AUC), subject leakage across folds, not beating supervised TransformEEG, no comparison to BIOT/LaBraM.
- **Requirements:** Fix baseline to reproduce 78.45%, show SSL gains >5% cross-dataset, provide CI via bootstrap.

### Rank 2: Option B — "Data-efficiency"
**"SSL pretraining enables Parkinson's EEG detection with 5-10x fewer labeled examples"**

- Publishable at: IEEE JBHI (strong fit), IEEE EMBC (shorter version)
- **What makes it publishable:** Learning curves at 10%/25%/50%/100% labeled data showing SSL reaches supervised performance with far fewer labels. Directly addresses clinical bottleneck (expert EEG labeling is expensive).
- **What gets it rejected:** Noisy curves, no statistical tests, no strong supervised baseline comparison.
- **Requirements:** Run labeled data subsampling experiments, statistical testing across multiple seeds.

### Rank 3: Option C — "Mechanistic"
**"What does self-supervised EEG pretraining learn about Parkinson's disease?"**

- Publishable at: IEEE JBHI (longer format required)
- **What makes it publishable:** Quantitative evidence that SSL encoder representations align with known PD biomarkers (beta-band desynchronization, theta power increase) using UMAP, CKA similarity, or attention analysis. Statistical tests required.
- **What gets it rejected:** Qualitative visualizations only, no statistical tests, no causal claim.
- **Requirements:** PD spectral biomarker literature review, quantitative representation analysis, high additional effort.

### Recommendation
**Lead with Rank 1 framing, include Rank 2 as a key result section.** This gives the paper two distinct contributions: the benchmark + the practical data-efficiency argument. Mechanistic analysis is "future work" unless results are striking enough to dedicate a full section.

---

## 5. Key Open Questions Before Finalizing Methodology

1. **Can we reproduce 78.45% with the fixed baseline?** This gates everything. If we can't get within ~3% of the paper's reported performance, we need to debug further (run their exact codebase) before SSL runs.

2. **Do ds003490 and ds004584 share channel space with ds002778?** Channel counts differ (67, 64, 40). For cross-dataset evaluation with a single unified encoder, we need a channel alignment strategy. Options: (a) subset to common channels, (b) per-dataset encoders with projection layer, (c) channel interpolation.

3. **Is 18k segments enough for SimCLR?** Empirical question — run it. If improvement is <2% absolute on cross-dataset, the data-efficiency story becomes the primary contribution rather than the generalization story.

4. **What augmentation policy won't destroy PD biomarkers?** Frequency masking over 8-30Hz risks masking theta/beta bands. Time shift, amplitude jitter, and Gaussian noise are safer starting points. Need augmentation sweep.

5. **Do any of the four datasets have overlapping subjects or data leakage risk?** Need to verify all subject IDs are unique across datasets before running cross-dataset evaluation. A single overlapping subject breaks the eval.

6. **What is the right comparison baseline for reviewers?** Minimum: supervised TransformEEG (what we're implementing). Ideal: also run BIOT or LaBraM on these same datasets for a fair SSL comparison. This is significant additional work but would preempt the strongest reviewer objection.

7. **Is there a simpler baseline we should beat first?** SVM/LDA on frequency-band features (theta power, beta desynchronization) is the traditional approach for PD EEG. If SSL doesn't beat even classical features, the story collapses.
