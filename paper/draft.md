# Site Confounds, Calibration, and the Limits of Self-Supervision in Cross-Dataset Parkinson's EEG Detection

> Working draft. **Owners:** Methods/Results/Discussion — Edward + Claude. Related Work (§2) and the
> Introduction hook (¶1) — Saanvi & Alex. Numbers are 19-channel LODO, subject-level, seeded where noted.
> Figures referenced are produced by `experiments/make_figures.py` → `paper/figures/`.

## Abstract

Electroencephalography (EEG) is a low-cost, non-invasive candidate biomarker for Parkinson's disease (PD), and deep-learning models report high balanced accuracy on a benchmark that pools four public PD datasets. We show that this pooled accuracy is **largely a site artifact**: because the datasets have strongly imbalanced and dataset-specific label distributions, a *site-prior null* that uses no EEG signal — predicting each dataset's majority class — reaches 0.927 segment-level balanced accuracy, matching or exceeding published models. Evaluating instead with leave-one-dataset-out (LODO), where the test site is unseen, fixed-threshold balanced accuracy collapses toward chance. We demonstrate this collapse is primarily a **calibration failure under domain shift, not an absence of signal**: supervised representations rank PD versus healthy controls on unseen sites at ROC-AUC 0.76 ± 0.03, and a fully deployable decision threshold — selected on the training sites — recovers 0.64 ± 0.03 balanced accuracy. Finally, we find that **self-supervised pretraining does not improve cross-site transfer** at the scales tested: pretraining on the four datasets or on a large disjoint clinical corpus (TUH) yields cross-site AUC of 0.58 and 0.53 respectively, far below the supervised model, and a fine-tuning data-efficiency analysis shows no advantage at any label budget. We release the evaluation tooling, including the site-prior null and calibration analysis, and argue that cross-site PD-EEG should be reported with LODO, threshold-independent metrics, and an explicit site-prior null.

## 1. Introduction *(¶1 hook — students; remainder — drafted below)*

*(Saanvi/Alex: open with the clinical motivation — PD prevalence, late diagnosis from motor symptoms, EEG as an accessible biomarker, and why a model must work across hospitals to matter.)*

Deep-learning models for EEG-based PD detection are commonly evaluated by pooling several public datasets and performing subject-level cross-validation, reporting balanced accuracies near 80–90%. This protocol holds out subjects but not *sites*: every dataset appears in both training and test folds. Because EEG recordings are trivially identifiable by site (amplifier, montage, filtering) and because each dataset carries a dataset-specific label distribution, a model can achieve high pooled accuracy by recognizing the dataset rather than the disease. Whether such accuracy reflects pathology — and whether it transfers to a genuinely unseen site — has not been directly tested.

We make three contributions. **(1)** We introduce a *site-prior null* and show that the pooled-protocol accuracy is confounded: a no-EEG baseline matches published models. **(2)** Using leave-one-dataset-out, we show that the resulting cross-site "failure" is mostly a calibration problem; supervised representations transfer (AUC ≈ 0.76) and a deployable threshold recovers useful balanced accuracy. **(3)** We report a careful negative result: self-supervised pretraining — including on a large disjoint corpus — does not improve cross-site transfer, across linear-probe, fine-tune, and data-efficiency evaluations.

## 2. Related Work *(STUDENTS — see `docs/related_works.md`)*

*(Saanvi — Track A: cross-site generalization & evaluation leakage in EEG-DL; PD-from-EEG. Alex — Track B: self-supervised learning for EEG; calibration under domain shift. Pull citations into Zotero; one paragraph per theme.)*

## 3. Methods

### 3.1 Datasets
We use four public OpenNeuro PD datasets (270 subjects: 140 PD, 130 healthy controls [HC]) and the Temple University Hospital (TUH) EEG corpus as an unlabeled pretraining source. The labeled datasets differ markedly in composition: **ds004148** contributes 60 HC and no PD (a resting/cognitive test–retest set); **ds002778** (UC San Diego) 15 PD / 16 HC; **ds003490** (UNM 3-Stim) 25 PD / 25 HC; and **ds004584** 100 PD / 49 HC. After windowing, ds004148 alone accounts for 12,369 of 18,721 segments (66%), all HC — a composition central to the confound analyzed in §4.1.

### 3.2 Preprocessing and channel harmonization
Recordings are band-pass filtered (1–45 Hz), resampled to 250 Hz, segmented into 16-second windows with 25% overlap, and z-scored per channel within each segment (no statistics shared across the train/test boundary). The four OpenNeuro datasets are mapped to a common 64-channel 10–20 layout. The original benchmark uses the 29 channels common to all four datasets; however, this set is OpenNeuro-specific. Measuring channel presence directly on TUH (clinical 10–20 montage), we find **only 15 of the 29 channels carry signal**: ten extended 10–10 positions (AF3/AF4, FC1/2/5/6, CP1/2/5/6) are absent, and four (T7/T8/P7/P8) are present but under the legacy names T3/T4/T5/T6. We add an old→new name remap and adopt the **19-channel TUH∩OpenNeuro montage** (verified 19/19 present in TUH), so that a TUH-pretrained encoder never learns channel filters on dead inputs. All cross-site experiments use the 19-channel montage.

### 3.3 Architecture
We use the TransformEEG encoder (Del Pup et al., 2025): a per-channel depthwise convolutional tokenizer followed by a two-layer transformer encoder and adaptive average pooling, producing a `C × 4`-dimensional feature (76-d at 19 channels). For supervised training a two-layer classification head is attached; for self-supervised pretraining a VICReg projector is attached to the pooled feature.

### 3.4 Evaluation protocols
We compare two protocols. **Combined N-LNSO** is the field-standard pooled protocol: all four datasets are merged and 10-fold subject-stratified cross-validation holds out ~27 subjects per fold; every dataset appears in every fold's train and test. **Leave-one-dataset-out (LODO)** holds out one entire dataset as the test site and trains on the other three; this removes the site shortcut and is our primary cross-site protocol. Because ds004148 is single-class, the three held-out test sites are the both-class PD datasets. Alongside every pooled number we report a **site-prior null**: a classifier that, for each test segment, predicts the majority class of that segment's dataset — using no EEG signal — and is scored on the same metric.

### 3.5 Metrics and calibration policies
We report subject-level balanced accuracy (segment scores averaged per subject before thresholding) and **ROC-AUC**, which is threshold-independent and isolates whether the model *ranks* PD above HC. Because a fixed 0.5 threshold mis-calibrates on an unseen site, we evaluate several decision-threshold policies: **fixed-0.5**; **train-transferred** (threshold maximizing balanced accuracy on the training sites, applied unchanged to the held-out site — fully deployable); **prevalence-matched** (threshold so the predicted positive rate equals the held-out site's PD prevalence); and **oracle** (threshold maximizing balanced accuracy on the held-out labels — the unachievable ceiling implied by the AUC). Variability is reported as mean ± standard deviation across random seeds.

### 3.6 Self-supervised pretraining
We pretrain the encoder with VICReg on unlabeled EEG, then evaluate three ways: a **linear probe** (encoder frozen, linear head trained per LODO fold); **fine-tuning** (encoder initialized from the SSL checkpoint and trained end-to-end, identical to the supervised recipe but with a pretrained initialization); and a **data-efficiency** sweep (fine-tuning on 10%, 25%, and 100% of training subjects). We pretrain on two sources: the four OpenNeuro datasets (same domain as evaluation; labels ignored) and a **disjoint TUH subset** (298 recordings, 19,883 segments) that shares no recordings with the evaluation data.

### 3.7 Training details
Supervised and fine-tuning runs use Adam (β₁ = 0.75, β₂ = 0.999), learning rate 2.5 × 10⁻⁴, ExponentialLR (γ = 0.99), batch size 32, 50 epochs, with class-balanced BCE (`pos_weight`). VICReg pretraining uses the same optimizer family with a cosine schedule. Reported cross-site results average three random seeds (encoder initialization, data shuffling). All experiments run on a single GPU (Apple MPS); no cloud GPUs were used for the results reported here.

## 4. Results

### 4.1 The pooled-protocol accuracy is a site artifact
On the combined N-LNSO protocol, the supervised model reaches 0.89 median segment-level balanced accuracy and the SSL probe 0.90–0.92 — apparently strong. However, the **site-prior null reaches 0.927** (segment) and 0.654 (subject) balanced accuracy using no EEG signal whatsoever (Fig. 1). Because ds004148 is entirely HC and dominates the segment pool, and because each PD dataset is PD-majority at the segment level, predicting the dataset's majority class alone matches or exceeds the trained models. The pooled metric therefore cannot distinguish "detects Parkinson's" from "detects the dataset," and any pooled number must be read against this null.

### 4.2 Honest cross-site evaluation: a calibration problem, not absent signal
Under LODO, fixed-0.5 subject-level balanced accuracy is 0.585 ± 0.014 — near chance, with the model predicting PD for almost all subjects on an unseen site (high sensitivity, near-zero specificity). This appears to be a generalization failure. It is not. The threshold-independent **ROC-AUC is 0.763 ± 0.034** (Fig. 3), i.e., the model ranks PD above HC well above chance on sites it never saw. The collapse is a *calibration* artifact: the score distribution shifts across sites, so 0.5 is the wrong operating point. Choosing the threshold on the training sites and transferring it (a fully deployable policy) recovers **0.643 ± 0.034**; matching the held-out site's prevalence reaches 0.686 ± 0.041; the oracle ceiling is 0.732 (Fig. 2). Cross-site PD detection from EEG is thus real but modest, and bottlenecked by calibration rather than representation.

### 4.3 Self-supervised pretraining does not improve cross-site transfer
Across every evaluation, SSL fails to help (Fig. 3, Fig. 4):
- **Linear probe.** Cross-site AUC is 0.581 ± 0.002 (OpenNeuro pretraining) and 0.526 ± 0.003 (disjoint TUH), versus 0.763 for the supervised model. The disjoint, larger-corpus encoder is the *weakest*, consistent with a domain mismatch between clinical TUH EEG and resting-state PD recordings.
- **Fine-tuning.** Initializing supervised training from the TUH encoder matches training from scratch at full labels (AUC 0.72 vs 0.73).
- **Data-efficiency.** Across label budgets of 10/25/100%, SSL-TUH fine-tuning tracks from-scratch training at every point; only same-domain OpenNeuro pretraining shows a small edge at 10% labels (Fig. 4). SSL does not reduce the label requirement (the supervised model reaches near-full cross-site AUC by ~25% of labels).
- **Augmentation.** Training-time augmentation gives no cross-site lift (AUC 0.763 vs 0.764).

The negative is stable across seeds (SSL AUC standard deviations ≈ 0.002), so it is not seed noise.

## 5. Discussion

Our results reframe what is known about cross-dataset PD-EEG. First, the widely reported pooled accuracy is substantially a measurement artifact; the appropriate baseline is not 0.5 but the site-prior null, and evaluation should hold out whole sites. Second, the apparent cross-site failure of supervised models is, to a large degree, a *calibration* problem — a practically important and partly solvable one: a threshold transferred from training sites already recovers most of the achievable balanced accuracy, and the residual gap to the oracle motivates site-aware calibration rather than better features. Third, self-supervised pretraining — the dominant recipe for transfer elsewhere — does not deliver a cross-site advantage here, even from a large disjoint corpus and even in the fine-tuning and low-label regimes where it is expected to shine. The most likely cause is domain mismatch (heterogeneous clinical TUH EEG versus resting-state research recordings), suggesting that *what* one pretrains on matters more than scale for this task; resting-state HC corpora are a natural next probe.

## 6. Conclusion
Cross-dataset PD-EEG accuracy, as conventionally reported, is largely a site artifact. Measured honestly with leave-one-dataset-out, threshold-independent metrics, and an explicit site-prior null, supervised detection transfers across sites at a real but modest level that is calibration-bound, and self-supervised pretraining does not improve it at the scales tested. We release the evaluation tooling to make these checks standard.

## Limitations
Four datasets and three held-out sites; a single architecture; small subject counts per held-out site (31/50/149); SSL pretraining bounded by single-GPU scale (the disjoint TUH subset is ~20k segments, not the full corpus — a full-scale run would strengthen but is not expected to overturn the negative). Cross-site results use three seeds; the data-efficiency grid is being extended to multiple seeds.
