# Paper Outline
## Site Confounds, Calibration, and the Limits of Self-Supervision in Cross-Dataset Parkinson's EEG Detection

**Target venue:** IEEE EMBC / IEEE JBHI / Neurocomputing
**Status:** Results largely in; data-efficiency sweep + writeup in progress.

> **NOTE — thesis changed.** The original framing ("SSL improves cross-dataset PD
> detection") was *not* supported by our experiments. This outline reflects what the
> data actually shows. Do not write toward the old thesis.

---

## Ownership (who drafts what — so we split the load)

| Section | Primary owner | Students can do |
|---|---|---|
| 1 Introduction | Mentor + Claude (claims) | draft the clinical-problem paragraph (¶1) |
| 2 Related Work | **Students** | the whole section (see `docs/related_works.md`) |
| 3 Datasets & Preprocessing | **Students** (draft) + Claude (montage finding) | dataset table, preprocessing prose |
| 4 Methods (protocols / null / calibration) | Mentor + Claude | — |
| 5 Results | Mentor + Claude (numbers) | **make the figures** from result JSONs |
| 6 Discussion / 7 Conclusion | Mentor + Claude | — |
| Figures + dataset table | — | **Students** (matplotlib from `results/`) |

---

## Title (working)
> **"Site Confounds, Not Signal: Re-evaluating Cross-Dataset Parkinson's Disease Detection from EEG"**

Alt: *"What Actually Transfers Across Sites in EEG-Based Parkinson's Detection? Confounds, Calibration, and the Limits of Self-Supervised Pretraining"*

---

## Core narrative (one paragraph)

EEG-based Parkinson's detection commonly reports high accuracy by pooling several public
datasets and evaluating with subject-level cross-validation. We show this number is
**confounded by dataset/site identity**: because label distribution is nearly determined
by which dataset a recording comes from, a classifier using *no EEG signal at all* — only
the site label — reaches ~0.93 balanced accuracy on the same protocol, matching or
exceeding published models. Under an honest **leave-one-dataset-out** (LODO) protocol,
where the test site is unseen, fixed-threshold balanced accuracy collapses toward chance.
But that collapse is largely a **calibration failure under domain shift**, not an absence
of transferable signal: supervised representations still rank PD vs HC on unseen sites
(AUC ≈ 0.76), and a deployable threshold recovers ≈ 0.64 balanced accuracy. Finally, we
find that **self-supervised pretraining** — including on a large, disjoint clinical corpus
(TUH) — and **training-time augmentation** do **not** improve cross-site transfer at the
scales tested. We argue cross-site PD-EEG should be evaluated with LODO, threshold-
independent metrics, and an explicit site-prior null.

---

## Contributions
1. **A site-prior null** that quantifies the confound in the standard pooled protocol: a
   no-EEG, dataset-majority classifier scores 0.93 (segment) / 0.65 (subject) balanced
   accuracy — ≥ published models on the same metric.
2. **An honest LODO evaluation** showing the cross-site "failure" is primarily calibration:
   supervised AUC ≈ 0.76 on unseen sites; a training-derived threshold recovers ≈ 0.64
   balanced accuracy (vs 0.50 chance).
3. **A negative result on SSL + augmentation:** neither self-supervised pretraining
   (OpenNeuro or disjoint TUH, linear-probe and fine-tune) nor augmentation improves
   cross-site transfer; with analysis of why (domain mismatch, scale, eval protocol).
4. **Open evaluation tooling:** site-prior null, LODO harness, calibration policies,
   reproducible montage handling.

---

## Abstract (fill last; numbers below are current best estimates)
- Problem: pooled-protocol PD-EEG accuracy conflates pathology with site identity.
- Finding 1: no-EEG site null ≈ 0.93 on the standard metric.
- Finding 2: under LODO, supervised AUC ≈ 0.76±0.03; calibrated bal-acc ≈ 0.64±0.03.
- Finding 3: SSL pretraining and augmentation do not improve cross-site transfer.
- Significance: a corrected evaluation protocol + an honest cross-site baseline for the field.

---

## 1. Introduction
- ¶1 *(students can draft)* — clinical problem: PD prevalence; EEG as accessible biomarker; models must work across sites/hardware to be deployable.
- ¶2 — what exists: DL for PD-EEG (EEGNet, transformers, TransformEEG 78.45%); all evaluate on a **pooled** multi-dataset protocol; cross-site generalization under-examined.
- ¶3 — the problem with pooling: when sites have imbalanced labels and are trivially identifiable, pooled accuracy can be reached without learning pathology (forward-reference our null).
- ¶4 — what we do: introduce the site-prior null; evaluate with LODO + AUC + calibration; test whether SSL closes the cross-site gap.
- ¶5 — contributions (above).

## 2. Related Work  *(STUDENTS — see docs/related_works.md)*
- 2.1 EEG-based Parkinson's detection (TransformEEG + the 4 datasets + DL methods).
- 2.2 Evaluation pitfalls in EEG-DL: subject/segment leakage, preprocessing variability, **site/batch effects** (ComBat), N-LNSO.
- 2.3 Self-supervised learning for EEG (LaBraM, BIOT, BENDR, SelfEEG, EEGPT) — and whether any show cross-site/domain-generalization gains.
- 2.4 Calibration under domain shift (temperature scaling; OOD calibration).

## 3. Datasets & Preprocessing  *(STUDENTS draft; Claude owns the montage subsection)*
- 3.1 Datasets — the characterization table (PD/HC, task, montage, sfreq, role). Use the table in `docs/related_works.md`.
- 3.2 Preprocessing — bandpass 1–45 Hz, resample, 16 s windows @ 25% overlap, per-segment per-channel z-score.
- 3.3 **Channel harmonization (our finding):** the OpenNeuro-derived 29-ch set leaves 14/29 channels dead in TUH; we use the 19-ch TUH∩OpenNeuro montage (with old→new 10-20 renaming). *(Claude)*

## 4. Methods  *(MENTOR + CLAUDE)*
- 4.1 Protocols: combined N-LNSO (the standard, site-confounded) vs **LODO** (honest cross-site). Subject-level aggregation.
- 4.2 **Site-prior null**: predict each dataset's majority class; report alongside every pooled number.
- 4.3 Metrics: balanced accuracy (subject-level, median±IQR), **ROC-AUC** (threshold-independent), bootstrap CIs.
- 4.4 **Calibration policies** under domain shift: fixed-0.5, train-transferred (deployable), prevalence-matched, oracle ceiling.
- 4.5 Models: supervised TransformEEG (from scratch); SSL (VICReg) pretraining → linear probe & fine-tune; augmentation ablation.

## 5. Results  *(MENTOR + CLAUDE for numbers; STUDENTS for figures)*
- **5.1 The confound** — Table: pooled combined-N-LNSO vs the site-prior null (0.93 seg / 0.65 subj). *Fig 1: no-EEG null vs published accuracy.*
- **5.2 Honest cross-site (LODO)** — fixed-threshold collapse → calibration recovery. Supervised: AUC 0.76±0.03, deployable 0.64±0.03, prevalence 0.69. *Fig 2: combined vs LODO + calibration policies.*
- **5.3 SSL does not help** — AUC: supervised 0.76, SSL-OpenNeuro 0.58, SSL-TUH 0.53 (linear probe); fine-tune + data-efficiency [PENDING battery]; augmentation no effect. *Fig 3: cross-site AUC by method; Fig 4: data-efficiency curves.*
- Table: full LODO results (AUC, calibrated bal-acc) × {supervised, SSL-OpenNeuro, SSL-TUH, aug} with seed error bars.

## 6. Discussion  *(MENTOR + CLAUDE)*
- Why pooled accuracy misleads; the site shortcut; why subject-level + LODO + null are necessary.
- Cross-site failure as a calibration problem — practical implication (deployable thresholds), and the limit (we still need site-aware calibration).
- Why SSL didn't help: domain mismatch (clinical TUH vs resting-state PD), scale, linear-probe vs fine-tune; what would be needed (resting-state HC pretraining, much larger scale).
- Limitations: 4 datasets / 3 held-out sites, single architecture, small subject counts, SSL scale ceiling on our hardware.

## 7. Conclusion
- The field's cross-dataset PD-EEG accuracy is substantially a site artifact; report the null.
- Honest cross-site detection is real but modest and calibration-bound (AUC ~0.76, deployable ~0.64).
- SSL pretraining is not a free cross-site win at these scales — a useful negative for the community.

---

## References (key)
- Del Pup et al. 2025 — TransformEEG · Del Pup et al. 2024 — SelfEEG / preprocessing variability
- Jiang et al. 2024 — LaBraM · Yang et al. 2023 — BIOT · Kostas et al. 2021 — BENDR
- Johnson et al. 2007 — ComBat · Guo et al. 2017 — calibration / temperature scaling
- Bardes et al. 2022 — VICReg · OpenNeuro: ds004148/002778/003490/004584 · Obeid & Picone 2016 — TUH
