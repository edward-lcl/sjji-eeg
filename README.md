# SJJI — Cross-Dataset Parkinson's Detection from EEG

**Site confounds, calibration, and the limits of self-supervision.**

A rigorous re-evaluation of EEG-based Parkinson's disease (PD) detection across datasets ("sites"). We find that the high accuracy commonly reported on the pooled multi-dataset benchmark is largely a **site artifact**; that honest cross-site detection is real but **calibration-bound**; and that self-supervised pretraining does **not** improve cross-site transfer at the scales tested.

📊 **Team dashboard:** https://sjji-eeg.exe.xyz · 📄 **Paper outline:** [`paper/OUTLINE.md`](paper/OUTLINE.md) · 📚 **Related works:** [`docs/related_works.md`](docs/related_works.md)

> **Note — the framing changed.** This project began as "does SSL pretraining improve cross-dataset PD detection?" The experiments did **not** support that. What they revealed instead (a measurement confound + a calibration story + a clean SSL-negative) is a stronger, more honest contribution. The headline numbers in older commits (89% / 92%) are now known to be site-confounded — see below.

## Research questions

1. **RQ1** — If we train on EEG from some hospitals and test on a *held-out* hospital, can the model still detect PD? *(generalization)*
2. **RQ2** — Does self-supervised pretraining on large unlabeled EEG improve that cross-hospital transfer?
3. **RQ3** — Is the high accuracy reported in this field actually measuring Parkinson's — or the site?

**Answers:** RQ1 — yes, but modestly (and only after calibration). RQ2 — no, not at our scale. RQ3 — substantially the site.

## Key findings

**1 · The pooled-protocol headline is site-confounded.** A **site-prior null** — predict each dataset's majority class using *no EEG signal at all* — scores **0.927** segment-level / **0.654** subject-level balanced accuracy on the standard pooled protocol, ≥ published models. One dataset (ds004148) is 100% healthy and 66% of the segments, so "which dataset" ≈ "the label."

**2 · Honest cross-site (leave-one-dataset-out) failure is mostly calibration, not absent signal.** Supervised representations still rank PD vs HC on an unseen site (**ROC-AUC 0.76 ± 0.03**, 3 seeds, 19-ch). Fixed-0.5 balanced accuracy collapses (it predicts one class), but a threshold chosen on the *training* sites — fully deployable — recovers **0.64 ± 0.03**; prevalence-aware reaches ~0.69.

**3 · Self-supervised pretraining does not help cross-site transfer.** Robust across linear-probe, fine-tune, and data-efficiency evaluations and multiple seeds:

| Method (19-ch LODO, cross-site AUC) | ROC-AUC |
|---|---|
| Supervised | **0.763 ± 0.034** |
| SSL · OpenNeuro pretrain | 0.581 ± 0.002 |
| SSL · TUH pretrain (disjoint) | 0.526 ± 0.003 |

Training-time augmentation also gives no lift (0.763 vs 0.764). In the data-efficiency sweep, SSL fine-tune ≈ from-scratch at every label budget (10/25/100%).

## Method

### Protocols
- **Combined N-LNSO** — the field's pooled protocol (all 4 datasets, subject-level CV). Site-confounded; we report the site-prior null beside it.
- **LODO (leave-one-dataset-out)** — train on 3 sites, test on the held-out 4th. The honest cross-site test.
- **Site-prior null** — no-EEG, dataset-majority baseline (`src/honest_eval.py`).
- **Metrics** — subject-level balanced accuracy (median ± IQR), **ROC-AUC** (threshold-independent), bootstrap CIs; calibration policies (fixed / train-transferred / prevalence-matched / oracle).

### Channel harmonization
The OpenNeuro-derived 29-channel set leaves **14/29 channels dead in TUH** (clinical 10-20 montage). We use a **19-channel TUH∩OpenNeuro montage** (verified 19/19 alive), with an old→new 10-20 name remap (T3→T7, T4→T8, T5→P7, T6→P8). Channel set is toggled via `SJJI_CH_SET={19,29}`.

### Architecture
**TransformEEG** (Del Pup et al. 2025): per-channel depthwise Conv1D tokenizer → 2-layer Transformer → AdaptiveAvgPool. Input `(B, C, 4000)` @ 250 Hz (16 s windows, 25% overlap), features = `C × 4`. SSL: VICReg with a 2-layer projector.

## Datasets

**Labeled (PD vs HC, 270 subjects):** [ds004148](https://openneuro.org/datasets/ds004148) (HC test-retest), [ds002778](https://openneuro.org/datasets/ds002778) (UC San Diego PD), [ds003490](https://openneuro.org/datasets/ds003490) (3-Stim), [ds004584](https://openneuro.org/datasets/ds004584) (PD).
**Unlabeled pretraining:** [TUH-EEG Corpus](https://isip.piconepress.com/projects/tuh_eeg/) (clinical, requires access).

## Repo structure

```
experiments/
  baseline_combined.py    # supervised combined-N-LNSO baseline (+ site-prior null, calibration)
  lodo_eval.py            # leave-one-dataset-out harness (supervised / probe / fine-tune;
                          #   SJJI_CH_SET, LODO_SEED, LODO_AUGMENT, LODO_INIT_ENCODER, LODO_LABEL_FRAC)
  ssl_29ch_local.py       # VICReg pretrain + combined-N-LNSO probe (OpenNeuro)
  tuh_pretrain_lodo.py    # TUH-only (disjoint) VICReg pretrain for LODO probing
src/
  honest_eval.py          # site_prior_null, subject-level metrics, calibration_report, bootstrap CIs
  model.py · pretrain.py · finetune.py · preprocess.py (+ COMMON19/COMMON29 montages, naming remap)
docs/
  related_works.md            # related-works scaffold (students)
  tuh_lodo_experiment_plan.md # the site-disjoint full-scale TUH × LODO plan
paper/OUTLINE.md          # paper outline (honest spine)
dashboard/index.html      # the team findings dashboard
```

## Reproduce (local, Apple GPU / CUDA)

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
# Data expected at data/processed_unified/  (set DATA_DIR to override)

export SJJI_CH_SET=19
python experiments/lodo_eval.py --mode supervised                 # honest cross-site baseline + calibration
python experiments/lodo_eval.py --mode probe \
    --encoder results/ssl/pretrained_encoder_19ch_opennero.pt     # SSL probe under LODO
python experiments/tuh_pretrain_lodo.py                           # pretrain a disjoint TUH encoder (then probe)
```

## Status & next

- ✅ Confound, calibration, and SSL-negative — established locally, seeded.
- ⏳ Paper writeup — targeting the **MICCAI 2026 AMAI workshop** (Springer LNCS, 8 pages), deadline **June 25, 2026**. Methods/Results/Discussion drafted ([`paper/draft.md`](paper/draft.md)); Related Work + figures in progress.
- ⏳ **Full-scale TUH pretrain** — *optional, confirmatory only*. AWS new-account GPU quota was **denied** (insufficient usage history); if pursued, it runs on a rented GPU pulling TUH fresh from NEDC (no AWS). The fair small-scale eval was already flat, so this would strengthen the negative ("even at scale"), not change the story. See [`docs/tuh_lodo_experiment_plan.md`](docs/tuh_lodo_experiment_plan.md).

## References
- TransformEEG: Del Pup et al. (2025) — [GitHub](https://github.com/MedMaxLab/TransformEEG)
- VICReg: Bardes et al. (2022) — [arXiv](https://arxiv.org/abs/2105.04906)
- TUH EEG Corpus: Obeid & Picone (2016)
