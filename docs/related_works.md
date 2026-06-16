# Related Works — working scaffold (SJJI / EEG)

> For: the two of you owning the related-works section. This is pre-seeded so you're
> **verifying + citing + summarizing**, not starting blank. Fill the blanks, fix any
> wrong details, and add papers you find. Use Semantic Scholar / Elicit / Consensus
> (links in the team doc) to pull exact authors/venue/year and the BibTeX.
>
> **Format for every row:** one paper, a 2–4 sentence summary, and a one-line
> "relevance to us." Put `[VERIFY]` on anything you haven't confirmed.

## What our paper actually argues (so you aim the related works correctly)

Our results did **not** support "SSL improves cross-dataset Parkinson's detection." The
honest findings are:
1. The pooled-4-dataset accuracy everyone reports (~89–92%) is **site-confounded** — a
   model using zero EEG (just "which dataset is this?") scores ~0.93 on the same metric.
2. Under leave-one-dataset-out (train on 3 sites, test on an unseen 4th), the apparent
   "chance" result is largely a **calibration** problem: supervised PD ranking transfers
   (AUC ~0.76) and a deployable threshold recovers ~0.64 balanced accuracy.
3. **SSL pretraining does not improve cross-site transfer** at the scales tested.

So related works should establish: (a) how the field evaluates PD-EEG and why pooled
accuracy can mislead (leakage / site effects), (b) what SSL-for-EEG has and hasn't shown,
and (c) calibration under domain shift. **Do not** frame it around "SSL is known to help
cross-site PD" — our own data is the counterexample.

---

## Track A — EEG evaluation, site effects, and Parkinson's-from-EEG  *(Student: ____)*

| Paper (author, year, venue) | Method / idea | Datasets | Key result | Relevance to us |
|---|---|---|---|---|
| Del Pup et al., 2025 — **TransformEEG** (arXiv 2507.07622) | depthwise-conv + transformer; 10×10 N-LNSO on 4 pooled PD datasets | ds004148/002778/003490/004584 | median bal-acc 78.45% (no aug), 80.10% (aug) | the architecture + protocol we build on; our baseline |
| Del Pup et al., 2024 — preprocessing variability in EEG-DL `[VERIFY exact title/venue]` | shows preprocessing choices swing accuracy a lot | multiple | accuracy 66%→75%→67% depending on pipeline | motivates "accuracy is fragile / protocol-dependent" |
| `[VERIFY]` the segment-leakage paper cited as ref [17] in TransformEEG | segments from same recording in train+test → near-perfect, drops >20% under cross-subject | — | biometric/segment leakage inflates accuracy | direct precedent for our **site**-leakage point (one level up from subject leakage) |
| Johnson et al., 2007 — **ComBat** (Biostatistics) | empirical-Bayes batch-effect harmonization | genomics; adopted in neuroimaging | removes site/batch effects | the standard "site effect" fix; contrast with our finding that the confound is *exploited*, not removed |
| Rockhill et al., 2021 — **ds002778** (UC San Diego PD) | dataset paper | ds002778 | 15 PD / 16 HC, resting eyes-open | one of our 4 eval sites |
| Cavanagh et al. — **ds003490** (UNM 3-Stim) `[VERIFY year]` | dataset paper | ds003490 | 25 PD / 25 HC, resting + oddball | eval site |
| Singh et al., 2023 — **ds004584** | dataset paper | ds004584 | 100 PD / 49 HC, resting | eval site (largest) |
| Wang et al., 2022 — **ds004148** (EEG test-retest) | dataset paper | ds004148 | 60 HC, resting + cognitive | our HC pool |
| ⭐ "Cross-Population Framework for Generalizable PD Detection" (arXiv 2604.23933, 2026) `[VERIFY]` | population-aware evaluation framework under distribution shift | PD-EEG, multi-site | robustness / clinical-reliability eval | **CLOSEST prior work — position against us**: we add the no-EEG site-prior null + the calibration reframe |
| ⭐ "Channel-Selected Stratified Nested CV for PD-EEG" (arXiv 2601.05276, 2026) `[VERIFY]` | unified eval (subject stratification, windowing, channel selection) to fix methodological flaws | 3 PD datasets | 80.6% cross-dataset | **CLOSE to us** — overlaps our channel + rigor work; we differ via the site-prior null, calibration, and the SSL ablation |
| GEPD: GAN-Enhanced Generalizable PD-EEG (arXiv 2508.14074, 2025) `[VERIFY]` | GAN augmentation for cross-dataset transfer | PD-EEG | cross-dataset gains | a *generative* take on the same problem; contrast with our (negative) SSL result |

> ⭐ **Novelty positioning (Saanvi — read this).** The field is actively working on cross-dataset PD-EEG and "fixing the evaluation." So our contribution is NOT "we noticed evaluation matters" — others have too. Our *distinct* contributions are: **(1)** the **site-prior null** (a no-EEG model scoring 0.93 quantifies the confound — most papers don't report this), **(2)** the **calibration reframe** (cross-site failure is a *threshold* problem, recoverable to ~0.64), and **(3)** a clean **SSL/augmentation negative**. In Related Work, name the two ⭐ papers explicitly and state in one sentence how we go further. If either already does one of our three things, flag it immediately — that changes our framing.

## Track B — Self-supervised learning for EEG + calibration under domain shift  *(Student: ____)*

| Paper (author, year, venue) | Method / idea | Datasets | Key result | Relevance to us |
|---|---|---|---|---|
| Jiang et al., 2024 — **LaBraM** (ICLR) | large-scale EEG foundation model, masked prediction | many EEG corpora | SOTA on several EEG tasks | the "SSL at scale helps EEG" claim we're testing for PD cross-site |
| Yang et al., 2023 — **BIOT** (NeurIPS) | biosignal transformer, cross-data tokenization | multiple biosignals | cross-dataset transfer | most relevant SSL cross-data precedent |
| Kostas et al., 2021 — **BENDR** (Front. Hum. Neurosci.) | BERT-style contrastive SSL for EEG | TUH + downstream | transfers to small downstream sets | template for "pretrain on TUH, fine-tune on small task" — exactly our setup |
| Del Pup et al., 2024 — **SelfEEG** `[VERIFY]` | SSL library for EEG (SimCLR/VICReg/etc.) | — | tooling, no PD-specific eval | the library our pretraining uses |
| `[VERIFY]` **EEGPT** (2024) | EEG foundation model | many | general EEG | another "scale helps" data point |
| Guo et al., 2017 — **On Calibration of Modern NNs** (ICML) | temperature scaling | vision | NNs are miscalibrated; temp scaling fixes it cheaply | underpins our calibration recovery; try temp-scaling on our cross-site scores |
| `[ADD]` domain-shift / OOD calibration paper (find 1–2) | calibration degrades under shift | | | supports "fixed 0.5 threshold fails across sites" |
| `[ADD]` 1 paper on whether SSL helps *domain generalization* (any modality) | | | | directly relevant to our negative result |

---

## Dataset characterization table  *(Track A deliverable — pre-filled from TransformEEG Table 1 + our own digging; VERIFY each cell)*

| Dataset | Subjects (PD/HC) | Task | Native channels & ref | Sampling | Role for us | Quirks we found |
|---|---|---|---|---|---|---|
| ds004148 | 0 / 60 | resting (eyes O/C) + cognitive | 64ch, FCz ref | 500 Hz | HC pool | paper uses session-1 resting-300s only; **we currently include all tasks** (→ more HC, worse site balance) |
| ds002778 | 15 / 16 | resting, eyes-open | 41ch, CMS/DRL ref | 512 Hz | PD eval | PD recorded off- & on-med; off-med used |
| ds003490 | 25 / 25 | resting + auditory oddball | 64ch, CPz ref | 500 Hz | PD eval | off-med used |
| ds004584 | 100 / 49 | resting, eyes-open | 63ch, Pz ref | 500 Hz | PD eval | largest; PD-majority |
| TUH-EEG | unlabeled (~69k recordings) | clinical (mixed pathology) | ~21–33ch, **old 10-20 naming (T3/T4/T5/T6)**, tcp montages | varies (we resample) | SSL pretrain | clinical ≠ resting-state research EEG (domain gap); only **19/29** common channels survive (we redefined a 19-ch montage; T3→T7 etc. remap) |

**Two things in this table are our own contributions to document precisely:** the 19-channel
TUH∩OpenNeuro montage (and why 10 channels are absent in TUH), and that the field's
combined protocol is site-confounded. Cite our `docs/` notes for these.

---

## Practical notes
- One row per paper; keep summaries to 2–4 sentences; always fill "relevance to us."
- Collect BibTeX in Zotero as you go (team doc has the link).
- Flag anything you can't verify with `[VERIFY]` and we'll resolve it at check-in.
- Target ~8–12 solid papers per track for a first pass.
