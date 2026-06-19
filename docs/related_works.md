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

## Track B: Self-supervised learning for EEG, and calibration under domain shift  *(Student: Alex)*

Status: done. Every paper below is checked against arXiv, the official proceedings, or the publisher (June 2026) and entered in `paper/refs.bib`; the cite key is in the last column. The write-up is the two Related Work subsections in `paper/main.tex`.

**Self-supervised and foundation models for EEG**

| Paper (name, authors, year, venue) | Method / idea | Key result | Relevance to us | cite key |
|---|---|---|---|---|
| **LaBraM** (Jiang et al., 2024, ICLR spotlight; arXiv 2405.18765) | large EEG foundation model, vector-quantized masked prediction | pretrained on about 2,500 hours of EEG; strong across several downstream tasks | the "SSL at scale helps EEG" claim we test for PD cross-site | `labram2024` |
| **EEGPT** (Wang et al., 2024, NeurIPS) | 10M-parameter masked self-supervised transformer with spatio-temporal alignment | strong linear-probe results across tasks | another data point for "scale helps" | `eegpt2024` |
| **BIOT** (Yang et al., 2023, NeurIPS; arXiv 2305.10351) | biosignal transformer that tokenizes mismatched channels and lengths | learns across datasets and formats, beats baselines | the most relevant cross-data SSL precedent | `biot2023` |
| **BENDR** (Kostas et al., 2021, Front. Hum. Neurosci. 15:653659; arXiv 2101.12037) | wav2vec/BERT-style contrastive SSL on large EEG | one model handles new hardware, subjects, and tasks; fine-tunes to small sets | matches the "pretrain on a big corpus, fine-tune on a small set" recipe we use | `bendr2021` |
| **CBraMod** (Wang et al., 2025, ICLR 2025; arXiv 2412.07236) | criss-cross transformer EEG foundation model, masked reconstruction | strong results across 10 BCI tasks and 12 datasets | the newest foundation model, so the framing stays current | `cbramod2025` |
| **Banville et al.** (2021, J. Neural Eng. 18:046020; arXiv 2007.16104) | temporal-context and contrastive-predictive-coding SSL on clinical EEG | SSL features beat supervised nets when labels are scarce, and match them at full labels | the exact claim our data-efficiency experiment tests | `banville2021` |
| **SSL-for-EEG survey** (Weng et al., 2024, arXiv 2401.05446; ACM CSUR) | systematic survey of SSL for EEG | taxonomy; positions SSL as the fix for label scarcity | the source we cite for the pro-SSL view we test | `weng2024sslsurvey` |
| **SelfEEG** (Del Pup et al., 2024, J. Open Source Software 9(95):6224) | SSL library for EEG (SimCLR, VICReg, and others) | tooling, with no PD-specific evaluation | the library our pretraining uses | `selfeeg2024` |
| **VICReg** (Bardes, Ponce, LeCun, 2022, ICLR; arXiv 2105.04906) | variance-invariance-covariance SSL objective, no negative pairs | on par with other SSL methods, but simpler | the SSL objective our pretraining uses | `vicreg2022` |
| **MCLPD** (Zhang et al., 2025, ECAI 2025; arXiv 2508.14073) | multi-view contrastive SSL for cross-dataset PD-EEG, evaluated leave-one-dataset-out | reports that SSL improves leave-one-dataset-out PD transfer | the closest prior work and our main counterpoint; we cite it and qualify it | `mclpd2025` |

**Critical evaluations and domain-generalization reality checks**

| Paper (name, authors, year, venue) | Method / idea | Key result | Relevance to us | cite key |
|---|---|---|---|---|
| **Are Large Brainwave Foundation Models Capable Yet?** (Lee et al., 2025, ICML 2025; arXiv 2507.01196) | fine-tuning study of recent EEG foundation models | about 1 percent gain over compact deep nets despite far more parameters | strong support for our negative, from a main-conference paper | `lee2025lbm` |
| **EEG Foundation Models: Progresses, Benchmarking, Open Problems** (Liu et al., 2026, arXiv 2601.17883) | benchmark of 12 EEG foundation models across 13 datasets and 9 paradigms | models trained from scratch stay competitive, larger models do not generalize better, linear probing is weak | an independent benchmark that supports our negative, and very recent | `liu2026eegfm` |
| **In Search of Lost Domain Generalization** (Gulrajani & Lopez-Paz, 2021, ICLR; arXiv 2007.01434) | DomainBed, a fair domain-generalization benchmark | well-tuned ERM matches or beats nine DG algorithms | the standard reference for "a careful baseline beats fancy methods" | `gulrajani2021` |
| **Ask Your Distribution Shift if Pre-Training is Right for You** (Cohen-Wang et al., 2024, arXiv 2403.00194) | separates the cases where pretraining helps under shift | pretraining fixes poor extrapolation but does not remove dataset biases or spurious correlations | explains why our negative is expected: a site confound is a dataset bias pretraining cannot remove | `cohenwang2024` |
| **Using SSL Can Improve Robustness and Uncertainty** (Hendrycks et al., 2019, NeurIPS; arXiv 1906.12340) | auxiliary self-supervision | SSL improves OOD detection and robustness in vision | part of why SSL was expected to help here | `hendrycks2019ssl` |

**Calibration and thresholds under shift**

| Paper (name, authors, year, venue) | Method / idea | Key result | Relevance to us | cite key |
|---|---|---|---|---|
| **On Calibration of Modern Neural Networks** (Guo et al., 2017, ICML; arXiv 1706.04599) | temperature scaling | modern networks are overconfident; temperature scaling corrects it cheaply | underpins our calibration recovery and Alex's Thread-1 temperature scaling | `guo2017calibration` |
| **Isotonic calibration** (Zadrozny & Elkan, 2002, KDD; doi 10.1145/775047.775151) | isotonic regression for probability calibration | the standard origin of isotonic recalibration | the reference for Alex's Thread-1 isotonic arm | `zadrozny2002` |
| **Can You Trust Your Model's Uncertainty?** (Ovadia et al., 2019, NeurIPS; arXiv 1906.02530) | benchmark of calibration under dataset shift | accuracy and calibration both degrade under shift; a calibrator fit in-distribution does not transfer | shows that a fixed 0.5 threshold fails across sites | `ovadia2019` |
| **Robust Calibration with Multi-domain Temperature Scaling** (Yu et al., 2022, NeurIPS; arXiv 2206.02757) | fit temperature across several training domains | calibration that holds up on both in- and out-of-distribution data | backs fitting temperature on our training hospitals, and underpins Alex's Thread-1 | `yu2022multidomain` |
| **On Calibration and Out-of-Domain Generalization** (Wald et al., 2021, NeurIPS; arXiv 2102.10395) | links multi-domain calibration to OOD generalization; robust isotonic regression | multi-domain calibration implies fewer spurious correlations | backs both our calibration reframe and Alex's Thread-1 isotonic arm | `wald2021` |
| **Deployment under Prevalence Shifts** (Godau et al., 2023, MICCAI; arXiv 2303.12540) | prevalence-aware recalibration of the operating point, without target labels | restores good decisions and reliable metrics across 30 medical tasks | the clinical analogue of our prevalence-matched and train-transferred thresholds, in the same venue family | `godau2023` |

> On the handoff's question of whether anyone shows SSL helping cross-site: only MCLPD (ECAI 2025) claims it for cross-dataset PD-EEG, and under a weaker protocol than ours, so we cite it and qualify it. The large EEG foundation-model papers (LaBraM, EEGPT, BIOT, BENDR, CBraMod) report transfer under pooled or in-distribution fine-tuning, or as downstream accuracy after very large pretraining; Banville's gains are in the low-label setting and Hendrycks is in vision. The 2025 and 2026 benchmarks (Lee, Liu) find these models barely beat compact supervised baselines, and that scale does not buy better generalization, and Cohen-Wang gives the reason: pretraining does not remove a dataset confound, which is what our site shortcut is. None of them test SSL under a strict leave-one-dataset-out clinical protocol at our scale, and that gap is what our negative fills (written up in `paper/main.tex`).

> Boundary note for Saanvi (Track A): the search also turned up strong site-confound and leakage papers that fit your track rather than mine, so I left them for you to avoid double-citing. Brookshire et al. 2024 (Data leakage in translational EEG, Front. Neurosci., doi 10.3389/fnins.2024.1373515); Souza et al. 2023 (site acts as a shortcut in multisite PD MRI and survives harmonization, JAMIA, doi 10.1093/jamia/ocad171); and a ComBat-for-EEG study (Jaramillo-Jimenez et al. 2024, Clin. Neurophysiol.). They support the point that pooled accuracy is a confound.

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
