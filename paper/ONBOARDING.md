# SJJI EEG Project — Team Onboarding

Welcome aboard. Read this before our kickoff call and you'll know what we're building, *why* the approach looks the way it does, and where you can plug in from day one. It's meant to be skimmable in ~10 minutes — we'll go deeper live.

---

## 1. What are we building?

We're training a model that detects **Parkinson's Disease (PD) from EEG** — brainwave recordings taken from scalp electrodes.

The hard part isn't getting good accuracy on one dataset. It's that **a model trained at one hospital falls apart at a different hospital.** It quietly learns the *recording setup* — the amplifier, the electrode montage, the local noise — instead of the actual disease signal. So it looks great in the lab and fails in the clinic.

Our fix is the same recipe that powers modern AI: **pretrain on a large pile of unlabeled data first, then fine-tune on the small labeled task.**

> Analogy: you teach someone to *read* before you ask them to read medical charts. The general skill transfers, and you need far fewer labeled charts to get them useful.

---

## 2. Why this problem is real (the motivating experiment)

Before doing anything clever, we ran a sanity check: **can a model tell which dataset a recording came from, with the disease labels hidden?**

It hit **94.8% balanced accuracy** at guessing the *hospital/dataset of origin*. That's the whole problem in one number — the recordings carry a loud "site fingerprint" that has nothing to do with Parkinson's. Any normal supervised model will happily latch onto that shortcut. Beating it is the point of the project.

*(Result: `results/fingerprint/dataset_fingerprint_balanced_*.json`, balanced_accuracy = 0.948.)*

---

## 3. The research cycle we're in — and why it matters beyond this project

This is the most useful thing to understand for the kickoff, because almost every task you'll touch lives somewhere in this loop. We're not just "training a model" — we're running the standard **foundation-model research cycle**, scaled down to clinical EEG:

```
   ┌──────────────────────────────────────────────────────────────┐
   │  1. BASELINE      Build an honest supervised model.           │
   │                   Know exactly what "no SSL" gets you.        │
   │        ↓                                                       │
   │  2. DIAGNOSE      Find the real failure mode, not the         │
   │                   convenient one. (→ the 94.8% fingerprint)   │
   │        ↓                                                       │
   │  3. HYPOTHESIZE   SSL pretraining should learn site-robust    │
   │                   features → better cross-hospital transfer.  │
   │        ↓                                                       │
   │  4. PILOT         Cheap, small-scale test for *signal*.       │
   │                   Is the effect even there? (Yes — see §4.)   │
   │        ↓                                                       │
   │  5. SCALE         Pretrain on a big disjoint unlabeled corpus  │
   │                   (TUH, hundreds of thousands of segments).    │
   │        ↓                                                       │
   │  6. EVALUATE      The real test: train on hospitals A+B,       │
   │                   test on hospital C. Cross-dataset.          │
   │        ↓                                                       │
   │  7. TIGHTEN       Kill confounds, fix the protocol, re-run.    │
   │                   (Most of the actual work lives here.)       │
   └───────────────────────────── ↺ ──────────────────────────────┘
```

**Why this is the cycle, and what it means in wider ML:**

- **Pretrain → fine-tune is the dominant paradigm in modern ML.** It's how LLMs work (pretrain on the internet, fine-tune on a task), how vision worked with ImageNet, how speech and protein models work. The bet everywhere is the same: *labels are scarce and expensive, but raw data is abundant — so learn structure from the abundant stuff first.* Clinical EEG is a near-perfect case for this: labeled PD recordings number in the thousands; unlabeled EEG exists in the millions.

- **Self-supervised learning (SSL) is how you pretrain without labels.** Methods like SimCLR / VICReg train the model to give the *same* representation to two augmented views of the same signal. No human annotation needed — the data supervises itself. This is what lets us exploit the giant unlabeled TUH corpus.

- **Why we expect it to fix the site-shift problem:** a model pretrained on huge, diverse EEG sees many recording setups before it ever sees a Parkinson's label. That broad exposure pushes it toward *general neural patterns* and away from any single site's fingerprint — which is exactly the robustness clinical deployment needs.

- **Why the loop, not a straight line:** the hardest part of ML research isn't the model — it's making the *evaluation trustworthy*. A huge fraction of our commits are bug fixes and protocol tightening (data leakage between pretrain and test, broken cross-validation splits, etc.). A flashy number from a leaky evaluation is worth nothing. Step 7 is where credibility is won or lost, and it's where careful new contributors add a lot of value.

If someone asks "what's the project about" in one sentence: **we're testing whether the foundation-model recipe — pretrain on lots of unlabeled brain data, then fine-tune — makes Parkinson's detection actually generalize across hospitals.**

---

## 4. Where we are right now (honest current results)

All numbers below use our rigorous protocol (**combined N-LNSO**: pool all PD subjects, leave whole groups of subjects out for testing, never letting a test subject leak into training).

| Stage | What it is | Median balanced accuracy |
|---|---|---|
| **TransformEEG paper** (Del Pup et al. 2025) | The supervised model we build on | 78.5% |
| **Our supervised baseline** | Same architecture, correct protocol, no SSL | **89.1%** |
| **SSL pilot** (VICReg pretrain + linear probe) | Small-scale, pretrain data overlaps the test data | **92.3%** |

Two honest caveats so nobody over-reads these:
1. **The SSL pilot is a lower bound, not a clean result.** Its pretraining data overlaps with the evaluation data, so the encoder has "seen" the test distribution. The real number comes from pretraining on **TUH only** (zero overlap with the PD datasets) — that's the experiment we're spinning up next.
2. **These are within-cohort (N-LNSO), not yet cross-dataset.** The headline clinical test — *train on hospitals A+B, test on hospital C* — is step 6 and still pending. That's the number that ultimately matters.

So: the supervised baseline already **beats the paper by ~10 points**, and SSL adds signal on top. Now we scale and run the clean, disjoint, cross-hospital version.

---

## 5. The datasets

| Dataset | What it is | Subjects | Role |
|---|---|---|---|
| ds002778 | PD patients + healthy controls | 31 | Labeled evaluation |
| ds003490 | PD patients + healthy controls | 50 | Labeled evaluation |
| ds004584 | PD patients + healthy controls | 149 | Labeled evaluation |
| ds004148 | Healthy controls only | 29 | Unlabeled pretraining |
| **TUH EEG corpus** | Huge general hospital EEG (not PD-specific) | ~69k recordings (~1.2 TB) | Unlabeled pretraining at scale |

The first three are how we *grade* the model. The last two are what we *pretrain* on. TUH is the engine for the "scale" step — it's deliberately **disjoint** from the PD data so there's no leakage.

> Note: PPMI was investigated as a 4th evaluation site and **dropped** — it turns out PPMI has no EEG data (MRI/SPECT/clinical only). If you see it in old drafts, it's stale.

---

## 6. Where help is needed right now

**No code required for any of these.** These are the highest-leverage things for new contributors this week.

### Task A — Related works: domain shift in biosignal ML
Find 8–12 papers on dataset bias / domain shift / site artifacts in EEG or ECG classification. Starting queries:
- "EEG domain shift generalization"
- "biosignal dataset bias site artifact"
- "cross-dataset EEG classification generalization"

For each, note: the task (which disease/signal), what *caused* the shift (hardware/site/population), what they proposed to fix it, and whether they evaluated cross-dataset. Drop notes in the `paper/` folder or the shared doc.

### Task B — Related works: SSL for EEG
Read and summarize:
- **SelfEEG** (Del Pup et al., 2024) — the SSL library we actually use
- **LaBraM** (Jiang et al., 2024) — large-scale SSL EEG foundation model
- **BIOT** (Yang et al., 2023) — cross-data SSL for biosignals

For each: which SSL method, what data, what downstream task, what results, and was it ever applied to a neurological-disease classification?

### Task C — Paper draft review
Paper structure lives in `paper/OUTLINE.md`. Once related works land, we need a pass: does our framing still hold, and which citations are we missing?

---

## 7. How to run things (optional — writing tasks come first)

```bash
git clone https://github.com/edward-lcl/sjji-eeg
cd sjji-eeg
pip install -r requirements.txt        # Python 3.10+

ls data/processed_unified/             # check data is present
python experiments/baseline_combined.py # supervised baseline (~30min GPU, ~2-3h CPU)
```

Large-scale pretraining runs on AWS SageMaker (GPU), not your laptop — ask Edward before launching anything cloud-side; it costs money and we're mid-migration to a new AWS account.

---

## 8. Key files to read first

| File | What it is |
|---|---|
| `README.md` | Current project status + headline results |
| `RESEARCH.md` | Full research compass — thesis, roadmap, open questions |
| `EXPERIMENT_LOG.md` | Every experiment we've run, with results and compute hours |
| `paper/OUTLINE.md` | Paper structure |

---

## 9. Glossary (quick version)

- **EEG** — electroencephalogram; electrical brain activity measured at the scalp.
- **Balanced accuracy** — accuracy corrected for class imbalance (50% = chance, 100% = perfect).
- **SSL / self-supervised learning** — training without labels; the data's own structure is the supervision signal.
- **SimCLR / VICReg** — SSL methods that train the model to give matching representations to two augmented views of the same signal.
- **Pretrain → fine-tune** — learn general features on lots of unlabeled data, then specialize on the small labeled task. The foundation-model recipe.
- **Linear probe** — freeze the pretrained encoder, train only a simple linear classifier on top; tests whether the learned representations are actually useful.
- **N-LNSO** — leave-N-subjects-out cross-validation; test subjects are *never* seen in training. Our within-cohort protocol.
- **Cross-dataset eval** — train on hospitals A+B, test on hospital C. The real clinical generalization test.
- **Site fingerprint / domain shift** — the recording-setup signal a model latches onto instead of the disease. The thing we're fighting.

---

## Questions?

Drop them in Slack or ping Edward. We'll walk through this doc together on the call.
