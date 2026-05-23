# Original SJJI Context (Pre-Pivot)

These documents are preserved as historical context for the SJJI project from before Edward + Andrew took it over.

## Files

- **`og_proposal.md`** — Original research proposal: "Utilizing TransformEEG Technology with SSL to improve model generalizability in Parkinson's Disease Detection." Drafted by the original team (Johnathan, Inesh Bose, Jithin Suresh, Karolina Torbus + mentor Kiran). Core idea: pretrain TransformEEG with SimCLR on TUH-EEG, fine-tune on 4 labeled PD datasets.

- **`og_paper.md`** — Full paper draft from the original team. Notably, this version pivoted toward a **LoRA fine-tuning study** rather than the SSL pretraining direction in the proposal. Title implied in the methods: "How and why does fine-tuning (particularly via LoRA) impact generalizability in TransformEEG for Parkinson's disease detection?" Contains methods on full vs LoRA vs partial fine-tuning. **Results section was empty** — "(ADD METRICS SECTION HERE)".

## Why It Was Rejected

This paper was **declined by Kevin (PI)**, forcing a pivot. The most likely reason: no implementation, no experiments, no actual results — just a paper shell with placeholders. A paper without numbers is not a paper.

## What We're Doing Differently

Our new direction (see `paper/OUTLINE.md`):

1. **Actually run experiments** — preprocessing, pretraining, fine-tuning, evaluation pipeline is implemented and ready
2. **Reframe the contribution** — "EEG foundation model for cross-dataset Parkinson's detection" instead of either "we added SSL to TransformEEG" or "we tried different fine-tuning methods"
3. **Make cross-dataset generalization the headline metric** — the eval that matters clinically, not a secondary concern
4. **Open-source pipeline** — full reproducibility

The science still builds on TransformEEG and uses SimCLR pretraining, but the framing is more ambitious and the execution is the difference between a rejected paper and an accepted one.
