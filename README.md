# SJJI — SSL-Enhanced TransformEEG for Parkinson's Disease Detection

Self-supervised pretraining of TransformEEG on unlabeled EEG data (TUH-EEG corpus) followed by fine-tuning on labeled Parkinson's disease EEG datasets.

## Research Question

Can self-supervised pretraining improve the generalizability of TransformEEG for Parkinson's disease detection?

## Datasets

**Pretraining (unlabeled):**
- TUH-EEG Corpus — Temple University Hospital (requires access request)
- Fallback: OpenNeuro general EEG datasets

**Fine-tuning (labeled PD):**
- ds004148 — EEG test-retest
- ds002778 — UC San Diego Parkinson's
- ds003490 — EEG 3-Stim
- ds004584 — Parkinson's EEG dataset

## Method

1. Wrap TransformEEG encoder with SelfEEG's SimCLR contrastive learning
2. Pretrain on large unlabeled EEG corpus (TUH-EEG)
3. Fine-tune on 4 labeled PD datasets (same as original TransformEEG paper)
4. Evaluate: balanced accuracy, sensitivity, specificity vs TransformEEG baseline

## Structure

```
data/
  raw/          # downloaded datasets (gitignored)
  processed/    # preprocessed .npy arrays
src/
  preprocess.py # MNE-based preprocessing pipeline
  model.py      # TransformEEG encoder + SSL wrapper
  pretrain.py   # SimCLR pretraining loop
  finetune.py   # supervised fine-tuning + N-LNSO CV
  evaluate.py   # metrics: balanced acc, sensitivity, specificity
  utils.py
notebooks/
  explore.ipynb
configs/
  pretrain.yaml
  finetune.yaml
results/        # experiment outputs (gitignored)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Long Experiments

Run experiments through the Mac watchdog so crashes and stalls are visible:

```bash
./venv/bin/python scripts/mac_experiment_watchdog.py --name baseline_native -- ./venv/bin/python -u baseline.py
```

For overnight runs, install the watchdog as a macOS LaunchAgent:

```bash
./venv/bin/python scripts/mac_launch_experiment.py --name baseline_native --load -- ./venv/bin/python -u baseline.py
```

See [docs/EXPERIMENT_WATCHDOG.md](docs/EXPERIMENT_WATCHDOG.md).

Start the local control console:

```bash
./venv/bin/python scripts/mac_launch_console.py --load --port 8765
```

Open `http://127.0.0.1:8765` to inspect watchdog status, events, and logs.

## References

- TransformEEG: Del Pup et al. (2025), Neurocomputing. <https://github.com/MedMaxLab/TransformEEG>
- SelfEEG: Del Pup et al. (2024). <https://github.com/MedMaxLab/selfEEG>
- SSL for EEG survey: Weng et al. (2024)
