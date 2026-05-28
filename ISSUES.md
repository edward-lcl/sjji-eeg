# ISSUES.md — Known Bugs and Blockers

---

## CRITICAL: Channel Cyclic Padding Corrupts Baseline

**File:** `src/preprocess.py`, `align_channels()` function
**Status:** Open
**Priority:** CRITICAL — blocks all valid results

### What's Happening
```python
def align_channels(raw, target_n: int = TARGET_CHANNELS):
    n = len(raw.ch_names)
    if n >= target_n:
        raw.pick(raw.ch_names[:target_n])
    else:
        data, times = raw.get_data(return_times=True)
        pad = np.tile(data, (target_n // n + 1, 1))[:target_n]  # ← BUG
        ...
```

When a recording has fewer than 61 channels (e.g., 32 channels), `np.tile` repeats the data cyclically to reach 61 channels. The encoder then receives 29 duplicate channels alongside the real 32.

### Why This Breaks Everything
TransformEEG's Conv1DEncoder uses **depthwise convolution** (`groups=Chans`), which applies a separate convolutional filter per channel. Duplicate channels produce identical features, creating a redundant and misleading representation. The transformer then operates on this corrupted token set.

This likely explains:
- Balanced accuracy at 52% (random chance) rather than 78.45%
- Sensitivity collapsed to near-zero (model predicts HC for everything)
- Specificity near 1.0 (consistent with default-HC prediction)

### What TransformEEG Actually Does
The original paper uses **per-dataset channel-specific tokenization**. Each dataset is preprocessed with its own channel count, and the encoder is instantiated separately per dataset with `Chan=N` matching that dataset's actual channel count.

From the TransformEEG paper: channel-specific tokenization means each channel gets its own convolutional tokenizer — the architecture is explicitly designed to handle variable channel counts.

### Fix Options

**Option A (recommended): Per-dataset encoders with actual channel counts**
- Preprocess each dataset retaining its actual channels (no padding)
- Instantiate `build_encoder(Chan=actual_chan_count)` per dataset
- Train each dataset's encoder separately for the supervised baseline
- Upside: faithful to original paper's approach, no information corruption
- Downside: cross-dataset eval needs a channel-alignment step at test time

**Option B: Channel subset selection**
- Select the smallest common channel set across all datasets (e.g., 10-channel 10-20 system subset)
- All datasets use the same channels → same encoder
- Upside: clean unified model
- Downside: loses information, may underperform the full-channel approach

**Option C: Channel interpolation to common montage**
- Use MNE's `set_montage` + `interpolate_bads` to interpolate all datasets to a standard 64-channel 10-20 montage
- Upside: principled, standard in EEG literature
- Downside: interpolation adds noise, more complex preprocessing

**Recommendation:** Start with Option A (per-dataset, actual channels) to reproduce the paper's baseline. Then explore Option C for the cross-dataset model that needs a unified representation.

---

## HIGH: Training Protocol Mismatch

**File:** `baseline.py`
**Status:** Open
**Priority:** HIGH

### Current baseline.py settings:
```python
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
N_OUTER = 5
```

### What configs/finetune.yaml specifies (intended):
```yaml
training:
  epochs: 50
  batch_size: 32
  lr: 1e-3
  n_outer_folds: 10
```

### TransformEEG paper settings (from paper/original_context):
- 10-fold N-LNSO
- Need to verify exact epoch count and batch size from paper

### Fix:
Update `baseline.py` to use `N_OUTER=10`, `EPOCHS=50`, `BATCH_SIZE=32` to match config and paper.

---

## HIGH: HC Augmentation Protocol Unclear

**File:** `baseline.py`, `mode_per_dataset()`
**Status:** Open — needs verification against paper
**Priority:** HIGH

### What we do:
All 12,369 segments from ds004148 (HC-only dataset) are added as training HC in every N-LNSO fold.

### What the paper likely does:
The TransformEEG N-LNSO protocol uses the dataset's own HC subjects for training. ds004148 may be used as additional unlabeled context, not as labeled HC augmentation.

### Risk:
If we're adding 12k extra HC segments to training but only a few hundred PD segments, we create extreme class imbalance that the BCEWithLogitsLoss doesn't compensate for (no `pos_weight`).

### Fix:
1. Add `pos_weight` to BCEWithLogitsLoss proportional to class imbalance
2. Verify HC augmentation approach against the MedMaxLab/transformeeg codebase
3. Consider downsampling HC to match PD count in training set

---

## MEDIUM: Missing Metrics

**File:** `src/evaluate.py`, `src/finetune.py`
**Status:** Open
**Priority:** MEDIUM (needed before paper submission)

### Missing:
- ROC-AUC
- PR-AUC (precision-recall)
- Bootstrap confidence intervals on all metrics
- Per-fold standard deviation (currently only mean is reported)

### Fix:
Add `roc_auc_score` and `average_precision_score` from sklearn to `compute_metrics()`. Add bootstrap CI computation in `evaluate.py`.

---

## LOW: MPS Workaround for AdaptiveAvgPool1d

**File:** `src/model.py`, `TransformEEGEncoder.forward()`
**Status:** Open (working workaround in place)
**Priority:** LOW

```python
# MPS workaround for AdaptiveAvgPool1d
dev = x.device
x = self.pool_lay(x.cpu()).to(dev)
```

This CPU round-trip slows training on Apple Silicon. Not a correctness issue. When moving to a GPU machine (for longer runs), this should be removed.
