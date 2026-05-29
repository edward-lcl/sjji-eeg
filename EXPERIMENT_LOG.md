# SJJI Experiment Log

Tracking compute hours for Algoverse reporting.
Hardware: Apple MacBook Pro (MPS / Apple Silicon GPU).

---

## Session 1 — 2026-05-24 / 2026-05-25

| Experiment | Started (UTC) | Ended (UTC) | Duration | Device | Status | Notes |
|---|---|---|---|---|---|---|
| baseline_native | 2026-05-24 21:03 | 2026-05-25 00:48 | **3h 45m** | MPS | ✅ Done | Corrected native-channel supervised baseline; 3 datasets × 10 folds; aggregate bal_acc 0.572 |
| baseline_unified | 2026-05-25 16:05 | 2026-05-25 23:10 | **7h 05m** | MPS | ✅ Done | 64-channel unified supervised baseline + cross-dataset; per-dataset agg bal_acc 0.538; cross-dataset agg 0.503 (near chance) |

### Session 1 total: ~15h 50m compute

---

## How to update this log

When an experiment finishes, add its actual end time and final duration. Pull from:

```bash
cat runs/watchdog/<name>.status.json | python3 -c "
import json,sys,datetime
d=json.load(sys.stdin)
start=datetime.datetime.fromisoformat(d['started_at'])
end=datetime.datetime.fromisoformat(d['updated_at'])
print('elapsed:', end-start)
"
```

---

## Cumulative total (all time)

| Period | Compute Hours |
|---|---|
| 2026-05-24 / 2026-05-25 | ~8h 45m (running) |
| **Total** | **~8h 45m** |

## Session 2 — 2026-05-25/26 overnight

| Experiment | Started (UTC) | Ended (UTC) | Duration | Device | Status | Notes |
|---|---|---|---|---|---|---|
| fingerprint (unbalanced) | 2026-05-26 01:23 | 2026-05-26 01:49 | **26m** | MPS | ✅ Done | Dataset origin classifier; bal_acc 0.579 — majority-class collapse to ds003490 |
| fingerprint_balanced | 2026-05-26 02:08 | 2026-05-26 07:03 | **~5h** | MPS | ✅ Done | Results at `results/fingerprint/dataset_fingerprint_balanced_20260525_230328.json` |
| subject_aggregation | 2026-05-26 02:08 | 2026-05-26 09:26 | **~7h** | MPS | ✅ Done | Results at `results/subject_aggregation/subject_aggregation_20260526_042553.json` |
| ssl_pilot | 2026-05-26 02:08 | 2026-05-26 10:44 | **~8.5h** | MPS | ✅ Done | SimCLR 100 epochs on OpenNeuro only (~18k segs); per-dataset agg bal_acc 0.574; cross-dataset 0.556; encoder at `results/ssl/pretrained_encoder.pt` |

### Session 2 total: ~21h compute

---

## Session 3 — 2026-05-27/28 (SageMaker)

### Data infrastructure
| Step | Status | Notes |
|---|---|---|
| TUH EEG raw → S3 | ✅ Done | 69,205 EDF files, 1.6 TiB at `data/raw/tuh_eeg/v2.0.1/edf/` in `sagemaker-us-east-2-506145782110` |
| OpenNeuro 4 datasets → S3 processed | ✅ Done | ds002778/003490/004148/004584 at `data/processed_unified/` — 64ch unified, labels.csv included |
| TUH EEG → S3 processed | 🔄 In progress | `sm_preprocess.py` reads raw EDF from S3, writes .npy back. Only ~1,076 files done so far. `sjji-eeg-preprocess-1780064052` running as of 2026-05-29. |

### SageMaker training runs
| Job | Started | Ended | Duration | Instance | Status | Notes |
|---|---|---|---|---|---|---|
| sjji-eeg-pretrain-1779893153 | 2026-05-27 | — | — | g5.4xlarge | ❌ Failed | Early run — code/config issues |
| sjji-eeg-pretrain-1779902972 | 2026-05-27 | — | — | g5.4xlarge | ❌ Failed | Early run — code/config issues |
| sjji-eeg-pretrain-1779975122 | 2026-05-28 09:32 | — | — | g5.4xlarge | ❌ Failed | Disk space — raw TUH channel too large for instance |
| sjji-eeg-pretrain-1779978279 | 2026-05-28 10:24 | — | — | g5.4xlarge | ❌ Failed | `ModuleNotFoundError: baseline` not in source staging |
| sjji-eeg-pretrain-1779982169 | 2026-05-28 11:29 | — | — | g5.4xlarge | ❌ Failed | Tensor shape mismatch — TUH variable channels, truncate-only in FlatDataset |
| sjji-eeg-pretrain-1779983239 | 2026-05-28 11:47 | 2026-05-28 23:53 | **~12h** | g5.4xlarge spot | ✅ **COMPLETE** | **100 epochs, final loss 4.3297**. Encoder at `runs/sjji-eeg-pretrain-1779983239/output/.../model.tar.gz` (2.96MB). This is the baseline pretrained encoder. |
| sjji-eeg-pretrain-1780059304 | 2026-05-29 08:56 | 2026-05-29 10:51 | **~2h** | g5.4xlarge spot | 🛑 Stopped (manual) | Redundant — same dataset as 1779983239. Stopped to save credits. |

### Current state (2026-05-29)
- **Pretrained encoder (pilot scale):** ✅ Complete. 100 epochs on OpenNeuro ~18k segments. Loss 4.3297. S3: `runs/sjji-eeg-pretrain-1779983239/output/`
- **Preprocessing (TUH full scale):** 🔄 Running. `sjji-eeg-preprocess-1780064052` on ml.r5.4xlarge. ~68k EDF files remaining. Resume-safe. ETA: 2026-05-29 evening (may need 2× 6h runs).
- **Next step:** After preprocessing completes → run `sjji-eeg-pretrain` warm-started from existing encoder on full TUH-scale processed dataset. Use `--max-hours 24`, 10–20 epochs (not 100).

### Session 3 compute (SageMaker)
| Instance | Hours | Cost est. |
|---|---|---|
| g5.4xlarge spot (pretrain attempts) | ~18h total across failed+completed runs | ~$9–11 |
| r5.4xlarge on-demand (preprocess, multiple attempts) | ~12h across failed runs | ~$12 |
| r5.4xlarge on-demand (preprocess, current run) | ongoing | ~$1.60/hr |

---

## Key results so far

| Experiment | bal_acc (per-dataset) | bal_acc (cross-dataset) | Notes |
|---|---|---|---|
| Supervised baseline (native channels) | 0.572 | — | ds002778/003490/004584 |
| Supervised baseline (64ch unified) | 0.538 | 0.503 | Near chance on cross-dataset |
| SSL pilot (OpenNeuro only, linear probe) | 0.574 | 0.556 | SimCLR encoder pretrained on ~18k segs |

**The SSL pilot shows marginal improvement over supervised on cross-dataset (0.556 vs 0.503). The hypothesis is that TUH-scale pretraining will meaningfully widen this gap.**
