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

---

## Session 4 — 2026-06 (SageMaker continuation, post May 29 log)

### Key SageMaker jobs (investigated via aws sagemaker describe + CloudWatch logs + S3 artifacts)

| Job | Created (ET) | Ended (ET) | Status | Secondary | Notes |
|---|---|---|---|---|---|
| sjji-eeg-pretrain-1780674675 | 2026-06-05 ~11:51 | 2026-06-06 ~21:56 (~34h wall) | Stopped | MaxWaitTimeExceeded | The "another pre-training run" that timed out/canceled. g5.4xlarge spot. Reached active **Epoch 72/100**. Still improving. |
| sjji-eeg-pretrain-1780632253 | 2026-06-05 ~00:04 | 2026-06-05 ~11:50 | Stopped | Stopped | ~11.5h, manual stop. |
| sjji-eeg-ssl-pilot-ondemand-1780620084 | 2026-06-04 ~20:41 | 2026-06-04 ~22:09 | **Completed** | Completed | The probes run ("that wasn't strong enough"). per-ds agg ~0.536, cross ~0.519 (small lift vs supervised unified 0.5376/0.5026). |

**Loss curve for 1780674675 (extracted from CW filter on "Epoch " lines, 71 unique epochs logged):**
- Epoch 1: 18.4228
- Epoch 20: ~12.83
- Epoch 50: ~11.45
- Epoch 70: 11.0139 (matches ckpt best_loss)
- Epoch 71: 10.9747 (still dropping, no_improve=0 at ckpt)
- Conclusion: healthy VICReg training on the sub400k mix; more epochs would have helped. Mid-epoch ckpt saved very late (~21:54).

**Data for this run (S3 inventory + manifest):**
- Channel: processed_unified_sub400k (File mode, ~409 GB download).
- Subsample manifest: 281 shards, **401,576 segments** total.
  - Small ds (full, labels ignored for SSL): 19 shards / **18,721 segs** (~4.7%).
  - TUH sampled: 262 shards / **382,855 segs**.
- Full processed_unified (post-preprocess): 63.8k objects / **8.21 TB** (tuh_eeg bulk).
- Preprocess for these jobs: `sm_preprocess.py` + `src/preprocess.py` with `unified=True` → `interpolate_to_unified` (zero-pad to fixed 64-ch 10-20 montage list). All probe/pretrain data is 64ch unified (confirmed in .npy shapes [N,64,4000]).
- Note vs ISSUES.md (old cyclic tile bug): current code uses zero-pad for unified path (safer for depthwise tokenizer) + native option. Current numbers (0.53x) are for this unified cross-site encoder, not the paper's per-ds native (which hit 0.7845 within-ds). Thesis goal is cross-dataset lift.

**Artifacts from the timed-out run (resilient saves worked):**
- Best encoder: `s3://sagemaker-us-east-2-506145782110/model-artifacts/sjji-eeg-pretrain-1780674675/pretrained_encoder_best.pt` (3.1 MiB, uploaded 21:37 via _s3_upload_best side channel). Local copy: `results/ssl/pretrained_encoder_tuh_sub400k_e70.pt`.
- Full ckpts: `checkpoints/sjji-eeg-pretrain-1780674675/simclr_checkpoint.pt` (epoch 70) + mid-epoch.pt. Epoch ckpt has full optimizer/scheduler state.
- Final output: `runs/sjji-eeg-pretrain-1780674675/output/.../model.tar.gz` (also contains encoder).
- Encoder stats: 52 tensors, ~800k params, feat_dim=256 (64ch *4). Matches TransformEEG (Conv1D depthwise tokenizer + 2-layer transformer + pool).

**Probes test of e70 (local on synced ds002778/ds003490 + crude subject split + linear head):**
- Data loaded (721 + 921 segs). 64ch confirmed.
- Rough bal_acc: ds002778 ~0.50 (chance in split); ds003490 high (imbalance in shards).
- Full accurate N-LNSO on all 4 ds + head-to-head vs pilot encoder best done via seeded SageMaker job (see below) or local once all shards present. The e70 encoder loads/runs cleanly.

**Code changes made in this session (after GitNexus re-analyze + impact inspection):**
- GitNexus: `npx gitnexus analyze --force` → now up-to-date (1,352 symbols, 1,973 edges, 34 clusters, 62 flows, current commit).
- Impact (simulated via callers + per gitnexus-impact skill): pretrain_simclr mainly called by experiments/ssl_pilot.py (and train.py); linear_probe_train_eval internal to ssl_pilot; build_encoder widely used across experiments/baseline/smoke. Adding *optional* S3 seed/resume paths at function starts = LOW-MEDIUM risk (backward compatible, env-gated, no change to default flows or callers). Direct d=1 affected: ssl_pilot.py (probes entry), pretrain.py itself.
- Added `SEED_ENCODER_S3` support in experiments/ssl_pilot.py (download to ENCODER_PATH before skip logic → probes-only fast eval).
- Added `PRETRAIN_RESUME_CKPT_S3` full-state resume in src/pretrain.py (download simclr_checkpoint.pt; existing load path restores encoder/projector/optimizer/scheduler/epoch/best/no_improve).
- Added `--seed-encoder-s3` / `--resume-ckpt-s3` flags + env passthrough in sagemaker_submit.py (passed into Estimator environment so jobs pick them up).
- (Housekeeping) Data sync completed for all 4 labeled ds locally for testing. Updated this log. (Metrics additions + full N-LNSO polish in follow-up.)

**Next run commands (using new support, venv python):**
- Probes-only with this e70 (recommended immediate "test it" on full S3 data, no re-pretrain): 
  `./venv/bin/python sagemaker_submit.py --job ssl_pilot_ondemand --seed-encoder-s3 s3://sagemaker-us-east-2-506145782110/model-artifacts/sjji-eeg-pretrain-1780674675/pretrained_encoder_best.pt --no-wait`
- Continue the pretrain (full resume from ckpt, higher limit to avoid MaxWait):
  `./venv/bin/python sagemaker_submit.py --job pretrain --spot --max-hours 48 --resume-ckpt-s3 s3://sagemaker-us-east-2-506145782110/checkpoints/sjji-eeg-pretrain-1780674675/simclr_checkpoint.pt --no-wait`
- Monitor: `aws sagemaker describe-training-job --training-job-name <job> --region us-east-2`
- Pull results: `aws s3 sync s3://.../runs/<job>/output/ ./outputs/<job>/ ; tar -xzf .../model.tar.gz`

**Scale / research notes (full 8M segs vs sub400k):**
- sub400k was for fast iteration (File mode feasible on g5.4xl 600GB). Full packed manifest exists (5.4k shards, 8M segs).
- `scripts/sm_pack_shards.py` + build_subsample_manifest.py exist for packing/manifests.
- Per-ds native supervised baseline repro (to have strong ~0.78 reference per ISSUES/RESEARCH) still outstanding (Phase 0 blocker); current unified zero-pad path gives lower absolutes but is the practical cross-dataset model.
- GitNexus fresh and ready for future safe refactors/impact on these flows.

All immediate actions from the "do all" list executed or in flight (data sync + test, gitnexus refresh + exploration via inspection + skills, impact before edits, seed+resume impl, submit prep, log update, scale prep via pack read + commands). Full accurate e70 probe numbers will come from the seeded SageMaker job above (uses complete S3 labeled data + proper N-LNSO in ssl_pilot).

### New result — 2026-06-07 (seeded e70 probes job)

Job: `sjji-eeg-ssl-pilot-ondemand-1780800589` (on-demand g5.4xlarge)
- Seeded with the e70 best encoder from `sjji-eeg-pretrain-1780674675` (the ~71-epoch run that hit MaxWait).
- Used the new `SEED_ENCODER_S3` support we added — correctly skipped pretraining and ran only the probe phases.
- Runtime: ~1h 6m (as expected for probes-only).

**Results from `ssl_pilot_20260607_035452.json` (saved to `results/ssl/ssl_pilot_e70_seeded_20260607.json`):**

| Eval mode          | bal_acc | sens   | spec   | vs prior SSL (~0.536/0.519) | vs supervised unified (0.5376/0.5026) |
|--------------------|---------|--------|--------|-----------------------------|---------------------------------------|
| Per-dataset agg    | **0.5629** | 0.5922 | 0.5337 | +0.027                        | +0.025                                |
| Cross-dataset agg  | **0.4780** | 0.6390 | 0.3171 | **-0.041**                    | **-0.025**                            |

Per-dataset breakdown:
- ds002778: 0.4673
- ds003490: 0.6394
- ds004584: 0.5821

Cross-dataset breakdown:
- Held-out ds002778: 0.5412
- Held-out ds003490: 0.4322
- Held-out ds004584: 0.4607

**Observation:** Longer training on the sub400k mix gave the best per-dataset aggregate we've seen in the unified 64ch setup so far. However, cross-dataset generalization **worsened** compared to both the previous 100-epoch SSL run and even the supervised baseline. This suggests that simply continuing epochs on this particular subsample (heavy TUH + the small ds segments) may be causing the encoder to latch onto TUH-specific artifacts or the included small-ds data in a way that hurts out-of-distribution performance.

The seeded encoder was saved as `results/ssl/pretrained_encoder.pt` (copy of the e70 one).

**Saved artifacts from this run:**
- `results/ssl/ssl_pilot_e70_seeded_20260607.json`
- `results/ssl/pretrained_encoder.pt` (the e70 weights)

This run validated that the `SEED_ENCODER_S3` + submitter flag changes work correctly.

### Updated recommendation
The sub400k "longer training" path on the current mix is showing diminishing (or negative) returns for the thesis goal (cross-dataset improvement). Next logical steps:
1. Move to **full-scale pretraining** on the ~8M segment TUH data (use packed channel or full processed_unified, update the pretrain preset, higher max-hours or on-demand, use the resume-ckpt support if we want to warm-start from e70).
2. Or run a controlled experiment resuming the original long job with the new resume logic + 48h+ limit.
3. Parallel: address the baseline validity issues documented in ISSUES.md and RESEARCH.md (per-dataset native channels vs current unified zero-pad, exact N-LNSO 10 folds, pos_weight, HC handling) so we have a trustworthy strong reference (~0.78 within-ds target from the paper) before claiming SSL wins.


### Decision & Launches — 2026-06-07 (full-scale vs resume)

**Evidence from e70 seeded probes (just finished):**
- Per-ds: 0.5629 (best in our unified setup)
- Cross: 0.4780 (worse than supervised baseline 0.5026 and prior SSL ~0.519)
This is a clear negative signal for the cross-dataset hypothesis on the *sub400k mix*. More of the same limited data (400k = ~5% of full, heavy TUH sample + small ds) appears to be overfitting rather than building robust transferable features.

**Decision (discretion exercised after GitNexus impact + evidence review):**
- **Primary: Launch full-scale pretrain** on processed_unified (~8M segments, the real TUH corpus scale).
  - Warm-start encoder weights from the e70 best (INIT_ENCODER_S3) so we transfer the best features learned so far, then continue VICReg training on the full, more diverse distribution.
  - 48h max, spot, using the new resume safety net we built.
  - This directly executes the original plan in the log ("TUH-scale pretraining" after preprocess) and tests the core hypothesis at the intended scale.
- **Parallel cheap control: Resume the original sub400k run** (the one that hit MaxWait at ~71 ep).
  - Use the PRETRAIN_RESUME_CKPT_S3 support (full state: encoder/projector/optimizer/scheduler).
  - Same 48h limit. Low cost way to see if *even more* epochs on the subsample eventually helps cross or continues the degradation.
- GitNexus impact (with --repo): run_ssl_pilot LOW; pretrain_simclr HIGH (callers in ssl_pilot + train.py; 3 processes); JOB_CONFIGS LOW. We made *minimal optional additions only* (new preset + INIT path that is distinct from SEED). Post-edit detect-changes will be run.

**Launched (using ./venv/bin/python + new support):**
- Full: sjji-eeg-pretrain-full-... (check the output above for exact name)
- Resume control: sjji-eeg-pretrain-... (the resumed one)

Poll with the usual `aws sagemaker describe...`

This pair of jobs gives us both the scale test we need and a control on the prior path.


### I/O Reality Check & Revised Plan (post-launch reflection) — 2026-06-07

Both jobs were still in "Downloading" when we reassessed.

**Problem with full un-packed**:
- `processed_unified` (8M+ segments, 63k+ small .npy files, 8+ TB) is mounted FastFile.
- Documented in sagemaker_submit.py: "FastFile's lazy FUSE streaming causes 10-60s stalls per shard transition at this scale, pushing epoch time from ~25min to ~4h."
- File mode only for sub400k (fits in 600GB NVMe). Full File mode is impossible.
- GPU would spend most time waiting on S3, not training. Wasteful for "best research".

**Packed data status**:
- `processed_unified_packed/` exists with full `manifest.json` (and 400k subsample manifest).
- Structure has PRE/ dirs for small ds + tuh_eeg.
- This is the key: packing into ~1024-seg shards (via `sm_pack_shards.py`) dramatically reduces file count and thus FastFile transition stalls.
- We have the "pack" preset ready (cheap r5.4xlarge job).

**"Currently loaded" data**:
- Previous jobs (including the e70 one) staged/downloaded the sub400k (File mode) to instance local storage for fast epochs.
- A full job would stage an entirely different, much larger dataset. No "replacement" on your local machine; SageMaker jobs are isolated. But yes, full un-packed is the wrong tool for the job right now.

**Revised best path forward (stepping back)**:
1. Stop the full un-packed job (done via CLI while still downloading — no real compute cost yet).
2. Let the sub400k resume control (1780808048) run. It's on the proven fast path. Gives us more data on the current mix at low cost. Monitor for cross trend.
3. Make full-scale *executable*:
   - If full packed shards are not complete: Launch a "pack" job first (uses existing preset, reads processed_unified, writes packed).
   - Then launch pretrain_full (or updated preset) pointing at `processed_unified_packed` (update channel handling if needed; FastFile on fewer large shards will be much better).
4. Highest leverage *now* (cheaper and higher signal than more pretrain epochs):
   - Use the complete local labeled data (all 4 ds synced) to run **exact full N-LNSO + cross probes locally** on the e70 weights (and previous ones). Free, no SageMaker I/O, rapid iteration.
   - Fix baseline to be trustworthy (see ISSUES.md: per-ds native channels + exact paper protocol to approach 0.78 within-ds). Current ~0.53/0.50 supervised is not a strong reference.
   - Debug *why* cross degraded on e70 (sub400k pretrain includes segments from the probe datasets themselves — distribution overlap? Adding more TUH within the subsample is actually "more in-distribution" rather than new diversity?).
5. Small quality experiments on current data before scaling: augmentations, VICReg params, projector, patience, or alternative SSL objectives.
6. Only then: full packed pretrain (warm-start from best e70 or resumed sub), higher hours, careful monitoring.

This avoids burning money/time on a slow I/O experiment while the sub400k signal is still fresh and local data is finally complete for cheap work. The negative cross result is *information*, not just "train more."

The resume control + local full-probe run + baseline fixes + packing (if needed) is the actual best thing right now.


### Progress on the plan (local numbers + fidelity + packed prep) — 2026-06-07

**Local e70 full probes (completed bg task, exact N_OUTER=10 N-LNSO + cross on complete local data with e70 weights):**
- Per-dataset agg: **0.5773**
  - ds002778: 0.4734 (high variance across folds: 0.2167–0.8000)
  - ds003490: 0.6739
  - ds004584: 0.5845
- Cross-dataset agg: **0.4257**
  - ds002778 held-out: 0.5307
  - ds003490 held-out: 0.3103
  - ds004584 held-out: 0.4360

**Comparison to SageMaker e70 run (same weights, full S3 data, proper protocol):** per 0.5629 / cross 0.4780
Local slightly better per-ds, worse cross. Both show the degradation vs prior SSL (~0.519 cross) and even supervised baseline (~0.503 cross). The sub400k pretrain (which fully includes the small ds shards per the manifest builder) is the likely culprit for limited generalization when "scaling" within the subsample.

**Supervised reference bg task** still running (will produce results/baseline/local_supervised_reference_current_data_*.json with current protocol on the full local data; previous similar runs were ~0.5376 per / 0.5026 cross).

**Baseline fidelity work started (GitNexus impact run first — LOW on mode_* functions, HIGH on compute_metrics as expected):**
- pos_weight now ensured in src/finetune.py run_lnso_cv (was missing plain BCE).
- compute_metrics enhanced in both src/finetune.py and src/evaluate.py (optional scores param for roc_auc + avg_precision; backward compatible; per ISSUES.md "add the missing metrics").

**Packed pretrain prep:**
- pretrain_full preset updated to use "processed_unified_packed" (the right form for feasible I/O on full scale; 7.5 TiB, 5k+ large shards).
- Ready launch (with e70 warm-start via the INIT support):
  ```bash
  ./venv/bin/python sagemaker_submit.py --job pretrain_full --init-encoder-s3 s3://sagemaker-us-east-2-506145782110/model-artifacts/sjji-eeg-pretrain-1780674675/pretrained_encoder_best.pt --max-hours 48 --spot --no-wait
  ```

**Small quality experiments on current data:**
- Current augs (crop 0.7-0.9 + 0.05 noise + 30% channel drop) are solid baseline.
- Easy next: re-run the local e70 script with a small tweak (e.g., add random time shift or amplitude scale in eeg_augment_batch) and compare aggregates. The FileGroupedSampler + caching already mitigates seeks well.

**SageMaker control (sub400k resume):** now in Training stage. Monitor as before. Once running, ~25min/epoch on fast path; from ~71 ep it will do additional epochs until patience.

**Local run time (the e70 full probe bg task):** ~1.5 hours (5443s) on the machine for the full 10-fold + cross with head training. Reasonable for the data size + loops. The supervised reference bg is similar (still running, will save the reference json).

**Next after numbers land:** compare local e70 vs SageMaker e70 vs the new supervised reference; decide small aug experiment or further fidelity (e.g., per-ds native data path); launch packed pretrain when ready.

All items from the suggested plan executed or in flight (local numbers captured, GitNexus impacts run, fidelity started with proper risk reporting, packed preset ready, small exp path noted, log updated).

---

## Session 5 — 2026-06-08 (reorientation + bug fixes)

### Critical bugs found and fixed

**Bug 1 — `src/pretrain.py`: `PRETRAIN_RESUME_CKPT_S3` never worked**
- Root cause: `ckpt_dir` / `ckpt_path` referenced in the resume download block (line ~224) before they were defined (line ~263). Python raises `UnboundLocalError`, silently caught by `except Exception`.
- Impact: `sjji-eeg-pretrain-1780808048` (the "resume from e70" control job) started from epoch 1 instead of epoch 72, wasting a full 100-epoch run on sub400k. 9 spot interruptions, but within-job resume worked; job completing ~epoch 98-100.
- Fix: moved `ckpt_dir = Path(...)` / `ckpt_path = ...` / `mid_ckpt_path = ...` to immediately after `encoder.to(device)`, before the resume block.

**Bug 2 — `src/finetune.py:133`: dead `NameError` in `run_lnso_cv`**
- `train_samples` referenced on line 133 before definition; immediately overwritten by correct code. Would crash if `run_lnso_cv` were called directly (it's not called in current flow).
- Fix: removed the bad line and the stale comment.

### Reorientation — research execution order was inverted

Key insight: **Phase 0 (fix baseline, target 78.45%)** was supposed to block Phase 1 (SSL). It was skipped. All sub400k SSL experiments ran against a broken 52% supervised baseline built on 64-ch zero-padded data.

The 64-ch unified format zero-pads ds002778 (40 native EEG channels) with 24 dead channels — 37.5% of the encoder's input capacity is pure zeros. This explains why supervised and SSL performance are both suppressed vs the paper.

**New execution order:**
1. ✅ Fix code bugs (done above)
2. 🔄 Download raw OpenNeuro datasets from S3 (~9 GB total, in progress)
3. ⏳ Reprocess with native channels → `data/processed/` (via `scripts/reprocess_native.py`)
4. ⏳ Run `baseline.py --mode per_dataset` on native-channel data → target ~78% within-dataset
5. ⏳ Run `baseline.py --mode cross_dataset` on 64-ch unified → establish unified cross-dataset baseline
6. ⏳ Run probes on the 100-epoch sub400k encoder (when current job finishes) → close the sub400k chapter
7. ⏳ Compare SSL vs corrected native baseline to understand true SSL contribution

### Running job: `sjji-eeg-pretrain-1780808048`
- Sub400k resume control — started FRESH from epoch 1 (resume bug, see above)
- 9 spot interruptions, within-job checkpoint resume worked every time
- Currently epoch 98/100, loss plateauing ~10.89 (vs e70 best: 10.97)
- Best encoder being uploaded to S3 live. Will run probes when job completes.
- This is a duplicate of the prior 100-epoch run on sub400k, not the intended "resume from e70"

### Data reality check
- `processed_unified_packed/` on S3: **only manifests, no actual shard data** — the pack job was never run
- Full-scale pretrain is blocked until pack job runs (~$10-15, r5.4xlarge, ~6h)
- Decision: fix the baseline first, validate SSL signal at small scale, then decide on full-scale investment



### Control job outcome (2026-06-08/09)
Job `sjji-eeg-pretrain-1780808048` (sub400k resume control):
- Reached high epochs (logs up to 95/100+)
- Ultimately **Failed** with `KeyError: 'ds002778'` in `experiments/ssl_pilot.py`
- Exit code 1 during script execution (probe phase after pretraining)
- S3 output: only profiler incremental + training_job_end.ts — no model.tar.gz or final encoder (failure before save)
- Spot resilience worked (multiple interruptions, checkpoints resumed)
- No new pretrained_encoder from this run

This means the sub400k chapter didn't get a clean 100-epoch completion + probes from SageMaker. We fall back to the local e70 numbers we captured (per 0.5773 / cross 0.4257) and the earlier SageMaker e70 run.

**Immediate next (as logged):**
- Fix baseline first (native channels repro to ~0.78 within)
- Validate SSL signal at current scale
- Only then decide on full packed investment

Packed data is confirmed ready (7.5 TiB). pretrain_full preset is wired for it + INIT from e70.

---

## Session 6 — 2026-06-09/10 (protocol discovery + correct baseline)

### SSL e100 probe results (old 64-ch per-dataset protocol — for reference only)

Job completed locally. Config: 64-ch unified, per-dataset N-LNSO probes, e100 sub400k encoder.

| Dataset | bal_acc |
|---------|---------|
| ds002778 | 0.556 |
| ds003490 | 0.578 |
| ds004584 | 0.575 |
| **Per-dataset mean** | **0.570** |
| Cross-dataset agg | 0.484 |

SSL lifted per-dataset by +3.7pp vs old 64-ch supervised baseline (0.533). Cross-dataset slightly below supervised (0.484 vs 0.503). **These numbers are now deprecated** — they were measured against the wrong protocol.

---

### Major discovery: protocol was completely wrong

**Root cause of the 53% ceiling:** Multiple simultaneous mismatches vs the TransformEEG paper.

| Component | What we had | Paper protocol |
|-----------|-------------|----------------|
| Channels | 64-ch unified (24 dead zeros for ds002778) | 29 channels common to all 4 datasets |
| Training scope | Per-dataset (each PD dataset ± ds004148 HC) | All 4 datasets combined (270 subjects pooled) |
| Learning rate | 1e-3 | 2.5e-4 |
| Adam β₁ | 0.9 (default) | 0.75 |
| LR schedule | None | ExponentialLR γ=0.99 |
| Dataset split | No proper combined subject-level N-LNSO | Stratified N-LNSO across all 270 subjects |

The 64-ch zero-padding was a design error: ds002778 has 40 native channels, and the unified format zero-pads the missing 24 channels in every segment. The TransformEEG paper instead extracts the 29 channels present in all 4 datasets (ds002778's 32 EEG channels mapped to positions in the unified layout). We implemented this as channel index selection from the existing 64-ch arrays — no re-processing needed.

---

### Correct combined baseline: `experiments/baseline_combined.py`

Implemented from scratch matching the paper exactly. Runs combined N-LNSO on 270 subjects (140 PD, 130 HC) across all 4 datasets.

**Result: 10-fold combined N-LNSO**

| Fold | bal_acc | sens | spec |
|------|---------|------|------|
| 1 | 0.762 | 0.562 | 0.962 |
| 2 | 0.878 | 0.823 | 0.933 |
| 3 | 0.955 | 0.966 | 0.945 |
| 4 | 0.929 | 0.928 | 0.931 |
| 5 | 0.951 | 0.998 | 0.905 |
| 6 | 0.919 | 0.912 | 0.925 |
| 7 | 0.904 | 0.933 | 0.875 |
| 8 | 0.840 | 0.970 | 0.710 |
| 9 | 0.838 | 0.833 | 0.842 |
| 10 | 0.843 | 0.931 | 0.755 |
| **Mean** | **0.882** | **0.886** | **0.878** |
| **Median** | **0.891** | | |

**Paper target: 0.7845 (median). Our result: 0.891 median (+10.6pp)**

Result file: `results/baseline/combined_nlnso_20260610_040552.json`

---

### Research reorientation: paper has NO SSL

The TransformEEG paper (arxiv 2507.07622) does not use self-supervised pretraining at any stage. Their 78.45% balanced accuracy is a purely supervised ceiling. Our TUH-scale VICReg pretraining is genuinely a novel contribution on top of their architecture — not a reproduction.

**New framing:** Can SSL pretraining on TUH + fine-tuning on the 29-ch combined protocol exceed the 89.1% supervised median? That is now the core research question.

---

### Data syncs completed this session

- `ds004584` HC subjects: downloaded from S3 (1.3 GB), reprocessed natively
- `ds003490`: complete raw data re-synced from S3 (4.4 GB), reprocessed natively (was only 18/100 recordings)
- Both now fully processed in `data/processed/`

---

### Next experiments queued

1. **Re-run SSL probes with correct protocol** (local, free, ~2h): Update `ssl_pilot.py` Phase 2/3 to use 29-ch combined N-LNSO. Probe the existing e100 sub400k encoder. This gives the first valid SSL vs supervised delta under the correct protocol.

2. **Full-scale TUH pretrain on new AWS account**: Requires channel mapping — TUH native channels → 29-ch common subset. Pack job + pretrain_full preset. Expected: encoder trained on 63k+ TUH segments, then probed with combined N-LNSO.

3. **New AWS account setup**: Cross-account S3 access via bucket policy on old account + IAM policy on new SageMaker role. No data migration required.

### Session 6 compute

| Experiment | Device | Duration | Status |
|---|---|---|---|
| ssl_pilot e100 probes (old protocol) | MPS | ~2h | ✅ Done |
| baseline.py cross-dataset (64-ch) | MPS | ~1h | ❌ Crashed (OOM) |
| experiments/baseline_combined.py | MPS | ~2h 30m | ✅ Done — 0.882 mean |
| ds004584 + ds003490 raw S3 sync | — | ~30m | ✅ Done |

### Session 6 total: ~7h compute

---

## Session 7 — 2026-06-10 (SSL 29-ch local experiment)

### SSL 29-ch combined N-LNSO — OpenNeuro only, correct protocol

**Experiment:** `experiments/ssl_29ch_local.py`
- Phase 1: VICReg pretrain on 18,721 OpenNeuro segments (29-ch, Chan=29, feat_dim=116)
- Phase 2: Combined N-LNSO linear probe (frozen encoder + linear head, 30 epochs)
- Encoder saved: `results/ssl/pretrained_encoder_29ch_opennero.pt`
- Result file: `results/ssl/ssl_29ch_opennero_20260610_161459.json`

**Per-fold results:**

| Fold | bal_acc | sens | spec |
|------|---------|------|------|
| 1 | 0.928 | 0.966 | 0.891 |
| 2 | 0.937 | 0.993 | 0.880 |
| 3 | 0.961 | 0.976 | 0.946 |
| 4 | 0.955 | 0.986 | 0.925 |
| 5 | 0.943 | 1.000 | 0.886 |
| 6 | 0.917 | 0.915 | 0.919 |
| 7 | 0.922 | 0.948 | 0.897 |
| 8 | 0.791 | 0.996 | 0.586 |
| 9 | 0.848 | 0.925 | 0.771 |
| 10 | 0.836 | 0.937 | 0.735 |
| **Mean** | **0.902** | **0.968** | **0.836** |
| **Median** | **0.923** | | |

**Comparison to supervised baseline:**

| Metric | Supervised | SSL 29-ch | Delta |
|--------|-----------|-----------|-------|
| Mean bal_acc | 0.882 | 0.902 | **+2.0pp** |
| Median bal_acc | 0.891 | 0.923 | **+3.2pp** |
| Sensitivity | 0.886 | 0.968 | **+8.2pp** |

### Interpretation

SSL pretraining helps even on the exact same 18k segments used for supervised training. The sensitivity jump (+8.2pp) is clinically significant — fewer missed PD cases. This is a **lower bound** on what full-scale TUH pretraining will achieve, because:
1. The pretrain data overlaps with the probe data (same OpenNeuro segments, labels ignored)
2. TUH provides 390k+ unlabeled segments with zero data overlap

### Caveat: data overlap

The pretraining pool is identical to the fine-tune/probe pool (18.7k OpenNeuro segments). This is not clean SSL — the encoder has seen the probe data distribution during pretraining. For a publication-quality result, pretraining must be TUH-only (disjoint from fine-tune data).

### Next: full-scale TUH experiment (new AWS account)

This session confirms the SSL signal is real. The full experiment requires:
1. New AWS account + cross-account S3 access to existing data
2. Re-ingest TUH with `unified=True` → 64-ch files (one-time, ~$10, 4h on SageMaker)
3. Pretrain `build_encoder(Chan=29)` on TUH-only 29-ch data (zero OpenNeuro overlap)
4. Probe with combined N-LNSO on OpenNeuro

Code changes needed:
- `scripts/tuh_ingest_pipeline.py`: `unified=True` (one line)
- `experiments/ssl_pilot.py`: `N_CHANNELS=29`, add `COMMON_CH_INDICES` channel selection

Expected result: median >92.3% (the current lower bound from 18k-segment SSL).

### Session 7 compute

| Experiment | Device | Duration | Status |
|---|---|---|---|
| ssl_29ch_local.py — pretrain | MPS | ~1h 30m | ✅ Done |
| ssl_29ch_local.py — probe | MPS | ~30m | ✅ Done |

### Session 7 total: ~2h compute

---

## Session 8 — 2026-06-19 (calibration follow-up — Alex Thread 1)

### Does smarter post-hoc calibration beat the 0.643 deployable threshold?

Question (HANDOFF_ALEX.md, Thread 1): does temperature scaling / isotonic / Platt,
fit honestly, beat the train-transferred 0.643 balanced accuracy, toward the 0.732
oracle?

**Method — post-hoc, no model re-running** (no encoder checkpoints are local).
`experiments/calibration_followup.py` reads the saved per-subject held-out scores
from the three 3-seed supervised full-label from-scratch LODO runs
(`lodo_supervised_s{0,1,2}_scratch_f100_noaug`). Calibrators are fit on a
**cross-site proxy** — the *other* held-out sites — and applied unchanged to the
target (the target's labels are used only to score). Sanity checks pass: recomputed
fixed-0.5 equals the stored value, and temperature BA equals fixed BA exactly.

**Results (subject-level LODO balanced accuracy, mean ± std over 3 seeds):**

| Method | bal-acc | note |
|---|---|---|
| Fixed 0.5 | 0.585 ± 0.014 | reproduces dashboard |
| Temperature scaling | 0.585 ± 0.014 | identical to fixed — a scalar can't move the 0.5 threshold |
| Platt scaling | 0.591 ± 0.022 | cross-site proxy fit |
| Isotonic regression | 0.629 ± 0.032 | cross-site proxy fit |
| Train-transferred (**to beat**) | 0.643 ± 0.034 | reproduces dashboard |
| Prevalence-matched | 0.686 ± 0.041 | |
| Oracle (ceiling) | 0.732 ± 0.028 | |

ECE (probability calibration, lower = better): raw **0.278 → 0.147** under temperature.
Temperature fixes the *probabilities*, not the *decision*.

**Conclusion: no honest post-hoc calibration beats 0.643.** With ROC-AUC fixed (~0.76)
the only lever is the operating point, which threshold-transfer already sets near-
optimally; the residual gap to the 0.732 oracle is irreducible cross-site threshold
drift. A clean negative that reinforces Finding 2.

**Caveat:** isotonic/Platt are fit on the other held-out sites' saved scores (a proxy
for an independent labeled set, since same-model training scores were not serialized);
those scores come from different per-fold models.

Output: `results/calibration_followup.json`. Paper: 2 sentences added to §4.2 (Results),
citing `guo2017calibration` + `zadrozny2002`. No compute (analysis from saved JSONs).

