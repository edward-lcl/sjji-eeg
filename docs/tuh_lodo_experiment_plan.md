# TUH-pretrain × LODO — the experiment that tests the actual thesis

> Status: planned. Gated on the new AWS account quota (cross-account S3 + SageMaker).
> Companion to `RESEARCH.md`. This is the *only* experiment that can produce a
> defensible headline result for the paper.

## 1. The one question

**Does self-supervised pretraining on a site-disjoint corpus (TUH) lift
leave-one-dataset-out (LODO) Parkinson's detection above chance?**

Everything else is secondary. We already know (measured 2026-06-12, `results/lodo/`):

| Encoder | Combined N-LNSO (segment) | **LODO (subject, macro)** |
|---|---|---|
| Site-prior null (no EEG) | 0.927 | — (n/a, unseen site) |
| Supervised (end-to-end) | ~0.89 | **0.569** |
| SSL on OpenNeuro (overlapping) | 0.923 | **0.496 (chance)** |
| **SSL on TUH (this plan)** | ? | **? ← the result that matters** |

The combined N-LNSO column is site-confounded and uninformative (a no-EEG null beats
it). The LODO column is the thesis. The current SSL encoder is at chance there because
it pretrains on the *same* OpenNeuro segments it probes — no new distribution, and the
site shortcut is the only thing on offer. TUH breaks both: it is a different corpus
(no probe overlap) at ~50× the scale.

## 2. Disjoint-site guarantee (the integrity constraint)

- **Pretrain pool:** TUH-EEG corpus ONLY. No OpenNeuro segments, labeled or unlabeled.
- **Eval pool:** the 4 OpenNeuro datasets ONLY (ds004148/ds002778/ds003490/ds004584).
- These never mix. This is the difference from `ssl_29ch_local.py`, whose own log
  flags the overlap as a confound. Verify with a manifest assertion in the pretrain
  job: no S3 key under the OpenNeuro prefixes enters the pretrain channel.

## 3. Channel mapping — the #1 risk, now MEASURED (2026-06-13)

**Finding (measured on raw TUH EDFs through `process_eeg_file(unified=True)`):**
of the 29 OpenNeuro-derived common channels, **only 15 survive in TUH; 14 are dead.**

- TUH uses the classic clinical 10-20 montage: `FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8
  T3 T4 T5 T6 FZ CZ PZ OZ`.
- **4 dead are recoverable** — `T7/T8/P7/P8` are TUH's `T3/T4/T5/T6` under old 10-20
  naming. Add aliases in `src/preprocess.py::_normalize_ch` (T3→T7, T4→T8, T5→P7, T6→P8).
- **10 dead are genuinely absent** in TUH: `AF3 AF4 FC5 FC1 FC2 FC6 CP5 CP1 CP2 CP6`
  (extended 10-10 positions TUH does not record). Do NOT interpolate — exclude them.

**Why it matters:** the depthwise tokenizer has one filter per channel. Pretraining
`Chan=29` on TUH trains 14 filters on pure zeros; at LODO eval those channels carry real
OpenNeuro signal → train/eval mismatch on half the inputs → transfer fails for reasons
unrelated to SSL. This would manufacture a false-negative result and waste GPU credits.

**Design change — use a 19-channel TUH∩OpenNeuro montage** (the 15 alive + the 4 renamed):
`Fp1 Fp2 F7 F3 Fz F4 F8 T7 C3 Cz C4 T8 P7 P3 P4 P8 O1 Oz O2`. Indices into the unified
64-ch layout (verified 19/19 alive on TUH after the naming fix landed in `_normalize_ch`):

```python
COMMON19_CH_INDICES = [0, 1, 6, 8, 10, 12, 14, 26, 28, 30, 32, 34, 46, 48, 52, 54, 60, 61, 62]
```

Rebuild `build_encoder(Chan=19)` and use it for BOTH pretrain and eval so channels match. NOTE: the existing OpenNeuro-only baselines
(`baseline_combined.py`, `ssl_29ch_local.py`, `lodo_eval.py`) use the 29-ch set — re-run
them at 19ch so the supervised/SSL baselines are at the same channel count the TUH
experiment will use. (Also worth caveating: TUH `processed_unified/tuh_eeg/` on local disk
is NOT actually 64-ch unified — it's native montage. The unified re-ingest is genuinely
unstarted.)

---

Original requirement (still holds, now at 19ch): for TUH features to transfer,
**the channels must be the same scalp positions in the same order** in both corpora:

1. Re-ingest TUH with `unified=True` (`scripts/tuh_ingest_pipeline.py`) → the standard
   64-ch 10-20 layout (`src/preprocess.py::UNIFIED_64_CHANNELS`), identical layout the
   OpenNeuro `processed_unified` arrays already use.
2. Select the same `COMMON_CH_INDICES` (the 29 indices into the 64-ch layout) at load
   time — exactly as `lodo_eval.py` / `baseline_combined.py` do for OpenNeuro.
3. **Assert** before pretraining: a handful of TUH 29-ch arrays and OpenNeuro 29-ch
   arrays index to the same channel *names* (`UNIFIED_64_CHANNELS[i] for i in
   COMMON_CH_INDICES`). A silent off-by-one here makes the whole run meaningless and
   looks like "SSL just doesn't transfer." Cheap check, do it first.

Note: TUH montages vary (tcp_le/tcp_ar/etc.); `interpolate_to_unified` zero-pads missing
channels. Confirm the 29 common channels are actually *present* (non-zero) in the bulk of
TUH recordings, or the encoder pretrains on partially-dead inputs.

## 4. Pretrain (SageMaker)

- Objective: VICReg (existing `src/pretrain.py`), 29-ch, feat_dim=116.
- Data: TUH `processed_unified` → **packed shards** (`scripts/sm_pack_shards.py`) for
  FastFile I/O sanity (the un-packed 63k-file path stalls the GPU; see EXPERIMENT_LOG.md).
- Scale: start with the ~390k-segment subsample for a fast first answer; escalate to the
  full ~8M only if the subsample shows life. Don't pay for 8M before the channel mapping
  and the LODO harness are proven on the subsample.
- Warm-start: optional from the e70/e100 encoders, but a clean random-init run is the
  cleaner story; prefer clean unless compute-constrained.
- Save encoder to S3 on every improvement (the resilient-save path already exists).

## 5. Eval (local, harness already built)

```bash
# Pull the TUH-pretrained 29-ch encoder, then:
python experiments/lodo_eval.py --mode probe --encoder results/ssl/<tuh_encoder>.pt
# Context only (will be site-confounded, report null alongside):
python experiments/ssl_29ch_local.py --probe-only   # after pointing ENCODER_SAVE at it
```

Primary metric: **LODO subject-level macro, median + IQR + bootstrap CI**, vs the 0.50
chance line. `src/honest_eval.py` already produces all of this; `lodo_eval.py` already
reports per-held-out + macro.

## 6. Decision tree

- **LODO subject macro clears 0.50 (CI excludes chance):** positive result. SSL
  pretraining is what enables cross-site PD detection on the TransformEEG benchmark —
  a genuine, novel contribution. Proceed to full-scale + data-efficiency.
- **LODO at chance:** negative result. Report it honestly inside the paper's own
  generalizability framing ("even TUH-scale SSL does not transfer across these sites; the
  combined-protocol gains in prior work are site artifacts"). Still publishable, and the
  site-prior-null analysis becomes the paper's spine.

## 7. Controls & strengthening (LODO has only 3 folds — thin)

- **Negative control:** random-init `build_encoder(Chan=29)` under LODO probe → must be
  ~chance. (Sanity that the probe isn't leaking.)
- **Comparators (already have):** supervised LODO 0.569, OpenNeuro-SSL LODO 0.496.
- **Tighten the estimate:** (i) subject-level bootstrap CI *within* each held-out site
  (already supported), (ii) 2–3 pretrain seeds → report spread, (iii) ideally add a
  genuinely external 4th site (RESEARCH.md lists NMT / Predict-PD / De Novo PD) as a true
  held-out — the strongest possible generalization claim.
- **Data-efficiency axis (RESEARCH.md's highest-impact narrative):** subsample probe
  *subjects* to 100/50/25/10%, plot TUH-SSL vs supervised LODO. SSL reaching parity with
  fewer labels is the clinically compelling result even if absolute LODO is modest.

## 8. Pitfalls checklist (verify before spending GPU credits)

- [x] Channel coverage measured (Section 3): only 15/29 alive in TUH → switch to 19ch.
- [ ] Old/new naming aliases added (T3→T7, T4→T8, T5→P7, T6→P8) — recovers 4 channels.
- [ ] `COMMON19` index set defined; OpenNeuro baselines re-run at 19ch.
- [ ] 19 common channels non-zero in TUH recordings (re-verify after unified re-ingest).
- [ ] Zero OpenNeuro keys in the pretrain manifest (disjointness asserted).
- [ ] Packed shards built (no un-packed FastFile run).
- [ ] Sample rate / window length consistent between pretrain and probe (currently 250Hz /
      16s / 4000-sample everywhere — keep it consistent; note paper used 125Hz/2000).
- [ ] Eval reports subject-level median+IQR+CI against site-prior null, not segment mean.
```
