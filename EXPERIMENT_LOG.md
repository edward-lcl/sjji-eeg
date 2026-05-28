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
| fingerprint_balanced | 2026-05-26 02:08 | in progress | TBD | MPS | 🔄 Running | Same with WeightedRandomSampler — tests if site artifacts are real or majority-class bias |
| subject_aggregation | 2026-05-26 02:08 | in progress | TBD | MPS | 🔄 Running | Per-subject majority-vote re-analysis of per-dataset CV |
| ssl_pilot | 2026-05-26 02:08 | in progress | TBD | MPS | 🔄 Running | SimCLR pretrain 100 epochs + linear probe cross-dataset vs supervised baseline |

### Session 2 total: ongoing
