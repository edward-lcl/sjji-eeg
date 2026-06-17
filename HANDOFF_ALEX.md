# Handoff — Alex

*2026-06-17 · async-friendly · repo: `github.com/edward-lcl/sjji-eeg` · dashboard: sjji-eeg.exe.xyz · paper: `paper/main.tex` (LNCS, MICCAI AMAI workshop, due Jun 25)*

The experiments and the heavy sections are settled. This runbook is your three open threads — the **calibration experiment**, **Related Work (Track B)**, and the **figures** — plus the locked context so you're not re-deriving it.

## The result (what the paper claims — run/write toward it)
- **Pooled accuracy is a site artifact.** A no-EEG model scores **0.927** on the same protocol.
- **Honest cross-site (LODO) "failure" is calibration, not signal.** Supervised ROC-AUC **0.76**; a deployable (train-transferred) threshold recovers **0.64**; oracle ceiling **0.73**.
- **SSL + augmentation don't improve cross-site transfer** (AUC 0.58 / 0.53 vs 0.76; data-efficiency flat across 3 seeds).

## Settled — don't relitigate
| Thing | State |
|---|---|
| The 3 findings above | done, 3-seed, error-barred |
| Methods / Results / Discussion | drafted in `paper/main.tex` |
| Threshold policies (fixed / train-transferred / prevalence / oracle) | built — `src/honest_eval.py:calibration_report` |
| Venue | MICCAI AMAI 2026 · LNCS · 8 pages · **Jun 25** |

## Thread 1 — Better calibration *(open — design is yours)*
**Question:** under LODO we lifted balanced accuracy from 0.59 (fixed 0.5) to **0.64** (threshold transferred from the training sites). Does a *smarter* calibration — **temperature scaling** or **isotonic regression**, fit on the training sites — beat 0.64? How close to the 0.73 oracle can an *honest* method get?

Runbook:
- The per-subject scores + labels are already saved in `results/lodo/*.json` (fields `subject_scores` / `subject_labels`) — no re-running models.
- `src/honest_eval.py:calibration_report` shows how the existing policies are computed; add yours next to them.
- Temperature scaling = fit one scalar on the *training* scores; isotonic = a monotonic mapping.
- **Honesty rule:** fit on training-site data only, apply *unchanged* to the held-out site. Never touch held-out labels — that's the oracle (the ceiling), not a deployable method.

Report: the new method's balanced accuracy vs 0.64, and one line on why it helps (or doesn't) given the AUC is fixed.

## Thread 2 — Related Work, Track B (SSL for EEG + calibration under shift)
- Source list in `docs/related_works.md`. Write ~2 short paragraphs into Related Work in `paper/main.tex`; BibTeX into `paper/refs.bib`.
- For SSL: note whether anyone shows SSL *helping* cross-site (relevant to our negative).
- **Verify every citation is a real paper.** Agents invent references.

## Thread 3 — Figures
- `python experiments/make_figures.py` regenerates Figs 1–4 in `paper/figures/` from the result JSONs.
- Improve styling in `experiments/make_figures.py` and re-run: consistent fonts/sizes, clear labels, readable at print size, colorblind-safe. Each figure should make its point in ~3 seconds.

## Inputs
`paper/main.tex` (your sections stubbed) · `docs/related_works.md` · `src/honest_eval.py` · `results/lodo/*.json` · `experiments/make_figures.py`, `paper/figures/` · the dashboard for the full picture.

## Deliverable
The calibration number + your two Related-Work paragraphs + refs + cleaned figures. Quality over the Jun 25 date. Blockers → Wednesday check-in.
