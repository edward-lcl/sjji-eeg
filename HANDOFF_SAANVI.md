# Handoff — Saanvi

*2026-06-17 · async-friendly · repo: `github.com/edward-lcl/sjji-eeg` · dashboard: sjji-eeg.exe.xyz · paper: `paper/main.tex` (LNCS, MICCAI AMAI workshop, due Jun 25)*

The experiments and the heavy sections are settled. This runbook is your two open threads — the **site-identifiability experiment** and **Related Work (Track A)** — plus the locked context so you're not re-deriving it.

## The result (what the paper claims — run/write toward it)
- **Pooled accuracy is a site artifact.** A no-EEG model predicting each dataset's majority class scores **0.927** on the same protocol — ≥ the published models.
- **Honest cross-site (LODO) "failure" is calibration, not signal.** Supervised ROC-AUC **0.76**; a deployable threshold recovers **0.64** balanced accuracy (vs 0.50 chance).
- **SSL + augmentation don't improve cross-site transfer** (AUC 0.58 / 0.53 vs 0.76).

## Settled — don't relitigate
| Thing | State |
|---|---|
| The 3 findings above | done, 3-seed, error-barred |
| Methods / Results / Discussion | drafted in `paper/main.tex` |
| Eval harness, 19-ch montage, site-prior null | built — `experiments/lodo_eval.py`, `src/honest_eval.py` |
| Venue | MICCAI AMAI 2026 · LNCS · 8 pages · **Jun 25** |

## Thread 1 — Site identifiability *(open — design is yours)*
**Question:** can a classifier predict *which dataset* a segment came from (ds002778 / ds003490 / ds004584 / ds004148) from the EEG alone — no PD label? How well?

If it's near-perfect, that's the mechanism behind finding (1): the model can recognize the site instead of the disease. Goes in Results as a figure; it's your number.

Runbook:
- Data: `data/processed_unified/<dataset>/`. Borrow the segment loader + channel selection from `experiments/lodo_eval.py` — don't rewrite it.
- Model: reuse `src/model.py:build_encoder` with a 4-way head, or a simpler classifier — your call.
- **Split by subject, not segment** — same person in train and test is leakage and inflates the number.
- **Balance the classes / report balanced accuracy** — ds004148 is huge, raw accuracy would lie.
- Chance = 25% (4-way). Runs fine on your M4 Air — lighter than the runs already done.

Report: balanced accuracy (+ a confusion matrix is nice — which datasets are most confusable), and one line on what a near-perfect score means for the pooled PD number.

## Thread 2 — Related Work, Track A (EEG eval / site effects + PD-from-EEG)
- Source list + the two closest 2026 competitors are in `docs/related_works.md`.
- Write ~2 short paragraphs into the Related Work section of `paper/main.tex`; BibTeX into `paper/refs.bib`.
- Name the two competitor papers (arXiv:2604.23933, arXiv:2601.05276) and say in one sentence how we go further (site-prior null + calibration). **If either already does our site-null or calibration analysis, flag it — that changes our framing.**
- **Verify every citation is a real paper.** Agents invent references.

## Thread 3 — Introduction hook (¶1)
The opening paragraph of `paper/main.tex` (replace the `[HOOK PARAGRAPH]` stub): 4–6 sentences on the clinical motivation — Parkinson's prevalence, late diagnosis from motor symptoms, EEG as a cheap/accessible biomarker, and why a model has to work across *hospitals* to be useful. Plain and motivating, no hype.

## Inputs
`paper/main.tex` (your sections stubbed) · `docs/related_works.md` · `experiments/lodo_eval.py`, `src/honest_eval.py`, `src/model.py` · `results/lodo/*.json` · the dashboard for the full picture.

## Deliverable
The site-ID number/figure + your two Related-Work paragraphs + the intro hook + refs. Quality over the Jun 25 date — if the experiment runs long, the writing ships first. Blockers → Wednesday check-in.
