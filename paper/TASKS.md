# Paper tasks — pick one up

> **Target: MICCAI AMAI 2026 (Springer LNCS, 8 pages) · due June 25, 2026.**
> The infra is done: `paper/main.tex` is the LNCS submission file with our sections
> already written and figures wired. Each task below is **self-contained** — copy the
> grey prompt block into Claude Code (or do it by hand), then **verify the output yourself.**
> The repo is public: github.com/edward-lcl/sjji-eeg
>
> **One rule that can't slip:** every citation must be a real paper that actually says what
> we claim. Agents invent references — you cite only papers you (or the agent + you) verified.

---

## Task 1 — Related Work: EEG evaluation & PD  ·  owner: **Saanvi**
**Goal:** ~2 short paragraphs + the dataset table, going into `\section{Related Work}` of `main.tex`.

```
Read docs/related_works.md (Track A) and paper/draft.md in this repo. Using ONLY the
papers listed there (TransformEEG; the preprocessing/leakage papers; ComBat; the four
dataset papers; and the two starred 2026 competitors arXiv:2604.23933 and arXiv:2601.05276),
write two short LaTeX paragraphs for the Related Work section:
 (a) EEG-based Parkinson's detection, and
 (b) evaluation pitfalls & site effects in EEG deep learning.
For each cited paper add a BibTeX entry to paper/refs.bib and cite with \cite{key}.
Explicitly name the two 2026 competitor papers and state in ONE sentence how our work
goes further (the no-EEG site-prior null, the calibration reframe). Also verify/finish
the dataset characterization table in related_works.md.
Output: edit paper/main.tex (Related Work subsections (a),(b)) and paper/refs.bib.
```
🧠 **Make it yours:** does each paper *support* or *challenge* our findings? If a competitor already reports a site-prior null or a calibration analysis, **flag it to Edward immediately** — it changes our framing. **Verify every BibTeX entry is a real paper.**

---

## Task 2 — Related Work: SSL & calibration  ·  owner: **Alex**
**Goal:** ~2 short paragraphs into `\section{Related Work}` of `main.tex`.

```
Read docs/related_works.md (Track B) and paper/draft.md. Using the listed papers
(LaBraM, BIOT, BENDR, SelfEEG; Guo et al. on calibration; + one domain-shift calibration
paper you find and verify), write two short LaTeX paragraphs:
 (c) self-supervised learning for EEG (note whether anyone shows SSL HELPING cross-site —
     relevant to our negative result), and
 (d) calibration under domain shift.
Add BibTeX entries to paper/refs.bib and cite with \cite{key}.
Output: edit paper/main.tex (Related Work subsections (c),(d)) and paper/refs.bib.
```
🧠 **Make it yours:** in one sentence — why do *you* think SSL didn't transfer here? **Verify every citation.**

---

## Task 3 — Figures  ·  owner: **Alex**
**Goal:** clean, readable Figures 1–4 for the paper.

```
Run `python experiments/make_figures.py` — it regenerates Figs 1–4 in paper/figures/ from
the result JSONs. Then improve styling in experiments/make_figures.py and re-run: consistent
fonts/sizes, clear axis labels, readable at print size, colorblind-friendly colors. Keep the
titles short. Re-run until each figure makes its point in ~3 seconds.
Output: updated experiments/make_figures.py + regenerated paper/figures/*.png.
```
🧠 **Make it yours:** show each figure to someone and ask "what's the takeaway?" If they can't say it instantly, fix it.

---

## Task 4 — Introduction hook (¶1)  ·  owner: **Saanvi or Alex**
**Goal:** the opening paragraph of the Introduction in `main.tex`.

```
Write the opening hook paragraph for paper/main.tex (replace [HOOK PARAGRAPH ...]):
4–6 sentences on the clinical motivation — Parkinson's prevalence, late diagnosis from
motor symptoms, EEG as a cheap/accessible biomarker, and why a model must work across
different hospitals to be clinically useful. Plain, motivating, no hype. LaTeX.
Output: edit paper/main.tex Introduction.
```
🧠 **Make it yours:** would a clinician care about the first sentence? Make it matter.

---

## Task 5 — References / BibTeX hygiene  ·  owner: **both (ongoing)**
As you each add citations, keep `paper/refs.bib` consistent (one key per paper, no
duplicates). Before submission, one of you does a final pass: every `\cite{}` resolves,
every entry is a real, correctly-described paper.

---

## Edward + Claude (infra / assembly — not student tasks)
- LNCS skeleton + our drafted sections ✅ (`paper/main.tex`)
- Author list & affiliations · weave the Related Work in · trim to 8 pages · final read · submit by Jun 25.

---

### How to pick up a task
1. Open the repo in Claude Code (or your editor + agent).
2. Copy a grey prompt block above into the agent.
3. **Review what it produces** — is it accurate? does it support our 3 findings? are the citations real?
4. Commit your section (or hand it to Edward). Stuck? Drop a note in the weekly check-in doc.
