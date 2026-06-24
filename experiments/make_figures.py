"""
Figure generator for the paper.

Reads the result JSONs in results/lodo/ and emits print-ready figures to
paper/figures/. Data parsing is unchanged; styling is colorblind-safe (Okabe--Ito),
300-DPI, with a consistent font hierarchy and figure size across all four panels.

  python experiments/make_figures.py            # all figures
  python experiments/make_figures.py fig3       # just one

Figures:
  fig1  the confound        — no-EEG site null vs reported accuracies (bars)
  fig2  calibration         — LODO balanced accuracy by threshold policy (bars)
  fig3  SSL doesn't help     — cross-site ROC-AUC by method (bars, mean±std)
  fig4  data-efficiency     — cross-site AUC vs label budget, scratch vs SSL (lines)
"""

import sys
import glob
import json
import numpy as np
from pathlib import Path

import os
# Output format defaults to PNG (raster, via the Agg backend) — the paper includes
# the .png figures. Where the Agg backend is unavailable, set FIG_FORMAT=pdf (or
# =svg) to emit a vector figure and rasterize it to PNG separately.
FMT = os.environ.get("FIG_FORMAT", "png").lower()
try:
    import matplotlib
    matplotlib.use({"png": "Agg", "pdf": "pdf", "svg": "svg"}.get(FMT, "Agg"))
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib not installed — run:  pip install matplotlib")

OUT = Path("paper/figures"); OUT.mkdir(parents=True, exist_ok=True)

# Consistent, print-ready defaults: 300-DPI output, clear font hierarchy, a light
# dashed y-grid only, and no top/right spines.
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.titlepad": 8,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.35,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Okabe--Ito colorblind-safe palette (https://jfly.uni-koeln.de/color/).
C = {"orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
     "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
     "gray": "#999999"}

FIGSIZE = (5.6, 3.5)   # one shared size so the side-by-side floats line up
BARLAB = 9             # bar value-label font size
ANN = 9                # annotation ("chance") font size


# ── load + bucket every LODO result ───────────────────────────────────────────
def load_lodo():
    rows = []
    for f in glob.glob("results/lodo/lodo_*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        c = d.get("config", {}); cal = d.get("macro", {}).get("calibration", {})
        enc = (d.get("encoder_path") or c.get("init_encoder") or "") or ""
        if d.get("mode") == "supervised" and not c.get("init_encoder"):
            method = "supervised"
        elif "tuh" in enc:
            method = "SSL·TUH"
        elif "opennero" in enc or "openneuro" in enc:
            method = "SSL·OpenNeuro"
        else:
            method = "supervised"
        rows.append({
            "method": method, "seed": c.get("seed", 0),
            "frac": float(c.get("label_frac", 1.0)),
            "init": "scratch" if not c.get("init_encoder") else ("TUH" if "tuh" in (c.get("init_encoder") or "") else "OpenN"),
            "auc": cal.get("roc_auc"), "deploy": cal.get("train_transferred"),
            "prev": cal.get("prevalence_matched"), "fixed": cal.get("fixed_0.5"),
            "oracle": cal.get("oracle_youden"), "mode": d.get("mode"),
        })
    return rows


def _ms(vals):
    v = [x for x in vals if x is not None]
    return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, 0.0)


def _chance(ax, x):
    """Dotted chance line at 0.5 with a small label at x."""
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.text(x, 0.51, "chance", color="k", fontsize=ANN, va="bottom")


def _save(fig, name):
    p = OUT / f"{name}.{FMT}"; fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


# ── fig1 — the confound ───────────────────────────────────────────────────────
def fig1(_rows):
    # Reference values (segment-level balanced accuracy on the pooled protocol).
    # null is also in any LODO json under site_prior_null; reported numbers are from
    # results/baseline + results/ssl (combined N-LNSO).
    labels = ["No-EEG\nsite null", "SSL\n(pooled)", "Supervised\n(pooled)", "TransformEEG\npaper"]
    vals = [0.927, 0.923, 0.891, 0.785]
    cols = [C["vermillion"], C["purple"], C["blue"], C["gray"]]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    b = ax.bar(labels, vals, color=cols, width=0.66)
    ax.bar_label(b, fmt="%.3f", padding=3, fontsize=BARLAB)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("Balanced accuracy (segment)")
    ax.set_title("A no-EEG baseline matches the best models")
    _chance(ax, 3.35)
    _save(fig, "fig1_confound")


# ── fig2 — calibration recovery ───────────────────────────────────────────────
def fig2(rows):
    sup = [r for r in rows if r["method"] == "supervised" and r["frac"] == 1.0 and not r.get("init") == "TUH"]
    pol = ["fixed", "deploy", "prev", "oracle"]
    names = ["Fixed 0.5\n(collapse)", "Train-transferred\n(deployable)", "Prevalence\n-matched", "Oracle\n(ceiling)"]
    means = [_ms([r[p] for r in sup])[0] for p in pol]
    errs = [_ms([r[p] for r in sup])[1] for p in pol]
    cols = [C["vermillion"], C["green"], C["skyblue"], C["gray"]]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    b = ax.bar(names, means, yerr=errs, color=cols, width=0.64, capsize=4)
    ax.bar_label(b, fmt="%.3f", padding=3, fontsize=BARLAB)
    ax.set_ylim(0.4, 0.85); ax.set_ylabel("Balanced accuracy (subject, LODO)")
    ax.set_title("Cross-site 'failure' is a calibration problem")
    _chance(ax, 3.35)
    _save(fig, "fig2_calibration")


# ── fig3 — SSL doesn't help (AUC by method) ───────────────────────────────────
def fig3(rows):
    # Linear-probe comparison only (frozen encoder). Supervised = trained from scratch at
    # full labels; SSL bars = the frozen-probe runs (mode=probe). Fine-tune is fig4.
    order = ["supervised", "SSL·OpenNeuro", "SSL·TUH"]
    data = [
        [r["auc"] for r in rows if r["method"] == "supervised" and r["init"] == "scratch"
         and r["frac"] == 1.0 and r["mode"] == "supervised"],
        [r["auc"] for r in rows if r["mode"] == "probe" and r["method"] == "SSL·OpenNeuro"],
        [r["auc"] for r in rows if r["mode"] == "probe" and r["method"] == "SSL·TUH"],
    ]
    means = [_ms(d)[0] for d in data]; errs = [_ms(d)[1] for d in data]
    cols = [C["green"], C["orange"], C["purple"]]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    b = ax.bar(order, means, yerr=errs, color=cols, width=0.6, capsize=4)
    ax.bar_label(b, fmt="%.3f", padding=3, fontsize=BARLAB)
    ax.set_ylim(0.45, 0.85); ax.set_ylabel("Cross-site ROC-AUC (LODO)")
    ax.set_title("Self-supervised pretraining does not transfer")
    _chance(ax, 2.35)
    _save(fig, "fig3_ssl_auc")


# ── fig4 — data-efficiency curves ─────────────────────────────────────────────
def fig4(rows):
    fracs = [0.10, 0.25, 1.0]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # Distinct color + marker + linestyle so the two curves separate even in grayscale.
    for init, col, mk, ls, lab in [("scratch", C["green"], "o", "-", "from scratch"),
                                    ("TUH", C["purple"], "s", "--", "SSL·TUH init")]:
        ys, es = [], []
        for fr in fracs:
            mu, sd = _ms([r["auc"] for r in rows if r["init"] == init and abs(r["frac"] - fr) < 1e-6 and r["mode"] == "supervised"])
            ys.append(mu); es.append(sd)
        x = [10, 25, 100]
        ax.errorbar(x, ys, yerr=es, marker=mk, linestyle=ls, color=col, capsize=3,
                    label=lab, lw=2, markersize=6)
    ax.set_xscale("log"); ax.set_xticks([10, 25, 100]); ax.set_xticklabels(["10%", "25%", "100%"])
    ax.set_xlabel("Labeled training subjects"); ax.set_ylabel("Cross-site ROC-AUC (LODO)")
    ax.set_title("SSL gives no data-efficiency advantage"); ax.set_ylim(0.5, 0.8)
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.legend(frameon=True, loc="upper left")
    _save(fig, "fig4_data_efficiency")


if __name__ == "__main__":
    rows = load_lodo()
    print(f"loaded {len(rows)} LODO result(s)")
    which = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4"]
    for name in which:
        fn = globals().get(name)
        if fn:
            print(f"[{name}]"); fn(rows)
        else:
            print(f"unknown figure: {name}")
    print(f"\nFigures in {OUT}/ — open them, then tweak styling in this script and re-run.")
