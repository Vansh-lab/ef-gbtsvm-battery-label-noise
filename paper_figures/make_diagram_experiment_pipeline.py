"""Experiment-pipeline diagram for the EF-GBTSVM battery paper.

One box-and-arrow figure showing the paper's data -> label -> noise -> LOBO/tuning ->
models -> metrics -> statistical-tests pipeline, built purely from the quoted sentences
in paper/ieee/paper.tex (sections "Data", "Experimental protocol" and "Statistical
comparison and overfitting check") and REPORT.md / PREREG_FINEGRID.md. No step, number
or claim is added beyond what those sections state; no numeric results (e.g. the
Friedman chi-squared value) are drawn on the figure itself.

    python paper_figures/make_diagram_experiment_pipeline.py
"""
import os
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.size": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False})


def wrap(text, box_w_in, fontsize, bold=False):
    """Hard-wrap each line of `text` (which may already contain \n) to fit box_w_in
    inches at the given font size, using an empirical chars-per-inch estimate for a
    serif font so lines do not overflow the box they are drawn in."""
    k = 0.60 if bold else 0.50  # average glyph width as a fraction of font size (pt)
    chars_per_line = max(6, int(box_w_in * 72 / (fontsize * k)))
    out_lines = []
    for raw_line in text.split("\n"):
        out_lines.extend(textwrap.wrap(raw_line, width=chars_per_line) or [""])
    return "\n".join(out_lines)

# Palette reused from paper_figures/make_diagnostic_figs2.py
BLUE = "#1f5fbf"      # primary / PF
GRAY = "#6b6b6b"      # neutral / data
GREEN = "#2a9d6b"     # metrics
ORANGE = "#c98a2a"    # labels / noise
PURPLE = "#9467bd"    # split + tuning
RED = "#d62728"       # statistical tests

# ---------------------------------------------------------------------------------------
# Seven pipeline stages, each box's text traceable to a quoted sentence (see report).
# Layout: a two-row "S" (snake) flow, top row left-to-right, bottom row right-to-left,
# chosen to keep the figure compact at two-column IEEE width (7.1in) while giving each
# box enough room for legible text at ~8-8.5pt.
# ---------------------------------------------------------------------------------------
# Figure geometry: data-unit x spans 0..XMAX inches*SCALE; SCALE converts data units to
# inches so wrap() can size text correctly for each box's actual printed width.
FIG_W_IN, FIG_H_IN = 7.1, 3.15
XMAX, YMAX = 11.6, 4.9
SCALE = FIG_W_IN / XMAX  # inches per data unit (x and y use the same scale)

TITLE_FS, BODY_FS, ASIDE_FS = 8.3, 7.0, 5.8

boxes = [
    # (key, x, y, w, h, color, title, body, aside)
    ("data", 0.25, 2.95, 2.55, 1.65, GRAY,
     "NASA PCoE data",
     "B0005, B0006, B0007, B0018 -- 636 discharge cycles", None),
    ("label", 3.05, 2.95, 2.3, 1.65, ORANGE,
     "Label generation",
     "degraded (1): SOH < 0.80; healthy (0): otherwise", None),
    ("noise", 5.6, 2.95, 2.85, 1.65, ORANGE,
     "Label noise injection (training labels only)",
     "rho in {0, 0.10, 0.20}; seed = 1000 + fold; test labels never flipped",
     "fine-grid aside: rho in {0.02, 0.04, 0.06, 0.08}, identical mechanism"),
    ("split", 8.7, 2.95, 2.65, 1.65, PURPLE,
     "LOBO outer split + battery-grouped inner CV",
     "each cell held out once, trained on other three; scaling + tuning on training data only, refit on whole training fold", None),
    ("models", 7.75, 0.4, 3.6, 1.65, BLUE,
     "15 models evaluated",
     "EF-GBTSVM family + ablations, RBF-SVM, RVFL, random forest, MLP, XGBoost, single-feature threshold",
     "sensitivity pool k = 17: adds L8 and J as separate entries"),
    ("metrics", 3.95, 0.4, 3.55, 1.65, GREEN,
     "Metrics",
     "ACC, BACC, F1 (degraded class), AUC -- mean +/- SD over 4 test batteries",
     "+ overfitting check: clean-train accuracy minus test accuracy"),
    ("stats", 0.25, 0.4, 3.45, 1.65, RED,
     "Statistical tests",
     "Friedman -> Nemenyi critical difference -> Wilcoxon signed-rank (pairwise)", None),
]

fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
ax.set_xlim(0, XMAX)
ax.set_ylim(0, YMAX)
ax.axis("off")

PAD_IN = 0.09  # inner text margin, inches, on each side of a box

centers = {}
for key, x, y, w, h, color, title, body, aside in boxes:
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                          linewidth=1.1, edgecolor=color, facecolor=color, alpha=0.14,
                          zorder=2)
    ax.add_patch(box)
    # a slightly stronger left accent bar to make the colour grouping legible in print
    ax.add_patch(FancyBboxPatch((x, y), 0.06, h, boxstyle="round,pad=0,rounding_size=0.02",
                                 linewidth=0, facecolor=color, zorder=3))
    cx = x + w / 2
    centers[key] = (x, y, w, h)
    text_w_in = w * SCALE - 2 * PAD_IN

    title_wrapped = wrap(title, text_w_in, TITLE_FS, bold=True)
    body_wrapped = wrap(body, text_w_in, BODY_FS, bold=False)
    n_title_lines = title_wrapped.count("\n") + 1
    n_body_lines = body_wrapped.count("\n") + 1

    title_y = y + h - 0.20
    ax.text(cx, title_y, title_wrapped, ha="center", va="top", fontsize=TITLE_FS,
             fontweight="bold", color="#1a1a1a", zorder=4, linespacing=1.3)
    body_y = title_y - n_title_lines * (TITLE_FS / 72 * 1.35) - 0.12
    ax.text(cx, body_y, body_wrapped, ha="center", va="top", fontsize=BODY_FS,
             color="#1a1a1a", zorder=4, linespacing=1.4)
    if aside:
        aside_wrapped = wrap(aside, text_w_in, ASIDE_FS, bold=False)
        aside_y = y + 0.16
        ax.text(cx, aside_y, aside_wrapped, ha="center", va="bottom", fontsize=ASIDE_FS,
                 style="italic", color="#444444", zorder=4, linespacing=1.25)


def arrow(key_from, key_to, side_from, side_to, color):
    xf, yf, wf, hf = centers[key_from]
    xt, yt, wt, ht = centers[key_to]
    points = {
        "right": (xf + wf, yf + hf / 2),
        "left": (xf, yf + hf / 2),
        "bottom": (xf + wf / 2, yf),
        "top": (xf + wf / 2, yf + hf),
    }
    start = points[side_from]
    points_to = {
        "right": (xt + wt, yt + ht / 2),
        "left": (xt, yt + ht / 2),
        "bottom": (xt + wt / 2, yt),
        "top": (xt + wt / 2, yt + ht),
    }
    end = points_to[side_to]
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                         linewidth=1.3, color=color, zorder=5,
                         connectionstyle="arc3,rad=0.0", shrinkA=1, shrinkB=1)
    ax.add_patch(a)


# Required node sequence (top row left->right, down, bottom row right->left):
# data -> label -> noise -> split -> models -> metrics -> stats
arrow("data", "label", "right", "left", GRAY)
arrow("label", "noise", "right", "left", ORANGE)
arrow("noise", "split", "right", "left", ORANGE)
arrow("split", "models", "bottom", "top", PURPLE)
arrow("models", "metrics", "left", "right", BLUE)
arrow("metrics", "stats", "left", "right", GREEN)

fig.tight_layout(pad=0.3)

os.makedirs("results", exist_ok=True)
fig.savefig("results/diagram_experiment_pipeline.png", dpi=300)
fig.savefig("results/diagram_experiment_pipeline.pdf")
plt.close(fig)

print("wrote results/diagram_experiment_pipeline.png and .pdf")
