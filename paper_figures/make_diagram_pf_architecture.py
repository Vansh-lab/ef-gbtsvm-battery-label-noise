"""Architecture diagram for EF-GBTSVM-PF: the per-fold pipeline from raw training
samples to a prediction, built only from Section 3 ("Method") and Table 1 of
paper/ieee/paper.tex. No step, number or claim is added beyond what those two
sources state; see the inline comments for the exact sentence each box is
grounded in.

Six boxes, left-to-right then down-and-back ("S" flow) to keep the figure
compact at two-column width:

  1. Granular ball generation         -- Section 3, "Granular balls" paragraph
  2. Purity-threshold splitting (tau)  -- same paragraph (the 2-means/tau rule)
  3. RVFL-style enhanced features      -- Section 3, "Enhanced features" paragraph
  4. Twin SVM dual QP (two hyperplanes)-- Section 3, "Twin hyperplanes" paragraph
  5. Per-fold variant selection (L8 vs J) -- Section 3, "Per-fold selection,
     EF-GBTSVM-PF" paragraph, with the two candidates' tau-rules from Table 1
  6. Prediction / decision rule        -- the y_hat(x) = argmin_i ... equation

    python paper_figures/make_diagram_pf_architecture.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Palette (reused from paper_figures/make_diagnostic_figs2.py's figure style)
# ---------------------------------------------------------------------------
BLUE = "#1f5fbf"     # primary / PF core model stage
GRAY = "#6b6b6b"     # granular-ball construction stage
PURPLE = "#9467bd"   # enhanced-feature stage
ORANGE = "#c98a2a"   # per-fold variant selection stage (L8 vs J)
GREEN = "#2a9d6b"    # final prediction / decision rule

TEXT_DARK = "#1a1a1a"


def box(ax, cx, cy, w, h, lines, facecolor, fontsizes=None, zorder=3):
    """Draw a rounded box centered at (cx, cy) with one or more text lines.

    `lines` is a list of (text, fontsize, weight, color) tuples stacked
    top-to-bottom inside the box.
    """
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1, edgecolor=facecolor, facecolor=facecolor + "26",
        zorder=zorder,
    )
    ax.add_patch(patch)
    n = len(lines)
    # distribute lines evenly within the box height
    top = cy + h / 2 - h / (n + 1)
    step = h / (n + 1)
    for i, (text, fs, weight, color) in enumerate(lines):
        ax.text(cx, top - i * step, text, ha="center", va="center",
                 fontsize=fs, weight=weight, color=color, zorder=zorder + 1)
    return patch


def arrow(ax, p0, p1, color=TEXT_DARK):
    a = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=9,
        linewidth=1.1, color=color, zorder=2,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(7.1, 2.9))
ax.set_xlim(0, 14.2)
ax.set_ylim(0, 6.0)
ax.axis("off")

# Column centers / widths, shared between the top and bottom row so boxes
# that belong to the same column (3 over 4, 2 over 5, 1 over 6) line up.
X1, W1 = 2.55, 3.9   # columns for box 1 / box 6
X2, W2 = 7.10, 3.7   # columns for box 2 / box 5
X3, W3 = 11.65, 3.9  # columns for box 3 / box 4

# ---------------------------------------------------------------------------
# Row 1 (top, left -> right): granular balls -> purity splitting -> features
# ---------------------------------------------------------------------------
y_top = 4.6
h_top = 1.9

# Box 1: "Ball generation starts from the whole training set as one ball and
# splits every ball whose purity is below the threshold tau into two balls by
# 2-means clustering, until no impure ball is left."
box(ax, X1, y_top, W1, h_top, [
    ("Granular ball", 7.8, "bold", GRAY),
    ("generation", 7.8, "bold", GRAY),
    ("whole training set starts", 6.6, "normal", TEXT_DARK),
    ("as one ball", 6.6, "normal", TEXT_DARK),
], GRAY)

# Box 2: same sentence -- the repeated split-by-purity/tau step, by 2-means,
# "until no impure ball is left". Each ball's centre c_j, label t_j and
# purity pi_j are computed from its member training samples.
box(ax, X2, y_top, W2, h_top, [
    ("Purity-threshold", 7.8, "bold", GRAY),
    ("splitting", 7.8, "bold", GRAY),
    (r"split if $\pi_j<\tau$ by 2-means,", 6.6, "normal", TEXT_DARK),
    ("until no impure ball is left", 6.6, "normal", TEXT_DARK),
], GRAY)

# Box 3: "Stack the k centres as C. A random hidden layer with weights W,
# bias b and activation phi is drawn once and never trained, as in RVFL:
# Z = phi(CW + 1 b^T); T = [Z, C]." T_1 / T_2 split by ball label.
box(ax, X3, y_top, W3, h_top, [
    ("RVFL-style enhanced", 7.8, "bold", PURPLE),
    ("features", 7.8, "bold", PURPLE),
    (r"$Z=\phi(CW+\mathbf{1}b^\top)$, $T=[Z,C]$", 6.8, "normal", TEXT_DARK),
    (r"random $W,b$ never trained; $T_1,T_2$ by label", 6.2, "normal", TEXT_DARK),
], PURPLE)

arrow(ax, (X1 + W1 / 2 + 0.08, y_top), (X2 - W2 / 2 - 0.08, y_top))
arrow(ax, (X2 + W2 / 2 + 0.08, y_top), (X3 - W3 / 2 - 0.08, y_top))

# ---------------------------------------------------------------------------
# Row 2 (bottom, right -> left): twin-SVM QP -> per-fold selection -> decision
# ---------------------------------------------------------------------------
y_bot = 1.4
h_bot = 2.0

# Box 4 (aligned under Box 3): "EF-GBTSVM solves two problems, one per
# hyperplane": two primal QPs, converted to two dual box-constrained QPs,
# "one per ball of the other class, so the problem size is the number of
# balls, not the number of samples."
box(ax, X3, y_bot, W3, h_bot, [
    ("Twin SVM dual QP", 7.8, "bold", BLUE),
    ("(two hyperplanes)", 7.8, "bold", BLUE),
    ("one dual QP per ball of the", 6.4, "normal", TEXT_DARK),
    ("other class -- size = #balls", 6.4, "normal", TEXT_DARK),
], BLUE)
arrow(ax, (X3, y_top - h_top / 2 - 0.05), (X3, y_bot + h_bot / 2 + 0.05))

# Box 5: "Which variant to use is decided inside each outer fold, before any
# test battery is touched ... every candidate ... is tuned on the other two
# [training] batteries with 150 candidates and scored on the held-out one by
# balanced accuracy, on noisy labels. The candidate with the highest mean
# validation balanced accuracy ... is used for that fold." Candidates'
# tau-rules from Table 1 (EF-GBTSVM-L8, EF-GBTSVM-J).
box(ax, X2, y_bot, W2, h_bot, [
    ("Per-fold variant selection", 7.5, "bold", ORANGE),
    ("(L8 vs J), by validation", 6.8, "bold", ORANGE),
    (r"L8: $\tau\!\in\!\{.55,\ldots,.98\}$ (8 vals)", 6.0, "normal", TEXT_DARK),
    (r"J: $\tau=\mathrm{clip}(.95-\hat\eta+\Delta,.55,.98)$", 5.8, "normal", TEXT_DARK),
], ORANGE)
arrow(ax, (X3 - W3 / 2 - 0.08, y_bot), (X2 + W2 / 2 + 0.08, y_bot))

# Box 6: decision rule, "y_hat(x) = argmin over i of |w_i^T psi(x) + b_i| /
# ||w_i||, i in {1,2}."
box(ax, X1, y_bot, W1, h_bot, [
    ("Prediction /", 7.8, "bold", GREEN),
    ("decision rule", 7.8, "bold", GREEN),
    (r"$\hat y(x)=\arg\min_i \dfrac{|w_i^\top\psi(x)+b_i|}{\|w_i\|}$", 6.8, "normal", TEXT_DARK),
], GREEN)
arrow(ax, (X2 - W2 / 2 - 0.08, y_bot), (X1 + W1 / 2 + 0.08, y_bot))

fig.tight_layout(pad=0.25)

os.makedirs("results", exist_ok=True)
fig.savefig("results/diagram_pf_architecture.png", dpi=300)
fig.savefig("results/diagram_pf_architecture.pdf")
print("Saved results/diagram_pf_architecture.png and .pdf")
