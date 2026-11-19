"""Vector diagrams for the manuscript: workflow (Fig. 1), cell schematic (Fig. 2),
EF-GBTSVM-PF architecture (Fig. 5), and the submission-system graphical abstract.

Only the steps that the implemented method actually performs are drawn.  Text inside
boxes is kept to short key points so that nothing overflows at print size.  The one
embedded data panel (stage 3 of Fig. 1) is drawn from the frozen result summaries.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.015,
})

INK = "#1d2b36"
EDGE = "#4a6983"
BLUE, GREEN, AMBER, VIOLET, ROSE = "#e6f0f9", "#e7f4ee", "#fdf2de", "#f1ecfa", "#fbeaea"
ACCENT = "#2b6ca3"
RED = "#b2182b"


def band(ax, x, y, w, h, fc, ec=EDGE, lw=1.1, z=1, r=0.016):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.006,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def head(ax, x, y, w, text, fs=8.6, color=INK):
    fit(ax, x, y, w, [text], fs, color=color, weight="bold")


def item(ax, x, y, w, h, lines, fs=6.5, fc="white", ec="#9db5c8"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.010",
                                fc=fc, ec=ec, lw=0.7, zorder=3))
    fit(ax, x, y + h / 2, w, lines, fs)


def arrow(ax, a, b, color=ACCENT, lw=1.3, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11,
                                 lw=lw, color=color, zorder=5,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def inset(fig, ax, x, y, w, h):
    """Axes placed at diagram coordinates (the main axes fills the figure)."""
    return fig.add_axes([x, y, w, h])


def fit(ax, x, y, w, lines, fs, color=INK, weight="normal", va="center"):
    """Draw centred text and shrink it until it fits inside width w."""
    t = ax.text(x + w / 2, y, "\n".join(lines), ha="center", va=va,
                fontsize=fs, color=color, fontweight=weight,
                linespacing=1.38, zorder=4)
    ax.figure.canvas.draw()
    for _ in range(14):
        bb = t.get_window_extent().transformed(ax.transData.inverted())
        if bb.width <= w * 0.93:
            break
        fs -= 0.25
        t.set_fontsize(fs)
        ax.figure.canvas.draw()
    return t


# ===================================================================== Figure 1
def figure1():
    fig, ax = canvas(7.1, 3.55)
    cols = [0.012, 0.345, 0.678]
    W, Y, H = 0.310, 0.135, 0.735
    titles = [("1  ·  DATA AND LEAKAGE AUDIT", BLUE),
              ("2  ·  PHYSICS FEATURES", GREEN),
              ("3  ·  NESTED LOBO EVALUATION", AMBER)]
    for x, (t, c) in zip(cols, titles):
        band(ax, x, Y, W, H, c)
        head(ax, x, Y + H - 0.045, W, t, fs=7.6)

    # --- stage 1
    x = cols[0]
    item(ax, x + 0.018, Y + 0.515, W - 0.036, 0.145,
         ["NASA PCoE: 4 cells, 636 cycles",
          "B0005 · B0006 · B0007 · B0018"])
    item(ax, x + 0.018, Y + 0.345, W - 0.036, 0.145,
         ["Audit: B0005–B0007 share one",
          "discharge time vector"])
    item(ax, x + 0.018, Y + 0.175, W - 0.036, 0.145,
         ["Drop whole-record timing",
          "Label: SOH < 0.80 is degraded"])
    # four cell icons
    for i, lab in enumerate(["B0005", "B0006", "B0007", "B0018"]):
        cx = x + 0.047 + i * 0.058
        ax.add_patch(FancyBboxPatch((cx, Y + 0.052), 0.030, 0.082,
                                    boxstyle="round,pad=0.002,rounding_size=0.008",
                                    fc="#ccd9e4", ec=EDGE, lw=0.7, zorder=3))
        ax.add_patch(Rectangle((cx + 0.010, Y + 0.134), 0.010, 0.012,
                               fc="#9fb3c4", ec=EDGE, lw=0.5, zorder=3))
        ax.text(cx + 0.015, Y + 0.030, lab, ha="center", fontsize=5.3, color=INK)

    # --- stage 2
    x = cols[1]
    item(ax, x + 0.018, Y + 0.515, W - 0.036, 0.145,
         ["Loaded segment only, before",
          "the cell's voltage cut-off"])
    item(ax, x + 0.018, Y + 0.345, W - 0.036, 0.145,
         ["6 base measures: $t_{v38}$, $t_{v35}$,",
          "$\\Delta t_{38,35}$, $V_{300}$, $V_{1000}$, $\\Delta T_{35}$"])
    item(ax, x + 0.018, Y + 0.175, W - 0.036, 0.145,
         ["+ 6 forms relative to each",
          "cell's first cycle = 12 inputs"])
    # tiny discharge-curve icon
    axi = inset(fig, ax, 0.420, 0.163, 0.160, 0.092)
    t = np.linspace(0, 1, 120)
    axi.plot(t, 4.2 - 0.55 * t - 0.75 * t ** 6, color=ACCENT, lw=1.1)
    axi.axhline(3.5, color=RED, lw=0.6, ls=":")
    axi.text(0.03, 3.54, "3.5 V", fontsize=4.6, color=RED)
    axi.set_xticks([]); axi.set_yticks([])
    for s in axi.spines.values():
        s.set_linewidth(0.5); s.set_color("#9db5c8")
    axi.set_facecolor("white")
    axi.set_title("loaded discharge", fontsize=5.0, color=INK, pad=1.5)

    # --- stage 3
    x = cols[2]
    item(ax, x + 0.018, Y + 0.515, W - 0.036, 0.145,
         ["Outer: leave one battery out",
          "4 folds, test cell untouched"])
    item(ax, x + 0.018, Y + 0.345, W - 0.036, 0.145,
         ["Inner: grouped CV on the 3",
          "training cells picks the variant"])
    item(ax, x + 0.018, Y + 0.175, W - 0.036, 0.145,
         ["17 models, equal budget",
          "Flips on training labels only"])
    axi = inset(fig, ax, 0.753, 0.163, 0.160, 0.092)
    comb = pd.read_csv(os.path.join(RES, "finegrid_plus_phaseb_0_10_combined.csv"))
    fg2 = pd.read_csv(os.path.join(RES, "finegrid2_summary_table.csv"))
    pf = [comb[(comb.model == "EF-GBTSVM-PF") & (comb.noise_pct == v)]["ACC_mean"].iloc[0]
          for v in (0, 6)]
    pf.append(fg2[(fg2.model == "EF-GBTSVM-PF") & (fg2.noise_pct == 68)]["ACC_mean"].iloc[0])
    axi.plot(range(3), pf, "o-", color=RED, lw=1.0, ms=2.6)
    axi.set_xticks(range(3))
    axi.set_xticklabels(["0", "6", "6.8"], fontsize=4.2)
    axi.set_yticks([])
    axi.tick_params(length=1, pad=1)
    for s in axi.spines.values():
        s.set_linewidth(0.5); s.set_color("#9db5c8")
    axi.set_title("PF accuracy (%)", fontsize=5.0, color=INK, pad=1.5)

    for x in (cols[0] + W, cols[1] + W):
        arrow(ax, (x + 0.004, Y + H / 2), (x + 0.027, Y + H / 2))

    ax.text(0.5, 0.055,
            "Reported conditions:  0 % and 6 % training-label noise,   ·   "
            "plus a 6.8 % point chosen after the 6 % result (exploratory)",
            ha="center", fontsize=6.6, color="#49606f")
    fig.savefig(os.path.join(OUT, "figure1_pipeline.pdf"))
    plt.close(fig)


# ===================================================================== Figure 2
def figure2():
    fig, ax = canvas(7.1, 2.85)
    # cylindrical cell with cut-away
    cx, cy, cw, ch = 0.045, 0.235, 0.105, 0.455
    ax.add_patch(FancyBboxPatch((cx, cy), cw, ch,
                                boxstyle="round,pad=0.002,rounding_size=0.030",
                                fc="#d6dde3", ec=INK, lw=1.2))
    ax.add_patch(Ellipse((cx + cw / 2, cy + ch), cw * 0.93, 0.045,
                         fc="#eef1f4", ec=INK, lw=1.0))
    ax.add_patch(Ellipse((cx + cw / 2, cy), cw * 0.93, 0.045,
                         fc="#c3cad0", ec=INK, lw=1.0))
    ax.add_patch(FancyBboxPatch((cx + cw * 0.33, cy + ch - 0.004), cw * 0.34, 0.048,
                                boxstyle="round,pad=0.001,rounding_size=0.008",
                                fc="#a9b4bc", ec=INK, lw=0.8))
    # spiral hint
    th = np.linspace(0, 4.6 * np.pi, 400)
    rr = np.linspace(0.004, cw * 0.36, 400)
    ax.plot(cx + cw / 2 + rr * np.cos(th), cy + ch + rr * np.sin(th) * 0.195,
            color="#8d99a3", lw=0.5)
    ax.text(cx + cw / 2, cy + ch + 0.085, "18650 cylindrical cell",
            ha="center", fontsize=7.6, fontweight="bold", color=INK)
    ax.text(cx + cw / 2, cy - 0.065, "jelly-roll winding", ha="center",
            fontsize=6.2, color="#55656f")
    ax.text(cx + cw / 2 + 0.002, cy + ch + 0.042, "+", ha="center", fontsize=8,
            color=INK, fontweight="bold")
    ax.text(cx + cw / 2, cy - 0.016, "−", ha="center", fontsize=9, color=INK,
            fontweight="bold")
    arrow(ax, (0.178, 0.46), (0.232, 0.46), lw=1.4)

    # layered cross-section
    xs = 0.255
    layers = [("Cu current\ncollector", 0.052, "#c88a52"),
              ("Graphite\nanode  (−)", 0.118, "#4a4a4a"),
              ("Separator +\nelectrolyte", 0.100, "#f4e0a1"),
              ("LiCoO$_2$\ncathode  (+)", 0.118, "#5b8fc4"),
              ("Al current\ncollector", 0.052, "#b8bfc6")]
    ylo, yhi = 0.325, 0.595
    x = xs
    for name, w, c in layers:
        ax.add_patch(Rectangle((x, ylo), w, yhi - ylo, fc=c, ec=INK, lw=0.8))
        tc = "white" if c in ("#4a4a4a", "#5b8fc4") else INK
        ax.text(x + w / 2, ylo - 0.055, name, ha="center", va="top",
                fontsize=6.2, color=INK, linespacing=1.3)
        x += w
    xend = x
    ax.text((xs + xend) / 2, 0.925, "Layered cross-section during discharge",
            ha="center", va="center", fontsize=7.8, fontweight="bold", color=INK)

    # Li+ transport inside the cell: anode -> separator -> cathode
    for k in range(3):
        yy = ylo + 0.062 + k * 0.073
        arrow(ax, (xs + 0.172, yy), (xs + 0.282, yy), color="#2e7d32", lw=1.0)
    ax.text(xs + 0.227, yhi + 0.038, "Li$^{+}$ across the separator",
            ha="center", va="center", fontsize=6.3, color="#2e7d32")

    # external circuit: anode (-) -> load -> cathode (+)
    ytop, xa, xc = 0.775, xs + 0.111, xend - 0.111
    xl0, xl1 = 0.430, 0.542
    ax.plot([xa, xa], [yhi + 0.072, ytop], color="#a0392c", lw=1.2)
    ax.plot([xc, xc], [yhi + 0.072, ytop], color="#a0392c", lw=1.2)
    ax.plot([xa, xl0], [ytop, ytop], color="#a0392c", lw=1.2)
    ax.plot([xl1, xc], [ytop, ytop], color="#a0392c", lw=1.2)
    arrow(ax, (0.382, ytop), (0.414, ytop), color="#a0392c", lw=1.2)
    arrow(ax, (0.556, ytop), (0.588, ytop), color="#a0392c", lw=1.2)
    ax.add_patch(FancyBboxPatch((xl0, ytop - 0.032), xl1 - xl0, 0.064,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc="white", ec=INK, lw=1.0))
    ax.text((xl0 + xl1) / 2, ytop, "load", ha="center", va="center",
            fontsize=6.6, color=INK)
    ax.text((xs + xend) / 2, ytop + 0.072, "e$^{-}$ through the external circuit",
            ha="center", va="center", fontsize=6.4, color="#a0392c")

    # measured signals panel
    bx = 0.735
    band(ax, bx, 0.235, 0.252, 0.475, "#f3f7fb")
    head(ax, bx, 0.665, 0.252, "What the recordings provide", fs=7.4)
    for i, (sym, txt) in enumerate([
            ("V(t)", "terminal voltage, loaded segment"),
            ("I(t)", "discharge current to cut-off"),
            ("T(t)", "surface temperature rise"),
            ("Q", "capacity per cycle → SOH → label")]):
        yy = 0.600 - i * 0.082
        ax.text(bx + 0.020, yy, sym, fontsize=6.6, fontweight="bold",
                color=ACCENT, va="center")
        ax.text(bx + 0.062, yy, txt, fontsize=5.6, color=INK, va="center")
    fit(ax, bx + 0.012, 0.283, 0.228,
        ["the 12 model inputs are built from V, I and T"], 5.6, color="#55656f")

    ax.text(0.5, 0.075,
            "Conceptual illustration of the discharge path and of the quantities the NASA records contain; "
            "it is not a measurement taken from those records.",
            ha="center", fontsize=6.3, color="#55656f")
    fig.savefig(os.path.join(OUT, "fig12_18650_discharge_schematic.pdf"))
    plt.close(fig)


# ===================================================================== Figure 5
def figure5():
    fig, ax = canvas(7.1, 3.5)
    ax.text(0.5, 0.965, "EF-GBTSVM-PF: per-fold variant selection inside the training cells",
            ha="center", va="top", fontsize=8.8, fontweight="bold", color=INK)

    # ---- row 1: the candidate pipeline
    band(ax, 0.012, 0.565, 0.976, 0.335, "#f7fafc", ec="#c3d2de")
    head(ax, 0.012, 0.862, 0.976, "A   One candidate variant, fitted on the three training cells", fs=7.4)
    steps = [
        (0.030, "Outer fold", ["3 cells train", "1 cell held out"], BLUE),
        (0.228, "Training-label flips", ["$\\lfloor\\rho n\\rfloor$ symmetric flips", "test labels stay clean"], AMBER),
        (0.426, "Granular balls", ["2-means splits until", "purity $\\pi_j \\geq \\tau$"], GREEN),
        (0.624, "Enhanced features", ["random hidden map", "$T=[\\phi(CW+b),\\,C]$"], VIOLET),
        (0.822, "Twin planes", ["two nonparallel", "class hyperplanes"], ROSE),
    ]
    for x, t, body, c in steps:
        band(ax, x, 0.615, 0.148, 0.205, c, lw=0.9)
        ax.text(x + 0.074, 0.775, t, ha="center", fontsize=6.9,
                fontweight="bold", color=INK)
        ax.text(x + 0.074, 0.683, "\n".join(body), ha="center", va="center",
                fontsize=6.0, color=INK, linespacing=1.4)
    for x in (0.178, 0.376, 0.574, 0.772):
        arrow(ax, (x + 0.004, 0.717), (x + 0.046, 0.717), lw=1.1)

    # ball-splitting icon
    rng = np.random.default_rng(3)
    axi = inset(fig, ax, 0.531, 0.621, 0.040, 0.040)
    for (mx, my, r, col) in [(0.3, 0.6, 0.20, "#4f9e6a"), (0.68, 0.66, 0.15, "#4f9e6a"),
                             (0.5, 0.26, 0.17, "#c2763f")]:
        axi.add_patch(Circle((mx, my), r, fc="none", ec=col, lw=0.8))
        p = rng.normal([mx, my], r * 0.4, size=(5, 2))
        axi.scatter(p[:, 0], p[:, 1], s=1.1, color=col)
    axi.set_xlim(0, 1); axi.set_ylim(0, 1)
    axi.set_xticks([]); axi.set_yticks([]); axi.axis("off")

    # ---- row 2: selection
    band(ax, 0.012, 0.115, 0.470, 0.395, "#eef4fa")
    head(ax, 0.012, 0.472, 0.470, "B   Inner CV uses training cells only", fs=7.4)
    item(ax, 0.034, 0.345, 0.426, 0.095,
         ["Rotate each of the 3 training cells as validation"], fs=6.3)
    item(ax, 0.034, 0.238, 0.426, 0.095,
         ["Score 5 candidates by mean inner BACC:",
          "EF-GBTSVM,  L8,  J,  M,  JM"], fs=6.3)
    item(ax, 0.034, 0.131, 0.426, 0.095,
         ["No label from the held-out cell enters selection"], fs=6.3)

    band(ax, 0.518, 0.115, 0.470, 0.395, "#fdf6ea")
    head(ax, 0.518, 0.472, 0.470, "C   Refit, then a single test evaluation", fs=7.4)
    item(ax, 0.540, 0.345, 0.426, 0.095,
         ["Refit the selected variant on all 3 noisy training cells"], fs=6.3)
    item(ax, 0.540, 0.238, 0.426, 0.095,
         ["Evaluate once on the untouched held-out cell"], fs=6.3)
    item(ax, 0.540, 0.131, 0.426, 0.095,
         ["Recorded picks:  L8 for B0005, B0007, B0018;  J for B0006"], fs=6.3)

    arrow(ax, (0.484, 0.313), (0.514, 0.313), lw=1.2)
    arrow(ax, (0.900, 0.610), (0.900, 0.515), lw=1.2)
    ax.text(0.912, 0.562, "per candidate", fontsize=5.8, color=ACCENT, va="center")
    arrow(ax, (0.100, 0.515), (0.100, 0.610), lw=1.2, rad=0.0)
    ax.text(0.112, 0.562, "repeat for 4 outer cells", fontsize=5.8,
            color=ACCENT, va="center")

    ax.text(0.5, 0.042,
            "The same procedure is repeated at every reported noise condition; "
            "variant choice, hyperparameters and standardization are all fitted inside the outer training fold.",
            ha="center", fontsize=6.2, color="#55656f")
    fig.savefig(os.path.join(OUT, "figure5_pf_nested_cv.pdf"))
    plt.close(fig)


# ============================================================ graphical abstract
def graphical_abstract():
    fig, ax = canvas(7.1, 2.0)
    items = [(0.012, "NASA PCoE", ["4 cells, 636 cycles", "shared-recording audit"], BLUE),
             (0.212, "12 features", ["loaded-discharge", "+ first-cycle relative"], GREEN),
             (0.412, "EF-GBTSVM-PF", ["granular balls, RVFL map,", "twin planes; per-fold pick"], VIOLET),
             (0.612, "LOBO benchmark", ["17 models, 4 held-out cells", "equal search budget"], AMBER),
             (0.812, "Label noise", ["0 %, 6 %, 6.8 %", "flips on training only"], ROSE)]
    for x, t, body, c in items:
        band(ax, x, 0.34, 0.176, 0.44, c)
        ax.text(x + 0.088, 0.690, t, ha="center", fontsize=7.6,
                fontweight="bold", color=INK)
        ax.text(x + 0.088, 0.500, "\n".join(body), ha="center", va="center",
                fontsize=6.2, color=INK, linespacing=1.45)
    for x in (0.188, 0.388, 0.588, 0.788):
        arrow(ax, (x + 0.004, 0.56), (x + 0.020, 0.56), lw=1.1)
    ax.text(0.5, 0.215,
            "EF-GBTSVM-PF ranks 1st of 17 at 6 % and at the exploratory 6.8 % point, "
            "and 12th of 17 on clean labels.",
            ha="center", fontsize=6.8, color=INK)
    ax.text(0.5, 0.115, "Four related cells from one dataset: the ranks are descriptive, not evidence of broad superiority.",
            ha="center", fontsize=6.2, color="#55656f")
    fig.savefig(os.path.join(OUT, "graphical_abstract.pdf"))
    plt.close(fig)


for f in (figure1, figure2, figure5, graphical_abstract):
    f()
    print("built", f.__name__)
