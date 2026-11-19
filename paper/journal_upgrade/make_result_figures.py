"""Result figures for the three reported noise conditions: 0 %, 6 % and 6.8 %.

Sources (nothing is recomputed from raw predictions):
  finegrid_plus_phaseb_0_10_combined.csv  -> summary at 0 % and 6 %
  finegrid2_summary_table.csv             -> summary at 6.8 % (noise_pct 68)
  phase_b_per_fold.csv                    -> per-cell values at 0 %
  finegrid_runs.csv / finegrid2_runs.csv  -> per-cell values at 6 % and 6.8 %

EF-GBTSVM-PF has no row of its own in the per-fold run files because it is a derived
model: it is the variant Phase A selected for that outer cell (L8 for B0005, B0007 and
B0018; J for B0006).  Reassembling it from those rows reproduces the published summary
means exactly, which is checked at the bottom of this file.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

ORDER = [
    "EF-GBTSVM", "EF-GBTSVM-A", "EF-GBTSVM-L", "EF-GBTSVM-W",
    "EF-GBTSVM-L8", "EF-GBTSVM-J", "EF-GBTSVM-PF",
    "GBTSVM", "HF-GBTSVM", "EF-TSVM", "TSVM",
    "SVM", "RVFL", "RandomForest", "MLP", "XGBoost", "Stump-fair",
]
LABEL = dict(zip(ORDER, [
    "EF-GBTSVM", "EF-GBTSVM-A", "EF-GBTSVM-L", "EF-GBTSVM-W",
    "EF-GBTSVM-L8†", "EF-GBTSVM-J†", "EF-GBTSVM-PF",
    "GBTSVM", "HF-GBTSVM", "EF-TSVM", "TSVM",
    "RBF-SVM", "RVFL", "Random forest", "MLP", "XGBoost",
    "Single-feature thr.",
]))
PF = "EF-GBTSVM-PF"
RED = "#b2182b"
ACC_CMAP = LinearSegmentedColormap.from_list("acc", ["#f7f7f7", "#9ecae1", "#2171b5"])

CELLS = ["B0005", "B0006", "B0007", "B0018"]
PF_PICK = {0: "EF-GBTSVM-L8", 1: "EF-GBTSVM-J",
           2: "EF-GBTSVM-L8", 3: "EF-GBTSVM-L8"}

COMB = pd.read_csv(os.path.join(RES, "finegrid_plus_phaseb_0_10_combined.csv"))
FG2 = pd.read_csv(os.path.join(RES, "finegrid2_summary_table.csv"))
CONDITIONS = [("0%", COMB, 0), ("6%", COMB, 6), ("6.8%", FG2, 68)]
XT = [c[0] for c in CONDITIONS]


def grid(col):
    out = []
    for m in ORDER:
        row = []
        for _, df, key in CONDITIONS:
            r = df[(df["model"] == m) & (df["noise_pct"] == key)]
            row.append(float(r[col].iloc[0]))
        out.append(row)
    return np.array(out)


def heat(ax, M, R, title, vmin, vmax):
    im = ax.imshow(M, cmap=ACC_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(XT)))
    ax.set_xticklabels(XT, fontsize=7.4)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([LABEL[m] for m in ORDER], fontsize=6.8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            shade = (M[i, j] - vmin) / max(vmax - vmin, 1e-9)
            c = "white" if shade > 0.62 else "#1a1a1a"
            ax.text(j, i - 0.15, f"{M[i,j]:.2f}", ha="center", va="center",
                    fontsize=6.2, color=c)
            ax.text(j, i + 0.27, f"[{int(R[i,j])}]", ha="center", va="center",
                    fontsize=5.4, color=c, alpha=0.85)
    k = ORDER.index(PF)
    ax.add_patch(plt.Rectangle((-0.5, k - 0.5), M.shape[1], 1, fill=False,
                               edgecolor=RED, lw=1.5, zorder=5))
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == LABEL[PF]:
            lbl.set_color(RED)
            lbl.set_fontweight("bold")
    ax.set_title(title, fontsize=7.8, pad=4)
    ax.set_xlabel("training-label noise", fontsize=7.2)
    ax.tick_params(length=2)
    return im


# ---------------------------------------------------------------- Figure R1
def fig_main():
    A, RA = grid("ACC_mean"), grid("rank_ACC_all17")
    B, RB = grid("BACC_mean"), grid("rank_BACC_all17")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.6))
    im = heat(axes[0], A, RA, "(a) Mean accuracy (%) and rank [1 = best of 17]", 84, 100)
    heat(axes[1], B, RB, "(b) Mean balanced accuracy (%) and rank", 84, 100)
    axes[1].set_yticklabels([])
    cb = fig.colorbar(im, ax=axes, fraction=0.024, pad=0.015)
    cb.ax.tick_params(labelsize=6.2)
    cb.set_label("mean over four held-out cells (%)", fontsize=6.6)
    fig.savefig(os.path.join(FIG, "figR1_main_sweep.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Figure R2
def fig_pf_vs_best():
    A = grid("ACC_mean")
    pf = A[ORDER.index(PF)]
    rk = grid("rank_ACC_all17")[ORDER.index(PF)]
    best = A.max(axis=0)
    med = np.median(A, axis=0)
    bname = [LABEL[ORDER[int(i)]] for i in A.argmax(axis=0)]
    x = np.arange(len(XT))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.fill_between(x, med, best, color="#d9e6f2", label="median-to-best band")
    ax.plot(x, best, "o-", color="#4d4d4d", lw=1.0, ms=3.2, label="best of 17")
    ax.plot(x, med, "s--", color="#9e9e9e", lw=0.9, ms=2.8, label="pool median")
    ax.plot(x, pf, "D-", color=RED, lw=1.6, ms=4.6, label="EF-GBTSVM-PF", zorder=5)
    for xi, yi, ri in zip(x, pf, rk):
        ax.annotate(f"rank {int(ri)}", (xi, yi), textcoords="offset points",
                    xytext=(0, -13 if ri > 3 else 8), ha="center",
                    fontsize=6.2, color=RED)
    ax.annotate(bname[0], (0, best[0]), textcoords="offset points",
                xytext=(4, 4), fontsize=5.8, color="#4d4d4d")
    ax.set_xticks(x)
    ax.set_xticklabels(XT, fontsize=7.4)
    ax.set_xlim(-0.35, len(XT) - 0.5)
    ax.set_xlabel("training-label noise", fontsize=7.5)
    ax.set_ylabel("mean test accuracy (%)", fontsize=7.5)
    ax.tick_params(labelsize=7, length=2)
    ax.set_ylim(90, 100.4)
    ax.grid(axis="y", lw=0.3, color="#dddddd")
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.0, frameon=False, loc="lower right", ncol=1,
              handlelength=1.6)
    fig.savefig(os.path.join(FIG, "figR2_pf_vs_best.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- per-cell data
def per_cell_table():
    """model -> condition -> {cell: ACC}."""
    out = {m: {} for m in ORDER}

    pb = pd.read_csv(os.path.join(RES, "phase_b_per_fold.csv"),
                     header=[0, 1], index_col=[0, 1])
    for m in ORDER:
        out[m]["0%"] = {c: float(pb.loc[(m, 0), ("ACC", c)]) for c in CELLS}

    for label, fname, key in [("6%", "finegrid_runs.csv", 6),
                              ("6.8%", "finegrid2_runs.csv", 68)]:
        d = pd.read_csv(os.path.join(RES, fname))
        d = d[d["noise_pct"] == key]
        for m in ORDER:
            vals = {}
            for run, cellname in enumerate(CELLS):
                src = PF_PICK[run] if m == PF else m
                sub = d[(d["model"] == src) & (d["run"] == run)]
                vals[cellname] = float(sub["ACC"].iloc[0])
            out[m][label] = vals
    return out


# ---------------------------------------------------------------- Figure R5
def fig_per_cell(pc):
    colors = ["#2171b5", "#e6a000", "#2e8b57", "#9b59b6"]
    y = np.arange(len(ORDER))
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.3), sharey=True)
    for ax, label in zip(axes, XT):
        for ci, (c, col) in enumerate(zip(CELLS, colors)):
            vals = [pc[m][label][c] for m in ORDER]
            ax.scatter(vals, y + (ci - 1.5) * 0.17, s=8, color=col,
                       edgecolor="none", label=c if label == "0%" else None, zorder=3)
        for i, m in enumerate(ORDER):
            vals = [pc[m][label][c] for c in CELLS]
            ax.plot([min(vals), max(vals)], [i, i], lw=0.6, color="#bbbbbb", zorder=1)
        ax.set_title(f"{label} training-label noise", fontsize=7.8, pad=3)
        ax.set_xlabel("held-out cell accuracy (%)", fontsize=7.2)
        ax.set_xlim(48, 102)
        ax.tick_params(labelsize=6.8, length=2)
        ax.grid(axis="x", lw=0.3, color="#e8e8e8")
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABEL[m] for m in ORDER], fontsize=6.8)
    axes[0].invert_yaxis()
    for lbl in axes[0].get_yticklabels():
        if lbl.get_text() == LABEL[PF]:
            lbl.set_color(RED)
            lbl.set_fontweight("bold")
    axes[0].legend(fontsize=6.2, frameon=False, loc="lower left", ncol=2,
                   handletextpad=0.2, columnspacing=0.8)
    fig.savefig(os.path.join(FIG, "figR5_per_cell.pdf"))
    plt.close(fig)


pc = per_cell_table()

# consistency check: per-cell means must reproduce the published summary means
print("per-cell mean vs. published summary mean (accuracy):")
for label, df, key in CONDITIONS:
    for m in [PF, "EF-GBTSVM", "RandomForest"]:
        got = np.mean([pc[m][label][c] for c in CELLS])
        want = float(df[(df.model == m) & (df.noise_pct == key)]["ACC_mean"].iloc[0])
        flag = "ok" if abs(got - want) < 0.005 else "**MISMATCH**"
        print(f"  {label:>5} {m:<14} {got:6.2f} vs {want:6.2f}  {flag}")

for fn in (fig_main, fig_pf_vs_best):
    fn()
    print("built", fn.__name__)
fig_per_cell(pc)
print("built fig_per_cell")

for stale in ["figR3_fine_sweep.pdf", "figR4_stress.pdf"]:
    p = os.path.join(FIG, stale)
    if os.path.exists(p):
        os.remove(p)
        print("removed stale", stale)
