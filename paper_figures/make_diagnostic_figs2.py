"""Three more diagnostic/comparison figures for the paper, from already-computed results only.
No new experiment is run and no number is invented.

  Figure A (fig3_train_test_gap): clean-train minus test accuracy per model, per noise level.
    Source: results/phase_b_train_gap.csv (15 models, column gap_clean), plus EF-GBTSVM-PF (its
    per-fold selected variant's gap, mapping B0005->L8, B0006->J, B0007->L8, B0018->L8, Section 3)
    and Stump-fair (phys_stump.stump_runs, same definition). Both re-derived values were checked to
    reproduce the published results/phase_b_claims.txt gap table exactly.
  Figure B (fig4_dlite_comparison): stress-test accuracy (mean +/- SD) over N30/N40/BND.
    Source: results/dlite_summary_table.csv. A readable subset of 6 of the 17 models is shown
    (Table 7 already lists all 17); the caption says so.
  Figure C (fig5_class_prior): per-battery-fold train vs test degraded-class share.
    Arithmetic aggregation of the paper's Table 2 cycle/label counts (no new computation).

    python paper_figures/make_diagnostic_figs2.py
"""
import os
import sys
sys.path.insert(0, os.getcwd())            # so phys_stump imports when run from repo root

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phys_stump import stump_runs

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False})

NOISES = [0, 10, 20]

# ======================================================================================
# Figure A -- train-test overfitting gap, two panels (family vs baselines)
# ======================================================================================
gap = pd.read_csv("results/phase_b_train_gap.csv")

# EF-GBTSVM-PF: per-fold selected variant's gap_clean (published mapping), then mean/SD over folds
PF_MAP = {0: "EF-GBTSVM-L8", 1: "EF-GBTSVM-J", 2: "EF-GBTSVM-L8", 3: "EF-GBTSVM-L8"}
pf_rows = []
for noise in NOISES:
    for run, variant in PF_MAP.items():
        r = gap[(gap.noise_pct == noise) & (gap.run == run) & (gap.model == variant)]
        pf_rows.append({"noise_pct": noise, "run": run, "model": "EF-GBTSVM-PF",
                        "gap_clean": r.iloc[0]["gap_clean"]})
# Stump-fair: official phys_stump.stump_runs, gap = clean-train-acc - test-acc
stump_rows = []
for noise in NOISES:
    fair = stump_runs(noise / 100, "fair")
    fair = fair.assign(gap_clean=fair["train_acc_clean"] - fair["ACC"])
    for _, fr in fair.iterrows():
        stump_rows.append({"noise_pct": noise, "run": int(fr["run"]), "model": "Stump-fair",
                           "gap_clean": fr["gap_clean"]})
gap_all = pd.concat([gap[["noise_pct", "run", "model", "gap_clean"]],
                     pd.DataFrame(pf_rows), pd.DataFrame(stump_rows)], ignore_index=True)

g = gap_all.groupby(["model", "noise_pct"])["gap_clean"].agg(["mean", "std"]).reset_index()

FAMILY = ["EF-GBTSVM", "EF-GBTSVM-W", "EF-GBTSVM-L", "EF-GBTSVM-A", "EF-GBTSVM-L8",
          "EF-GBTSVM-J", "EF-GBTSVM-PF"]
BASELINE = ["GBTSVM", "HF-GBTSVM", "EF-TSVM", "TSVM", "SVM", "RVFL", "RandomForest",
            "MLP", "XGBoost", "Stump-fair"]

fam_colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(FAMILY)))
base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 1.95), sharey=True)
for ax, group, cols, title in [(axes[0], FAMILY, fam_colors, "EF-GBTSVM family"),
                               (axes[1], BASELINE, base_colors, "baselines and ablations")]:
    for k, m in enumerate(group):
        r = g[g.model == m].sort_values("noise_pct")
        pf = m == "EF-GBTSVM-PF"
        ax.plot(r.noise_pct, r["mean"], marker=MARKERS[k % len(MARKERS)], markersize=3.5,
                lw=1.6 if pf else 0.9, color=cols[k], label=m, zorder=5 if pf else 3,
                alpha=1.0 if pf else 0.9)
    ax.axhline(0, color="#999999", lw=0.6, ls="-", zorder=1)
    ax.axhline(5, color="#cc6666", lw=0.6, ls="--", zorder=1)
    ax.set_xticks(NOISES)
    ax.set_xlabel("training-label noise (%)")
    ax.set_title(title, fontsize=7)
    ax.grid(axis="y", color="#eeeeee", lw=0.5)
    ax.legend(frameon=False, fontsize=5, ncol=2, handletextpad=0.3, columnspacing=0.8,
              labelspacing=0.25, loc="upper left")
axes[0].set_ylabel("train$-$test accuracy gap (pp)")
axes[0].text(0, 5.3, "5 pp", fontsize=5, color="#cc6666", ha="left", va="bottom")
fig.tight_layout()
fig.savefig("results/fig3_train_test_gap.png", dpi=300)
fig.savefig("results/fig3_train_test_gap.pdf")
plt.close(fig)

# ======================================================================================
# Figure B -- D-lite stress-test accuracy, subset of models
# ======================================================================================
dl = pd.read_csv("results/dlite_summary_table.csv")
COND_ORDER = ["N30", "N40", "BND"]
SHOW = ["EF-GBTSVM-PF", "EF-GBTSVM", "Stump-fair", "SVM", "RandomForest", "RVFL"]
LABELS_B = {"EF-GBTSVM-PF": "EF-GBTSVM-PF (primary)", "EF-GBTSVM": "EF-GBTSVM (original)",
            "Stump-fair": "Stump-fair (one-feature)", "SVM": "RBF-SVM", "RandomForest": "random forest",
            "RVFL": "RVFL"}
COLORS_B = {"EF-GBTSVM-PF": "#1f5fbf", "EF-GBTSVM": "#6b6b6b", "Stump-fair": "#2a9d6b",
            "SVM": "#c98a2a", "RandomForest": "#9467bd", "RVFL": "#d62728"}
MARKERS_B = {"EF-GBTSVM-PF": "o", "EF-GBTSVM": "s", "Stump-fair": "D", "SVM": "^",
             "RandomForest": "v", "RVFL": "P"}

fig, ax = plt.subplots(figsize=(3.5, 1.95))
xpos = np.arange(len(COND_ORDER))
for m in SHOW:
    sub = dl[dl.model == m].set_index("noise_pct").loc[COND_ORDER]
    pf = m == "EF-GBTSVM-PF"
    ax.errorbar(xpos, sub.ACC_mean, yerr=sub.ACC_std, label=LABELS_B[m], color=COLORS_B[m],
                marker=MARKERS_B[m], markersize=4, lw=1.6 if pf else 1.0, capsize=2, capthick=0.8,
                elinewidth=0.8, zorder=5 if pf else 3)
ax.set_xticks(xpos)
ax.set_xticklabels(["N30", "N40", "BND"])
ax.set_xlim(-0.4, 2.4)
ax.set_xlabel("stress-test condition")
ax.set_ylabel("test accuracy (%)")
ax.grid(axis="y", color="#dddddd", lw=0.5)
ax.legend(loc="lower left", frameon=False, fontsize=5.5, handletextpad=0.3, borderaxespad=0.2,
          labelspacing=0.3)
fig.tight_layout()
fig.savefig("results/fig4_dlite_comparison.png", dpi=300)
fig.savefig("results/fig4_dlite_comparison.pdf")
plt.close(fig)

# ======================================================================================
# Figure C -- per-battery-fold train vs test degraded-class share (Table 2 arithmetic)
# ======================================================================================
# Table 2 (published): cycles, degraded count per cell
TAB2 = {"B0005": (168, 66), "B0006": (168, 108), "B0007": (168, 45), "B0018": (132, 58)}
cells = ["B0005", "B0006", "B0007", "B0018"]
test_pct, train_pct = [], []
for test in cells:
    cyc_t, deg_t = TAB2[test]
    test_pct.append(100 * deg_t / cyc_t)
    others = [c for c in cells if c != test]
    cyc_tr = sum(TAB2[c][0] for c in others)
    deg_tr = sum(TAB2[c][1] for c in others)
    train_pct.append(100 * deg_tr / cyc_tr)

fig, ax = plt.subplots(figsize=(3.5, 1.9))
x = np.arange(len(cells))
w = 0.38
ax.bar(x - w / 2, train_pct, w, label="train split (other 3 cells)", color="#6b6b6b")
ax.bar(x + w / 2, test_pct, w, label="test split (held-out cell)", color="#1f5fbf")
for xi, (tr, te) in enumerate(zip(train_pct, test_pct)):
    ax.text(xi - w / 2, tr + 1, f"{tr:.0f}", ha="center", va="bottom", fontsize=5.5, color="#6b6b6b")
    ax.text(xi + w / 2, te + 1, f"{te:.0f}", ha="center", va="bottom", fontsize=5.5, color="#1f5fbf")
ax.axhline(50, color="#cccccc", lw=0.6, ls="--", zorder=1)
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n(test)" for c in cells])
ax.set_ylim(0, 78)
ax.set_ylabel("degraded-class share (%)")
ax.grid(axis="y", color="#eeeeee", lw=0.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False, fontsize=5.5,
          handletextpad=0.4, columnspacing=1.0)
fig.tight_layout()
fig.savefig("results/fig5_class_prior.png", dpi=300)
fig.savefig("results/fig5_class_prior.pdf")
plt.close(fig)

print("wrote fig3_train_test_gap, fig4_dlite_comparison, fig5_class_prior (png+pdf) in results/")
print("\nFigure A gap means (verify PF=1.48/2.21/0.88, Stump-fair=2.11/3.96/1.31):")
print(g[g.model.isin(["EF-GBTSVM-PF", "Stump-fair"])].round(2).to_string(index=False))
print("\nFigure C train/test degraded %:")
for c, tr, te in zip(cells, train_pct, test_pct):
    print(f"  {c}: train {tr:.1f}%  test {te:.1f}%")
