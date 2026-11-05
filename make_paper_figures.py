"""Figures and chart data for the paper (Phase C). Reads results only; trains nothing.

    python make_paper_figures.py      (after phase_b_analysis.py)

Writes into paper_figures/ (new folder; results/ is never touched):
  console_phase_b_analysis.txt   full stdout of `python phase_b_analysis.py`, captured verbatim
  console_*.png                  that captured stdout, cut into the blocks the paper shows and
                                 rendered in a monospace font ("terminal output" figures; the text
                                 is the program's own output, nothing retyped)
  chart_acc_by_noise.csv         mean ACC per model x noise (main pool), for the doc chart
  chart_pooled_ranks.csv         pooled Friedman average ranks + Nemenyi CD (main pool)
  chart_pf_vs_ef_folds.csv       per-fold ACC of EF-GBTSVM-PF vs original EF-GBTSVM
  chart_train_gap.csv            mean train-vs-test gap per model x noise
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import phase_b_analysis as pba

OUT = "paper_figures"
FONT_CANDIDATES = [r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\cour.ttf"]


def capture_console():
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    res = subprocess.run([sys.executable, "phase_b_analysis.py"], capture_output=True, text=True,
                         encoding="utf-8", env=env, check=True)
    with open(os.path.join(OUT, "console_phase_b_analysis.txt"), "w", encoding="utf-8") as f:
        f.write(res.stdout)
    return res.stdout.splitlines()


def block(lines, start, stop=None):
    """lines from the first line starting with `start` up to (not incl.) the next line starting
    with `stop` (or the first blank line when stop is None)."""
    i = next(k for k, l in enumerate(lines) if l.startswith(start))
    j = i + 1
    while j < len(lines) and not (lines[j].startswith(stop) if stop else lines[j].strip() == ""):
        j += 1
    return lines[i:j]


def render(lines, path, title):
    font = next((ImageFont.truetype(p, 15) for p in FONT_CANDIDATES if os.path.exists(p)), ImageFont.load_default())
    text = ["$ python phase_b_analysis.py   # " + title, ""] + lines
    w = max(font.getlength(l) for l in text) + 40
    lh = 20
    img = Image.new("RGB", (int(w), lh * len(text) + 30), (24, 24, 24))
    d = ImageDraw.Draw(img)
    for k, l in enumerate(text):
        d.text((20, 15 + k * lh), l, font=font, fill=(120, 200, 120) if k == 0 else (230, 230, 230))
    img.save(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = capture_console()
    shots = {
        "console_claims_main.png": ("claims (a)-(c), main pool",
                                    block(lines, "(a) ", "(b) [SENSITIVITY")),
        "console_friedman_main.png": ("Friedman + Nemenyi, main pool, pooled",
                                      block(lines, "--- pooled (4 folds x 3 noise): k=15")),
        "console_wilcoxon_main.png": ("Wilcoxon PF vs each, main pool, pooled",
                                      block(lines, "--- Wilcoxon signed-rank, EF-GBTSVM-PF vs each, pooled")),
        "console_train_gap.png": ("train-vs-test gap",
                                  block(lines, "Train-vs-test gap", "  gap > 5 pp")
                                  + block(lines, "  gap > 5 pp")),
    }
    for name, (title, blk) in shots.items():
        render(blk, os.path.join(OUT, name), title)

    df, _, stump_gap = pba.load()
    sel = pba.per_fold_selection()
    main_pool = [m for m in sorted(df["model"].unique()) if m not in pba.COMPONENTS]
    mean = (df[df.model.isin(main_pool)].groupby(["model", "noise_pct"])["ACC"].mean()
            .round(2).rename("acc_mean").reset_index())
    mean.to_csv(os.path.join(OUT, "chart_acc_by_noise.csv"), index=False)

    mat = pba.matrix(df, "ACC", pba.NOISES, main_pool)
    ranks = pba.avg_ranks(mat)
    _, cd = pba.nemenyi_cd(len(main_pool), mat.shape[1])
    pd.DataFrame({"model": ranks.index, "avg_rank": ranks.values.round(3),
                  "cd": round(cd, 3)}).to_csv(os.path.join(OUT, "chart_pooled_ranks.csv"), index=False)

    folds = df[df.model.isin([pba.PRIMARY, "EF-GBTSVM"])].copy()
    folds["test_cell"] = folds["run"].map(pba.FOLD_CELL)
    folds["ACC"] = folds["ACC"].round(2)
    folds[["model", "noise_pct", "test_cell", "ACC"]].sort_values(["noise_pct", "test_cell", "model"]) \
        .to_csv(os.path.join(OUT, "chart_pf_vs_ef_folds.csv"), index=False)

    gap = pd.read_csv(os.path.join("results", "phase_b_train_gap.csv"))
    gap = pba.add_per_fold(gap, sel, ["train_acc_noisy", "train_acc_clean", "test_acc", "gap_clean"])
    gap = pd.concat([gap, stump_gap], ignore_index=True)
    g = (gap[gap.model.isin(main_pool)].groupby(["model", "noise_pct"])["gap_clean"].mean()
         .round(2).rename("gap_mean").reset_index())
    g.to_csv(os.path.join(OUT, "chart_train_gap.csv"), index=False)
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
