"""Friedman + Nemenyi test across datasets (needs >= 3 models and >= 2 datasets).

python stats_tests.py results/*_noise0_runs.csv --metric ACC
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--metric", default="ACC")
    ap.add_argument("--per-file", action="store_true",
                    help="treat every input file as its own dataset (file stem), "
                         "e.g. noise0/noise10 runs that share the dataset name")
    a = ap.parse_args()

    frames = []
    for f in a.files:
        d = pd.read_csv(f)
        if a.per_file:
            d["dataset"] = os.path.basename(f).replace("_runs.csv", "")
        frames.append(d)
    df = pd.concat(frames)
    tab = df.groupby(["dataset", "model"])[a.metric].mean().unstack()   # datasets x models
    N, k = tab.shape
    ranks = np.vstack([rankdata(-row.values) for _, row in tab.iterrows()])  # rank 1 = best
    avg = pd.Series(ranks.mean(0), index=tab.columns).sort_values()

    stat, p = friedmanchisquare(*[tab[c].values for c in tab.columns])
    qa = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    cd = qa * np.sqrt(k * (k + 1) / (6 * N))

    print(tab.round(2).to_string(), "\n")
    print("Average ranks (1 = best):\n", avg.round(2).to_string(), "\n")
    print(f"Friedman chi2 = {stat:.3f}, p = {p:.4g}  (N={N} datasets, k={k} models)")
    print(f"Nemenyi critical difference (alpha=0.05) = {cd:.3f}")
    best = avg.index[0]
    for m in avg.index[1:]:
        d = avg[m] - avg[best]
        print(f"  {best} vs {m}: rank diff {d:.2f} -> {'significant' if d > cd else 'not significant'}")


if __name__ == "__main__":
    main()
