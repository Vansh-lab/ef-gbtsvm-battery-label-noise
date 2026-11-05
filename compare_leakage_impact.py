"""Compare old (nasa_battery.csv) vs corrected (nasa_battery_phys.csv) experiment results.

Usage (run after Phase B experiment completes):
    python compare_leakage_impact.py

Reads:
  OLD:  results/nasa_bigsearch_noise{0,10,20}_summary.csv   (EF-GBTSVM, -W, -L, -A; paper grids, 150 cand)
        results/nasa_noise{0,10,20}_summary.csv              (other models; paper grids, n_iter=30)
  NEW:  results/phys_phase_b_noise{0,10,20}_summary.csv     (15 models, equal grids, min(150, grid) cand)

CONFOUNDED: old and new differ in (1) the feature set -- 13 features vs 12 phys features, no
column shared -- and (2) the tuning budget/grids. In the old runs SVM/RF/MLP/XGB/TSVM fully
searched their small paper grids, but GBTSVM (30/40), HF-GBTSVM (30/1000), EF-TSVM (30/200) and
RVFL (30/200) did not; the new runs use larger equal grids. So a delta is NOT the effect of
leakage removal alone (for SVM@0% and RF@10/20% it is mostly the tuning change). EF-GBTSVM also
exists in nasa_noise (30 cand); the bigsearch value overrides it. Old ranks are over 13 models,
new ranks over 15, so rank columns are not comparable.
"""
import os
import pandas as pd
import numpy as np

NOISES = [0, 10, 20]

# Which summary file to prefer for the "old" numbers, per model
BIGSEARCH_MODELS = {"EF-GBTSVM", "EF-GBTSVM-W", "EF-GBTSVM-L", "EF-GBTSVM-A"}


def load_summary(path):
    """Load a summary CSV; return dict model -> row."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, index_col=0)
    return {k: v for k, v in df.iterrows()}


def old_numbers(noise):
    """Merge bigsearch (EF family) + nasa_noise (others) for the given noise level."""
    big = load_summary(f"results/nasa_bigsearch_noise{noise}_summary.csv")
    base = load_summary(f"results/nasa_noise{noise}_summary.csv")
    merged = {**base, **big}   # bigsearch overwrites base for EF-GBTSVM family
    return merged


def new_numbers(noise):
    return load_summary(f"results/phys_phase_b_noise{noise}_summary.csv")


def rank_by_acc(d):
    """Return dict model -> rank (1=best)."""
    items = sorted(d.items(), key=lambda kv: float(kv[1]["ACC"].split("±")[0]), reverse=True)
    return {m: i + 1 for i, (m, _) in enumerate(items)}


def main():
    rows = []
    for noise in NOISES:
        old = old_numbers(noise)
        new = new_numbers(noise)
        if not new:
            print(f"[noise={noise}%] phys_phase_b results not available — skipping")
            continue
        old_ranks = rank_by_acc(old) if old else {}
        new_ranks = rank_by_acc(new)
        # Union of model names
        all_models = sorted(set(old.keys()) | set(new.keys()))
        for m in all_models:
            o_acc = float(old[m]["ACC"].split("±")[0]) if m in old else float("nan")
            n_acc = float(new[m]["ACC"].split("±")[0]) if m in new else float("nan")
            o_f1  = float(old[m]["F1"].split("±")[0]) if m in old else float("nan")
            n_f1  = float(new[m]["F1"].split("±")[0]) if m in new else float("nan")
            o_rk  = old_ranks.get(m, float("nan"))
            n_rk  = new_ranks.get(m, float("nan"))
            rows.append(dict(
                noise=f"{noise}%",
                model=m,
                acc_old=round(o_acc, 2),
                acc_new=round(n_acc, 2),
                acc_delta=round(n_acc - o_acc, 2),
                f1_old=round(o_f1, 2),
                f1_new=round(n_f1, 2),
                f1_delta=round(n_f1 - o_f1, 2),
                rank_old=o_rk,
                rank_new=n_rk,
                rank_delta=float(o_rk - n_rk) if not (pd.isna(o_rk) or pd.isna(n_rk)) else float("nan"),
            ))
    if not rows:
        print("No new results available yet.")
        return
    df = pd.DataFrame(rows)
    df.to_csv("results/leakage_impact_comparison.csv", index=False)
    print("=== Old vs Corrected (CONFOUNDED: features AND grids/budget changed) ===")
    print("Columns: model | old ACC | new ACC | Δ ACC | old rank | new rank | Δ rank\n")
    for noise in NOISES:
        sub = df[df["noise"] == f"{noise}%"]
        if sub.empty:
            continue
        print(f"--- Noise = {noise}% ---")
        print(sub[["model","acc_old","acc_new","acc_delta","rank_old","rank_new","rank_delta"]]
              .sort_values("rank_new").to_string(index=False))
        print()
    # Highlight EF-GBTSVM family and stump/SVM specifically
    print("=== EF-GBTSVM-L8 highlights (new model for Phase B) ===")
    new_l8 = df[df["model"] == "EF-GBTSVM-L8"]
    if new_l8.empty:
        print("EF-GBTSVM-L8 not found in new results (may be under EF-GBTSVM-L or absent)")
    else:
        print(new_l8[["noise","acc_old","acc_new","acc_delta","rank_old","rank_new"]].to_string(index=False))
    print("\nSaved to results/leakage_impact_comparison.csv")


if __name__ == "__main__":
    main()
