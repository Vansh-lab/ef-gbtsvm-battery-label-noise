"""Phase F (fine-grid) analysis -- PREREG_FINEGRID.md section 3. Data only; no new statistical
test, no claim, nothing written to the paper (section 5's claim rule).

    python finegrid_analysis.py

Inputs : results/finegrid_N0{2,4,6,8}_runs.csv  (run_finegrid.py output: 15 models x 4 folds,
         plus Stump-fair/Stump-oracle from phys_stump.stump_runs, per condition)
         results/phase_a/phys2_selection.csv     (same per-fold PF selection as Phase B/D-lite)
Outputs: results/finegrid_runs.csv            the four condition files concatenated, unchanged rows
         results/finegrid_summary_table.csv   same columns as phase_b_summary_table.csv
         results/finegrid_train_gap.csv       same columns as phase_b_train_gap.csv

Unlike phase_b_train_gap.py, no refit is done here: run_finegrid.py already recorded
train_acc_noisy/train_acc_clean from the one fit each model received, and ACC there IS the test
accuracy from that same fit (test labels are never flipped). So test_acc/logged_test_acc below are
both just that ACC column, and reproduced is trivially True by construction -- this file does NOT
re-verify reproducibility the way phase_b_train_gap.py's refit does; it only extracts what
run_finegrid.py already computed, under the same gap_clean definition.

PF (EF-GBTSVM-PF) reconstruction and the main(15)/sensitivity(17) pools are the identical logic
and model lists phase_b_analysis.py uses for 0/10/20%, applied to noise_pct in {2,4,6,8}. Stump-
oracle is kept out of both the per-fold reuse and the pools (reported as upper-bound reference
only), matching phase_b_analysis.load()'s separation.
"""
import os

import pandas as pd

NOISES = [2, 4, 6, 8]
COND = {2: "N02", 4: "N04", 6: "N06", 8: "N08"}
METRICS = ["ACC", "BACC", "F1", "AUC"]
PRIMARY = "EF-GBTSVM-PF"
COMPONENTS = ["EF-GBTSVM-L8", "EF-GBTSVM-J"]
SELECTION = "results/phase_a/phys2_selection.csv"
OUT = "results"


def per_fold_selection():
    sel = pd.read_csv(SELECTION)
    return dict(zip(sel["fold"].astype(int), sel["selected"]))


def add_per_fold(df, sel, value_cols):
    pick = pd.concat([df[(df["run"] == r) & (df["model"] == m)] for r, m in sel.items()])
    cnt = pick.groupby(["noise_pct", "run"]).size()
    assert len(cnt) == len(NOISES) * len(sel) and (cnt == 1).all(), "need exactly one row per (noise, run)"
    pick = pick.assign(selected_variant=pick["model"], model=PRIMARY)
    return pd.concat([df, pick[["run", "model", "noise_pct", "selected_variant"] + value_cols]],
                     ignore_index=True)


def main():
    raw = pd.concat([pd.read_csv(f"results/finegrid_{COND[n]}_runs.csv") for n in NOISES],
                     ignore_index=True)
    raw.to_csv(os.path.join(OUT, "finegrid_runs.csv"), index=False)
    print(f"finegrid_runs.csv: {len(raw)} rows, noise_pct={sorted(raw['noise_pct'].unique())}, "
          f"models={sorted(raw['model'].unique())}")

    oracle = raw[raw["model"] == "Stump-oracle"].copy()
    df = raw[raw["model"] != "Stump-oracle"].copy()[["run", "model", "noise_pct"] + METRICS]
    df[METRICS] = df[METRICS].round(6)
    sel = per_fold_selection()
    df = add_per_fold(df, sel, METRICS)

    allmodels = sorted(df["model"].unique())
    assert len(allmodels) == 17, allmodels
    main_pool = [m for m in allmodels if m not in COMPONENTS]
    sens_pool = allmodels

    summ = df.groupby(["noise_pct", "model"])[METRICS].agg(["mean", "std"]).round(2)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ["in_main_pool"] = summ["model"].isin(main_pool)
    for met in ["ACC", "BACC"]:
        mp = summ[summ.in_main_pool]
        summ[f"rank_{met}_main15"] = (mp.groupby("noise_pct")[f"{met}_mean"]
                                      .rank(ascending=False, method="min").astype("Int64"))
        summ[f"rank_{met}_all17"] = (summ.groupby("noise_pct")[f"{met}_mean"]
                                     .rank(ascending=False, method="min").astype(int))
    summ.to_csv(os.path.join(OUT, "finegrid_summary_table.csv"), index=False)
    print(f"finegrid_summary_table.csv: {len(summ)} rows (main pool k={len(main_pool)}, "
          f"sensitivity pool k={len(sens_pool)})")
    print(f"Stump-oracle (upper-bound reference, not pooled/ranked) mean ACC by noise: "
          f"{oracle.groupby('noise_pct')['ACC'].mean().round(2).to_dict()}")

    # Matches phase_b_train_gap.csv's own column structure exactly: that raw file holds only the
    # 15 run_experiments.py models (no Stump-fair, no PF) -- phase_b_analysis.py adds Stump-fair
    # and reconstructs PF's gap only in memory, for the claims text, never re-saving it to CSV.
    # PF's/Stump-fair's gap_clean for these 4 conditions can be derived the same way from
    # finegrid_runs.csv + results/phase_a/phys2_selection.csv if needed later; not pre-computed here.
    base_models = [m for m in sorted(raw["model"].unique()) if m not in ("Stump-fair", "Stump-oracle")]
    assert len(base_models) == 15, base_models
    gap_raw = raw[raw["model"].isin(base_models)].copy()
    gap_raw["test_acc"] = gap_raw["ACC"]
    gap_raw["logged_test_acc"] = gap_raw["ACC"]
    gap_raw["reproduced"] = True
    gap_raw["gap_clean"] = gap_raw["train_acc_clean"] - gap_raw["test_acc"]
    gap_cols = ["noise_pct", "run", "model", "train_acc_noisy", "train_acc_clean", "test_acc",
                "logged_test_acc", "reproduced", "gap_clean"]
    gap = gap_raw[gap_cols].sort_values(["noise_pct", "run", "model"]).reset_index(drop=True)
    gap.to_csv(os.path.join(OUT, "finegrid_train_gap.csv"), index=False)
    print(f"finegrid_train_gap.csv: {len(gap)} rows (15 base models only, matching "
          f"phase_b_train_gap.csv's own column/row structure)")

    print("\n=== Raw summary, ALL models, ALL 4 new noise levels (no filtering) ===")
    show = summ.sort_values(["noise_pct", "ACC_mean"], ascending=[True, False])
    print(show.to_string(index=False))

    print("\n=== Who is rank 1 (ACC, main pool) at each new noise level ===")
    for n in NOISES:
        top = summ[(summ.noise_pct == n) & (summ.in_main_pool) & (summ.rank_ACC_main15 == 1)]
        print(f"  {n}%: {top['model'].tolist()}  (ACC={top['ACC_mean'].tolist()})")


if __name__ == "__main__":
    main()
