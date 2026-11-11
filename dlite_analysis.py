"""Phase D-lite analysis (PREREG_DLITE.md sections 3-6). Analysis only; nothing is tuned here.

    python dlite_analysis.py        (after run_dlite.py)

The statistics are the functions of phase_b_analysis.py, imported unchanged (Friedman, Nemenyi CD,
Wilcoxon + Holm, per-fold PF construction); only the "noise" column now holds the condition label
N30 / N40 / BND. Inputs: results/dlite_{N30,N40,BND}_runs.csv, results/dlite_flip_log.csv.
Outputs: results/dlite_per_fold.csv, dlite_summary_table.csv, dlite_stats.txt, dlite_claims.txt.
"""
import os

import numpy as np
import pandas as pd

import phase_b_analysis as pb

CONDS = ["N30", "N40", "BND"]
METRICS = pb.METRICS
PRIMARY = pb.PRIMARY
OUT = "results"


def load():
    runs = [pd.read_csv(f"{OUT}/dlite_{c}_runs.csv") for c in CONDS]
    d = pd.concat(runs, ignore_index=True).rename(columns={"condition": "noise_pct"})
    stump = d[d["model"].isin(["Stump-fair", "Stump-oracle"])]
    oracle = stump[stump.model == "Stump-oracle"]
    d = d[d["model"] != "Stump-oracle"].copy()
    gcols = ["run", "model", "noise_pct", "train_acc_noisy", "train_acc_clean"]
    gap = d[gcols + ["ACC"]].rename(columns={"ACC": "test_acc"})
    gap["gap_clean"] = gap["train_acc_clean"] - gap["test_acc"]
    df = d[["run", "model", "noise_pct"] + METRICS].copy()
    df[METRICS] = df[METRICS].round(6)
    sel = pb.per_fold_selection()
    gap_cols = ["train_acc_noisy", "train_acc_clean", "test_acc", "gap_clean"]
    df = pb.add_per_fold(df, sel, METRICS)
    gap = pb.add_per_fold(gap, sel, gap_cols)
    return df, oracle, gap, sel


def main():
    df, oracle, gap, sel = load()
    models = sorted(df["model"].unique())
    assert len(models) == 17, models
    main_pool = [m for m in models if m not in pb.COMPONENTS]       # 15
    sens_pool = models                                               # 17
    flips = pd.read_csv(f"{OUT}/dlite_flip_log.csv")

    df["test_cell"] = df["run"].map(pb.FOLD_CELL)
    df.pivot_table(index=["model", "noise_pct"], columns="test_cell", values=METRICS).round(2) \
        .to_csv(f"{OUT}/dlite_per_fold.csv")
    summ = df.groupby(["noise_pct", "model"])[METRICS].agg(["mean", "std"]).round(2)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ["in_main_pool"] = summ["model"].isin(main_pool)
    for met in ["ACC", "BACC"]:
        mp = summ[summ.in_main_pool]
        summ[f"rank_{met}_main15"] = mp.groupby("noise_pct")[f"{met}_mean"] \
            .rank(ascending=False, method="min").astype("Int64")
        summ[f"rank_{met}_all17"] = summ.groupby("noise_pct")[f"{met}_mean"] \
            .rank(ascending=False, method="min").astype(int)
    summ.to_csv(f"{OUT}/dlite_summary_table.csv", index=False)

    S = ["=== Phase D-lite statistics ===",
         f"Primary {PRIMARY} = per-fold Phase A selection {sel} (not re-selected).",
         f"MAIN pool k={len(main_pool)}; SENSITIVITY pool k={len(sens_pool)} (adds L8, J).",
         "CAVEATS: 4 folds per condition -> per-condition Wilcoxon n=4, minimum two-sided p = 0.125, so no",
         "per-condition test can be significant. The 3 conditions reuse the same 4 batteries/features and",
         "B0005/6/7 share one recording: pooled N=12 observations are NOT independent; pooled p-values are",
         "descriptive only. Nemenyi is post-hoc and descriptive. Absence of significance != equivalence.", ""]
    S.append("########## MAIN POOL ##########")
    S += [pb.friedman_block(df, main_pool, CONDS, "pooled (4 folds x 3 conditions)")[0], ""]
    for c in CONDS:
        S += [pb.friedman_block(df, main_pool, [c], f"condition={c}")[0], ""]
    S += [pb.wilcoxon_block(df, PRIMARY, main_pool, CONDS, "pooled"), ""]
    for c in CONDS:
        S += [pb.wilcoxon_block(df, PRIMARY, main_pool, [c], f"condition={c} (min p=0.125)"), ""]
    S += [pb.friedman_block(df, main_pool, CONDS, "pooled, BACC", metric="BACC")[0], ""]
    S.append("########## SENSITIVITY POOL (adds L8, J) ##########")
    S += [pb.friedman_block(df, sens_pool, CONDS, "pooled (4 folds x 3 conditions)")[0], ""]
    S += [pb.wilcoxon_block(df, PRIMARY, sens_pool, CONDS, "pooled"), ""]
    stats = "\n".join(S)
    open(f"{OUT}/dlite_stats.txt", "w", encoding="utf-8").write(stats)

    # claims (PREREG_DLITE.md section 6): strict rank 1 in mean ACC AND mean BACC, main pool k=15
    mean = df.groupby(["noise_pct", "model"])[METRICS].mean().round(6)
    C = ["=== D-lite claims (computed from results/dlite_*_runs.csv) ===",
         "Flip log (per fold): see results/dlite_flip_log.csv", flips.to_string(index=False), ""]
    verdict = {}
    for c in CONDS:
        row = {}
        for met in ["ACC", "BACC"]:
            for pool, name in ((main_pool, "main15"), (sens_pool, "sens17")):
                s = mean.loc[c].loc[pool, met].sort_values(ascending=False)
                top = s.iloc[0]
                tied_top = [m for m in s.index if s[m] == top]
                row[(met, name)] = dict(pf=s[PRIMARY], top_model=s.index[0], top=top, tied_top=tied_top,
                                        rank=int((s > s[PRIMARY]).sum()) + 1,
                                        strict_first=(tied_top == [PRIMARY]))
        verdict[c] = row
        best = row[("ACC", "main15")]["strict_first"] and row[("BACC", "main15")]["strict_first"]
        C.append(f"[{c}] main pool k=15:")
        for met in ["ACC", "BACC"]:
            r = row[(met, "main15")]
            C.append(f"  {met}: PF {r['pf']:.2f}, rank {r['rank']}/15, top {r['top_model']} {r['top']:.2f} "
                     f"(gap {r['pf'] - r['top']:+.2f}); tied for top: {r['tied_top']}")
        for met in ["ACC", "BACC"]:
            r = row[(met, "sens17")]
            C.append(f"  [sensitivity k=17] {met}: rank {r['rank']}/17, top {r['top_model']} {r['top']:.2f}")
        C.append("  => " + ("PF has the highest mean ACC and BACC on this condition (main pool, 4 folds); the "
                            "difference is not statistically testable at this N."
                            if best else "PF is NOT best on this condition."))
        C.append("")
    allbest = all(verdict[c][("ACC", "main15")]["strict_first"] and verdict[c][("BACC", "main15")]["strict_first"]
                  for c in CONDS)
    C.append("Across all three conditions: " + ("claim 'PF highest mean ACC and BACC on all three' allowed."
                                                 if allbest else "no across-conditions 'best' claim; per-condition statements only."))
    C += ["", "Mean ACC / BACC over folds (main + components + stump-fair):"]
    C.append(mean.loc[CONDS][["ACC", "BACC", "F1", "AUC"]].round(2).to_string())
    C += ["", "Train-vs-test gap (train ACC on TRUE train labels minus test ACC), mean over folds:"]
    C.append(gap.groupby(["noise_pct", "model"])["gap_clean"].mean().unstack(0).round(2).to_string())
    C.append("  gap > 5 pp: " + "; ".join(f"{m}@{n}={v:.1f}" for (n, m), v in
                                           gap.groupby(["noise_pct", "model"])["gap_clean"].mean().items() if v > 5))
    C += ["", "Stump-oracle upper bound (not ranked): " + ", ".join(
        f"{c} {oracle[oracle.noise_pct == c]['ACC'].mean():.2f}" for c in CONDS)]
    claims = "\n".join(C)
    open(f"{OUT}/dlite_claims.txt", "w", encoding="utf-8").write(claims)
    print(stats)
    print(claims)


if __name__ == "__main__":
    main()
