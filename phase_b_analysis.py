"""Phase B analysis (post-Phase-B instructions). Analysis only; nothing is tuned here.

    python phase_b_analysis.py          (after phase_b_train_gap.py)

Inputs : results/phys_phase_b_noise{0,10,20}_runs.csv   15 models x 4 folds x 3 noise
         results/phase_b_train_gap.csv                  from phase_b_train_gap.py
         Stump-fair / Stump-oracle rows computed via phys_stump.stump_runs
Outputs: results/phase_b_per_fold.csv     model x noise x fold, ACC/BACC/F1/AUC
         results/phase_b_summary_table.csv mean, SD and rank per model x noise
         results/phase_b_stats.txt        Friedman + Nemenyi CD + Wilcoxon
         results/phase_b_claims.txt       claims (a), (b), (c)

Primary model = EF-GBTSVM-PF, the pre-registered per-fold selection (REPORT.md "Phase B plan"):
on each outer fold use the variant that fold's own Phase A inner CV picked
(results/phase_a/phys2_selection.csv: L8 on B0005/B0007/B0018, J on B0006). The test cell was
excluded from that fold's inner CV, so no test label influences the variant used on it. A single
global majority choice (L8 on all folds) would let each test cell's labels, used as validation
data in the other folds' inner CVs, influence the variant used on it; it is reported only as a
non-primary reference.

Main pool (15) = the 13 non-variant run_experiments.py models + EF-GBTSVM-PF + Stump-fair. L8 and
J are PF's candidate inputs, not separate methods: the main pool ranks the method (per-fold
selection + fit), not its candidates. This is defensible but not neutral, so the sensitivity pool
is always reported next to it.
Sensitivity pool (17) = all 15 run_experiments.py models + PF + Stump-fair. Stump-oracle selects
its feature with test ACC, so it is reported only as an "oracle upper bound", never ranked.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon

from phys_stump import stump_runs

NOISES = [0, 10, 20]
FOLD_CELL = {0: "B0005", 1: "B0006", 2: "B0007", 3: "B0018"}
METRICS = ["ACC", "BACC", "F1", "AUC"]
PRIMARY = "EF-GBTSVM-PF"
COMPONENTS = ["EF-GBTSVM-L8", "EF-GBTSVM-J"]   # PF's per-fold building blocks
GB_TSVM_FAMILY = ["EF-GBTSVM", "EF-GBTSVM-W", "EF-GBTSVM-L", "EF-GBTSVM-A", PRIMARY,
                  "GBTSVM", "HF-GBTSVM", "EF-TSVM", "TSVM"]
SELECTION = "results/phase_a/phys2_selection.csv"
OUT = "results"


def per_fold_selection():
    sel = pd.read_csv(SELECTION)
    return dict(zip(sel["fold"].astype(int), sel["selected"]))


def add_per_fold(df, sel, value_cols):
    """Append PRIMARY rows: for each (noise, run) copy the row of the variant selected for that run."""
    pick = pd.concat([df[(df["run"] == r) & (df["model"] == m)] for r, m in sel.items()])
    cnt = pick.groupby(["noise_pct", "run"]).size()
    assert len(cnt) == len(NOISES) * len(sel) and (cnt == 1).all(), "need exactly one row per (noise, run)"
    pick = pick.assign(selected_variant=pick["model"], model=PRIMARY)
    return pd.concat([df, pick[["run", "model", "noise_pct", "selected_variant"] + value_cols]],
                     ignore_index=True)


def load():
    frames, oracle, stump_gap = [], [], []
    for n in NOISES:
        d = pd.read_csv(f"results/phys_phase_b_noise{n}_runs.csv")[["run", "model"] + METRICS]
        fair = stump_runs(n / 100, "fair")
        orc = stump_runs(n / 100, "oracle")
        for f in (d, fair, orc):
            f["noise_pct"] = n
        frames += [d, fair[["run", "model", "noise_pct"] + METRICS]]
        oracle.append(orc)
        stump_gap.append(fair.assign(test_acc=fair["ACC"],
                                     gap_clean=fair["train_acc_clean"] - fair["ACC"]))
    # metrics are count/n_test fractions; CSV vs live values differ in the last float bits, so
    # round before ranking/Wilcoxon or real ties go undetected (smallest true gap is ~0.3 pp)
    df = pd.concat(frames, ignore_index=True)
    df[METRICS] = df[METRICS].round(6)
    df = add_per_fold(df, per_fold_selection(), METRICS)
    gap_cols = ["run", "model", "noise_pct", "train_acc_noisy", "train_acc_clean", "test_acc", "gap_clean"]
    return df, pd.concat(oracle, ignore_index=True), pd.concat(stump_gap, ignore_index=True)[gap_cols]


def tied_position(ranks, model):
    """1-based position of `model` in a rank Series (lower = better), tie-aware ('a' or 'a-b')."""
    lo = int((ranks < ranks[model] - 1e-9).sum()) + 1
    hi = int((ranks <= ranks[model] + 1e-9).sum())
    ties = [m for m in ranks.index if m != model and abs(ranks[m] - ranks[model]) <= 1e-9]
    return (f"{lo}" if lo == hi else f"{lo}-{hi} (tied with {', '.join(ties)})")


def matrix(df, metric, noises, models):
    """rows = models, cols = (noise, fold) observations."""
    cols = [(n, f) for n in noises for f in range(4)]
    piv = df.pivot_table(index="model", columns=["noise_pct", "run"], values=metric)
    return piv.loc[models, cols]


def avg_ranks(mat):
    r = np.column_stack([rankdata(-mat.iloc[:, j].values) for j in range(mat.shape[1])])
    return pd.Series(r.mean(axis=1), index=mat.index).sort_values()


def nemenyi_cd(k, n_obs, alpha=0.05):
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q, q * np.sqrt(k * (k + 1) / (6 * n_obs))


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for i, j in enumerate(order):
        running = max(running, (len(pvals) - i) * pvals[j])
        adj[j] = min(1.0, running)
    return adj


def friedman_block(df, models, noises, label, metric="ACC"):
    mat = matrix(df, metric, noises, models)
    stat, p = friedmanchisquare(*[mat.loc[m].values for m in models])
    ranks = avg_ranks(mat)
    q, cd = nemenyi_cd(len(models), mat.shape[1])
    L = [f"--- {label}: k={len(models)} models, N={mat.shape[1]} observations, metric={metric} ---",
         f"Friedman chi2 = {stat:.3f}, p = {p:.4g}",
         f"Nemenyi q(0.05,k={len(models)}) = {q:.3f}, CD = {cd:.3f} rank units",
         "Average ranks (1 = best):"]
    top = ranks.iloc[0]
    for m, r in ranks.items():
        tag = " <- primary" if m == PRIMARY else (" <- PF component" if m in COMPONENTS else "")
        sig = "  (significantly worse than rank-1 model)" if r - top > cd else ""
        L.append(f"  {m:14s} {r:6.2f}{tag}{sig}")
    if PRIMARY in ranks:
        lo = [m for m, r in ranks.items() if abs(r - ranks[PRIMARY]) <= cd and m != PRIMARY]
        L.append(f"Models NOT significantly different from {PRIMARY} (|rank diff| <= CD): {len(lo)}/{len(models)-1}")
    return "\n".join(L), ranks


def wilcoxon_block(df, target, models, noises, label, metric="ACC"):
    mat = matrix(df, metric, noises, models)
    L = [f"--- Wilcoxon signed-rank, {target} vs each, {label} (nominal n={mat.shape[1]}, two-sided, exact/permutation) ---"]
    rows = []
    for m in models:
        if m == target:
            continue
        # round the differences too: subtracting rounded values reintroduces last-digit noise
        # (smallest true gap between distinct |d| is 100/1848 ~ 0.054 pp, so 4 dp is safe)
        d = np.round(mat.loc[target].values - mat.loc[m].values, 4)
        nz = int((d != 0).sum())
        if nz == 0:
            rows.append((m, np.nan, np.nan, 0, 0, 0, 0.0))
            continue
        res = wilcoxon(d, alternative="two-sided")
        rows.append((m, res.statistic, res.pvalue, nz, int((d > 0).sum()), int((d < 0).sum()), np.mean(d)))
    ps = np.array([r[2] for r in rows if not np.isnan(r[2])])
    adj = iter(holm(ps)) if len(ps) else iter([])
    for m, w, p, nz, wins, losses, md in rows:
        if np.isnan(p):
            L.append(f"  {m:14s} all differences zero (identical)")
            continue
        L.append(f"  {m:14s} W={w:5.1f}  p={p:.4f}  Holm-p={next(adj):.4f}  "
                 f"wins/losses/ties={wins}/{losses}/{mat.shape[1]-nz}  mean diff={md:+.2f}")
    return "\n".join(L)


def claims_block(mean, oracle, pool, pool_ranks, pool_name):
    """claims (b) and (c) for one ranking pool."""
    k = len(pool)
    fam = [m for m in GB_TSVM_FAMILY + COMPONENTS if m in pool]
    C = [f"(b) [{pool_name}] {PRIMARY} rank inside the GB/TSVM family ({len(fam)} models), by mean ACC:"]
    for n in NOISES:
        f = mean.loc[n].loc[fam, "ACC"].sort_values(ascending=False)
        rk = int(f.rank(ascending=False, method="min")[PRIMARY])
        other = f.drop(PRIMARY)
        C.append(f"  noise {n:2d}%: rank {rk}/{len(f)}  (PF {f[PRIMARY]:.2f}; best other: "
                 f"{other.index[0]} {other.iloc[0]:.2f}, gap {f[PRIMARY]-other.iloc[0]:+.2f})")
    fr = pool_ranks.loc[fam].sort_values()
    C.append(f"  pooled Friedman avg rank within all {k}: PF {fr[PRIMARY]:.3f} -> "
             f"{tied_position(fr, PRIMARY)} of {len(fr)} in family "
             f"(family best: {fr.index[0]} {fr.iloc[0]:.3f})")
    C.append("")
    C.append(f"(c) [{pool_name}] {PRIMARY} among all {k}, by mean ACC:")
    for n in NOISES:
        allm = mean.loc[n].loc[pool, "ACC"].sort_values(ascending=False)
        rk = int(allm.rank(ascending=False, method="min")[PRIMARY])
        orc = oracle[oracle.noise_pct == n]["ACC"].mean()
        C.append(f"  noise {n:2d}%: rank {rk}/{k}; PF {allm[PRIMARY]:.2f}; top {allm.index[0]} {allm.iloc[0]:.2f} "
                 f"(gap {allm[PRIMARY]-allm.iloc[0]:+.2f}); SVM {allm['SVM']:.2f} (gap {allm[PRIMARY]-allm['SVM']:+.2f}); "
                 f"Stump-fair {allm['Stump-fair']:.2f} (gap {allm[PRIMARY]-allm['Stump-fair']:+.2f}); "
                 f"[Stump-oracle upper bound {orc:.2f}]")
    top = [n for n in NOISES if mean.loc[n].loc[pool, "ACC"].idxmax() == PRIMARY]
    C.append(f"  pooled Friedman avg rank: PF {pool_ranks[PRIMARY]:.3f} -> position "
             f"{tied_position(pool_ranks, PRIMARY)} of {k}; best {pool_ranks.index[0]} {pool_ranks.iloc[0]:.3f}")
    C.append(f"  *** {PRIMARY} is rank 1 overall at: {top or 'NO noise level'} ***")
    if not top or pool_ranks.index[0] != PRIMARY:
        C.append("  -> 'best overall' claim is NOT supported by this data.")
    C.append("")
    return C


def main():
    df, oracle, stump_gap = load()
    sel = per_fold_selection()
    allmodels = sorted(df["model"].unique())
    assert len(allmodels) == 17, allmodels
    main_pool = [m for m in allmodels if m not in COMPONENTS]           # 15
    sens_pool = allmodels                                                # 17
    df["test_cell"] = df["run"].map(FOLD_CELL)

    # 1. per-fold table + summary ------------------------------------------------------
    pf = df.pivot_table(index=["model", "noise_pct"], columns="test_cell", values=METRICS).round(2)
    pf.to_csv(os.path.join(OUT, "phase_b_per_fold.csv"))
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
    summ.to_csv(os.path.join(OUT, "phase_b_summary_table.csv"), index=False)

    gap = pd.read_csv(os.path.join(OUT, "phase_b_train_gap.csv"))
    assert gap["reproduced"].all(), "train-gap refit does not reproduce logged test ACC"
    gap = add_per_fold(gap, sel, ["train_acc_noisy", "train_acc_clean", "test_acc", "gap_clean"])
    gap = pd.concat([gap, stump_gap], ignore_index=True)   # Stump-fair: same definition, computed live

    # 2. stats -------------------------------------------------------------------------
    S = ["=== Phase B statistics ===",
         f"Primary {PRIMARY} = per-fold Phase A selection {sel}.",
         f"MAIN pool k={len(main_pool)} (L8/J excluded as PF components); SENSITIVITY pool k={len(sens_pool)}.",
         "CAVEATS: (i) only 4 leave-one-battery-out folds -> per-noise Wilcoxon has n=4, minimum",
         "achievable two-sided p = 0.125 (all 4 pairs differ), so NO per-noise pairwise test can reach",
         "p<0.05; zero differences are dropped, so with ties the floor rises to 0.25 (n=3) / 0.5 (n=2);",
         "the effective n is the nonzero-pair count (wins+losses). With ties/zeros scipy uses an exact",
         "conditional sign-flip permutation test, not the exact Wilcoxon table (pin scipy version);",
         "(ii) the 3 noise settings reuse the same 4 batteries and the same features, so the pooled",
         "N=12 observations are NOT independent; pooled p-values are not valid inference, descriptive only;",
         "(iii) Nemenyi is a post-hoc test meant for after a significant Friedman test and assumes",
         "independent observations; at N=4 the CD is near the max possible rank gap (k-1).",
         "Friedman chi2 is scipy's tie-corrected statistic; metrics are rounded to 6 dp so exact ties count.", ""]
    S.append("########## MAIN POOL ##########")
    pooled_txt, pooled_ranks = friedman_block(df, main_pool, NOISES, "pooled (4 folds x 3 noise)")
    S += [pooled_txt, ""]
    for n in NOISES:
        S += [friedman_block(df, main_pool, [n], f"noise={n}%")[0], ""]
    S += [wilcoxon_block(df, PRIMARY, main_pool, NOISES, "pooled"), ""]
    for n in NOISES:
        S += [wilcoxon_block(df, PRIMARY, main_pool, [n], f"noise={n}% (min p=0.125)"), ""]
    S += [friedman_block(df, main_pool, NOISES, "pooled, BACC", metric="BACC")[0], ""]
    S.append("########## SENSITIVITY POOL (adds L8, J) ##########")
    sens_txt, sens_ranks = friedman_block(df, sens_pool, NOISES, "pooled (4 folds x 3 noise)")
    S += [sens_txt, ""]
    S += [wilcoxon_block(df, PRIMARY, sens_pool, NOISES, "pooled"), ""]
    stats = "\n".join(S)
    with open(os.path.join(OUT, "phase_b_stats.txt"), "w", encoding="utf-8") as f:
        f.write(stats)

    # 3. claims ------------------------------------------------------------------------
    mean = df.groupby(["noise_pct", "model"])[METRICS].mean()
    picks = ", ".join(f"{FOLD_CELL[r]}: {m}" for r, m in sel.items())
    C = ["=== Claims (computed from results/*.csv; nothing tuned on test) ===",
         f"Primary {PRIMARY}: per-fold variant ({picks})", ""]
    C.append(f"(a) {PRIMARY} vs original EF-GBTSVM (mean over 4 folds; fold wins = PF > EF):")
    for n in NOISES:
        a, b = mean.loc[(n, PRIMARY)], mean.loc[(n, "EF-GBTSVM")]
        fa = df[(df.noise_pct == n) & (df.model == PRIMARY)].sort_values("run")["ACC"].values
        fb = df[(df.noise_pct == n) & (df.model == "EF-GBTSVM")].sort_values("run")["ACC"].values
        C.append(f"  noise {n:2d}%: ACC {a.ACC:.2f} vs {b.ACC:.2f} (diff {a.ACC-b.ACC:+.2f}); "
                 f"BACC {a.BACC:.2f} vs {b.BACC:.2f} (diff {a.BACC-b.BACC:+.2f}); "
                 f"fold wins/losses/ties {int((fa>fb).sum())}/{int((fa<fb).sum())}/{int((fa==fb).sum())}")
    C.append("")
    C += claims_block(mean, oracle, main_pool, pooled_ranks, f"MAIN k={len(main_pool)}")
    C += claims_block(mean, oracle, sens_pool, sens_ranks, f"SENSITIVITY k={len(sens_pool)}")
    C.append("Non-primary reference (global majority choice; each test cell's labels influence it): "
             + "; ".join(f"L8-on-all-folds {n}%: {mean.loc[(n, 'EF-GBTSVM-L8')].ACC:.2f}" for n in NOISES))
    C.append("")

    C.append("Train-vs-test gap (refit with logged params; train ACC on TRUE train labels minus test ACC):")
    g = gap.groupby(["noise_pct", "model"])["gap_clean"].mean().unstack(0).round(2)
    C.append(g.to_string())
    C.append("  gap > 5 pp (flag): " + "; ".join(
        f"{m}@{n}%={v:.1f}" for (n, m), v in gap.groupby(["noise_pct", "model"])["gap_clean"].mean().items() if v > 5))
    claims = "\n".join(C)
    with open(os.path.join(OUT, "phase_b_claims.txt"), "w", encoding="utf-8") as f:
        f.write(claims)

    print(stats)
    print(claims)
    print("\nPhase C NOT started.")


if __name__ == "__main__":
    main()
