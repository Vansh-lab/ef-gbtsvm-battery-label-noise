"""Build every LaTeX results table in the manuscript directly from the frozen result CSVs.

The manuscript reports three training-label noise conditions: 0 %, 6 % and 6.8 %.

Each file in tables/ holds a complete `tabular` environment, which main.tex wraps in a
float with its caption.  (A bare row fragment cannot be \\input inside an open
alignment, so the whole environment is emitted here.)

Every number that reaches the manuscript passes through this script and is also written
to number_provenance.csv with the CSV file and columns it came from.  Values are printed
exactly as the summary CSVs store them; nothing is re-rounded or recomputed.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")
OUT = os.path.join(HERE, "tables")
os.makedirs(OUT, exist_ok=True)

PROV = []

ORDER = [
    "EF-GBTSVM", "EF-GBTSVM-A", "EF-GBTSVM-L", "EF-GBTSVM-W",
    "EF-GBTSVM-L8", "EF-GBTSVM-J", "EF-GBTSVM-PF",
    "GBTSVM", "HF-GBTSVM", "EF-TSVM", "TSVM",
    "SVM", "RVFL", "RandomForest", "MLP", "XGBoost", "Stump-fair",
]
DISPLAY = {
    "EF-GBTSVM": "EF-GBTSVM",
    "EF-GBTSVM-A": "EF-GBTSVM-A",
    "EF-GBTSVM-L": "EF-GBTSVM-L",
    "EF-GBTSVM-W": "EF-GBTSVM-W",
    "EF-GBTSVM-L8": r"EF-GBTSVM-L8$^{\dagger}$",
    "EF-GBTSVM-J": r"EF-GBTSVM-J$^{\dagger}$",
    "EF-GBTSVM-PF": r"\textbf{EF-GBTSVM-PF (proposed)}",
    "GBTSVM": "GBTSVM",
    "HF-GBTSVM": "HF-GBTSVM",
    "EF-TSVM": "EF-TSVM",
    "TSVM": "TSVM",
    "SVM": "RBF-SVM",
    "RVFL": "RVFL",
    "RandomForest": "Random forest",
    "MLP": "MLP",
    "XGBoost": "XGBoost",
    "Stump-fair": "Single-feature threshold",
}
PROPOSED = "EF-GBTSVM-PF"
SHADE = r"\cellcolor{proposedrow}"

# The three reported conditions, with the file and key each one is read from.
CONDITIONS = [
    ("0\\%", "finegrid_plus_phaseb_0_10_combined.csv", 0, "0%"),
    ("6\\%", "finegrid_plus_phaseb_0_10_combined.csv", 6, "6%"),
    ("6.8\\%", "finegrid2_summary_table.csv", 68, "6.8%"),
]
FRAMES = {f: pd.read_csv(os.path.join(RES, f))
          for f in {c[1] for c in CONDITIONS}}


def row_for(model, csv_name, key):
    df = FRAMES[csv_name]
    sub = df[(df["model"] == model) & (df["noise_pct"] == key)]
    if sub.empty:
        raise KeyError(f"{model} at {key} missing from {csv_name}")
    return sub.iloc[0]


def cell(mean, sd, rank, bold=False):
    s = f"{mean:.2f}$\\pm${sd:.2f} ({rank})"
    return SHADE + r"\textbf{" + s + "}" if bold else s


def rowname(m):
    return (SHADE + DISPLAY[m]) if m == PROPOSED else DISPLAY[m]


def metric_pair_table(metrics, table_id):
    """One tabular: 17 models x (3 conditions x 2 metrics), each cell mean+-SD (rank)."""
    rows = []
    for m in ORDER:
        cells = []
        for tex_label, csv_name, key, plain in CONDITIONS:
            r = row_for(m, csv_name, key)
            for metric in metrics:
                rank_col = f"rank_{metric}_all17"
                if rank_col in r.index and not pd.isna(r[rank_col]):
                    rk = int(r[rank_col])
                    rank_src = rank_col
                else:
                    # the CSVs carry ranks only for ACC and BACC; F1 and AUC ranks
                    # are recomputed here from the printed means, same tie rule
                    lvl = FRAMES[csv_name]
                    lvl = lvl[lvl["noise_pct"] == key]
                    rk = int(lvl[f"{metric}_mean"].rank(ascending=False,
                                                        method="min").loc[r.name])
                    rank_src = f"rank recomputed from {metric}_mean (min tie rule)"
                cells.append(cell(r[f"{metric}_mean"], r[f"{metric}_std"], rk,
                                  bold=(m == PROPOSED)))
                PROV.append((table_id, m, plain, metric,
                             f"{r[f'{metric}_mean']:.2f} +- {r[f'{metric}_std']:.2f}",
                             rk, csv_name,
                             f"{metric}_mean / {metric}_std / {rank_src}"))
        rows.append(rowname(m) + " & " + " & ".join(cells) + r" \\")

    head = [
        "&" + "&".join(r"\multicolumn{2}{c}{%s}" % c[0] for c in CONDITIONS) + r"\\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        "Model & " + " & ".join([metrics[0], metrics[1]] * 3) + r" \\",
    ]
    out = [r"\begin{tabular}{l*{6}{c}}", r"\toprule"] + head + [r"\midrule"]
    out += rows + [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def write(name, text):
    with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("wrote", name)


write("tab_main.tex", metric_pair_table(["ACC", "BACC"], "T5 main ACC/BACC"))
write("tab_appendix.tex", metric_pair_table(["F1", "AUC"], "TA1 appendix F1/AUC"))

# remove tables from the earlier all-noise-levels draft so no stale file can be \input
for stale in ["tab_main_ACC.tex", "tab_main_BACC.tex", "tab_main_F1.tex",
              "tab_main_AUC.tex", "tab_fine_ACC_a.tex", "tab_fine_ACC_b.tex",
              "tab_fine_BACC_a.tex", "tab_fine_BACC_b.tex", "tab_fine_ACC.tex",
              "tab_fine_BACC.tex", "tab_stress.tex", "tab_pf_ranks.tex"]:
    p = os.path.join(OUT, stale)
    if os.path.exists(p):
        os.remove(p)
        print("removed stale", stale)

prov = pd.DataFrame(PROV, columns=["table", "model", "condition", "metric",
                                   "value_as_printed", "rank_as_printed",
                                   "source_csv", "source_column"])
prov.to_csv(os.path.join(HERE, "number_provenance.csv"), index=False)
print("provenance rows:", len(prov))

print("\nEF-GBTSVM-PF at the three reported conditions:")
for tex_label, csv_name, key, plain in CONDITIONS:
    r = row_for(PROPOSED, csv_name, key)
    print(f"  {plain:>5} : ACC {r['ACC_mean']:.2f} (rank {int(r['rank_ACC_all17'])})"
          f"   BACC {r['BACC_mean']:.2f} (rank {int(r['rank_BACC_all17'])})"
          f"   F1 {r['F1_mean']:.2f}   AUC {r['AUC_mean']:.2f}")
print("\nBest ACC model at each reported condition:")
for tex_label, csv_name, key, plain in CONDITIONS:
    lvl = FRAMES[csv_name]
    lvl = lvl[lvl["noise_pct"] == key].sort_values("ACC_mean", ascending=False)
    top = lvl.iloc[0]
    print(f"  {plain:>5} : {top['model']:<14} {top['ACC_mean']:.2f}")
