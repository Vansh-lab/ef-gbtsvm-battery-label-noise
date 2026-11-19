"""Produce the two reviewer-facing deliverables:

  REFERENCES_40.md      - the 40 references with a clickable DOI/arXiv link each
  NUMBER_PROVENANCE.md  - every number printed in the manuscript, with the CSV
                          file and column it was read from
"""
import os
import re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def references():
    src = open(os.path.join(HERE, "references_40.tex"), encoding="utf-8").read()
    pat = re.compile(r"\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem|\\end\{thebibliography\})",
                     re.S)
    items = pat.findall(src)
    lines = [
        "# References (40 entries, ordered oldest first)",
        "",
        "Every DOI was resolved on 2026-11-19: 37 against the Crossref REST API and 3",
        "against the arXiv API. In each case the returned title, container title and year",
        "matched the entry below. One correction was made during this check: Deng et al.",
        "was listed as 2021 but the IEEE TPEL issue (vol. 37, no. 5) is dated 2022, so the",
        "entry was changed to 2022 and moved to its correct position in the year ordering.",
        "",
        "| # | Key | Reference | Link |",
        "|---|---|---|---|",
    ]
    for i, (key, body) in enumerate(items, 1):
        b = re.sub(r"\s+", " ", body).strip()
        m = re.search(r"\\href\{([^}]+)\}", b)
        url = m.group(1) if m else ""
        text = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", b)
        for a, c in [("\\emph{", ""), ("``", '"'), ("''", '"'), ("--", "-"),
                     ("\\&", "&"), ('\\"o', "o"), ("\\'e", "e"), ("{", ""), ("}", "")]:
            text = text.replace(a, c)
        text = re.sub(r"\s*doi:\S+\s*$", "", text).strip().rstrip(".")
        lines.append(f"| {i} | `{key}` | {text.replace('|', '/')}. | "
                     f"[{url.replace('https://doi.org/', 'doi:')}]({url}) |")
    open(os.path.join(HERE, "REFERENCES_40.md"), "w",
         encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print("wrote REFERENCES_40.md:", len(items), "entries")


def provenance():
    p = pd.read_csv(os.path.join(HERE, "number_provenance.csv"))
    lines = [
        "# Where every number in the manuscript comes from",
        "",
        "The manuscript reports three training-label noise conditions: **0 %, 6 % and",
        "6.8 %**. Every value in its tables and figures is read straight from the frozen",
        "result summaries in `results/` by `build_tables.py` and `make_result_figures.py`.",
        "Nothing is recomputed, re-rounded or typed by hand: the summary CSVs already store",
        "two-decimal percentages and the scripts print them unchanged.",
        "",
        "## Source files",
        "",
        "| Condition | CSV file | Key | Columns used |",
        "|---|---|---|---|",
        "| 0 % | `results/finegrid_plus_phaseb_0_10_combined.csv` | `noise_pct == 0` | `ACC_mean`, `ACC_std`, `BACC_mean`, `BACC_std`, `F1_mean`, `F1_std`, `AUC_mean`, `AUC_std`, `rank_ACC_all17`, `rank_BACC_all17` |",
        "| 6 % | `results/finegrid_plus_phaseb_0_10_combined.csv` | `noise_pct == 6` | same columns |",
        "| 6.8 % | `results/finegrid2_summary_table.csv` | `noise_pct == 68` (stored in tenths) | same columns |",
        "| Per-cell values (Fig. 10) | `results/phase_b_per_fold.csv` (0 %), `results/finegrid_runs.csv` (6 %), `results/finegrid2_runs.csv` (6.8 %) | `run` 0-3 = B0005, B0006, B0007, B0018 | `ACC` |",
        "",
        "Ranks printed in parentheses are the `rank_*_all17` columns of those same files,",
        "which rank all seventeen models shown (ties share the lower rank). For F1 and AUC",
        "the CSVs carry no rank column, so the rank is computed from the printed means by",
        "the same rule; that is the only derived quantity in any table.",
        "",
        "EF-GBTSVM-PF has no row of its own in the per-fold run files, because it is a",
        "derived model: per outer cell it is the variant Phase A selected (L8 for B0005,",
        "B0007 and B0018; J for B0006). `make_result_figures.py` reassembles it from those",
        "rows and asserts that the result reproduces the published summary means exactly",
        "before drawing anything.",
        "",
        "## Numbers quoted in the running text",
        "",
        "| Where | Value | Source |",
        "|---|---|---|",
        "| Abstract, Sections 4.1, 5 - proposed model, accuracy and rank | 95.25 (12), 98.01 (1), 97.69 (1) | row `EF-GBTSVM-PF`, `ACC_mean` / `rank_ACC_all17` |",
        "| Abstract, Section 4.1 - proposed model, balanced accuracy and rank | 95.24 (10), 98.10 (1), 97.67 (1) | row `EF-GBTSVM-PF`, `BACC_mean` / `rank_BACC_all17` |",
        "| Section 4.1 - models ahead of it on clean labels | TSVM 98.77, RVFL 97.75, XGBoost 97.69, MLP 97.61, random forest 97.39 | `ACC_mean` at `noise_pct == 0` |",
        "| Section 4.1 - unstable baseline | RBF-SVM `ACC_std` 23.25 at 0 %, `ACC_mean` 97.27 and rank 3 at 6.8 % | rows `SVM` |",
        "| Section 4.2 - models that fall | TSVM 98.77 / 95.63 / 91.10, ranks 1 / 9 / 16; RVFL 97.75 / 90.08 / 88.89, ranks 2 / 16 / 17 | `ACC_mean`, `rank_ACC_all17` |",
        "| Section 4.2 - models that hold | random forest 97.39-97.84; MLP 97.09-97.84 | `ACC_mean` |",
        "| Section 4.2 - model that improves | RBF-SVM 86.61 to 97.27 | `ACC_mean`, rows `SVM` |",
        "| Section 4.2 - granular-ball family split | EF-GBTSVM 94.83 / 97.42 / 97.12; L and W 95.25 to 93.76 and 93.52 | `ACC_mean` |",
        "| Section 4.2 - gap to the best model | 3.52 points on clean labels (98.77 - 95.25) | difference of two `ACC_mean` values |",
        "| Section 4.2 - no significance available | smallest attainable two-sided p = 0.125 with four paired observations | `REPORT.md`, section \"Statistics\" (a property of n = 4, not a computed test) |",
        "| Section 4.4 - ablation | TSVM 98.77; GBTSVM 96.43 / 94.12 / 91.37; EF-TSVM 96.67 / 91.13; EF-GBTSVM 94.83 / 97.42 / 97.12 | `ACC_mean` |",
        "| Section 4.4 - single-feature threshold | 96.94, 95.60, 95.16 | rows `Stump-fair`, `ACC_mean` |",
        "| Section 4.5 - confusion counts at 0 % | 335 / 24 / 6 / 271, 30 errors | saved 0 % predictions, `results/phase_b_per_fold.csv` |",
        "| Section 4.5 - AUC of the proposed model at 0 % | 99.59, SD 0.75 | `AUC_mean` / `AUC_std` |",
        "| Section 4.5 - train-test gap at 0 % | 1.48 points; 180 of 180 rows reproduced | `REPORT.md`, section \"Train-vs-test gap\" |",
        "| Section 4.3 - effect of one error | 0.60 points for B0005-B0007 (1/168), 0.76 for B0018 (1/132) | cell counts in Table 1 |",
        "| Section 3.2 - cell counts | 168 / 168 / 168 / 132 cycles; 66/102, 108/60, 45/123, 58/74 | `data/nasa_battery_phys.csv` |",
        "",
        "## Numbers taken from the three compared papers (Table 11)",
        "",
        "| Study | Value as printed | Where it appears in that paper |",
        "|---|---|---|",
        "| Xu et al. (2026) | RMSE reduced from 1.89 % to 1.32 % on NASA; memory 897 MB to 616 kB | abstract and conclusion |",
        "| Xu et al. (2026) | per-cell RMSE below 0.0036, R2 above 0.978 | text describing their Table 9 |",
        "| Xu et al. (2026) | within-cell split by training ratio, 30 % and 60 % reported | their Figs. 7 and 8, and Table 7 |",
        "| Lei et al. (2027) | RMSE 0.525 % within-battery; mean RMSE 0.800 % under LOBO | abstract |",
        "| Lei et al. (2027) | 13 HNEI NMC/LCO 18650 cells; within-battery, temporal and LOBO protocols | abstract |",
        "| Li and Fetene (2027) | RMSE 0.00768 / 0.01209 / 0.00652 and MAE 0.00503 / 0.00808 / 0.00458 for B0005 / B0006 / B0007 | their Table 4 |",
        "| Li and Fetene (2027) | R2 >= 0.99772 | their Table 4 |",
        "| Li and Fetene (2027) | cycle-level split of 30 % train, 15 % validation, 15 % test | their model-setup section, quoted as reported |",
        "",
        "No number from those three papers was converted, rescaled or combined with ours.",
        "",
        "## Full machine-readable list",
        "",
        f"`number_provenance.csv` lists every table cell individually ({len(p)} rows) with",
        "its model, condition, metric, printed value, printed rank, source CSV and source",
        "columns.",
        "",
        "## Completeness",
        "",
        "`check_final.py` confirms against the compiled PDF that all 204 `mean +- standard",
        "deviation` values for the three reported conditions appear verbatim, and that no",
        "value belonging to a condition the paper does not report (2, 4, 8, 10, 20 %, the",
        "6.2-7.8 % sweep, or the 30 / 40 / boundary stress conditions) leaks into the text.",
        "Those experiments still exist in `results/` and in the pre-registration documents;",
        "they are simply outside this manuscript's reported scope.",
    ]
    open(os.path.join(HERE, "NUMBER_PROVENANCE.md"), "w",
         encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print("wrote NUMBER_PROVENANCE.md")


references()
provenance()
