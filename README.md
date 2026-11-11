# A leakage-free, fair benchmark of EF-GBTSVM on NASA Li-ion battery data under label noise

Independent re-implementation inspired by Quadir et al., EF-GBTSVM [1], benchmarked on NASA battery
data. Not affiliated with the original authors.

This repository holds the code, per-fold results and lab notebook behind the paper
*A Leakage-Free, Fair Benchmark of EF-GBTSVM for Li-ion Battery Degradation Classification under
Label Noise* (Vansh Sachan, Faculty of Technology, University of Delhi). No file from the original
authors' repository is included; their reference code is at <https://github.com/mtanveer1/EF-GBTSVM>.

## Main findings

Four NASA cells (B0005, B0006, B0007, B0018; 636 discharge cycles, label = state of health below
0.80), leave-one-battery-out, battery-grouped nested tuning, the same cap of 150 hyperparameter
candidates per model (min(150, grid size); GBTSVM's 40-point and TSVM's 8-point grids are searched
in full), and 0%, 10% or 20% of training labels flipped.

- **Shared recording.** B0005, B0006 and B0007 have identical time vectors in every discharge
  cycle, so whole-record features (discharge time, mean voltage, ...) of one cell depend on another.
  This benchmark uses 12 physics-guided features computed only from the loaded part of each discharge.
- **EF-GBTSVM-PF** (the EF-GBTSVM variant chosen on each fold by inner CV on the training cells:
  L8 on B0005, B0007, B0018 and J on B0006) reaches 95.25%, 95.08% and 95.28% mean accuracy at
  0/10/20% noise: +0.42, +8.41 and +0.57 points over the original EF-GBTSVM, **not statistically
  significant** (pooled Wilcoxon p = 0.090, Holm 1.0).
- **It is not the best model.** It ranks 11th (tied with two other variants), 10th and 6th of 15.
  Inside the GB/TSVM family (9 models) it ranks 6th, 5th and 1st; the 20% lead over EF-GBTSVM-A is
  0.18 points. It is 2nd of 9 by pooled Friedman rank, behind TSVM (7.83 vs 6.96), and 2nd of 11 at
  20% when L8 and J are added. So "best in the family" is not supported. Random forest has the best
  pooled Friedman rank (5.29; PF 7.83). A single-feature threshold tuned by the same protocol reaches
  96.94 / 95.16 / 96.85 and matches or beats PF at every noise level (the 10% value depends on an
  exact tie, see `REPORT.md`).
- **No difference is significant after correction.** Friedman over 15 models, p = 0.123. The lowest
  pooled Wilcoxon p is PF vs random forest, 0.033 unadjusted (Holm 0.465). The 12 pooled
  observations (4 folds x 3 noise levels) are not independent, so pooled p-values are descriptive
  only. With 4 batteries, a per-noise Wilcoxon test cannot go below p = 0.125.
- **Overfitting check.** PF's mean train-test gap is at most 2.21 points (worst fold 8.70, B0006 at
  10% noise).
- **Heavier and boundary-concentrated noise (pre-registered stress test, Phase D-lite).** Three more
  training-label conditions on the same four batteries: 30% and 40% random flips (N30, N40) and a
  boundary condition (BND, half the training cycles with SOH in [0.75, 0.85] flipped). PF is not
  best in any of the three (ACC rank 8/5/12 of 15, BACC rank 6/6/11; best ACC: RBF-SVM 97.63 on N30,
  random forest 94.49 on N40, RVFL 90.90 on BND). PF's mean accuracy falls to 92.92 (N30) and 85.12
  (N40), and to 81.52 under boundary noise. Pooled over the 12 (condition, battery) observations the
  Friedman test rejects equality (chi^2 = 40.10, p = 0.0002; per-condition p = 0.035/0.011/0.165,
  not independently significant at 4 batteries), but no pairwise comparison against PF survives Holm
  correction except PF above the GBTSVM ablation. See `REPORT.md` ("Phase D-lite") and
  `PREREG_DLITE.md` for the pre-registered plan.
- **External validation attempts: closed, no eligible cell.** Two pre-registered attempts to validate
  on data outside the four dev batteries both found zero eligible cells: 0 of 30 candidate NASA
  "fresh" cells (`prereg/inclusion_rule.md`, written before any fresh label was read; the three
  candidates that passed the other rules had 0 degraded cycles), and 0 of 23 XJTU Battery Dataset
  cells (CC BY 4.0; `ADDENDUM_E1.md`/`PREREG_E.md`; the recorded cycles do not give the required
  10%-per-class balance against either capacity reference). Phase E (new EF-GBTSVM variants) is
  therefore dev-only: candidates can be designed and compared on the four dev batteries, but no
  "best on external data" claim is made. See `REPORT.md` ("Phase D, step 1", "XJTU validation
  attempt").

Every number, with the audit notes and disclosures, is in `REPORT.md` ("Phase A", "Phase B results",
"Phase D-lite", "XJTU validation attempt").

### Earlier results (kept for history)

The first release of this repository was a sensitivity study (10 models, "7th of 10, 83.2%", the
time-to-3.5 V baseline). Those runs used the older feature files (`nasa_battery.csv` and its `_t70`,
`_t85` and `_hard` variants), whose whole-record features are affected by the shared recording. They
also used smaller, unequal tuning budgets (`--n-iter 30`, paper grids). Their numbers are superseded
and are not comparable with Phase B. They are kept unchanged: `results/nasa_*`, `results/_*.log`,
`results/stats_friedman_nemenyi_*.txt`, `results/phase_a/base.log` and `base_*` (old 13 features),
`results/phase_a/phys.log` and `phys_*` (an earlier 14-feature physics file), and the `REPORT.md`
sections before "Phase B setup".

## Contents

| Path | What it is |
| --- | --- |
| `efgbtsvm/` | Re-implementation: granular balls, random hidden-layer features, linear twin SVM, purity-weighted and mass-weighted balls, noise-adaptive purity and its jointly searched offset (J) |
| `prep_nasa_battery.py` | Builds the older per-cycle feature CSV `nasa_battery.csv` (needed as a row/label check) |
| `prep_phys_features.py` | Builds the 12 physics-guided features, `data/nasa_battery_phys.csv` |
| `run_experiments.py` | Leave-one-battery-out experiments, randomised tuning (`--grids equal`), training-label noise |
| `phase_a_inner_cv.py` | Phase A: per-fold choice of the EF-GBTSVM variant by inner CV on the training cells only |
| `phase_b_train_gap.py` | Refits every Phase B row with its logged parameters; train-vs-test gap |
| `phys_stump.py` | Single-feature threshold: fair (inner CV) and oracle (upper bound) versions |
| `phase_b_analysis.py` | Phase B tables, Friedman/Nemenyi/Wilcoxon, claims; refits no logged model, but fits the single-feature stumps itself (via `phys_stump.py`) |
| `compare_leakage_impact.py` | Old vs new results table (confounded: features and tuning both changed) |
| `run_dlite.py` | Phase D-lite: three more training-label noise conditions (N30, N40, boundary) on the same four batteries, checkpointed per (condition, fold) |
| `dlite_analysis.py` | Phase D-lite tables, Friedman/Wilcoxon, claims |
| `PREREG_DLITE.md` | Pre-registered plan for the Phase D-lite stress test (frozen before any D-lite run) |
| `prereg/` | Pre-registration records: the NASA fresh-cell inclusion rule and inventory (0 of 30 eligible), the D-lite script version diffs, and the XJTU dataset licence/documentation-check notes (Phase E, still draft) |
| `make_paper_figures.py` | Chart CSVs and console-output images in `paper_figures/`; re-runs `phase_b_analysis.py` to capture its output, which rewrites `results/phase_b_*` |
| `paper_figures/` | `chart_*.csv`, `console_*.png`, `console_phase_b_analysis.txt`; `check_citations.py` and `build_ieee_tex.py` (both need doc exports that are not in this repository) |
| `stats_tests.py`, `proxy_checks.py` | Statistics and proxy checks of the earlier study |
| `results/phys_phase_b_noise{0,10,20}_*.csv`, `results/phys_phase_b.log` | Phase B per-fold runs, summaries, log |
| `results/phase_a/phys2.log`, `results/phase_a/phys2_*` | Phase A log, inner-CV scores and per-fold selection used by PF |
| `results/phase_b_*` | Per-fold table, summary table, statistics, claims, train-test gap |
| `results/dlite_*` | Phase D-lite per-fold runs, flip log, summary/claims/stats, console output |
| `results/leakage_impact_comparison.csv` | Output of `compare_leakage_impact.py` |
| `REPORT.md` | Lab notebook with every experiment, including negative results |

## Data

The NASA PCoE Battery Data Set [2] is not redistributed here, and neither are the processed CSVs:
the data.nasa.gov catalogue record lists its licence as "not specified". Download `B0005.mat`,
`B0006.mat`, `B0007.mat` and `B0018.mat` from the NASA PCoE data set repository (item 5, "Battery
Data Set",
<https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>)
into `data/raw/`, then build the CSVs as below. The repository asks publications that use its data
to acknowledge the data donors; the data set citation it gives is [2]. The 30 candidate NASA "fresh"
cells and the 23-cell XJTU Battery Dataset (Zenodo 10963339, CC BY 4.0) used in the two closed
external-validation attempts are likewise not redistributed here.

## Setup

Versions recorded for the first release: Python 3.13.9, numpy 2.2.6, pandas 2.3.3, scipy 1.15.3,
scikit-learn 1.8.0, cvxopt 1.3.3 and xgboost 3.1.3. `make_paper_figures.py` also needs Pillow, and
`paper_figures/build_ieee_tex.py` needs matplotlib.

    pip install numpy pandas scipy scikit-learn cvxopt xgboost pillow

Tuning runs in parallel with joblib. On Windows, set `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`, and run one experiment at a time; running several in
parallel exhausted process resources in our runs. The Phase B command uses `--n-jobs 4` for this reason.

## Reproduce

1. Build both feature files (`prep_phys_features.py` checks that its rows, battery order and labels
   match `data/nasa_battery.csv`):

       python prep_nasa_battery.py data/raw/B0005.mat data/raw/B0006.mat data/raw/B0007.mat data/raw/B0018.mat --out data/nasa_battery.csv
       python prep_phys_features.py data/raw/B0005.mat data/raw/B0006.mat data/raw/B0007.mat data/raw/B0018.mat --out data/nasa_battery_phys.csv

2. Phase A, per-fold variant selection (writes `results/phase_a/phys2_inner_cv.csv` and
   `phys2_selection.csv`):

       python phase_a_inner_cv.py --data data/nasa_battery_phys.csv --name phys2

3. Phase B, all models at three noise levels (bash loop as recorded in `REPORT.md`; writes
   `results/phys_phase_b_noise{0,10,20}_{runs,summary}.csv`; `results/phys_phase_b.log` is that
   run's captured console output):

       for noise in 0.0 0.1 0.2; do
         python run_experiments.py \
           --data data/nasa_battery_phys.csv \
           --noise $noise \
           --n-iter 150 \
           --grids equal \
           --n-jobs 4 \
           --group-col battery \
           --name phys_phase_b \
           --out results/
       done

4. Train-vs-test gap, then the analysis (the analysis computes the single-feature threshold itself
   through `phys_stump.py`), then the figures:

       python phase_b_train_gap.py
       python phase_b_analysis.py
       python make_paper_figures.py

   Optional: `python phys_stump.py` prints the per-feature threshold table, and
   `python compare_leakage_impact.py` writes the old-vs-new table.

5. Phase D-lite, the pre-registered stress test (plan in `PREREG_DLITE.md`; writes
   `results/dlite_{N30,N40,BND}_runs.csv`, checkpointed per condition/fold so an interruption loses
   only one fold):

       python run_dlite.py --data data/nasa_battery_phys.csv --n-jobs 2
       python dlite_analysis.py

## Citation

If you use this code, please cite the original method [1] and the data set [2]. The benchmark's
citation will be added when it is published.

## References

1. A. Quadir, M. Sajid, M. Tanveer and P. N. Suganthan, "Enhanced feature based granular ball twin
   support vector machine," arXiv:2410.05786, 2024 (v2, Feb. 2025).
   <https://arxiv.org/abs/2410.05786>. Authors' code: <https://github.com/mtanveer1/EF-GBTSVM>.
2. B. Saha and K. Goebel, "Battery data set," NASA Prognostics Data Repository, NASA Ames Research
   Center, Moffett Field, CA, 2007.

## License

The code in this repository is MIT-licensed (see `LICENSE`). The NASA data, the XJTU Battery
Dataset, and the original EF-GBTSVM code are not part of this repository and are not covered by
this license.
