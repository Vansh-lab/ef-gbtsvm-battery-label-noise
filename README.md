# EF-GBTSVM on NASA Li-ion battery data: a label-noise sensitivity study

Independent re-implementation inspired by Quadir et al., EF-GBTSVM [1], for a sensitivity study
on NASA battery data. Not affiliated with the original authors.

This repository holds the code, per-fold results and lab notebook behind the paper
*Label Noise and the Purity Threshold: A Sensitivity Study of EF-GBTSVM for Li-ion Battery Health
Classification* (Vansh Sachan, Faculty of Technology, University of Delhi). No file from the
original authors' repository is included; their reference code is at
<https://github.com/mtanveer1/EF-GBTSVM>.

## Main findings

On four NASA cells (B0005, B0006, B0007, B0018; 636 discharge cycles, label = state of health
below 0.80), evaluated leave-one-battery-out:

- EF-GBTSVM ranks 7th of 10 models on clean labels (83.2% accuracy; RBF-SVM 94.1%), and a
  five-times larger hyperparameter search does not close the gap.
- Under uniformly flipped training labels, its accuracy depends strongly on the range searched for
  the granular-ball purity threshold; no setting tested is best at 0%, 10% and 20% noise.
- A single threshold on one battery-management feature, time to reach 3.5 V, beats all ten models
  on average at every noise level (96.3% to 97.6%).

All numbers, including negative results, are documented in `REPORT.md`.

## Contents

| Path | What it is |
| --- | --- |
| `efgbtsvm/` | Re-implementation: granular balls, random hidden-layer features, linear twin SVM, purity-weighted balls, noise-adaptive purity |
| `prep_nasa_battery.py` | Builds the per-cycle feature CSV from the NASA `.mat` files |
| `run_experiments.py` | Leave-one-battery-out experiments, randomised tuning, training-label noise |
| `stats_tests.py` | Friedman and Nemenyi tests |
| `proxy_checks.py` | Capacity proxy check and the one-feature time-to-3.5 V baseline |
| `results/` | Per-fold results (`*_runs.csv`), summaries, Friedman output and run logs |
| `REPORT.md` | Lab notebook with every experiment |

## Data

The NASA PCoE Battery Data Set [2] is not redistributed here, and neither are the processed CSVs:
the data.nasa.gov catalogue record lists its licence as "not specified". Download `B0005.mat`,
`B0006.mat`, `B0007.mat` and `B0018.mat` from the NASA PCoE data set repository (item 5, "Battery
Data Set",
<https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>)
into `data/raw/`, then build the CSVs as below. The repository asks publications that use its data
to acknowledge the data donors; the data set citation it gives is [2].

## Setup

Tested with Python 3.13.9, numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, scikit-learn 1.8.0,
cvxopt 1.3.3 and xgboost 3.1.3.

    pip install numpy pandas scipy scikit-learn cvxopt xgboost

Tuning runs in parallel with joblib. On Windows, set `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`, and run one experiment at a time; running several in
parallel exhausted process resources in our runs.

## Reproduce

1. Build the CSVs (default SOH threshold 0.80; two other thresholds; a harder feature subset):

       python prep_nasa_battery.py data/raw/B0005.mat data/raw/B0006.mat data/raw/B0007.mat data/raw/B0018.mat --out data/nasa_battery.csv
       python prep_nasa_battery.py data/raw/B0005.mat data/raw/B0006.mat data/raw/B0007.mat data/raw/B0018.mat --soh-threshold 0.70 --out data/nasa_battery_t70.csv
       python prep_nasa_battery.py data/raw/B0005.mat data/raw/B0006.mat data/raw/B0007.mat data/raw/B0018.mat --soh-threshold 0.85 --out data/nasa_battery_t85.csv
       python prep_nasa_battery.py data/raw/B0005.mat data/raw/B0006.mat data/raw/B0007.mat data/raw/B0018.mat --drop-feature discharge_time,i_mean --out data/nasa_battery_hard.csv

2. Ten models, 30 search candidates (repeat the first line with `--noise 0.1` and `--noise 0.2`):

       python run_experiments.py --data data/nasa_battery.csv --name nasa --label-col degraded --group-col battery --noise 0 --n-iter 30
       python run_experiments.py --data data/nasa_battery_t70.csv --name nasa_t70 --label-col degraded --group-col battery --noise 0 --n-iter 30
       python run_experiments.py --data data/nasa_battery_t85.csv --name nasa_t85 --label-col degraded --group-col battery --noise 0 --n-iter 30
       python run_experiments.py --data data/nasa_battery_hard.csv --name nasa_hard --label-col degraded --group-col battery --noise 0 --n-iter 30

3. Purity-threshold study, each at `--noise 0`, `0.1` and `0.2` with
   `--data data/nasa_battery.csv --label-col degraded --group-col battery`:

   | `--name` | extra arguments |
   | --- | --- |
   | `nasa_w2` | `--n-iter 30 --models EF-GBTSVM EF-GBTSVM-W` |
   | `nasa_w3` | `--n-iter 30 --models EF-GBTSVM EF-GBTSVM-W --scoring-override EF-GBTSVM-W:gmean` |
   | `nasa_abl` | `--n-iter 30 --models EF-GBTSVM-L` |
   | `nasa_adaptive` | `--n-iter 30 --models EF-GBTSVM EF-GBTSVM-L EF-GBTSVM-A` |
   | `nasa_bigsearch` | `--n-iter 150 --models EF-GBTSVM EF-GBTSVM-W EF-GBTSVM-L EF-GBTSVM-A` |

   `results/nasa_w_*` came from an earlier version of `run_experiments.py` in which EF-GBTSVM-W
   searched the default purity grid {0.85, 0.9, 0.925, 0.95, 0.98}. The current EF-GBTSVM-W uses
   the wide grid, so reproducing `nasa_w_*` needs that grid swapped back.

4. Statistics and proxy checks:

       python stats_tests.py results/nasa_noise0_runs.csv results/nasa_noise10_runs.csv results/nasa_noise20_runs.csv results/nasa_t70_noise0_runs.csv results/nasa_t85_noise0_runs.csv results/nasa_hard_noise0_runs.csv --metric ACC --per-file
       python proxy_checks.py

Each run writes `results/<name>_noise<pct>_runs.csv` (every fold, chosen hyperparameters, times)
and `results/<name>_noise<pct>_summary.csv` (mean ± SD over folds).

## Citation

If you use this code, please cite the original method [1]. The battery study's citation will be
added when it is published.

## References

1. A. Quadir, M. Sajid, M. Tanveer and P. N. Suganthan, "Enhanced feature based granular ball twin
   support vector machine," arXiv:2410.05786, 2024 (v2, Feb. 2025).
   <https://arxiv.org/abs/2410.05786>. Authors' code: <https://github.com/mtanveer1/EF-GBTSVM>.
2. B. Saha and K. Goebel, "Battery data set," NASA Prognostics Data Repository, NASA Ames Research
   Center, Moffett Field, CA, 2007.

## License

The code in this repository is MIT-licensed (see `LICENSE`). The NASA data and the original
EF-GBTSVM code are not part of this repository and are not covered by this license.
