# EF-GBTSVM for Electrical Engineering data

Clean re-implementation of the EF-GBTSVM idea (granular balls -> random hidden
features -> Twin SVM) plus a full experiment pipeline. Original paper/code:
Quadir, Sajid, Tanveer, Suganthan, "Enhanced Feature Based Granular Ball Twin
Support Vector Machine", IEEE TETCI 2026 (arXiv:2410.05786). Cite it.

## Install
    pip install numpy pandas scipy scikit-learn cvxopt xgboost

## Run
UCI Smart Grid Stability (columns tau1-4, p1-4, g1-4, stab, stabf).
`stab` is a continuous version of the label, so it MUST be dropped (leakage):

    python run_experiments.py --data data/smart_grid_stability.csv --name smartgrid \
        --label-col stabf --drop stab --runs 5 --n-iter 30

Any other CSV (header row, last column = label by default):

    python run_experiments.py --data data/mydata.csv --label-col fault --runs 5
    python run_experiments.py --data raw.csv --no-header          # headerless file

Label-noise experiment (flips TRAIN labels only):  `--noise 0.1`  (0.05 ... 0.2)
Quick smoke test:  `--runs 1 --n-iter 5 --cv 3 --max-rows 2000`
Subset of models:  `--models EF-GBTSVM TSVM XGBoost`

Outputs: `results/<name>_noise<pct>_runs.csv` (every run, best params, times) and
`..._summary.csv` (mean ± std of ACC, BACC, PREC, REC, F1, G-mean, AUC, time).

Multi-dataset ranking (Friedman + Nemenyi):

    python stats_tests.py results/*_noise0_runs.csv --metric ACC


## NASA Li-ion battery (healthy / degraded)
1. Get the NASA PCoE battery data (B0005, B0006, B0007, B0018 `.mat` files) into `data/raw/`.
2. Build the CSV (one row per discharge cycle; label = SOH < 80%):

       python prep_nasa_battery.py data/raw/B0005.mat data/raw/B0006.mat \
              data/raw/B0007.mat data/raw/B0018.mat --out data/nasa_battery.csv

3. Leave-one-battery-out evaluation (one run per battery, group-aware CV):

       python run_experiments.py --data data/nasa_battery.csv --name nasa \
              --label-col degraded --group-col battery --n-iter 30
       python run_experiments.py ... --noise 0.1     # and 0.2

Why group-wise: random splits put neighbouring cycles of the same battery in
train and test, which inflates accuracy. `capacity`/`SOH` are never features
(they define the label); the cycle index is excluded unless `--keep-cycle`.
Also report other SOH thresholds (0.70, 0.75, 0.85) via `--soh-threshold`.

## Models
EF-GBTSVM (proposed), ablations GBTSVM / HF-GBTSVM / EF-TSVM / TSVM,
baselines SVM(RBF), RVFL, RandomForest, MLP, XGBoost. (GBSVM, which needs a PSO
solver, is not included.)

## Differences from the reference repo
* seeds are fixed everywhere; scaler fitted on train only
* decision rule uses the perpendicular distance |w'x+b|/||w|| (paper Eq. 22)
* 5-fold CV random search is built in; stratified repeated splits
* multiple metrics, ablation switches, label-noise switch
