"""Phase F (fine-grid) runner (PREREG_FINEGRID.md). Runs 2%, 4%, 6%, 8% training-label noise once,
on the dev batteries only -- the exact Phase B protocol, just four new noise values.

    python run_finegrid.py                 # all four conditions, in order
    python run_finegrid.py N02 N04          # a subset
    python run_finegrid.py --n-jobs 2       # override n_jobs (default 4, Windows process limit)

Nothing here changes a model, grid, seed, scorer or metric: build_models, load_dataset, metrics,
score_of and flip_labels come unchanged from run_experiments.py, and the per-fold loop below is the
loop of run_experiments.main() (leave-one-battery-out, GroupKFold inner CV, balanced accuracy, 150
candidates, grids="equal") with only the noise rate changed. Stump-fair / Stump-oracle use
phys_stump.stump_runs directly -- the same call phase_b_analysis.py makes for 0/10/20%, so the new
rates get identical treatment, not a reimplementation.

Checkpointed from fold 0 of the first condition (not only after a crash): results/finegrid_ckpt/
{condition}_fold{run}.csv. A restart reloads a finished fold's checkpoint instead of rerunning it.
"""
import hashlib
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phys_stump import stump_runs
from run_experiments import build_models, flip_labels, load_dataset, metrics, score_of

warnings.filterwarnings("ignore")

DATA = "data/nasa_battery_phys.csv"
OUT = "results"
N_ITER = 150
N_JOBS = 4
CONDITIONS = ["N02", "N04", "N06", "N08"]
RATES = {"N02": 0.02, "N04": 0.04, "N06": 0.06, "N08": 0.08}

FROZEN = {   # sha256 from PREREG_FINEGRID.md section 1 (identical to PREREG_DLITE.md section 1)
    "run_experiments.py": "a1c752165a85da7681db13b84310bd3fb1c873d20e8994a92b56769826e238d9",
    "phys_stump.py": "3fb0d0c73a90210f60f642e272e97d917d56f1e1ab6e099202e89012ff66b4b0",
    "efgbtsvm/model.py": "ebe73f7eef24d587bf2244769022d963f6d22e20d04016d977c25073a0498115",
    "efgbtsvm/granular_ball.py": "ed8242fe5ad62cc9b7dae9e06ccfb4f51a243383a714f178e6cc64a729c549ec",
    "efgbtsvm/noise.py": "bffee064ad9fb39913f9450ee4942034b41114965140ed04b43fd7c8f436f4f0",
    "efgbtsvm/rvfl_features.py": "f1e53b73b4575cb8cdd63cf3f24ad03668f86db452f46d0d85c62d9262660f68",
    "efgbtsvm/twsvm.py": "52805f30617dc5a3df8bdbb0e1acdcd95f00ac46421b74f50b7f4c51671fd614",
    "efgbtsvm/__init__.py": "c1d98b7ccc06c0775b0ac2e0761c74f4fde19b56ede6934f4c87dedc51ec6922",
    "data/nasa_battery_phys.csv": "7bceb4b0aaa4cc267f8d158c99d910bb9b78ca825f06a29f5b69fd3be1310274",
    "results/phase_a/phys2_selection.csv": "0997a1c69e473ece8aa7b8356aa1e082020409de91921043ef7df66c9918c1e6",
    "phase_b_analysis.py": "d8403a508ee3fbc59d34fc9ba03324181f8aa5c0cddd0210c584da220b531ffc",
}


def check_frozen():
    for p, h in FROZEN.items():
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if got != h:
            raise SystemExit(f"frozen file changed, not running: {p}\n  expected {h}\n  got      {got}")
    print("frozen hashes OK", flush=True)


def main():
    args, n_jobs = sys.argv[1:], N_JOBS
    if "--n-jobs" in args:
        i = args.index("--n-jobs")
        n_jobs = int(args[i + 1])
        del args[i:i + 2]
    conds = args or CONDITIONS
    assert all(c in CONDITIONS for c in conds), conds
    check_frozen()
    X, y, groups = load_dataset(DATA, "degraded", group_col="battery")
    idx_all = np.arange(len(X))
    splits = [(idx_all[groups != g], idx_all[groups == g]) for g in np.unique(groups)]
    print(f"{X.shape[0]} samples, {X.shape[1]} features, class balance {np.bincount(y)}; "
          f"folds={len(splits)}; n_iter={N_ITER}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "finegrid_ckpt"), exist_ok=True)

    for cond in conds:
        rate = RATES[cond]
        print(f"\n=== condition {cond} (noise={rate}) ===", flush=True)
        rows = []
        for run, (tr, te) in enumerate(splits):
            seed = run
            Xtr, Xte, yte = X[tr], X[te], y[te]
            ytr_clean, gtr = y[tr], groups[tr]
            ytr = flip_labels(ytr_clean, rate, np.random.RandomState(1000 + run))
            n_flipped = int((ytr != ytr_clean).sum())
            print(f"  fold {run}: flipped {n_flipped}/{len(tr)} ({100 * n_flipped / len(tr):.2f}%)",
                  flush=True)
            if len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
                print(f"run{run}: skipped (train or test has a single class)", flush=True)
                continue
            ck = os.path.join(OUT, "finegrid_ckpt", f"{cond}_fold{run}.csv")
            if os.path.exists(ck):
                rows += pd.read_csv(ck).to_dict("records")
                print(f"  fold {run}: loaded {ck} (already complete), skipping its model runs",
                      flush=True)
                continue
            fold_rows = []
            for mname, (est, grid) in build_models(seed, None, "equal").items():
                pipe = Pipeline([("scale", StandardScaler()), ("clf", est)])
                pgrid = {f"clf__{k}": v for k, v in grid.items()}
                n_comb = int(np.prod([len(v) for v in grid.values()]))
                cv = GroupKFold(min(5, len(np.unique(gtr))))
                search = RandomizedSearchCV(pipe, pgrid, n_iter=min(N_ITER, n_comb),
                                            scoring="balanced_accuracy", cv=cv, random_state=seed,
                                            n_jobs=n_jobs, refit=False)
                t0 = time.time()
                search.fit(Xtr, ytr, groups=gtr)
                best = search.best_params_
                final = Pipeline([("scale", StandardScaler()), ("clf", est)]).set_params(**best)
                t1 = time.time()
                final.fit(Xtr, ytr)
                train_time = time.time() - t1
                pred = final.predict(Xte)
                sc = score_of(final, Xte) if mname != "SVM" else final.decision_function(Xte)
                m = metrics(yte, pred, sc)
                ptr = final.predict(Xtr)
                m.update(noise_pct=int(round(rate * 100)), run=run, model=mname,
                         train_time=train_time, tune_time=t1 - t0,
                         train_acc_noisy=100 * accuracy_score(ytr, ptr),
                         train_acc_clean=100 * accuracy_score(ytr_clean, ptr),
                         params={k.replace("clf__", ""): v for k, v in best.items()})
                rows.append(m)
                fold_rows.append(m)
                print(f"run{run} {mname:13s} ACC={m['ACC']:.2f} BACC={m['BACC']:.2f} F1={m['F1']:.2f} "
                      f"AUC={m['AUC']:.2f} fit={train_time:.2f}s  {m['params']}", flush=True)
            pd.DataFrame(fold_rows).to_csv(ck + ".tmp", index=False)
            os.replace(ck + ".tmp", ck)
            print(f"  fold {run}: checkpoint saved {ck}", flush=True)

        # Stump-fair / Stump-oracle: identical call phase_b_analysis.py makes for 0/10/20%.
        for mode, name in (("fair", "Stump-fair"), ("oracle", "Stump-oracle")):
            fair = stump_runs(rate, mode)
            fair = fair.assign(noise_pct=int(round(rate * 100)))
            rows += fair.to_dict("records")

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT, f"finegrid_{cond}_runs.csv"), index=False)
        print(f"=== finished {cond} ({len(df)} rows) ===", flush=True)
    print("ALL CONDITIONS DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
