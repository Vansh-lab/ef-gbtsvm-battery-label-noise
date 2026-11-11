"""Phase D-lite runner (PREREG_DLITE.md). Runs N30, N40, BND once, on the dev batteries only.

    python run_dlite.py

Nothing here changes a model, grid, seed, scorer or metric: build_models, load_dataset, metrics and
score_of come unchanged from run_experiments.py, and the per-fold loop below is the loop of
run_experiments.main() (leave-one-battery-out, GroupKFold inner CV, balanced accuracy, 150
candidates, grids="equal") with only the label-flipping step replaced by the three D-lite
conditions. Stump-fair / Stump-oracle follow phys_stump.stump_runs with the same flipped labels.
The train-test gap is taken from the final fit itself (the refit that phase_b_train_gap.py repeats).
"""
import hashlib
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from phys_stump import INNER_SCORING, INNER_SPLITS, _fold_metrics
from prep_phys_features import load_battery
from run_experiments import build_models, flip_labels, load_dataset, metrics, score_of

warnings.filterwarnings("ignore")

DATA = "data/nasa_battery_phys.csv"
RAW = ["data/raw/B0005.mat", "data/raw/B0006.mat", "data/raw/B0007.mat", "data/raw/B0018.mat"]
OUT = "results"
N_ITER = 150
N_JOBS = 4
BAND = (0.75, 0.85)
BOUNDARY_FRACTION = 0.5
CONDITIONS = ["N30", "N40", "BND"]
RATES = {"N30": 0.30, "N40": 0.40}

FROZEN = {   # sha256 from PREREG_DLITE.md section 1
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


def soh_per_row():
    """SOH = Capacity / first-cycle Capacity per battery, aligned to the CSV (asserted)."""
    frames = []
    for p in RAW:
        b = load_battery(p)
        b["soh"] = b["capacity"] / b["capacity"].iloc[0]
        frames.append(b)
    s = pd.concat(frames, ignore_index=True)
    ref = pd.read_csv(DATA, usecols=["battery", "degraded"])
    ok = (len(s) == len(ref) == 636 and (s["battery"].values == ref["battery"].values).all()
          and ((s["soh"].values < 0.80).astype(int) == ref["degraded"].values).all())
    if not ok:
        raise SystemExit("SOH rows/labels do not match nasa_battery_phys.csv; stopping (prereg 2b.1)")
    print("SOH aligned with CSV: 636 rows, labels reproduced", flush=True)
    return s["soh"].values


def corrupt(cond, ytr_clean, soh_tr, fold):
    """Flipped training labels for one (condition, fold); returns (labels, n_boundary, n_flipped)."""
    rng = np.random.RandomState(1000 + fold)
    if cond in RATES:
        yb = flip_labels(ytr_clean, RATES[cond], rng)
        return yb, np.nan, int((yb != ytr_clean).sum())
    inb = np.where((soh_tr >= BAND[0] - 1e-12) & (soh_tr <= BAND[1] + 1e-12))[0]
    nflip = int(BOUNDARY_FRACTION * len(inb))
    yb = ytr_clean.copy()
    if nflip:
        idx = rng.choice(inb, nflip, replace=False)
        yb[idx] = 1 - yb[idx]
    return yb, len(inb), nflip


def stump_rows(X, y, groups, feats, folds, cond):
    """Stump-fair and Stump-oracle, logic of phys_stump.stump_runs with given flipped labels."""
    def fit_eval(fi, tr, te, ytr):
        clf = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X[tr][:, [fi]], ytr)
        return _fold_metrics(y[te], clf.predict(X[te][:, [fi]]), clf.predict_proba(X[te][:, [fi]])[:, 1])
    chosen = {}
    for run, tr, te, ytr in folds:
        gtr = groups[tr]
        cv = GroupKFold(min(INNER_SPLITS, len(np.unique(gtr))))
        sc = [cross_val_score(DecisionTreeClassifier(max_depth=1, random_state=0), X[tr][:, [fi]], ytr,
                              groups=gtr, cv=cv, scoring=INNER_SCORING).mean() for fi in range(len(feats))]
        chosen[run] = int(np.argmax(sc))
    mean_acc = [np.mean([fit_eval(fi, tr, te, ytr)["ACC"] for _, tr, te, ytr in folds])
                for fi in range(len(feats))]
    orc = int(np.argmax(mean_acc))
    rows = []
    for name, pick in (("Stump-fair", chosen), ("Stump-oracle", {r: orc for r, *_ in folds})):
        for run, tr, te, ytr in folds:
            fi = pick[run]
            m = fit_eval(fi, tr, te, ytr)
            tree = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X[tr][:, [fi]], ytr)
            p = tree.predict(X[tr][:, [fi]])
            m.update(train_acc_noisy=100 * accuracy_score(ytr, p), train_acc_clean=100 * accuracy_score(y[tr], p),
                     run=run, model=name, condition=cond, feature=feats[fi])
            rows.append(m)
    return rows


def main():
    check_frozen()
    X, y, groups = load_dataset(DATA, "degraded", group_col="battery")
    feats = [c for c in pd.read_csv(DATA, nrows=0).columns if c not in ("battery", "degraded")]
    soh = soh_per_row()
    idx_all = np.arange(len(X))
    splits = [(idx_all[groups != g], idx_all[groups == g]) for g in np.unique(groups)]
    print(f"{X.shape[0]} samples, {X.shape[1]} features, class balance {np.bincount(y)}; "
          f"folds={len(splits)}; n_iter={N_ITER}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    flip_log = []

    for cond in CONDITIONS:
        print(f"\n=== condition {cond} ===", flush=True)
        rows, folds = [], []
        for run, (tr, te) in enumerate(splits):
            seed = run
            Xtr, Xte, yte = X[tr], X[te], y[te]
            ytr_clean, gtr = y[tr], groups[tr]
            ytr, n_b, n_f = corrupt(cond, ytr_clean, soh[tr], run)
            flip_log.append(dict(condition=cond, fold=run, n_train=len(tr), n_boundary=n_b, n_flipped=n_f,
                                 flip_pct=100 * n_f / len(tr), train_pos_after=int(ytr.sum()),
                                 test_untouched=bool((y[te] == yte).all())))
            print(f"  fold {run}: flipped {n_f}/{len(tr)} ({100 * n_f / len(tr):.1f}%), boundary set {n_b}", flush=True)
            if len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
                print(f"run{run}: skipped (train or test has a single class)", flush=True)
                continue
            folds.append((run, tr, te, ytr))
            for mname, (est, grid) in build_models(seed, None, "equal").items():
                pipe = Pipeline([("scale", StandardScaler()), ("clf", est)])
                pgrid = {f"clf__{k}": v for k, v in grid.items()}
                n_comb = int(np.prod([len(v) for v in grid.values()]))
                cv = GroupKFold(min(5, len(np.unique(gtr))))
                search = RandomizedSearchCV(pipe, pgrid, n_iter=min(N_ITER, n_comb), scoring="balanced_accuracy",
                                            cv=cv, random_state=seed, n_jobs=N_JOBS, refit=False)
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
                m.update(condition=cond, run=run, model=mname, train_time=train_time, tune_time=t1 - t0,
                         train_acc_noisy=100 * accuracy_score(ytr, ptr),
                         train_acc_clean=100 * accuracy_score(ytr_clean, ptr),
                         params={k.replace("clf__", ""): v for k, v in best.items()})
                rows.append(m)
                print(f"run{run} {mname:13s} ACC={m['ACC']:.2f} BACC={m['BACC']:.2f} F1={m['F1']:.2f} "
                      f"AUC={m['AUC']:.2f} fit={train_time:.2f}s  {m['params']}", flush=True)
        rows += stump_rows(X, y, groups, feats, folds, cond)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"dlite_{cond}_runs.csv"), index=False)
        pd.DataFrame(flip_log).to_csv(os.path.join(OUT, "dlite_flip_log.csv"), index=False)
        print(f"=== finished {cond} ===", flush=True)
    print("ALL CONDITIONS DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
