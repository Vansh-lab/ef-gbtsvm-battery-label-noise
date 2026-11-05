"""Phase A: choose an EF-GBTSVM variant by nested inner cross-validation on the
training cells only.

    python phase_a_inner_cv.py --data data/nasa_battery_phys.csv --name phys
    python phase_a_inner_cv.py --data data/nasa_battery.csv --name base

For each outer fold the held-out test cell is removed first; nothing below uses it.
Within the three training cells, each cell is the validation cell once. Every model is
tuned by randomised search (up to --n-iter candidates or its full grid, leave-one-cell-out
CV on the other two training cells, balanced accuracy) and then scored on the
validation cell. Training labels are flipped exactly as in run_experiments.py
(seed 1000 + fold), so the validation cell carries the same noisy labels the tuning
would see in practice; its clean labels are recorded only as a diagnostic and are not
used for selection.

Selection rule (fixed before running): per outer fold, the candidate with the highest
mean validation balanced accuracy (noisy labels) over the three validation cells and the
three noise levels; an exact tie goes to the candidate with fewer added components.
RBF-SVM and a depth-1 tree (one-feature rule) are run the same way as references only.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from efgbtsvm import EFGBTSVM
from run_experiments import ACT_GRID, C_GRID, H_GRID, PUR_GRID, flip_labels, load_dataset

OFFSET_GRID = [-0.10, -0.05, 0.0, 0.05, 0.10]   # EF-GBTSVM-J: shift of the noise-set purity
MASS_GRID = [0.0, 0.5, 1.0]                      # EF-GBTSVM-M: exponent on ball mass n_i * p_i
PUR_GRID_WIDE = [0.55, 0.65, 0.75, 0.85, 0.9, 0.925, 0.95, 0.98]   # EF-GBTSVM-L8: fixed tau
                                                 # over the range J can reach (review F3)
SVM_C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
SVM_GAMMA = ["scale"] + [float(g) for g in np.logspace(-4, 1, 14)]   # 10 x 15 = 150


def build(seed):
    """name -> (estimator, grid, number of added components, is_candidate)."""
    ef = lambda **kw: EFGBTSVM(random_state=seed, **kw)
    base = dict(C1=C_GRID, n_hidden=H_GRID, activation=ACT_GRID)
    return {
        "EF-GBTSVM": (ef(), {**base, "purity": PUR_GRID}, 0, True),
        "EF-GBTSVM-L8": (ef(), {**base, "purity": PUR_GRID_WIDE}, 0, True),
        "EF-GBTSVM-J": (ef(purity_mode="adaptive_offset"),
                        {**base, "purity_offset": OFFSET_GRID}, 1, True),
        "EF-GBTSVM-M": (ef(ball_weight="mass"),
                        {**base, "purity": PUR_GRID, "mass_power": MASS_GRID}, 1, True),
        "EF-GBTSVM-JM": (ef(purity_mode="adaptive_offset", ball_weight="mass"),
                         {**base, "purity_offset": OFFSET_GRID, "mass_power": MASS_GRID},
                         2, True),
        "SVM": (SVC(kernel="rbf", random_state=seed), dict(C=SVM_C, gamma=SVM_GAMMA), 0, False),
        "Stump": (DecisionTreeClassifier(max_depth=1, random_state=seed), {}, 0, False),
    }


def select(df):
    """Apply the selection rule; returns one row per outer fold."""
    cand = df[df["candidate"]]
    score = cand.groupby(["fold", "test_cell", "model"])["bacc_noisy"].mean().reset_index()
    comps = cand.groupby("model")["components"].first()
    out = []
    for (fold, cell), s in score.groupby(["fold", "test_cell"]):
        s = s.assign(components=s["model"].map(comps))
        best = s.sort_values(["bacc_noisy", "components"], ascending=[False, True]).iloc[0]
        out.append(dict(fold=fold, test_cell=cell, selected=best["model"],
                        score=round(best["bacc_noisy"], 2),
                        **{m: round(v, 2) for m, v in zip(s["model"], s["bacc_noisy"])}))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--noise", type=float, nargs="*", default=[0.0, 0.1, 0.2])
    ap.add_argument("--n-iter", type=int, default=150)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--out", default="results/phase_a")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    X, y, groups = load_dataset(a.data, "degraded", group_col="battery")
    feat_names = [c for c in pd.read_csv(a.data, nrows=0).columns
                  if c not in ("battery", "degraded")]
    cell_names = np.array(sorted(pd.read_csv(a.data, usecols=["battery"])["battery"].unique()))
    print(f"[phase A {a.name}] {X.shape[0]} cycles, {X.shape[1]} features, "
          f"n_iter={a.n_iter}, noise={a.noise}", flush=True)

    rows = []
    for k, g_test in enumerate(np.unique(groups)):
        tr = np.where(groups != g_test)[0]               # the test cell is dropped here
        assert not np.isin(tr, np.where(groups == g_test)[0]).any()
        Xtr, ytr_clean, gtr = X[tr], y[tr], groups[tr]
        for noise in a.noise:
            ytr = (flip_labels(ytr_clean, noise, np.random.RandomState(1000 + k))
                   if noise > 0 else ytr_clean.copy())
            for v in np.unique(gtr):
                fit, val = gtr != v, gtr == v
                for name, (est, grid, ncomp, is_cand) in build(k).items():
                    if a.models and name not in a.models:
                        continue
                    pipe = Pipeline([("scale", StandardScaler()), ("clf", est)])
                    t0 = time.time()
                    if grid:
                        pgrid = {f"clf__{p}": vals for p, vals in grid.items()}
                        n_comb = int(np.prod([len(vals) for vals in grid.values()]))
                        search = RandomizedSearchCV(
                            pipe, pgrid, n_iter=min(a.n_iter, n_comb),
                            scoring="balanced_accuracy", cv=GroupKFold(2),
                            random_state=k, n_jobs=-1, refit=True)
                        search.fit(Xtr[fit], ytr[fit], groups=gtr[fit])
                        model = search.best_estimator_
                        params = {p.replace("clf__", ""): val
                                  for p, val in search.best_params_.items()}
                    else:
                        model = pipe.fit(Xtr[fit], ytr[fit])
                        params = {}
                    tune_time = time.time() - t0
                    pred = model.predict(Xtr[val])
                    clf = model.named_steps["clf"]
                    row = dict(
                        features=a.name, fold=k, test_cell=cell_names[g_test],
                        noise=noise, val_cell=cell_names[v], model=name,
                        candidate=is_cand, components=ncomp,
                        bacc_noisy=100 * balanced_accuracy_score(ytr[val], pred),
                        acc_noisy=100 * accuracy_score(ytr[val], pred),
                        bacc_clean=100 * balanced_accuracy_score(ytr_clean[val], pred),
                        acc_clean=100 * accuracy_score(ytr_clean[val], pred),
                        n_fit=int(fit.sum()), n_val=int(val.sum()),
                        tune_time=round(tune_time, 2), params=params,
                        n_balls=getattr(clf, "n_balls_", None),
                        purity_used=getattr(clf, "purity_used_", None),
                        est_noise=getattr(clf, "estimated_noise_", None),
                        stump_feature=(feat_names[clf.tree_.feature[0]]
                                       if name == "Stump" and clf.tree_.feature[0] >= 0
                                       else None),
                    )
                    rows.append(row)
                    extra = ""
                    if row["purity_used"] is not None:
                        extra = f" tau={row['purity_used']:.3f} balls={row['n_balls']}"
                        if row["est_noise"] is not None:
                            extra += f" est_noise={row['est_noise']:.3f}"
                    if row["stump_feature"]:
                        extra = f" feature={row['stump_feature']}"
                    print(f"fold{k} test={row['test_cell']} noise={noise:.1f} "
                          f"val={row['val_cell']} {name:13s} BACC={row['bacc_noisy']:6.2f} "
                          f"(clean {row['bacc_clean']:6.2f}) {tune_time:6.1f}s "
                          f"{params}{extra}", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(a.out, f"{a.name}_inner_cv.csv"),
                                      index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, f"{a.name}_inner_cv.csv"), index=False)
    sel = select(df)
    sel.to_csv(os.path.join(a.out, f"{a.name}_selection.csv"), index=False)
    print("\n=== Mean validation BACC (noisy labels) by fold, model and noise ===")
    print(df.pivot_table(index=["model"], columns=["fold", "noise"], values="bacc_noisy",
                         aggfunc="mean").round(2).to_string())
    print("\n=== Selection (mean over validation cells and noise levels) ===")
    print(sel.to_string(index=False))


if __name__ == "__main__":
    main()
