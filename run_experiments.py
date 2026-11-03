"""Full experimental pipeline for EF-GBTSVM on any binary CSV dataset.

Example
-------
python run_experiments.py --data data/smart_grid_stability.csv --name smartgrid \
       --label-col stabf --drop stab --runs 5 --n-iter 30

Protocol (same spirit as the paper, but stricter):
  * stratified 70/30 split, repeated over `--runs` seeds
  * scaler fitted on train only (no leakage)
  * hyper-parameters tuned by 5-fold CV on the train part (randomised search)
  * optional label noise on the TRAIN labels only (`--noise 0.1`)
  * metrics: ACC, balanced ACC, precision, recall, F1, G-mean, AUC, train time
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, make_scorer, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import (GroupKFold, RandomizedSearchCV,
                                     StratifiedKFold, train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from efgbtsvm import EFGBTSVM, RVFLClassifier

warnings.filterwarnings("ignore")

C_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
H_GRID = [23, 63, 103, 143, 183]
ACT_GRID = [1, 2, 3, 4, 10]            # selu, relu, sigmoid, sin, tanh
PUR_GRID = [0.85, 0.9, 0.925, 0.95, 0.98]
PUR_GRID_W = [0.55, 0.65, 0.75, 0.85, 0.95]  # EF-GBTSVM-W only: looser, so the
                                              # purity-weighting has impure balls
                                              # to act on (attempt 2, see REPORT.md)


def _gmean_score(y_true, y_pred):
    """Same G-mean definition as metrics() below, as an sklearn scorer
    (attempt 3: optional alternate scoring for purity selection, see REPORT.md)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return np.sqrt(sens * spec)


GMEAN_SCORER = make_scorer(_gmean_score)


def _parse_overrides(pairs):
    """'Model:value' strings -> {model: value}. Used for --cv-override /
    --scoring-override, both no-ops unless a model name matches."""
    out = {}
    for p in pairs:
        model, _, value = p.partition(":")
        if not value:
            raise SystemExit(f"override must be 'Model:value', got {p!r}")
        out[model] = value
    return out


def _resolve_scoring(name):
    return GMEAN_SCORER if name == "gmean" else name


def load_dataset(path, label_col=None, drop=(), header="infer", sep=",", group_col=None):
    df = pd.read_csv(path, header=0 if header == "infer" else None, sep=sep)
    groups = None
    if group_col is not None:                       # e.g. battery id: never a feature
        groups = LabelEncoder().fit_transform(df[group_col])
        df = df.drop(columns=[group_col])
    if label_col is None:
        label_col = df.columns[-1]
    elif label_col not in df.columns:
        label_col = df.columns[int(label_col)]          # allow index
    df = df.drop(columns=[c for c in drop if c in df.columns])
    y = LabelEncoder().fit_transform(df[label_col])
    X = df.drop(columns=[label_col])
    X = pd.get_dummies(X, drop_first=True).astype(float)   # encode categoricals
    X = X.fillna(X.median())
    if len(np.unique(y)) != 2:
        raise SystemExit(f"Binary labels required, found {len(np.unique(y))} classes")
    return X.values, y, groups


def build_models(seed, include=None):
    ef = lambda **kw: EFGBTSVM(random_state=seed, **kw)
    ef_grid = dict(C1=C_GRID, n_hidden=H_GRID, activation=ACT_GRID, purity=PUR_GRID)
    models = {
        # proposed + ablations -------------------------------------------------
        "EF-GBTSVM": (ef(), ef_grid),
        "EF-GBTSVM-W": (ef(weighted=True),           # purity-weighted balls (novelty)
                        dict(C1=C_GRID, n_hidden=H_GRID, activation=ACT_GRID,
                             purity=PUR_GRID_W)),     # looser grid: EF-GBTSVM's own
                                                       # grid/results (ef_grid, PUR_GRID)
                                                       # are untouched
        # ablation: same loose grid as EF-GBTSVM-W but weighting OFF, to separate
        # "looser threshold" from "purity weighting" (see REPORT.md)
        "EF-GBTSVM-L": (ef(), dict(C1=C_GRID, n_hidden=H_GRID, activation=ACT_GRID,
                                   purity=PUR_GRID_W)),
        # attempt 4: purity set inside fit() from a kNN noise estimate, so it is
        # not in the grid; noise_alpha fixed at 1.0 (chosen a priori, see REPORT.md)
        "EF-GBTSVM-A": (ef(purity_mode="adaptive", noise_alpha=1.0),
                        dict(C1=C_GRID, n_hidden=H_GRID, activation=ACT_GRID)),
        "GBTSVM": (ef(features="original"), dict(C1=C_GRID, purity=PUR_GRID)),
        "HF-GBTSVM": (ef(features="hidden"), ef_grid),
        "EF-TSVM": (ef(use_balls=False), dict(C1=C_GRID, n_hidden=H_GRID, activation=ACT_GRID)),
        "TSVM": (ef(use_balls=False, features="original"), dict(C1=C_GRID)),
        # classical baselines ---------------------------------------------------
        "SVM": (SVC(kernel="rbf", random_state=seed),
                dict(C=[0.1, 1, 10, 100], gamma=["scale", 0.001, 0.01, 0.1])),
        "RVFL": (RVFLClassifier(random_state=seed),
                 dict(C=C_GRID, n_hidden=H_GRID, activation=ACT_GRID)),
        "RandomForest": (RandomForestClassifier(random_state=seed, n_jobs=-1),
                         dict(n_estimators=[100, 300], max_depth=[None, 10, 20])),
        "MLP": (MLPClassifier(max_iter=300, random_state=seed),
                dict(hidden_layer_sizes=[(32,), (64,), (64, 32)], alpha=[1e-4, 1e-2])),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = (XGBClassifier(random_state=seed, n_jobs=-1, verbosity=0,
                                           eval_metric="logloss"),
                             dict(n_estimators=[100, 300], max_depth=[3, 6],
                                  learning_rate=[0.05, 0.1, 0.3]))
    except Exception:
        pass
    if include:
        models = {k: v for k, v in models.items() if k in include}
    return models


def score_of(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return dict(
        ACC=100 * accuracy_score(y_true, y_pred),
        BACC=100 * balanced_accuracy_score(y_true, y_pred),
        PREC=100 * precision_score(y_true, y_pred, zero_division=0),
        REC=100 * recall_score(y_true, y_pred, zero_division=0),
        F1=100 * f1_score(y_true, y_pred, zero_division=0),
        GMEAN=100 * np.sqrt(sens * spec),
        AUC=100 * roc_auc_score(y_true, y_score),
    )


def flip_labels(y, rate, rng):
    y = y.copy()
    idx = rng.choice(len(y), int(rate * len(y)), replace=False)
    y[idx] = 1 - y[idx]
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--label-col", default=None, help="column name or index (default: last)")
    ap.add_argument("--drop", nargs="*", default=[], help="columns to drop (e.g. leaky targets)")
    ap.add_argument("--no-header", action="store_true")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--n-iter", type=int, default=30, help="random-search candidates")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--scoring", default="balanced_accuracy")
    ap.add_argument("--cv-override", nargs="*", default=[],
                    help="per-model CV fold override, e.g. --cv-override EF-GBTSVM-W:10 "
                         "(every other model keeps using --cv unchanged)")
    ap.add_argument("--scoring-override", nargs="*", default=[],
                    help="per-model scoring override for hyperparameter search, e.g. "
                         "--scoring-override EF-GBTSVM-W:gmean (use 'gmean' for the "
                         "G-mean scorer, or any sklearn scoring string such as 'f1'; "
                         "every other model keeps using --scoring unchanged)")
    ap.add_argument("--noise", type=float, default=0.0, help="train label-noise rate")
    ap.add_argument("--group-col", default=None,
                    help="column with group ids (e.g. battery_id). Enables leave-one-group-out "
                         "evaluation (one run per group) and group-aware CV; column is not a feature.")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--max-rows", type=int, default=None, help="subsample for quick tests")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cv_override = _parse_overrides(a.cv_override)
    scoring_override = _parse_overrides(a.scoring_override)

    name = a.name or os.path.splitext(os.path.basename(a.data))[0]
    X, y, groups = load_dataset(a.data, a.label_col, a.drop,
                                "none" if a.no_header else "infer", group_col=a.group_col)
    if a.max_rows and len(X) > a.max_rows:
        sel = np.random.RandomState(0).choice(len(X), a.max_rows, replace=False)
        X, y = X[sel], y[sel]
        groups = groups[sel] if groups is not None else None
    print(f"[{name}] {X.shape[0]} samples, {X.shape[1]} features, "
          f"class balance {np.bincount(y)}, noise={a.noise}")
    os.makedirs(a.out, exist_ok=True)

    # ---- splits: stratified random 70/30 repeats, or leave-one-group-out ----------
    idx_all = np.arange(len(X))
    if groups is None:
        splits = [train_test_split(idx_all, test_size=0.3, stratify=y, random_state=r)
                  for r in range(a.runs)]
    else:
        splits = [(idx_all[groups != g], idx_all[groups == g]) for g in np.unique(groups)]
        print(f"leave-one-group-out over {len(splits)} groups "
              f"(group sizes {np.bincount(groups)}); --runs is ignored")

    rows = []
    for run, (tr, te) in enumerate(splits):
        seed = run
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        gtr = groups[tr] if groups is not None else None
        if len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
            print(f"run{run}: skipped (train or test has a single class)")
            continue
        if a.noise > 0:
            ytr = flip_labels(ytr, a.noise, np.random.RandomState(1000 + seed))
        for mname, (est, grid) in build_models(seed, a.models).items():
            pipe = Pipeline([("scale", StandardScaler()), ("clf", est)])
            pgrid = {f"clf__{k}": v for k, v in grid.items()}
            n_comb = int(np.prod([len(v) for v in grid.values()]))
            cv_n = int(cv_override.get(mname, a.cv))
            scoring_m = _resolve_scoring(scoring_override.get(mname, a.scoring))
            cv = (GroupKFold(min(cv_n, len(np.unique(gtr)))) if gtr is not None
                  else StratifiedKFold(cv_n, shuffle=True, random_state=seed))
            search = RandomizedSearchCV(pipe, pgrid, n_iter=min(a.n_iter, n_comb),
                                        scoring=scoring_m, cv=cv, random_state=seed,
                                        n_jobs=-1, refit=False)
            t0 = time.time()
            search.fit(Xtr, ytr, groups=gtr)
            best = search.best_params_
            # time a clean refit with the best params = reported training time
            final = Pipeline([("scale", StandardScaler()), ("clf", est)]).set_params(**best)
            t1 = time.time()
            final.fit(Xtr, ytr)
            train_time = time.time() - t1
            clf = final.named_steps["clf"]
            if mname == "EF-GBTSVM-W":   # diagnostics: is the weighting mechanism
                                          # actually seeing impure balls this fold?
                pur = clf.ball_purities_
                n_impure = int((pur < 0.95).sum())
                print(f"  [EF-GBTSVM-W diag] purity_param={best.get('clf__purity')} "
                      f"n_balls={len(pur)} n_impure(<0.95)={n_impure} "
                      f"min={pur.min():.3f} mean={pur.mean():.3f}")
            if mname == "EF-GBTSVM-A":   # diagnostics: estimated noise -> purity
                print(f"  [EF-GBTSVM-A diag] est_noise={clf.estimated_noise_:.4f} "
                      f"purity_used={clf.purity_used_:.4f} n_balls={clf.n_balls_}")
            pred = final.predict(Xte)
            sc = score_of(final, Xte) if mname != "SVM" else final.decision_function(Xte)
            m = metrics(yte, pred, sc)
            m.update(dataset=name, noise=a.noise, run=run, model=mname,
                     train_time=train_time, tune_time=t1 - t0,
                     params={k.replace("clf__", ""): v for k, v in best.items()})
            rows.append(m)
            print(f"run{run} {mname:13s} ACC={m['ACC']:.2f} F1={m['F1']:.2f} "
                  f"AUC={m['AUC']:.2f} fit={train_time:.2f}s  {m['params']}")

    df = pd.DataFrame(rows)
    tag = f"{name}_noise{int(a.noise * 100)}"
    df.to_csv(os.path.join(a.out, f"{tag}_runs.csv"), index=False)
    cols = ["ACC", "BACC", "PREC", "REC", "F1", "GMEAN", "AUC", "train_time"]
    g = df.groupby("model")[cols]
    summ = g.mean().round(2).astype(str) + " ± " + g.std().round(2).astype(str)
    summ = summ.loc[df["model"].unique()]
    summ.to_csv(os.path.join(a.out, f"{tag}_summary.csv"))
    print("\n=== Summary (mean ± std over runs) ===")
    print(summ.to_string())


if __name__ == "__main__":
    main()
