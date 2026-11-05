"""Single-feature threshold (depth-1 decision tree) on the physics features.

Reports two versions:
  (1) fair stump   -- best feature chosen per outer fold by inner CV on the TRAINING FOLD ONLY,
                      with the same inner-CV protocol as every run_experiments.py model:
                      GroupKFold over the 3 training batteries (one battery per inner fold),
                      balanced-accuracy scoring; exact ties go to the first feature in CSV column
                      order (a different feature may be chosen each fold)
  (2) oracle stump -- best feature chosen by max(mean test-fold ACC) across all folds
                      (uses test information; labelled "oracle upper bound" in tables)

Only version (1) is used as the comparison baseline in Phase B analysis tables.
Version (2) sets an upper bound on what a single-feature rule could ever achieve
given perfect knowledge of test performance.

History: the first fair-stump version used StratifiedKFold(3, shuffle) + accuracy, which mixes
cycles of one battery across inner folds and differs from the other models' protocol. Switched to
GroupKFold + balanced accuracy on Vansh's decision (Nov 2026); see REPORT.md "Phase B results".

    python phys_stump.py

Prints leave-one-battery-out ACC for every phys feature at every noise level,
then the fair and oracle summaries, under the same folds and label flips as
run_experiments.py (seed 1000 + fold).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from run_experiments import flip_labels, load_dataset

NOISES = [0.0, 0.1, 0.2]
INNER_SPLITS = 3                    # run_experiments.py uses GroupKFold(min(--cv=5, 3 train batteries)) = 3
INNER_SCORING = "balanced_accuracy"  # same as run_experiments.py --scoring default


def _fold_metrics(y_true, pred, score):
    return dict(ACC=100 * accuracy_score(y_true, pred),
                BACC=100 * balanced_accuracy_score(y_true, pred),
                F1=100 * f1_score(y_true, pred, zero_division=0),
                AUC=100 * roc_auc_score(y_true, score))


def stump_runs(noise, mode, path="data/nasa_battery_phys.csv"):
    """Per-fold rows for the stump baseline, same folds/flips as run_experiments.py.
    mode='fair':   feature picked per fold by GroupKFold inner CV (balanced accuracy) on the
                   (noisy) training fold, grouped by battery.
    mode='oracle': one feature for all folds, picked by max mean TEST-fold ACC (upper bound)."""
    X, y, groups = load_dataset(path, "degraded", group_col="battery")
    feats = [c for c in pd.read_csv(path, nrows=0).columns if c not in ("battery", "degraded")]
    idx = np.arange(len(X))
    folds = []
    for run, g in enumerate(np.unique(groups)):
        tr, te = idx[groups != g], idx[groups == g]
        ytr = y[tr].copy()
        if noise > 0:
            ytr = flip_labels(ytr, noise, np.random.RandomState(1000 + run))
        folds.append((run, tr, te, ytr))

    def fit_eval(fi, tr, te, ytr):
        clf = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X[tr][:, [fi]], ytr)
        return _fold_metrics(y[te], clf.predict(X[te][:, [fi]]),
                             clf.predict_proba(X[te][:, [fi]])[:, 1])

    if mode == "oracle":
        mean_acc = [np.mean([fit_eval(fi, tr, te, ytr)["ACC"] for _, tr, te, ytr in folds])
                    for fi in range(len(feats))]
        chosen = {run: int(np.argmax(mean_acc)) for run, *_ in folds}
    elif mode == "fair":
        chosen = {}
        for run, tr, te, ytr in folds:
            gtr = groups[tr]
            cv = GroupKFold(min(INNER_SPLITS, len(np.unique(gtr))))
            cv_score = [cross_val_score(DecisionTreeClassifier(max_depth=1, random_state=0),
                                        X[tr][:, [fi]], ytr, groups=gtr, cv=cv,
                                        scoring=INNER_SCORING).mean()
                        for fi in range(len(feats))]
            chosen[run] = int(np.argmax(cv_score))
    else:
        raise ValueError(mode)

    rows = []
    for run, tr, te, ytr in folds:
        fi = chosen[run]
        m = fit_eval(fi, tr, te, ytr)
        tree = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X[tr][:, [fi]], ytr)
        pred_tr = tree.predict(X[tr][:, [fi]])
        m.update(train_acc_noisy=100 * accuracy_score(ytr, pred_tr),
                 train_acc_clean=100 * accuracy_score(y[tr], pred_tr))
        m.update(run=run, feature=feats[fi],
                 model="Stump-fair" if mode == "fair" else "Stump-oracle")
        rows.append(m)
    return pd.DataFrame(rows)


def main():
    X, y, groups = load_dataset("data/nasa_battery_phys.csv", "degraded", group_col="battery")
    feat_names = [c for c in pd.read_csv("data/nasa_battery_phys.csv", nrows=0).columns
                  if c not in ("battery", "degraded")]
    idx = np.arange(len(X))

    # ── full per-feature table (each row = one feature × noise level) ─────────
    print(f"{'Feature':20s}  {'Noise':>6s}  {'F0':>7s}  {'F1':>7s}  {'F2':>7s}  {'F3':>7s}  {'Mean':>7s}  {'SD':>6s}")
    for noise in NOISES:
        for fi, feat in enumerate(feat_names):
            accs = []
            for run, g in enumerate(np.unique(groups)):
                tr, te = idx[groups != g], idx[groups == g]
                ytr = y[tr].copy()
                if noise > 0:
                    ytr = flip_labels(ytr, noise, np.random.RandomState(1000 + run))
                clf = DecisionTreeClassifier(max_depth=1, random_state=0)
                clf.fit(X[tr][:, [fi]], ytr)
                accs.append(100 * accuracy_score(y[te], clf.predict(X[te][:, [fi]])))
            print(f"{feat:20s}  {int(noise*100):>5}%  "
                  f"{'  '.join(f'{a:7.2f}' for a in accs)}  {np.mean(accs):7.2f}  {np.std(accs, ddof=1):6.2f}")
        print()

    # ── clean summary table (same code path as the Phase B analysis) ─────────
    print("\n=== Clean summary: fair vs oracle stump per noise level ===")
    print(f"{'Noise':>7s}  {'Fair-stump mean ACC':>22s}  {'Fair features (F0..F3)':60s}  "
          f"{'Oracle ACC':>16s}  {'Oracle feat':20s}")
    for noise in NOISES:
        fair, orc = stump_runs(noise, "fair"), stump_runs(noise, "oracle")
        fair_str = f"{fair.ACC.mean():.2f} +/- {fair.ACC.std(ddof=1):.2f}"
        oracle_str = f"{orc.ACC.mean():.2f} +/- {orc.ACC.std(ddof=1):.2f}"
        print(f"{int(noise*100):>6}%  {fair_str:>22s}  {str(list(fair.feature)):60s}  "
              f"{oracle_str:>16s}  {orc.feature.iloc[0]:20s}")


if __name__ == "__main__":
    main()
