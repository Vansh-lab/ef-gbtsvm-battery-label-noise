"""How much of the label the BMS features already give away (paper, Sections 4, 5, 7).

    python proxy_checks.py

1. Capacity proxy: |mean current| x discharge time is the discharged charge, i.e.
   capacity. Taken relative to each cell's first cycle and thresholded at 0.80
   (the SOH threshold), it reproduces the `degraded` label almost exactly.
2. Harder subset (data/nasa_battery_hard.csv, discharge_time and i_mean removed):
   a single threshold (depth-1 tree) on one feature, fitted on the three
   training cells, scored leave-one-battery-out.
3. Within-cell correlation of discharge_time with time_to_v3p5.
4. The time_to_v3p5 rule as a Figure 1 baseline row: same rule on the full
   13-feature data (it only reads time_to_v3p5, so it matches step 2), with
   per-fold ACC/F1, mean +- sample SD, and the threshold learned in each fold.
5. The same rule under training-label noise, with exactly the folds and label
   flips of run_experiments.py (flip seed 1000 + fold index), for Figure 2.
6. How constant the discharge current is (mean |I| per cycle, by cell).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from run_experiments import flip_labels, load_dataset


def main():
    d = pd.read_csv("data/nasa_battery.csv")
    q = d["i_mean"].abs() * d["discharge_time"]
    ok = 0
    for _, g in d.assign(q=q).groupby("battery", sort=False):
        pred = (g["q"] / g["q"].iloc[0] < 0.80).astype(int)
        ok += int((pred == g["degraded"]).sum())
    print(f"capacity proxy reproduces the label for {ok} of {len(d)} cycles "
          f"({100 * ok / len(d):.1f}%)")

    h = pd.read_csv("data/nasa_battery_hard.csv")
    for feat in ["time_to_v3p5", "time_to_tmax"]:
        accs = []
        for b in h["battery"].unique():
            tr, te = h[h["battery"] != b], h[h["battery"] == b]
            clf = DecisionTreeClassifier(max_depth=1, random_state=0).fit(tr[[feat]], tr["degraded"])
            accs.append(100 * (clf.predict(te[[feat]]) == te["degraded"]).mean())
        print(f"one threshold on {feat}: folds {np.round(accs, 1).tolist()}, "
              f"mean {np.mean(accs):.1f}% (leave-one-battery-out)")

    cors = [np.corrcoef(g["discharge_time"], g["time_to_v3p5"])[0, 1]
            for _, g in d.groupby("battery", sort=False)]
    print(f"within-cell corr(discharge_time, time_to_v3p5): {np.round(cors, 3).tolist()}")

    accs, f1s, thr = [], [], []
    for b in d["battery"].unique():                 # full data, clean labels
        tr, te = d[d["battery"] != b], d[d["battery"] == b]
        clf = DecisionTreeClassifier(max_depth=1, random_state=0).fit(tr[["time_to_v3p5"]], tr["degraded"])
        pred = clf.predict(te[["time_to_v3p5"]])
        accs.append(100 * (pred == te["degraded"]).mean())
        f1s.append(100 * f1_score(te["degraded"], pred))
        side = "below" if clf.tree_.value[1].argmax() == 1 else "above"   # left child = <= threshold
        thr.append(f"{b}: degraded if time_to_v3p5 {'<=' if side == 'below' else '>'} {clf.tree_.threshold[0]:.1f} s")
    print(f"Figure 1 row, single-feature threshold (time_to_v3p5), clean labels, full data: "
          f"ACC folds {np.round(accs, 2).tolist()}, mean {np.mean(accs):.2f} +- {np.std(accs, ddof=1):.2f}; "
          f"F1 mean {np.mean(f1s):.2f} +- {np.std(f1s, ddof=1):.2f}")
    print("learned thresholds (held-out cell):", "; ".join(thr))

    X, y, groups = load_dataset("data/nasa_battery.csv", "degraded", (), "infer", group_col="battery")
    j = [c for c in d.columns if c not in ("battery", "degraded")].index("time_to_v3p5")
    idx = np.arange(len(X))
    for noise in [0.0, 0.1, 0.2]:
        accs = []
        for run, g in enumerate(np.unique(groups)):         # same split order as run_experiments
            tr, te = idx[groups != g], idx[groups == g]
            ytr = y[tr].copy()
            if noise > 0:
                ytr = flip_labels(ytr, noise, np.random.RandomState(1000 + run))
            clf = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X[tr][:, [j]], ytr)
            accs.append(100 * (clf.predict(X[te][:, [j]]) == y[te]).mean())
        print(f"rule with {int(noise * 100)}% training-label noise: folds {np.round(accs, 2).tolist()}, "
              f"mean {np.mean(accs):.2f} +- {np.std(accs, ddof=1):.2f}")

    cur = d.assign(a=d["i_mean"].abs()).groupby("battery", sort=False)["a"].agg(["mean", "std", "min", "max"])
    print("mean |I| per cycle (A), by cell:\n" + cur.round(3).to_string())


if __name__ == "__main__":
    main()
