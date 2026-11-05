"""Train-vs-test gap for Phase B (overfitting check), no re-tuning.

    python phase_b_train_gap.py

For every (noise, fold, model) row in results/phys_phase_b_noise{0,10,20}_runs.csv, rebuilds
the same estimator (build_models(seed=fold, grids='equal')), sets the LOGGED best params,
refits on the same training fold with the same label flips, and records:
  train_acc_noisy  accuracy on the training labels the model was fitted to (flipped)
  train_acc_clean  accuracy on the true training labels
  test_acc         recomputed test accuracy (must match the logged ACC -> reproducibility)
Writes results/phase_b_train_gap.csv.
"""
import ast
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_experiments import build_models, flip_labels, load_dataset

warnings.filterwarnings("ignore")
NOISES = [0, 10, 20]


def main():
    X, y, groups = load_dataset("data/nasa_battery_phys.csv", "degraded", group_col="battery")
    idx = np.arange(len(X))
    rows = []
    for n in NOISES:
        runs = pd.read_csv(f"results/phys_phase_b_noise{n}_runs.csv")
        for run, g in enumerate(np.unique(groups)):
            tr, te = idx[groups != g], idx[groups == g]
            ytr_clean = y[tr]
            ytr = flip_labels(ytr_clean, n / 100, np.random.RandomState(1000 + run)) if n else ytr_clean.copy()
            models = build_models(run, None, "equal")
            for _, r in runs[runs["run"] == run].iterrows():
                est, _ = models[r["model"]]
                params = {f"clf__{k}": v for k, v in ast.literal_eval(r["params"]).items()}
                pipe = Pipeline([("scale", StandardScaler()), ("clf", est)]).set_params(**params)
                pipe.fit(X[tr], ytr)
                rows.append(dict(
                    noise_pct=n, run=run, model=r["model"],
                    train_acc_noisy=100 * accuracy_score(ytr, pipe.predict(X[tr])),
                    train_acc_clean=100 * accuracy_score(ytr_clean, pipe.predict(X[tr])),
                    test_acc=100 * accuracy_score(y[te], pipe.predict(X[te])),
                    logged_test_acc=r["ACC"]))
        print(f"noise {n}% done")
    df = pd.DataFrame(rows)
    df["reproduced"] = np.isclose(df["test_acc"], df["logged_test_acc"], atol=1e-6)
    df["gap_clean"] = df["train_acc_clean"] - df["test_acc"]
    df.to_csv("results/phase_b_train_gap.csv", index=False)
    print(f"reproduced logged test ACC: {df['reproduced'].sum()}/{len(df)}")
    if not df["reproduced"].all():
        print(df[~df["reproduced"]][["noise_pct", "run", "model", "test_acc", "logged_test_acc"]].to_string())
    print(df.groupby(["noise_pct", "model"])[["train_acc_noisy", "train_acc_clean", "test_acc", "gap_clean"]]
          .mean().round(2).to_string())


if __name__ == "__main__":
    main()
