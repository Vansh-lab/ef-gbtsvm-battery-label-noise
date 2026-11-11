# PREREG_DLITE.md — Phase D-lite: three new label-noise conditions on the dev batteries

Written 2026-11-05, before any model was run under these conditions. Nothing in this document was
tuned or chosen after seeing results of the new conditions. Status when written: **no D-lite run
exists**; no D-lite label flips have been generated or inspected.

## 0. What this is and is not

* It is a **robustness / stress test** on the same four dev batteries (B0005, B0006, B0007, B0018) that
  were already used in Phase A, Phase B and Phase C. It is **not external validation**: the fresh-cell
  route was closed (REPORT.md "Phase D, step 1": 0 eligible cells).
* The four batteries are not independent: B0005/B0006/B0007 share one recording (identical time
  vectors). The three conditions below reuse the same batteries and features, so observations pooled
  across conditions are not independent either (see section 5).
* Disclosure carried over from Phase B: the Phase A candidate list (L8 purity grid, J adaptive offset)
  was shaped by earlier test-fold results on these same batteries (old feature file).

## 1. Frozen (nothing new, no retuning)

| Item | Frozen value |
|---|---|
| Data | `data/nasa_battery_phys.csv` (636 rows: B0005 168, B0006 168, B0007 168, B0018 132), 12 physics features: t_v38, t_v35, dt_38_35, v_300, v_1000, temp_rise_35 and their `_rel` versions; label `degraded` (SOH < 0.80); group = `battery` (not a feature) |
| Split | leave-one-battery-out, the same 4 folds (fold k = k-th battery in sorted order: 0 B0005, 1 B0006, 2 B0007, 3 B0018) |
| Models (15, `run_experiments.build_models(seed=fold, grids="equal")`) | EF-GBTSVM, EF-GBTSVM-W, EF-GBTSVM-L, EF-GBTSVM-A, EF-GBTSVM-L8, EF-GBTSVM-J, GBTSVM, HF-GBTSVM, EF-TSVM, TSVM, SVM (RBF), RVFL, RandomForest, MLP, XGBoost |
| Derived models | **EF-GBTSVM-PF** = per fold the variant picked by Phase A (`results/phase_a/phys2_selection.csv`): B0005 -> L8, B0006 -> J, B0007 -> L8, B0018 -> L8. **Not re-selected for the new conditions.** **Stump-fair** = depth-1 tree, feature chosen by inner CV on the (noisy) training fold, as `phys_stump.py` fair mode. Stump-oracle is computed only as a labelled upper bound and is never ranked |
| Grids / budget | the grids in `run_experiments.py` (EQUAL_GRIDS for SVM/RF/MLP/XGBoost; model grids otherwise); `--n-iter 150` (a model with fewer grid points searches its full grid) |
| Inner CV | scaler -> classifier pipeline; `RandomizedSearchCV`, `GroupKFold(3)` over the 3 training batteries, scoring balanced accuracy, `random_state = fold`, estimator `random_state = fold`; refit on the whole noisy training fold |
| Command shape | the Phase B command: `--n-iter 150 --grids equal --n-jobs 4 --group-col battery` |
| Code hashes (sha256, frozen) | `run_experiments.py` a1c752165a85da7681db13b84310bd3fb1c873d20e8994a92b56769826e238d9; `phys_stump.py` 3fb0d0c73a90210f60f642e272e97d917d56f1e1ab6e099202e89012ff66b4b0; `efgbtsvm/model.py` ebe73f7eef24d587bf2244769022d963f6d22e20d04016d977c25073a0498115; `granular_ball.py` ed8242fe5ad62cc9b7dae9e06ccfb4f51a243383a714f178e6cc64a729c549ec; `noise.py` bffee064ad9fb39913f9450ee4942034b41114965140ed04b43fd7c8f436f4f0; `rvfl_features.py` f1e53b73b4575cb8cdd63cf3f24ad03668f86db452f46d0d85c62d9262660f68; `twsvm.py` 52805f30617dc5a3df8bdbb0e1acdcd95f00ac46421b74f50b7f4c51671fd614; `efgbtsvm/__init__.py` c1d98b7ccc06c0775b0ac2e0761c74f4fde19b56ede6934f4c87dedc51ec6922; `data/nasa_battery_phys.csv` 7bceb4b0aaa4cc267f8d158c99d910bb9b78ca825f06a29f5b69fd3be1310274; `results/phase_a/phys2_selection.csv` 0997a1c69e473ece8aa7b8356aa1e082020409de91921043ef7df66c9918c1e6; `phase_b_analysis.py` d8403a508ee3fbc59d34fc9ba03324181f8aa5c0cddd0210c584da220b531ffc |
| Environment | Python 3.13.9, scikit-learn 1.8.0, numpy 2.2.6, scipy 1.15.3, pandas 2.3.3, xgboost 3.1.3 |

Two thin new scripts are needed and are the only new code: `run_dlite.py` (runs the three conditions
by importing `build_models`, `load_dataset`, `metrics`, `score_of` from `run_experiments.py` unchanged,
and the Stump-fair routine from `phys_stump.py`; adds only the boundary-noise function of section 2b)
and `dlite_analysis.py` (the Phase B analysis functions copied unchanged, applied to the new
conditions). Both are written after approval and their sha256 is recorded in REPORT.md **before** the
run. They may not change any model, grid, seed, scorer or metric above. If a model/grid file listed
above differs from its hash at run time, the run does not start.

## 2. New conditions (training labels only; test labels are never read for flipping and never changed)

For fold k the training set = the 3 batteries other than the test battery. Flips are applied to the
training labels once per (condition, fold), before any inner CV; the inner CV and the final fit both
see the flipped labels, exactly as in Phase B. The same flipped vector is given to every model
(including PF components and Stump-fair).

* **(a) Random label noise 30% and 40%.** `flip_labels(ytr, rate, np.random.RandomState(1000 + k))`
  from `run_experiments.py`, rate 0.30 and 0.40 (`int(rate * n_train)` cycles, no replacement) — the
  Phase B seed rule. Conditions `N30`, `N40`.
* **(b) Boundary noise, condition `BND`.**
  1. Per-cycle SOH = Capacity / (first discharge-cycle Capacity of the same battery), the quantity that
     defines the label. It is recomputed from `data/raw/B000{5,6,7}.mat`, `B0018.mat` with
     `prep_phys_features.load_battery`; before use the script must assert that the 636 rows (battery,
     cycle order) match `nasa_battery_phys.csv` and that `SOH < 0.80` reproduces `degraded` on every
     row. Mismatch -> stop and report, do not guess.
  2. Boundary set B_k = training-fold cycles with `0.75 <= SOH <= 0.85` (closed band, 1e-12
     tolerance), indexed in training-array order.
  3. Flip `int(0.5 * |B_k|)` of them, drawn without replacement with
     `np.random.RandomState(1000 + k).choice(B_k_indices, int(0.5 * |B_k|), replace=False)`; all other
     training labels stay clean. SOH is used only to choose which training labels to flip; it is not a
     feature and never reaches a model.
  4. |B_k| and the number flipped are logged per fold. The overall flip fraction is therefore small
     and depends on fold; it is reported, not tuned. If |B_k| = 0 or flipping leaves one class in the
     training labels, that fold is reported as skipped for every model (as `run_experiments.py` does
     for single-class folds) and PF/claims are then marked "incomplete (n folds)".

Not rerun: 0%, 10% and 20% random noise (Phase B results stand and are not re-analysed together with
these).

## 3. Outputs and metrics (all models, all conditions)

Per (condition, fold, model): ACC, BACC, F1, AUC (as `metrics()` in `run_experiments.py`, test
battery only) and the train-test gap exactly as `phase_b_train_gap.py` (train ACC on the **true**
training labels minus test ACC, refit with the logged best params; reproduction of the logged test ACC
is checked and any non-reproduced row is reported). Every condition is reported, whatever the result;
no condition, model or fold is dropped after seeing results.

## 4. Pools and ranks

Main pool (k = 15) = the 13 non-variant models (the 15 above minus L8 and J) + EF-GBTSVM-PF +
Stump-fair. Sensitivity pool (k = 17) adds EF-GBTSVM-L8 and EF-GBTSVM-J. Rank = rank of the mean over
the 4 folds, descending, metrics rounded to 6 decimals.

## 5. Statistics (same functions as `phase_b_analysis.py`)

* Friedman (tie-corrected) over models, blocks = folds: per condition N = 4; pooled over the three
  conditions N = 12. Nemenyi critical difference with q(0.05, k) from the studentized range; Nemenyi is
  descriptive only because it follows Friedman.
* Wilcoxon signed-rank, two-sided, PF vs each of the other 14 models (main pool), Holm correction over
  14: per condition (n = 4 folds; the minimum attainable two-sided p is 0.125, so **no per-condition
  test can be significant**) and pooled (n = 12).
* **The observations are not independent**: all three conditions use the same 4 batteries and
  features, and B0005/B0006/B0007 share one recording. Pooled p-values are descriptive only. The design
  is underpowered for differences of a few percentage points; absence of significance is not
  equivalence.

## 6. Claim table (fixed now)

Rank 1 means the strictly highest mean over the 4 folds at 6 decimals, in the main pool (k = 15). A tie
for the top does **not** count as rank 1 (reported as "tied for first").

| Result on a condition (N30, N40, BND each evaluated separately) | Wording that will be used |
|---|---|
| PF is rank 1 in mean ACC **and** rank 1 in mean BACC | "PF has the highest mean ACC and BACC on <condition> (main pool, 4 folds); the difference is not statistically testable at this N" — **"best" is permitted only for this case**, and always with that qualifier |
| Anything else (rank 1 in only one of the two, not rank 1, tie, or incomplete folds) | "PF is not best on <condition>"; its rank, its mean, and the gap to the top model are reported |

No rule uses the Friedman or Wilcoxon outcome to upgrade a claim. Because per-condition tests cannot
reach significance, a "best" claim never means statistically better. A claim over all three conditions
("PF best across conditions") is allowed only if the first row holds on all three; otherwise only the
per-condition statements are made. Sensitivity-pool (k = 17) ranks are reported next to the main
pool; if the main-pool claim does not hold in the sensitivity pool the paper says so.

## 7. Execution rules

* Run **once**, in the order N30, N40, BND, with the frozen command shape above. No reruns for
  performance reasons, no change of seeds, grids, models or conditions after results exist.
* If a bug is found: fix it, then rerun **all** models, **all** conditions, equally; keep the
  first-run outputs, and write the deviation (what, why, which numbers changed) in REPORT.md.
* A crash or timeout is reported as it happened; no partial result is substituted.
* Results go to new files `results/dlite_*` (no existing file is overwritten).

## 8. Freeze

The sha256 of this file is recorded in REPORT.md. The directory is not a git repository (checked
2026-11-05), so no commit is made unless the user asks for `git init`. The run does not start until the
user says yes.
