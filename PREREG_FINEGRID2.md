# PREREG_FINEGRID2.md — Phase F2: post-hoc fine-grid extension around the 6% point (6.2%-7.8%)

Written 2026-11-14, before any model was run under these conditions. Status when written: **no
finegrid2 run exists**; no finegrid2 label flips have been generated or inspected.

## 0. What this is, and why it is NOT a blind sweep (read this before anything else)

* **This round is explicitly post-hoc / exploratory, not a blind pre-registered sweep.** The nine
  noise values below (6.2%, 6.4%, 6.6%, 6.8%, 7.0%, 7.2%, 7.4%, 7.6%, 7.8%) were chosen *because*
  `results/finegrid_summary_table.csv` (Phase F, `PREREG_FINEGRID.md`) showed EF-GBTSVM-PF ranking
  1st of 15 (main pool) at the 6% point, and these new values sit immediately around and after that
  single observed noise level. Phase F's own six noise points (0/2/4/6/8/10%) were chosen *before*
  any of them were run -- that is what made Phase F a blind sweep. Phase F2 is the opposite: the
  choice of *where* to look was made *after seeing* a result, on the same four batteries and the
  same underlying data. This is a textbook garden-of-forking-paths / multiple-comparisons setup:
  zooming in on a neighbourhood because it looked good inflates the apparent chance of finding more
  "PF is best" points there, even if nothing about PF is actually better at that noise region. A
  rank-1 finding in this document carries **less evidentiary weight** than a rank-1 finding at a
  value chosen blind, and must never be described in the paper or in chat as if it were.
* It is **not** a new stress test (`PREREG_DLITE.md`) and **not** external validation (closed).
* 0%, 2%, 4%, 6%, 8%, 10%, 20% are **not rerun**: they already exist in Phase B and Phase F. Only
  the nine new values below are run.
* Same non-independence disclosure as Phase B / D-lite / Phase F: B0005/B0006/B0007 share one
  recording, and all noise levels (old and new) reuse the same four batteries and features, so
  pooling observations across noise levels is descriptive, not independent-sample inference. That
  caveat is now compounded by the post-hoc selection above: even the *descriptive* picture at these
  nine points is biased toward whatever made 6% look good (sampling noise in which battery/fold
  combinations happened to favour PF, or a real but narrow effect -- this data cannot tell the two
  apart).
* Granularity capped at 10% by design, same as Phase F; nothing above 10% is run.

## 1. Frozen (identical to Phase B / D-lite / Phase F; nothing retuned, re-verified unchanged)

| Item | Frozen value |
|---|---|
| Data | `data/nasa_battery_phys.csv` (636 rows: B0005 168, B0006 168, B0007 168, B0018 132), 12 physics features: t_v38, t_v35, dt_38_35, v_300, v_1000, temp_rise_35 and their `_rel` versions; label `degraded` (SOH < 0.80); group = `battery` (not a feature) |
| Split | leave-one-battery-out, the same 4 folds (fold k = k-th battery in sorted order: 0 B0005, 1 B0006, 2 B0007, 3 B0018) |
| Models (15, `run_experiments.build_models(seed=fold, grids="equal")`) | EF-GBTSVM, EF-GBTSVM-W, EF-GBTSVM-L, EF-GBTSVM-A, EF-GBTSVM-L8, EF-GBTSVM-J, GBTSVM, HF-GBTSVM, EF-TSVM, TSVM, SVM (RBF), RVFL, RandomForest, MLP, XGBoost |
| Derived models | **EF-GBTSVM-PF** = per fold the variant picked by Phase A (`results/phase_a/phys2_selection.csv`): B0005 -> L8, B0006 -> J, B0007 -> L8, B0018 -> L8. **Not re-selected for these new noise values** (same rule as D-lite / Phase F). **Stump-fair** = depth-1 tree, feature chosen by inner CV on the (noisy) training fold, via `phys_stump.stump_runs` (the same call `phase_b_analysis.py` and `run_finegrid.py` make). Stump-oracle is computed only as a labelled upper bound and is never ranked |
| Grids / budget | the grids in `run_experiments.py` (EQUAL_GRIDS for SVM/RF/MLP/XGBoost; model grids otherwise); `--n-iter 150` (a model with fewer grid points searches its full grid) |
| Inner CV | scaler -> classifier pipeline; `RandomizedSearchCV`, `GroupKFold(3)` over the 3 training batteries, scoring balanced accuracy, `random_state = fold`, estimator `random_state = fold`; refit on the whole noisy training fold |
| Label flip | `flip_labels(ytr, rate, np.random.RandomState(1000 + fold))` from `run_experiments.py` — the Phase B / D-lite / Phase F seed rule, applied to training labels only; test labels are never touched |
| Metrics | ACC, BACC, F1, AUC (`run_experiments.metrics`, test battery only); train-test gap exactly as `phase_b_train_gap.py` / `run_finegrid.py` (train ACC on the **true** training labels minus test ACC, from the same fit, no separate refit) |
| Command shape | the Phase B / D-lite / Phase F command: `--n-iter 150 --grids equal --n-jobs 4 --group-col battery` |
| Code hashes (sha256, frozen, identical to `PREREG_FINEGRID.md` section 1, re-verified unchanged on 2026-11-14 before writing this document) | `run_experiments.py` a1c752165a85da7681db13b84310bd3fb1c873d20e8994a92b56769826e238d9; `phys_stump.py` 3fb0d0c73a90210f60f642e272e97d917d56f1e1ab6e099202e89012ff66b4b0; `efgbtsvm/model.py` ebe73f7eef24d587bf2244769022d963f6d22e20d04016d977c25073a0498115; `granular_ball.py` ed8242fe5ad62cc9b7dae9e06ccfb4f51a243383a714f178e6cc64a729c549ec; `noise.py` bffee064ad9fb39913f9450ee4942034b41114965140ed04b43fd7c8f436f4f0; `rvfl_features.py` f1e53b73b4575cb8cdd63cf3f24ad03668f86db452f46d0d85c62d9262660f68; `twsvm.py` 52805f30617dc5a3df8bdbb0e1acdcd95f00ac46421b74f50b7f4c51671fd614; `efgbtsvm/__init__.py` c1d98b7ccc06c0775b0ac2e0761c74f4fde19b56ede6934f4c87dedc51ec6922; `data/nasa_battery_phys.csv` 7bceb4b0aaa4cc267f8d158c99d910bb9b78ca825f06a29f5b69fd3be1310274; `results/phase_a/phys2_selection.csv` 0997a1c69e473ece8aa7b8356aa1e082020409de91921043ef7df66c9918c1e6; `phase_b_analysis.py` d8403a508ee3fbc59d34fc9ba03324181f8aa5c0cddd0210c584da220b531ffc |
| Environment | Python 3.13.9, scikit-learn 1.8.0, numpy 2.2.6, scipy 1.15.3, pandas 2.3.3, xgboost 3.1.3 |

One new script, `run_finegrid2.py`, is the only new execution code: it is `run_finegrid.py` with
only the `CONDITIONS`/`RATES` constants changed to the nine values below -- `build_models`,
`load_dataset`, `metrics`, `score_of`, `flip_labels` still come unchanged from `run_experiments.py`,
and Stump-fair/oracle still go through the unchanged `phys_stump.stump_runs`. Its sha256 is recorded
in REPORT.md **before** the run. It may not change any model, grid, seed, scorer or metric above. If
a model/grid file listed above differs from its hash at run time, the run does not start (same
`check_frozen()` mechanism as `run_finegrid.py` / `run_dlite.py`).

## 2. New noise conditions

For fold k the training set = the 3 batteries other than the test battery. The flip is applied to
the training labels once per (noise value, fold), before any inner CV; the inner CV and the final
fit both see the flipped labels, exactly as in Phase B / D-lite / Phase F. The same flipped vector is
given to every model (including PF's components and Stump-fair).

| Condition label | Noise rate |
|---|---|
| N062 | 0.062 |
| N064 | 0.064 |
| N066 | 0.066 |
| N068 | 0.068 |
| N070 | 0.070 |
| N072 | 0.072 |
| N074 | 0.074 |
| N076 | 0.076 |
| N078 | 0.078 |

`flip_labels(ytr, rate, np.random.RandomState(1000 + k))`, `int(rate * n_train)` cycles flipped
without replacement — identical mechanism to every earlier phase.

Not rerun: 0%, 2%, 4%, 6%, 8%, 10% (Phase B / Phase F), N30/N40/BND (D-lite), 20% (Phase B). Not run:
anything above 10%.

## 3. Outputs and files

Per (condition, fold, model): ACC, BACC, F1, AUC and the train-test gap, as section 1 states. Every
condition is reported, whatever the result; no condition, model or fold is dropped after seeing
results -- this is the single most important rule of this document, given section 0's disclosure:
having gone looking near an already-good point, every one of the nine new points must be reported,
not only the ones that confirm what 6% showed. Checkpointed **from fold 0 of the first condition**
(the D-lite / Phase F lesson), correctly backgrounded this time (`run_finegrid.py`'s own tracking
mistake and fix, logged in REPORT.md, is not repeated: no `nohup`/`disown` inside an already
backgrounded call): `results/finegrid2_ckpt/{condition}_fold{run}.csv`, written as soon as that
fold's 15 models finish, and reloaded (not rerun) on any restart.

Files (new; no existing file is overwritten):
* `results/finegrid2_N0{62,64,66,68,70,72,74,76,78}_runs.csv` — per-condition raw rows
* `results/finegrid2_runs.csv` — the nine conditions concatenated, same row shape as the per-condition files
* `results/finegrid2_summary_table.csv` — same column structure as `results/phase_b_summary_table.csv`
  / `results/finegrid_summary_table.csv` (`noise_pct, model, ACC_mean, ACC_std, BACC_mean, BACC_std,
  F1_mean, F1_std, AUC_mean, AUC_std, in_main_pool, rank_ACC_main15, rank_ACC_all17,
  rank_BACC_main15, rank_BACC_all17`); `noise_pct` for these nine conditions is stored as tenths of a
  percent (62, 64, ..., 78) since the existing integer-percent column cannot represent fractional
  points — documented in the script and in the report back to Vansh, not a silent unit change
* `results/finegrid2_train_gap.csv` — same column structure and row scope as
  `results/finegrid_train_gap.csv` (15 base models only, no refit; extracted from the run's own
  logged train_acc_noisy/train_acc_clean/ACC, same as Phase F)

## 4. Pools and ranks

Main pool (k = 15) = the 13 non-variant models (the 15 above minus L8 and J) + EF-GBTSVM-PF +
Stump-fair. Sensitivity pool (k = 17) adds EF-GBTSVM-L8 and EF-GBTSVM-J. Rank = rank of the mean over
the 4 folds at that noise level, descending, metrics rounded to 6 decimals — identical definition to
every earlier phase. Stump-oracle is never ranked.

## 5. Claim rule (fixed now; this phase makes no new significance claim, and less than Phase F did)

This phase's purpose is **descriptive, exploratory follow-up only**, with the post-hoc caveat of
section 0 attached to everything it produces:

* **No "best" or "robust" claim is made from this data alone**, and in particular **no claim that
  EF-GBTSVM-PF is reliable "around 6-8% noise" may be made from this round even if it ranks 1st at
  several of the nine points** — that would be exactly the forking-paths error section 0 warns
  about: the neighbourhood was picked because one point in it already looked good, so more good
  points nearby are expected from chance alone, not evidence of a real effect. The rank at EACH of
  the nine points is reported purely descriptively ("PF is/is not rank 1 at 7.2% noise"), same
  wording as Phase F.
* **No claim is upgraded by combining noise levels** (within this batch, or with Phase B/Phase F's
  six levels) unless it is separately true at every single point combined — and even then, the
  post-hoc selection of this neighbourhood must be disclosed alongside any such statement.
* **No new statistical test.** Friedman/Nemenyi/Wilcoxon, if ever run on this data, use the identical
  functions already in `phase_b_analysis.py` — no new test family. Whether/how to combine this data
  with earlier phases statistically is explicitly deferred to Vansh.
* **Nothing is inserted into the paper or its tables by this phase.** Running it only produces data
  and a report.
* **The full rank table for all nine points is reported, unfiltered** — including every point where
  EF-GBTSVM-PF is not rank 1, and whatever fraction of the nine points that turns out to be. Stopping
  early, or reporting only the confirming points, is exactly the failure mode this document exists
  to rule out.

## 6. Execution rules

* Run **once**, in ascending noise order (N062 -> N078), with the frozen command shape above. No
  reruns for performance reasons, no change of seeds, grids, models or conditions after results
  exist.
* If a bug is found: fix it, then rerun **all** models, **all** nine new conditions, equally; keep
  the first-run outputs if any exist, and write the deviation (what, why, which numbers changed) in
  REPORT.md — the same rule as D-lite/Phase F section 6/7.
* A crash, sleep, or reboot is reported as it happened; fold-level checkpoints mean only the
  in-progress fold is lost, not the whole condition.
* Run in the background as a single tracked command (`python run_finegrid2.py ... | tee
  results/finegrid2.log`, launched directly under the harness's own background tracking) — not
  `nohup ... & disown`, which detached the Phase F launcher from tracking and produced a false
  "completed" notification before any real work had happened (REPORT.md, "Tool-tracking
  double-backgrounding bug"). No polling once launched; wait for the real completion notification.

## 7. Freeze

The sha256 of this file is recorded in REPORT.md immediately after this is written, before
`run_finegrid2.py` exists. The run does not start until the user says yes.
