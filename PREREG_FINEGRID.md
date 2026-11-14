# PREREG_FINEGRID.md — Phase F: finer label-noise granularity (2%, 4%, 6%, 8%)

Written 2026-11-14, before any model was run under these conditions. Nothing in this document was
tuned or chosen after seeing results of the new conditions. Status when written: **no fine-grid run
exists**; no fine-grid label flips have been generated or inspected.

## 0. What this is and is not

* It is **additional noise-level resolution** on the same four dev batteries (B0005, B0006, B0007,
  B0018), under the exact Phase B protocol, so the paper can show a 0/2/4/6/8/10% curve instead of
  only 0/10/20%. It is **not** a new stress test (that is Phase D-lite, `PREREG_DLITE.md`, already
  frozen and run) and **not** external validation (closed, REPORT.md "Phase D, step 1" and "XJTU
  validation attempt").
* 0% and 10% are **not rerun**: `results/phys_phase_b_noise0_*` and `results/phys_phase_b_noise10_*`
  (Phase B) already exist and stand as written. Only four new noise values are run: 0.02, 0.04, 0.06,
  0.08. The paper-facing 0/2/4/6/8/10% picture is assembled by placing these alongside the existing
  Phase B 0%/10% rows, not by regenerating them.
* Same non-independence disclosure as Phase B and D-lite: B0005/B0006/B0007 share one recording, and
  all six noise levels (old and new) reuse the same four batteries and features, so pooling
  observations across noise levels is descriptive, not independent-sample inference.
* Granularity capped at 10% by design (Vansh's instruction): no new value above 0.10 is run here.

## 1. Frozen (identical to Phase B / Phase D-lite; nothing retuned)

| Item | Frozen value |
|---|---|
| Data | `data/nasa_battery_phys.csv` (636 rows: B0005 168, B0006 168, B0007 168, B0018 132), 12 physics features: t_v38, t_v35, dt_38_35, v_300, v_1000, temp_rise_35 and their `_rel` versions; label `degraded` (SOH < 0.80); group = `battery` (not a feature) |
| Split | leave-one-battery-out, the same 4 folds (fold k = k-th battery in sorted order: 0 B0005, 1 B0006, 2 B0007, 3 B0018) |
| Models (15, `run_experiments.build_models(seed=fold, grids="equal")`) | EF-GBTSVM, EF-GBTSVM-W, EF-GBTSVM-L, EF-GBTSVM-A, EF-GBTSVM-L8, EF-GBTSVM-J, GBTSVM, HF-GBTSVM, EF-TSVM, TSVM, SVM (RBF), RVFL, RandomForest, MLP, XGBoost |
| Derived models | **EF-GBTSVM-PF** = per fold the variant picked by Phase A (`results/phase_a/phys2_selection.csv`): B0005 -> L8, B0006 -> J, B0007 -> L8, B0018 -> L8. **Not re-selected for the new noise values** (same rule as D-lite). **Stump-fair** = depth-1 tree, feature chosen by inner CV on the (noisy) training fold, `phys_stump.py` fair-mode logic (as used in `run_dlite.stump_rows`). Stump-oracle is computed only as a labelled upper bound and is never ranked |
| Grids / budget | the grids in `run_experiments.py` (EQUAL_GRIDS for SVM/RF/MLP/XGBoost; model grids otherwise); `--n-iter 150` (a model with fewer grid points searches its full grid) |
| Inner CV | scaler -> classifier pipeline; `RandomizedSearchCV`, `GroupKFold(3)` over the 3 training batteries, scoring balanced accuracy, `random_state = fold`, estimator `random_state = fold`; refit on the whole noisy training fold |
| Label flip | `flip_labels(ytr, rate, np.random.RandomState(1000 + fold))` from `run_experiments.py` — the Phase B / D-lite seed rule, applied to training labels only; test labels are never touched |
| Metrics | ACC, BACC, F1, AUC (`run_experiments.metrics`, test battery only); train-test gap exactly as `phase_b_train_gap.py` (train ACC on the **true** training labels minus test ACC, refit with the logged best params) |
| Command shape | the Phase B / D-lite command: `--n-iter 150 --grids equal --n-jobs 4 --group-col battery` |
| Code hashes (sha256, frozen, identical to `PREREG_DLITE.md` section 1 and verified unchanged on 2026-11-14) | `run_experiments.py` a1c752165a85da7681db13b84310bd3fb1c873d20e8994a92b56769826e238d9; `phys_stump.py` 3fb0d0c73a90210f60f642e272e97d917d56f1e1ab6e099202e89012ff66b4b0; `efgbtsvm/model.py` ebe73f7eef24d587bf2244769022d963f6d22e20d04016d977c25073a0498115; `granular_ball.py` ed8242fe5ad62cc9b7dae9e06ccfb4f51a243383a714f178e6cc64a729c549ec; `noise.py` bffee064ad9fb39913f9450ee4942034b41114965140ed04b43fd7c8f436f4f0; `rvfl_features.py` f1e53b73b4575cb8cdd63cf3f24ad03668f86db452f46d0d85c62d9262660f68; `twsvm.py` 52805f30617dc5a3df8bdbb0e1acdcd95f00ac46421b74f50b7f4c51671fd614; `efgbtsvm/__init__.py` c1d98b7ccc06c0775b0ac2e0761c74f4fde19b56ede6934f4c87dedc51ec6922; `data/nasa_battery_phys.csv` 7bceb4b0aaa4cc267f8d158c99d910bb9b78ca825f06a29f5b69fd3be1310274; `results/phase_a/phys2_selection.csv` 0997a1c69e473ece8aa7b8356aa1e082020409de91921043ef7df66c9918c1e6; `phase_b_analysis.py` d8403a508ee3fbc59d34fc9ba03324181f8aa5c0cddd0210c584da220b531ffc |
| Environment | Python 3.13.9, scikit-learn 1.8.0, numpy 2.2.6, scipy 1.15.3, pandas 2.3.3, xgboost 3.1.3 |

One new script, `run_finegrid.py`, is the only new code: it imports `build_models`, `load_dataset`,
`metrics`, `score_of`, `flip_labels` from `run_experiments.py` unchanged, and the Stump-fair/oracle
routine of `run_dlite.stump_rows` unchanged (imported, not copied, so a future fix to that function
is not silently duplicated). It adds only the outer loop over the four new noise rates and per-fold
checkpointing. Its sha256 is recorded in REPORT.md **before** the run. It may not change any model,
grid, seed, scorer or metric above. If a model/grid file listed above differs from its hash at run
time, the run does not start (same `check_frozen()` mechanism as `run_dlite.py`).

## 2. New noise conditions

For fold k the training set = the 3 batteries other than the test battery. The flip is applied to the
training labels once per (noise value, fold), before any inner CV; the inner CV and the final fit both
see the flipped labels, exactly as in Phase B / D-lite. The same flipped vector is given to every
model (including PF's components and Stump-fair).

| Condition label | Noise rate |
|---|---|
| N02 | 0.02 |
| N04 | 0.04 |
| N06 | 0.06 |
| N08 | 0.08 |

`flip_labels(ytr, rate, np.random.RandomState(1000 + k))`, `int(rate * n_train)` cycles flipped
without replacement — identical mechanism to Phase B's 10%/20% and D-lite's N30/N40.

Not rerun: 0%, 10%, 20% (Phase B), N30/N40/BND (D-lite). Not run: anything above 10%.

## 3. Outputs and files

Per (condition, fold, model): ACC, BACC, F1, AUC and the train-test gap, as section 1 states. Every
condition is reported, whatever the result; no condition, model or fold is dropped after seeing
results. Checkpointed **from fold 0 of the first condition**, not only after a crash (the D-lite
lesson in REPORT.md): `results/finegrid_ckpt/{condition}_fold{run}.csv`, written as soon as that
fold's 15 models finish, and reloaded (not rerun) on any restart.

Files (new; no existing file is overwritten):
* `results/finegrid_N0{2,4,6,8}_runs.csv` — per-condition raw rows (parallel to `results/dlite_N30_runs.csv` etc.)
* `results/finegrid_runs.csv` — the four conditions concatenated, same row shape as the per-condition files
* `results/finegrid_summary_table.csv` — same column structure as `results/phase_b_summary_table.csv`
  (`noise_pct, model, ACC_mean, ACC_std, BACC_mean, BACC_std, F1_mean, F1_std, AUC_mean, AUC_std,
  in_main_pool, rank_ACC_main15, rank_ACC_all17, rank_BACC_main15, rank_BACC_all17`), built by the
  same `add_per_fold` (PF) + Stump-fair-concat + per-noise-level rank logic as `phase_b_analysis.py`,
  applied to these four conditions only (ranks are computed within each noise level, so this is
  well-defined without needing the other noise levels in the same table)
* `results/finegrid_train_gap.csv` — same column structure as `results/phase_b_train_gap.csv`

## 4. Pools and ranks

Main pool (k = 15) = the 13 non-variant models (the 15 above minus L8 and J) + EF-GBTSVM-PF +
Stump-fair. Sensitivity pool (k = 17) adds EF-GBTSVM-L8 and EF-GBTSVM-J. Rank = rank of the mean over
the 4 folds at that noise level, descending, metrics rounded to 6 decimals — identical definition to
Phase B and D-lite.

## 5. Claim rule (fixed now; this phase makes no new significance claim)

This phase's purpose is **descriptive granularity**: showing EF-GBTSVM-PF's and every other model's
accuracy at 2%, 4%, 6% and 8% noise alongside the existing 0%/10%/20% points, nothing more. Before
running:

* **No "best" or "robust" claim is made from this data alone.** The existing Phase B claim rule
  (best = strictly rank 1 in mean ACC and BACC, main pool) may be reported per new noise level purely
  descriptively ("PF is/is not best at 4% noise"), the same wording Table 4 already uses for 0/10/20%.
  No claim is upgraded by combining noise levels (e.g. "PF is best across 0-10% noise") unless it is
  separately true at every one of the six points — if that is not the case, only the per-point
  statements are made.
* **No new statistical test.** If Friedman/Nemenyi/Wilcoxon are later run on the combined six-point
  data, they use the identical functions already defined in `phase_b_analysis.py` — no new test family
  is introduced for this phase. That decision (whether/how to pool six noise levels statistically) is
  explicitly deferred to Vansh; this document does not pre-commit to it.
* **Nothing is inserted into the paper or its tables by this phase.** Running it only produces data
  and a report; the paper's text, tables and figures are touched in a later, separate step after
  Vansh reviews the raw numbers.

## 6. Execution rules

* Run **once**, in the order N02, N04, N06, N08, with the frozen command shape above. No reruns for
  performance reasons, no change of seeds, grids, models or conditions after results exist.
* If a bug is found: fix it, then rerun **all** models, **all** four new conditions, equally; keep the
  first-run outputs if any exist, and write the deviation (what, why, which numbers changed) in
  REPORT.md — the same rule as D-lite section 7.
* A crash, sleep, or reboot is reported as it happened; fold-level checkpoints mean only the
  in-progress fold is lost, not the whole condition. No partial result is substituted for a result
  that was not actually computed.
* Run in the background (`run_in_background`), monitored periodically; not expected to finish quickly
  (D-lite's three conditions took multiple real-world sessions with interruptions — four more
  conditions at the same per-model budget is comparable or greater compute).

## 7. Freeze

The sha256 of this file is recorded in REPORT.md immediately after this is written, before
`run_finegrid.py` exists. The run does not start until the user says yes.
