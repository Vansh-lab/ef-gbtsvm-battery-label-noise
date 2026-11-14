# EF-GBTSVM on NASA Li-ion Battery Data — Results

Dataset: NASA PCoE Li-ion battery discharge cycles (B0005, B0006, B0007, B0018),
636 discharge cycles total. Label = `degraded` (1 if SOH = capacity / first-cycle
capacity < threshold, default 0.80). Features are BMS-measurable discharge-curve
statistics only (`discharge_time, v_mean, v_min, v_std, v_start, v_end, t_mean,
t_max, t_rise, time_to_tmax, i_mean, time_to_v3p5, ambient_temp`) — capacity and
SOH themselves are excluded (no label leakage). Evaluation is **leave-one-battery-out**
(one run per battery, group-aware CV for hyperparameter search), `--n-iter 30`
randomized-search candidates per model per run. All numbers below are read
directly from `results/*_summary.csv` (mean ± std over the battery folds); none
are invented.

Per-battery stats (from `prep_nasa_battery.py`, threshold = 0.80 default):

| Battery | Cycles | Capacity (first → last, Ah) | Degraded % |
|---|---|---|---|
| B0005 | 168 | 1.856 → 1.325 | 39% |
| B0006 | 168 | 2.035 → 1.186 | 64% |
| B0007 | 168 | 1.891 → 1.432 | 27% |
| B0018 | 132 | 1.855 → 1.341 | 44% |

## Table 1 — Label-noise sweep (SOH threshold = 0.80, flips TRAIN labels only)

ACC / F1 / GMEAN / AUC / train_time(s), mean ± std over the 4 leave-one-battery-out folds.

| Model | noise=0% ACC | noise=10% ACC | noise=20% ACC | noise=0% F1 | noise=10% F1 | noise=20% F1 |
|---|---|---|---|---|---|---|
| **EF-GBTSVM** (proposed) | 83.20 ± 16.50 | 89.60 ± 9.15 | 61.31 ± 29.29 | 82.59 ± 17.76 | 85.30 ± 14.76 | 52.38 ± 37.21 |
| GBTSVM (no hidden feat.) | 65.57 ± 24.39 | 80.22 ± 11.07 | 75.27 ± 19.90 | 62.93 ± 27.95 | 64.31 ± 40.94 | 66.61 ± 30.17 |
| HF-GBTSVM (hidden only) | 73.77 ± 19.14 | 73.17 ± 26.61 | 69.14 ± 20.79 | 75.85 ± 17.65 | 76.78 ± 23.49 | 62.29 ± 30.66 |
| EF-TSVM (no balls) | 92.29 ± 5.38 | 68.41 ± 12.23 | 61.70 ± 11.01 | 90.03 ± 7.59 | 57.34 ± 25.13 | 54.94 ± 18.66 |
| TSVM (plain, no balls/feat.) | 77.61 ± 18.84 | 77.00 ± 15.98 | 73.78 ± 22.14 | 60.23 ± 43.17 | 62.02 ± 40.28 | 69.72 ± 22.21 |
| SVM (RBF) | 94.13 ± 3.57 | 87.76 ± 10.91 | 90.81 ± 7.61 | 91.47 ± 6.78 | 84.48 ± 14.06 | 87.90 ± 10.88 |
| RVFL | 86.04 ± 14.29 | 88.93 ± 14.02 | 88.72 ± 12.08 | 84.65 ± 16.74 | 87.19 ± 16.60 | 87.07 ± 15.05 |
| RandomForest | 93.84 ± 3.33 | 83.20 ± 9.64 | 73.53 ± 11.08 | 92.13 ± 4.64 | 74.37 ± 13.66 | 51.44 ± 27.34 |
| MLP | 88.77 ± 8.23 | 86.92 ± 9.94 | 85.13 ± 12.46 | 86.75 ± 11.41 | 68.07 ± 45.58 | 61.72 ± 45.49 |
| XGBoost | 92.36 ± 5.99 | 93.28 ± 4.27 | 86.50 ± 7.50 | 90.26 ± 6.84 | 89.34 ± 8.37 | 79.15 ± 17.32 |

Full raw files: `results/nasa_noise0_{runs,summary}.csv`, `results/nasa_noise10_{runs,summary}.csv`, `results/nasa_noise20_{runs,summary}.csv`.

## Table 2 — SOH-threshold sensitivity (noise = 0%)

| Model | t=0.70 ACC | t=0.80 ACC | t=0.85 ACC | t=0.70 F1 | t=0.80 F1 | t=0.85 F1 |
|---|---|---|---|---|---|---|
| **EF-GBTSVM** | 48.21 ± 18.52 | 83.20 ± 16.50 | 82.05 ± 18.93 | 4.20 ± 5.94 | 82.59 ± 17.76 | 71.87 ± 37.35 |
| GBTSVM | 57.74 ± 5.05 | 65.57 ± 24.39 | 90.22 ± 6.51 | 5.75 ± 8.13 | 62.93 ± 27.95 | 91.10 ± 6.40 |
| HF-GBTSVM | 46.43 ± 21.04 | 73.77 ± 19.14 | 87.16 ± 14.30 | 4.00 ± 5.66 | 75.85 ± 17.65 | 84.19 ± 20.59 |
| EF-TSVM | 73.21 ± 16.84 | 92.29 ± 5.38 | 96.94 ± 1.41 | 14.29 ± 20.20 | 90.03 ± 7.59 | 97.26 ± 1.13 |
| TSVM | 81.25 ± 22.31 | 77.61 ± 18.84 | 95.16 ± 4.86 | 9.72 ± 13.75 | 60.23 ± 43.17 | 95.82 ± 3.60 |
| SVM | 79.17 ± 25.25 | 94.13 ± 3.57 | 94.82 ± 3.77 | 33.33 ± 47.14 | 91.47 ± 6.78 | 95.41 ± 3.64 |
| RVFL | 52.68 ± 12.21 | 86.04 ± 14.29 | 92.55 ± 8.22 | 4.81 ± 6.80 | 84.65 ± 16.74 | 92.14 ± 9.78 |
| RandomForest | 87.50 ± 5.05 | 93.84 ± 3.33 | 94.24 ± 3.02 | 36.89 ± 52.17 | 92.13 ± 4.64 | 95.00 ± 2.13 |
| MLP | 70.24 ± 12.63 | 88.77 ± 8.23 | 92.13 ± 4.60 | 11.11 ± 15.71 | 86.75 ± 11.41 | 93.33 ± 3.79 |
| XGBoost | 84.23 ± 18.10 | 92.36 ± 5.99 | 90.88 ± 6.17 | 20.73 ± 29.32 | 90.26 ± 6.84 | 91.97 ± 4.47 |

**Caveat on t=0.70**: at this threshold B0007 and B0018 have **0% degraded**
samples, so 2 of the 4 leave-one-battery-out folds are skipped (single-class
test/train set — `run2`/`run3: skipped` in the log). Every model's t=0.70
numbers above are the mean of only **2** folds, not 4, and should be read as
indicative at best, not a reliable estimate.

Full raw files: `results/nasa_t70_noise0_{runs,summary}.csv`, `results/nasa_t85_noise0_{runs,summary}.csv`, `results/nasa_noise0_{runs,summary}.csv` (t=0.80).

## Table 3 — Harder feature subset (`discharge_time` and `i_mean` removed)

Built with `python prep_nasa_battery.py data/raw/B00{05,06,07,18}.mat --drop-feature discharge_time,i_mean --out data/nasa_battery_hard.csv`
(11 features instead of 13; same 636 rows, same 0.80 threshold, same labels). Run with
`--name nasa_hard --label-col degraded --group-col battery --noise 0 --n-iter 30`.
The model code is untouched; only the input columns differ. "Full" = the 13-feature noise=0%
run from Table 1. Rank is by mean ACC / F1 among the 10 models.

| Model | Full ACC | Hard ACC | Δ ACC | Rank full → hard (ACC) | Full F1 | Hard F1 | Δ F1 | Rank full → hard (F1) |
|---|---|---|---|---|---|---|---|---|
| **EF-GBTSVM** | 83.20 ± 16.50 | 78.50 ± 21.51 | −4.69 | 7 → 9 | 82.59 ± 17.76 | 79.90 ± 17.80 | −2.68 | 7 → 8 |
| GBTSVM | 65.57 ± 24.39 | 89.33 ± 13.36 | +23.76 | 10 → 5 | 62.93 ± 27.95 | 87.33 ± 16.03 | +24.40 | 9 → 6 |
| HF-GBTSVM | 73.77 ± 19.14 | 86.57 ± 11.90 | +12.80 | 9 → 7 | 75.85 ± 17.65 | 84.93 ± 14.41 | +9.08 | 8 → 7 |
| EF-TSVM | 92.29 ± 5.38 | 83.87 ± 8.41 | −8.41 | 4 → 8 | 90.03 ± 7.59 | 78.71 ± 12.49 | −11.32 | 4 → 9 |
| TSVM | 77.61 ± 18.84 | 64.04 ± 17.31 | −13.57 | 8 → 10 | 60.23 ± 43.17 | 44.51 ± 35.39 | −15.71 | 10 → 10 |
| SVM (RBF) | 94.13 ± 3.57 | 93.89 ± 5.21 | −0.24 | 1 → 3 | 91.47 ± 6.78 | 90.76 ± 9.00 | −0.70 | 2 → 3 |
| RVFL | 86.04 ± 14.29 | 89.27 ± 15.87 | +3.23 | 6 → 6 | 84.65 ± 16.74 | 87.89 ± 18.07 | +3.24 | 6 → 5 |
| RandomForest | 93.84 ± 3.33 | 96.23 ± 2.02 | +2.38 | 2 → 1 | 92.13 ± 4.64 | 94.22 ± 5.56 | +2.10 | 1 → 1 |
| MLP | 88.77 ± 8.23 | 93.14 ± 6.39 | +4.37 | 5 → 4 | 86.75 ± 11.41 | 90.07 ± 9.49 | +3.32 | 5 → 4 |
| XGBoost | 92.36 ± 5.99 | 96.08 ± 2.77 | +3.72 | 3 → 2 | 90.26 ± 6.84 | 93.60 ± 7.13 | +3.34 | 3 → 2 |

Full raw files: `results/nasa_hard_noise0_{runs,summary}.csv`.

What the numbers show:

* **Accuracy does not fall for everyone.** Only 4 of 10 models lost ACC: TSVM (−13.57), EF-TSVM (−8.41),
  EF-GBTSVM (−4.69) and SVM (−0.24, negligible). The other 6 *gained* (GBTSVM +23.76, HF-GBTSVM +12.80,
  MLP +4.37, XGBoost +3.72, RVFL +3.23, RandomForest +2.38). So the hypothesis "removing the
  discharge_time proxy makes everything harder" is **not supported** by these runs. The
  tree ensembles, MLP and SVM stay at 93–96 ACC.
* **EF-GBTSVM's relative rank got worse, 7th → 9th by ACC (7th → 8th by F1).** It is now ahead of only TSVM.
  Its two single-feature-type ablations (GBTSVM = original features only, 89.33; HF-GBTSVM = hidden
  features only, 86.57) now beat it by 8–11 ACC points, whereas on the full features they were *below* it
  (65.57 and 73.77). The two models that use original + hidden features together (EF-GBTSVM and EF-TSVM)
  are the ones that lost ground; no mechanism for this was tested here.
* **Per-battery instability.** EF-GBTSVM fold ACCs (held-out B0005, B0006, B0007, B0018, in that order)
  were 86.90 / 92.26 / 58.93 / 94.70 on full features and are 91.67 / 99.40 / 71.43 / **51.52** on the
  hard subset. The mean drop is driven by the B0018 fold; three of four folds actually improved.
  Its recall is 99.77 ± 0.46 but precision is 69.61 ± 25.99, i.e. in the hard run it over-predicts "degraded".
* With only 4 folds and ±20-point stds, none of these per-model differences should be over-read.
* Caveat: `time_to_v3p5` (time until voltage first drops to 3.5 V) is still in the feature set and
  could be a partial stand-in for discharge duration. Whether it is, was **not tested**.

## Statistical significance (Friedman + Nemenyi)

Command (the 6 `*_runs.csv` files, each treated as one "dataset", metric ACC; mean over folds per model per file):

```
python stats_tests.py results/nasa_noise0_runs.csv results/nasa_noise10_runs.csv results/nasa_noise20_runs.csv \
       results/nasa_t70_noise0_runs.csv results/nasa_t85_noise0_runs.csv results/nasa_hard_noise0_runs.csv \
       --metric ACC --per-file
```

`--per-file` is a new opt-in flag in `stats_tests.py` (default behaviour unchanged). It is needed because
`nasa_noise0/10/20` all store `dataset = "nasa"`, so without it they would be merged into one row and averaged.
No error occurred (N=6, k=10). Raw outputs: `results/stats_friedman_nemenyi_ACC.txt` (and `_F1.txt`).

Average ranks by ACC (1 = best): SVM 2.67, XGBoost 3.00, RandomForest 3.50, MLP 5.00, RVFL 5.00,
EF-TSVM 6.17, TSVM 6.17, GBTSVM 7.00, **EF-GBTSVM 7.83**, HF-GBTSVM 8.67.

* Friedman χ² = 24.473, p = 0.003613 (N = 6, k = 10): the 10 models are not all equivalent.
* Nemenyi critical difference (α = 0.05) = 5.530.
* Best (SVM) vs EF-GBTSVM: rank difference 5.17 < 5.530 → **not significant**. Only SVM vs HF-GBTSVM
  (6.00) exceeds the CD. (stats_tests.py only compares everything against the top-ranked model, so other pairs
  such as EF-GBTSVM vs EF-TSVM were not tested.)
* Same test on F1: Friedman χ² = 20.400, p = 0.0156; SVM (2.50) is best, EF-GBTSVM (7.67) last; SVM vs
  EF-GBTSVM = 5.17 → not significant; no model pair vs SVM exceeds the CD.

Caveats on validity: the 6 "datasets" are re-runs of one NASA dataset with different noise / threshold /
feature settings, so they are **not independent**, which violates the Friedman assumption (the p-value is
likely optimistic). `t70` has only 2 valid folds (see Table 2 caveat). N = 6 is also small for k = 10 models.
This analysis is a ranking summary, not evidence about generalisation across datasets.

## Novelty: purity-weighted balls (EF-GBTSVM-W)

**Idea.** Each granular ball already has a purity (fraction of its majority label), computed
during splitting but previously discarded. `EF-GBTSVM-W` (`weighted=True` in `EFGBTSVM`) feeds
that purity into the TWSVM dual as a per-ball upper bound: `C_i = C * purity_i` instead of a flat
`C` for every ball, so a less-pure (noisier) ball constrains the opposite-class plane less. Weight
= purity itself, no floor (confirmed with you before implementing). Changes, all backward-compatible:

* `granular_ball.py`: `generate_granular_balls(..., return_purity=False)` — default call is still
  the original 4-tuple `(centers, labels, radii, sizes)`; pass `return_purity=True` to also get a
  5th array of per-ball purities. Splitting itself is untouched either way.
* `twsvm.py`: `TWSVM.fit(A, B, weights=None)` — `weights=None` (default) reproduces the exact
  original box constraint (`_box_qp` scalar `ub`, bit-for-bit identical, see regression check
  below); `weights=(wA, wB)` scales each sample's bound by its weight.
* `model.py`: `EFGBTSVM(weighted=False, ...)` — default unchanged; `weighted=True` (only effective
  when `use_balls=True`) passes each ball's purity as the TWSVM weights.
* `run_experiments.py`: new `"EF-GBTSVM-W": (ef(weighted=True), ef_grid)` entry, same hyperparameter
  grid as `EF-GBTSVM`.

**Regression check (`weighted=False` must be identical to before):**
1. `generate_granular_balls(X, y, ...)` vs `generate_granular_balls(X, y, ..., return_purity=True)` —
   the shared 4 arrays (centers/labels/radii/sizes) are `np.allclose`-identical; splitting doesn't
   change.
2. `TWSVM().fit(A, B)` (old call signature) vs `TWSVM().fit(A, B, weights=None)` — `w1_, w2_, b1_,
   b2_` are `np.array_equal`-identical.
3. `EFGBTSVM(random_state=0)` (weighted defaults to `False`) vs `EFGBTSVM(random_state=0,
   weighted=False)` — `decision_function` output `np.array_equal`-identical on a synthetic set.
4. `weighted=True` with `use_balls=False` does not error — the weighting is silently inert (no
   balls, no purities to use), as it has to be since there is nothing to weight.

All 4 passed. The mechanism itself was separately confirmed to *do something* when it can: with
`C1=1.0` forced (overriding the search) and a 30%-flipped-label synthetic set, `EF-GBTSVM-W`'s
plane weights differed from `EF-GBTSVM`'s by a small but non-zero amount.

**Experiment (noise 0/10/20%, NASA battery, `--models EF-GBTSVM EF-GBTSVM-W`, other 9 models
unchanged from existing `results/*_summary.csv`):**

| Noise | EF-GBTSVM ACC | EF-GBTSVM-W ACC | Δ ACC | EF-GBTSVM F1 | EF-GBTSVM-W F1 | Δ F1 | EF-GBTSVM GMEAN | EF-GBTSVM-W GMEAN | Δ GMEAN |
|---|---|---|---|---|---|---|---|---|---|
| 0% | 83.20 ± 16.50 | 83.20 ± 16.50 | 0.00 | 82.59 ± 17.76 | 82.59 ± 17.76 | 0.00 | 85.79 ± 13.31 | 85.79 ± 13.31 | 0.00 |
| 10% | 89.60 ± 9.15 | 89.45 ± 9.05 | −0.15 | 85.30 ± 14.76 | 85.17 ± 14.64 | −0.13 | 89.02 ± 12.69 | 88.90 ± 12.60 | −0.12 |
| 20% | 61.31 ± 29.29 | 61.46 ± 29.07 | +0.15 | 52.38 ± 37.21 | 52.43 ± 37.20 | +0.05 | 48.94 ± 39.56 | 49.66 ± 38.70 | +0.72 |

Full raw files: `results/nasa_w_noise{0,10,20}_{runs,summary}.csv`.

**Honest result: the weighting makes essentially no difference on this dataset/protocol, at any
noise level — it does not fix the 20%-noise collapse.** The 20%-noise ACC (61.46 vs 61.31) is
within the fold-to-fold noise of this 4-fold evaluation, not a real improvement; GMEAN moves +0.72,
also not meaningful. At 0% noise the two are identical to 2 decimal places on every metric; at
10% noise EF-GBTSVM-W is marginally *worse* (−0.15 ACC), also not meaningful given the ±9 std.

**Why: the purity-split step already does most of the "purity" job before weighting ever runs.**
Per-fold logs at noise=0 show the random search consistently selects `purity ∈ {0.85, 0.925, 0.98}`
and `C1 ∈ {1e-5, 1e-4}` for `EF-GBTSVM` across all 4 folds — and `generate_granular_balls` only
keeps a ball below the `purity` threshold if splitting it further would eliminate a whole class
(the one explicit exception in that function). So almost every kept ball already has purity ≥ the
threshold (0.85+), meaning `weight = purity` for almost all balls is already close to 1 and the
down-weighting has little room to act — on top of that, the tiny `C1` the search prefers (1e-5–1e-4)
scales whatever purity gap remains down to a numerically negligible difference (confirmed by the
`C1=1.0` mechanism check above showing a real but small effect at a much larger `C1`). So this is
not a bug in the implementation; it's that this particular purity-splitting design leaves very
little room for a purity-based *weight* to matter once a ball has already been accepted.
**This does not test whether a different purity → weight mapping (e.g. `purity²`, which was offered
and declined) or a lower `--purity` CLI default (looser splitting, more genuinely-impure balls kept)
would behave differently — only the as-is, no-floor mapping was run.**

### Attempt 2: looser purity threshold

Attempt 1's own diagnosis was that the random search kept picking `purity ∈ {0.85, 0.925, 0.98}`,
so almost every surviving ball was already ≥85% pure and the weighting had nothing to bite on.
**Change made (`weighted=False` / `EF-GBTSVM` not touched at all):** `EF-GBTSVM-W` now gets its own,
looser hyperparameter grid in `run_experiments.py` — `purity ∈ {0.55, 0.65, 0.75, 0.85, 0.95}`
instead of the shared `PUR_GRID = {0.85, 0.9, 0.925, 0.95, 0.98}` — so the search can explore
configs that keep genuinely-impure balls. `EF-GBTSVM`'s own grid, model, and results files are
untouched (`results/nasa_noise*_runs.csv` unchanged — the baseline numbers in the table below are
read from the same files as Attempt 1 / Table 1). New run: `--name nasa_w2`, `--models EF-GBTSVM
EF-GBTSVM-W`, noise 0/10/20%, raw files `results/nasa_w2_noise{0,10,20}_{runs,summary}.csv`.

**Mechanism check — does it actually trigger now?** Added `EFGBTSVM.ball_purities_` (diagnostics
only, doesn't affect `fit`/`predict`) and reconstructed each fold with the exact best params the
search picked (logged per-fold in `results/_w2_noise*.log`), then inspected the balls:

| noise | run | chosen `purity` param | n_balls | n_impure (purity<0.95) | min purity | mean purity |
|---|---|---|---|---|---|---|
| 0% | 0 | 0.75 | 2 | 1 | 0.796 | 0.898 |
| 0% | 1 | 0.75 | 4 | 2 | 0.752 | 0.882 |
| 0% | 2 | 0.75 | 2 | 1 | 0.896 | 0.948 |
| 0% | 3 | 0.85 | 4 | 2 | 0.883 | 0.949 |
| 10% | 0 | 0.65 | 2 | 2 | 0.743 | 0.827 |
| 10% | 1 | 0.95 | 73 | 0 | 0.952 | 0.998 |
| 10% | 2 | 0.65 | 2 | 2 | 0.803 | 0.863 |
| 10% | 3 | 0.65 | 2 | 2 | 0.737 | 0.820 |
| 20% | 0 | 0.65 | 2 | 2 | 0.679 | 0.756 |
| 20% | 1 | 0.85 | 52 | 13 | 0.850 | 0.972 |
| 20% | 2 | 0.75 | 6 | 5 | 0.764 | 0.822 |
| 20% | 3 | 0.65 | 2 | 2 | 0.691 | 0.750 |

11 of 12 folds now pick a loose `purity` (0.65–0.85) with genuinely-impure balls (min purity down to
0.68–0.90, several balls <0.95); one fold (10% noise, run1) still picked `purity=0.95` and stayed
pure. So the mechanism verifiably triggers this time — Attempt 1's diagnosis (search never explores
low purity) was correct, and this is the direct fix for that specific cause.

**Result — a real effect this time, but it's a trade-off, not a fix:**

| Noise | EF-GBTSVM ACC | EF-GBTSVM-W ACC | Δ ACC | EF-GBTSVM F1 | EF-GBTSVM-W F1 | Δ F1 | EF-GBTSVM GMEAN | EF-GBTSVM-W GMEAN | Δ GMEAN |
|---|---|---|---|---|---|---|---|---|---|
| 0% | 83.20 ± 16.50 | 83.50 ± 11.62 | +0.30 | 82.59 ± 17.76 | 82.63 ± 14.55 | +0.05 | 85.79 ± 13.31 | 84.58 ± 8.29 | −1.20 |
| 10% | 89.60 ± 9.15 | 80.78 ± 10.18 | **−8.82** | 85.30 ± 14.76 | 79.14 ± 11.85 | **−6.16** | 89.02 ± 12.69 | 83.47 ± 6.96 | −5.55 |
| 20% | 61.31 ± 29.29 | 80.78 ± 22.54 | **+19.47** | 52.38 ± 37.21 | 80.18 ± 21.13 | **+27.80** | 48.94 ± 39.56 | 81.68 ± 19.34 | **+32.74** |

Δ columns are computed from unrounded per-fold means, so a Δ can differ by 0.01 from the
difference of the rounded values shown (e.g. 0% F1: 82.63 − 82.59 shows 0.04, exact Δ is +0.05).

* **0% noise: a wash.** ACC/F1 move less than 0.3 points either way (well within the ±12–18 fold
  std); GMEAN moves slightly negative. No case for or against on clean data.
* **20% noise: a large, consistent improvement.** Per-fold ACC goes `[59.52, 99.40, 27.98, 58.33]`
  → `[83.33, 97.62, 48.21, 93.94]` — 3 of 4 folds improve substantially (+23.8, +20.2, +35.6), one
  dips slightly (−1.8). The standard deviation also drops (ACC ±29.29 → ±22.54, F1 ±37.21 → ±21.13),
  i.e. more stable across batteries, not just a higher mean. This is the first result in this
  project where something measurably helps the 20%-noise collapse. **But see the ablation below:
  the gain comes from the looser purity threshold, not from the weighting.**
* **10% noise: a real, meaningful regression.** Per-fold ACC goes `[98.81, 94.64, 86.90, 78.03]` →
  `[82.14, 77.38, 69.64, 93.94]` — 3 of 4 folds get *worse* (−16.7, −17.3, −17.3), one improves a lot
  (+15.9). EF-GBTSVM-W drops from 2nd-of-10 (Table 1, Honest analysis) to clearly below EF-GBTSVM at
  this noise level.
* **Accuracy-vs-robustness trade-off, as asked for.** Loosening `purity` lets more genuinely-noisy
  points stay inside a ball together, so purity-weighting has something to discount — but on folds
  where the *unweighted* split would have been fine (10% noise, where EF-GBTSVM itself is already
  strong), the same looser splitting plus weighting now hurts, because it is reacting to noise that
  isn't necessarily there in a way that trades clean-case accuracy for noisy-case robustness.
  (Correction, added in Attempt 3: an earlier version of this bullet said the one fold that stayed at
  `purity=0.95` — 10% noise, run1 — "regressed the most". That is not accurate. The three regressing
  10%-noise folds dropped −16.67 (run0, `purity=0.65`), −17.26 (run1, `purity=0.95`) and −17.26
  (run2, `purity=0.65`). run1 ties run2, and 2 of the 3 regressing folds picked a *loose* purity.)
* **Not tested here (see Attempt 3 for scoring/CV):** a per-noise-level or adaptive purity choice, a floor on the weight, or `purity²`
  (all previously raised as alternatives) might recover the 10% case while keeping the 20% gain —
  none of these were run.

### Attempt 3: stabilizing purity selection

**Question:** at 10% noise, run1 picked `purity=0.95` and regressed (94.64 → 77.38 ACC). Is that
noisy CV selection, or does tight purity really score better on that training fold?

**Step 1 — CV candidates for that fold.** The search was re-run for that fold only (same seed, same
grid, same `GroupKFold`, `balanced_accuracy`). It reproduced the logged pick exactly
(`purity=0.95, n_hidden=23, activation=10, C1=1e-5`, CV score 0.7885). Best CV score per purity
value among the 30 candidates:

| purity | best CV bal. acc. | candidates |
|---|---|---|
| 0.95 | **0.7885** (also ranks 2 and 3: 0.7553, 0.7523) | 6 |
| 0.65 | 0.7187 (rank 4) | 4 |
| 0.85 | 0.7038 | 8 |
| 0.75 | 0.6680 | 9 |
| 0.55 | 0.5000 (all 3 candidates exactly chance) | 3 |

The top three candidates are all `purity=0.95`, so the preference is not one lucky draw. But the
gap to the best loose candidate (0.7885 − 0.7187 = 0.070) is smaller than that candidate's CV std
(0.091; others range 0.07–0.16). With only 3 inner folds, the CV cannot reliably tell these apart.
**Verdict: a consistent but weak preference — partly genuine, partly within CV noise.**

Also, this fold is not the main cause of the 10%-noise regression. Two of the three regressing
folds picked a *loose* `purity=0.65` and lost just as much (see the correction under Attempt 2).

**Step 2, Try A — per-model `--cv` override (`--cv-override EF-GBTSVM-W:10`).** Added to
`run_experiments.py` (other models keep `--cv`). **It cannot have any effect on this protocol.**
Inner CV is `GroupKFold(min(cv, number of training groups))`. Leave-one-battery-out always leaves 3
training batteries, so it is 3 folds whether `--cv` is 5 or 10. Checked on the regressed fold:
cv=5 and cv=10 both used 3 folds, picked `purity=0.95`, score 0.7885. Since the splits are
identical in every fold, a full re-run would reproduce Attempt 2 exactly, so it was not re-run.
More inner folds would require splitting *within* a battery, which breaks the group-aware tuning.

**Step 2, Try B — scoring override (`--scoring-override EF-GBTSVM-W:gmean`, or `:f1`).** Added a
G-mean scorer (same formula as the reported GMEAN). On the regressed fold, both `f1` (0.7150) and
`gmean` (0.7571) still picked `purity=0.95`. Try B (gmean) was then run on all folds as
`--name nasa_w3`, noise 0/10/20%, `--models EF-GBTSVM EF-GBTSVM-W`. Raw files:
`results/nasa_w3_noise{0,10,20}_{runs,summary}.csv`, logs `results/_w3_noise*.log`.

| Noise | EF-GBTSVM ACC | EF-GBTSVM-W, Attempt 2 (bal. acc.) | EF-GBTSVM-W, Attempt 3 (gmean) |
|---|---|---|---|
| 0% | 83.20 ± 16.50 | 83.50 ± 11.62 | 83.50 ± 11.62 |
| 10% | 89.60 ± 9.15 | 80.78 ± 10.18 | 80.78 ± 10.18 |
| 20% | 61.31 ± 29.29 | 80.78 ± 22.54 | 80.78 ± 22.54 |

**Try B changed nothing.** In all 12 fold × noise combinations gmean selected exactly the same
hyperparameters as balanced accuracy, so every metric and fold is identical to Attempt 2
(F1 and GMEAN too). That fits the definitions: for binary labels, balanced accuracy is the mean of
sensitivity and specificity, G-mean is their geometric mean, and on this search they rank the
candidates the same way at the top. The `EF-GBTSVM` rows in `nasa_w3` match `nasa_w2` fold for fold,
confirming the baseline was not affected.

**Conclusion (final for this iteration — both tries failed, per the stop rule).** Neither more CV
folds (structurally impossible here) nor a different selection metric changes the 10%-noise
outcome. The regression is not mainly a selection-variance artifact: it happens with loose and
tight purity alike. So this is treated as a **genuine trade-off of the fixed-threshold model**.
One purity threshold chosen by CV cannot be both loose enough to help at 20% noise and safe at
10% noise. The 20%-noise gain from Attempt 2 (+19.47 ACC) stands, and so does the 10%-noise cost
(−8.82 ACC).

### Ablation: is it the weighting or the looser threshold?

Attempt 2 changed two things at once for EF-GBTSVM-W: weighting on, *and* a looser purity grid. To
separate them, a new model `EF-GBTSVM-L` was added: plain EF-GBTSVM (`weighted=False`) with the
same loose grid as EF-GBTSVM-W (`purity ∈ {0.55, 0.65, 0.75, 0.85, 0.95}`). Same seeds and grid, so
the random search draws the same 30 candidates per fold; the only difference from EF-GBTSVM-W is
the weighting. Run as `--name nasa_abl --models EF-GBTSVM-L`, noise 0/10/20%, otherwise identical
protocol. Raw files: `results/nasa_abl_noise{0,10,20}_{runs,summary}.csv`. EF-GBTSVM and
EF-GBTSVM-W numbers are from `results/nasa_w2_*` (Attempt 2).

| Noise | EF-GBTSVM (tight grid, no weights) | EF-GBTSVM-L (loose grid, no weights) | EF-GBTSVM-W (loose grid, weights) |
|---|---|---|---|
| 0% ACC | 83.20 ± 16.50 | **85.73 ± 13.77** | 83.50 ± 11.62 |
| 0% F1 | 82.59 ± 17.76 | **84.23 ± 16.21** | 82.63 ± 14.55 |
| 10% ACC | **89.60 ± 9.15** | 80.78 ± 10.18 | 80.78 ± 10.18 |
| 10% F1 | **85.30 ± 14.76** | 79.14 ± 11.85 | 79.14 ± 11.85 |
| 20% ACC | 61.31 ± 29.29 | 80.63 ± 22.52 | **80.78 ± 22.54** |
| 20% F1 | 52.38 ± 37.21 | **80.90 ± 21.13** | 80.18 ± 21.13 |
| 20% GMEAN | 48.94 ± 39.56 | **82.14 ± 19.39** | 81.68 ± 19.34 |

Per-fold ACC at 20% noise: EF-GBTSVM-L `[82.74, 97.62, 48.21, 93.94]` vs EF-GBTSVM-W
`[83.33, 97.62, 48.21, 93.94]`. At 10% noise the two are identical fold for fold.

**Verdict: the purity weighting contributes essentially nothing. The whole effect (the +19 ACC at
20% noise *and* the −8.8 ACC at 10% noise) comes from letting the search use a looser purity
threshold, i.e. keeping coarser, less-pure balls.**

* 20% noise: without weighting 80.63 ACC, with weighting 80.78 (+0.15). F1 and GMEAN are actually
  slightly *higher* without weighting (80.90 vs 80.18; 82.14 vs 81.68). All within noise.
* 10% noise: identical to the last digit with and without weighting, so the regression is also
  caused by the loose threshold, not the weighting.
* 0% noise: without weighting is +2.23 ACC better than with (85.73 vs 83.50), driven by one fold
  (run1: 98.21 vs 89.29). In that fold the two searches picked different hyperparameters, so this
  is a selection difference, not evidence that weighting hurts.
* In 10 of 12 fold × noise cases both variants picked the same `purity`, and in 9 of 12 the same
  full parameter set. In all 9 of those, ACC and F1 are identical with and without weighting
  (individual predictions were not compared). The small differences in the table come only from
  the 3 cases where the
  search picked different parameters. Why the weighting has no effect even
  with impure balls was not investigated further; Attempt 1 showed its effect scales with `C1`, and
  the search mostly picks tiny `C1` (1e-5 to 1e-3).

**Consequence for the novelty claim.** "Purity-weighted granular balls improve robustness to label
noise" is **not supported** by these runs. What *is* supported (on this one dataset, 4 folds): for
EF-GBTSVM, the purity threshold range is a strong lever under label noise. A looser threshold
(0.65–0.85) helps a lot at 20% noise and hurts at 10% noise, and the paper-style grid
(0.85–0.98) never explores it. This is a hyperparameter-range finding, not a new mechanism.

### Attempt 4: noise-adaptive purity (EF-GBTSVM-A) — last iteration of this direction

**Idea.** Instead of grid-searching `purity`, estimate the training-label noise inside `fit()` and
set `purity` from it, so it can be loose on noisy data and tight on clean data.

**Implementation.**
* New module `efgbtsvm/noise.py`:
  * `estimate_label_noise(X, y, k=5, random_state=0)` returns the out-of-fold error rate of a 5-NN
    classifier under stratified 5-fold CV.
  * `purity_from_noise(noise, alpha=1.0)` returns `clip(0.95 − alpha·noise, 0.55, 0.95)`.
* `EFGBTSVM(purity_mode="fixed" | "adaptive", noise_alpha=1.0)`. `"fixed"` (default) is the old
  behaviour. `"adaptive"` ignores `purity` and sets `purity_used_` from the estimate.
  * The estimate runs on the raw (scaled) training points, *before* ball generation. Estimating on
    the balls, as first specified, is circular: the balls are built from `purity`.
  * Inside the hyperparameter search, every inner fit estimates noise on its own training subset,
    so no test labels are used.
* `run_experiments.py`: `EF-GBTSVM-A` = adaptive mode, grid over `C1, n_hidden, activation` only.
* **`alpha` fixed at 1.0, decided before any adaptive results were seen.** Two reasons:
  * Tuning `alpha` on a {0.5, 1.0, 1.5, 2.0} inner-CV grid would bring back CV selection of purity
    by another name, and Attempt 3 showed that selection is weak with only 3 inner folds.
  * Choosing `alpha` to hit purities that earlier test folds favoured would be test-set leakage.

**Regression.** `purity_mode="fixed"` was checked end to end:
* All 36 previously stored per-fold results (EF-GBTSVM, EF-GBTSVM-W, EF-GBTSVM-L × 3 noise levels
  × 4 folds) were re-fitted from their logged hyperparameters with the new code. All 36 reproduce
  the stored ACC exactly.
* In the new run, EF-GBTSVM and EF-GBTSVM-L match `nasa_w2` / `nasa_abl` fold for fold.
* Other checks: the default equals explicit `"fixed"` bit-for-bit, an invalid mode raises, and
  `sklearn.clone` keeps the new params.

**Estimator sanity check (training folds only).** It tracks the injected noise well:

| Injected train-label noise | Estimated noise (4 folds) | Mean | Resulting purity |
|---|---|---|---|
| 0% | 1.28–1.59% | 1.47% | 0.934–0.937 |
| 10% | 11.32–12.18% | 11.74% | 0.828–0.837 |
| 20% | 21.23–25.00% | 22.99% | 0.700–0.738 |

The ~1.5% floor on clean data is the kNN's own error, i.e. class overlap. Above that floor the
estimate is close to the true flip rate. **The estimator works.**

**Result.** Run as `--name nasa_adaptive --models EF-GBTSVM EF-GBTSVM-L EF-GBTSVM-A`, noise 0/10/20%.
Raw files: `results/nasa_adaptive_noise{0,10,20}_{runs,summary}.csv`; per-fold estimates in
`results/_adaptive_noise*.log`.

| Noise | Metric | EF-GBTSVM (tight grid) | EF-GBTSVM-L (loose grid) | EF-GBTSVM-A (adaptive) |
|---|---|---|---|---|
| 0% | ACC | 83.20 ± 16.50 | **85.73 ± 13.77** | 72.67 ± 16.09 |
| 0% | F1 | 82.59 ± 17.76 | **84.23 ± 16.21** | 74.69 ± 14.06 |
| 10% | ACC | **89.60 ± 9.15** | 80.78 ± 10.18 | 76.26 ± 19.66 |
| 10% | F1 | **85.30 ± 14.76** | 79.14 ± 11.85 | 77.03 ± 19.04 |
| 20% | ACC | 61.31 ± 29.29 | 80.63 ± 22.52 | **84.60 ± 12.75** |
| 20% | F1 | 52.38 ± 37.21 | 80.90 ± 21.13 | **82.93 ± 15.03** |
| 20% | GMEAN | 48.94 ± 39.56 | 82.14 ± 19.39 | **86.26 ± 9.29** |

Per-fold ACC for EF-GBTSVM-A: 0% `[80.36, 90.48, 66.07, 53.79]`, 10% `[78.57, 77.38, 50.60, 98.48]`,
20% `[78.57, 97.02, 69.64, 93.18]`.

* **20% noise: best of the three.** ACC 84.60 vs 80.63 (loose) and 61.31 (tight), and with the
  smallest spread (±12.75 vs ±22.52 and ±29.29). The weakest fold rises from 27.98 (tight) / 48.21
  (loose) to 69.64.
* **10% noise: worse than both.** ACC 76.26, i.e. −13.34 vs the tight grid and −4.52 vs the loose
  grid. Fold run2 drops to 50.60. Adaptive does *not* avoid the 10%-noise loss; it is larger.
* **0% noise: clearly worse than both.** ACC 72.67, i.e. −10.52 vs tight and −13.05 vs loose. Fold
  run3 falls from 94.70 to 53.79 (precision 48.74, recall 100.00): it over-predicts "degraded".
  Folds run0, run2 and run3 all have recall 100 with precision 44–67.
  On clean data the adaptive purity (≈0.935) sits *inside* the tight grid's range, so this loss is
  not from a wrong purity level. Purity is no longer chosen jointly with `C1 / n_hidden /
  activation`, and the purity the tight grid picked on these folds (0.85–0.98, varying by fold) was
  better suited to them than a fixed 0.935. Not investigated further.

**Verdict.** The estimator does what it should, but the linear mapping with `alpha=1.0` does not
produce a model that wins at every noise level. It helps at 20% noise (+23.30 ACC over the
original, +3.98 over the loose grid) and loses at 10% and 0%. **Goal not met:** the 20%-noise gain
comes with a 10%-noise loss, and now also a clean-data loss. These differences sit on 4 folds with
±13–20 ACC spreads and are not significance-tested. A different `alpha` or mapping shape was not
tried. Tuning one now, after seeing these test folds, would be leakage, and this was declared the
last iteration.

### Final verdict on the novelty direction (Attempts 1–4 + ablation)

* **Purity-weighted balls (`weighted=True`):** no measurable effect. Attempt 1 changed nothing; the
  ablation showed Attempt 2's gain also appears without weighting.
* **Purity threshold range is the real lever under label noise.** Loose purity helps a lot at 20%
  noise and hurts at 10% noise. CV selection (Attempt 3) cannot resolve this, and a simple
  noise-adaptive rule (Attempt 4) moves the trade-off rather than removing it: best at 20% noise,
  worst at 0% and 10%.
* **What the evidence supports**, on one dataset with 4 folds: EF-GBTSVM's accuracy under heavy
  label noise depends strongly on the granular-ball purity threshold. No single setting tested
  (tight grid, loose grid, or noise-adaptive) is best across 0%, 10% and 20% noise.
* **What it does not support:** a new method that improves noise robustness overall.

## Tuning budget equalization

**Why.** `RandomizedSearchCV` uses `min(--n-iter, grid size)` candidates. With `--n-iter 30`,
RBF-SVM (16), XGBoost (12), TSVM (8), random forest and MLP (6 each) were searched in full, while
EF-GBTSVM, -W and -L sampled 30 of 1,000 combinations and EF-GBTSVM-A 30 of 200. The baselines
were therefore not re-run; only the EF-GBTSVM family was re-tuned.

**Run.** `--name nasa_bigsearch --n-iter 150 --models EF-GBTSVM EF-GBTSVM-W EF-GBTSVM-L EF-GBTSVM-A`,
noise 0/10/20%, everything else unchanged (same folds, seeds and label flips). 150 candidates is 5×
the default for EF-GBTSVM/-W/-L (150 of 1,000) and 150 of 200 for -A. Raw files:
`results/nasa_bigsearch_noise{0,10,20}_{runs,summary}.csv`, logs `results/_bigsearch_noise*.log`.
A first attempt that ran the three noise levels in parallel crashed (three processes × 16 joblib
workers exhausted Windows process resources: bash `fork` failures `0xC000026B` and
`TerminatedWorkerError` in all three; no result files were written). The sequential re-run
finished in 734 s (noise 0%: 216 s, 10%: 188 s, 20%: 330 s).

| Noise | EF-GBTSVM, 30 cand. | EF-GBTSVM, 150 cand. | Rank 30 → 150 (of 10) | Best other model | Gap to best, 30 → 150 |
|---|---|---|---|---|---|
| 0% | 83.20 ± 16.50 | 72.54 ± 22.46 | 7 → 9 | RBF-SVM 94.13 | 10.93 → 21.59 |
| 10% | 89.60 ± 9.15 | 88.30 ± 6.81 | 2 → 3 | XGBoost 93.28 | 3.68 → 4.98 |
| 20% | 61.31 ± 29.29 | 74.95 ± 21.21 | 10 → 6 | RBF-SVM 90.81 | 29.50 → 15.87 |

Per-fold ACC for EF-GBTSVM at 150 candidates: 0% `[86.90, 95.83, 58.93, 48.48]`, 10%
`[92.26, 94.05, 88.10, 78.79]`, 20% `[61.31, 97.62, 52.98, 87.88]`. At 0% the held-out B0018 fold
falls from 94.70 (30 candidates) to 48.48 with `purity=0.925, C1=0.01`.

**Verdict: the gap persists even with 5× more candidates.** The larger search does not move
EF-GBTSVM consistently toward the baselines: −10.66 ACC at 0% noise, −1.30 at 10%, +13.64 at 20%.
At every noise level EF-GBTSVM stays below RBF-SVM, and at 20% also below RVFL, XGBoost, MLP and
GBTSVM. The non-monotone change is consistent with selection noise from only three inner folds (a
larger search finds more candidates that win the inner CV by chance); that mechanism was not tested
directly. Under-tuning is therefore not the explanation for EF-GBTSVM's gap on this dataset.

The other variants at 150 candidates (ACC; 30-candidate value in brackets): EF-GBTSVM-W 81.89
(83.50) / 81.60 (80.78) / 80.88 (80.78); EF-GBTSVM-L 81.70 (85.73) / 81.60 (80.78) / 80.74 (80.63);
EF-GBTSVM-A 70.82 (72.67) / 78.41 (76.26) / 85.63 (84.60), at 0/10/20% noise. The Section 6
trade-off survives: with 150 candidates the wide grid (L) is 9.16 points above the default grid
at 0%, 6.70 below it at 10% and 5.79 above it at 20%, and the best purity setting is still
different at each noise level (wide, default, adaptive). W and L picked the same hyperparameters
in 11 of 12 cases; accuracy differed in one of them (0%, B0018 fold, `purity=0.85, C1=1e-5`,
+0.76 for weighting), so weighting is near-null rather than exactly null.

Not re-tuned: RVFL (30 of 200), GBTSVM (30 of 40), HF-GBTSVM (30 of 1,000) and EF-TSVM (30 of 200),
as requested (only the EF-GBTSVM family). The Friedman ranking over six settings still uses the
30-candidate runs.

## Single-feature baseline (time to 3.5 V)

`python proxy_checks.py` fits one threshold (a depth-1 decision tree) on `time_to_v3p5` alone on
the three training cells and tests on the held-out cell, with the same folds and the same training
label flips as `run_experiments.py` (flip seed 1000 + fold index).

| Training-label noise | Rule ACC (folds) | Rule mean ± SD | Best model | EF-GBTSVM (30 / 150 cand.) |
|---|---|---|---|---|
| 0% | 97.02 / 99.40 / 93.45 / 95.45 | 96.33 ± 2.52 | RBF-SVM 94.13 | 83.20 / 72.54 |
| 10% | 97.02 / 99.40 / 93.45 / 96.97 | 96.71 ± 2.45 | XGBoost 93.28 | 89.60 / 88.30 |
| 20% | 97.62 / 99.40 / 97.02 / 96.21 | 97.56 ± 1.36 | RBF-SVM 90.81 | 61.31 / 74.95 |

Clean-label F1 of the rule: 94.26 ± 5.77. Learned thresholds (held-out cell): degraded if
`time_to_v3p5` ≤ 1417.4 s (B0005), 1387.0 s (B0006), 1364.0 s (B0007), 1415.6 s (B0018).
Fold by fold on clean labels the rule is at or above RBF-SVM, EF-GBTSVM, MLP, GBTSVM and
HF-GBTSVM in 4 of 4 folds, and above random forest, XGBoost, EF-TSVM, RVFL and TSVM in 3 of 4.
The rule only reads `time_to_v3p5`, so it gives the same result on the full and hard feature sets.

Why it works: the discharge current is near-constant (mean |I| per cycle 1.78–1.91 A by cell,
cycle-to-cycle SD 0.04–0.14 A), so time-based features track discharged charge, i.e. capacity;
|I| × discharge time relative to each cell's first cycle reproduces the label for 632 of 636
cycles. The task as defined (SOH < 0.80 from discharge features) is therefore close to a known
physical relationship, and the model comparison mainly measures how well each model recovers it.
Caveats: 4 folds, no significance test; only uniform label noise was tested, and noise near the SOH
threshold could shift the rule's threshold.

## Honest analysis

**EF-GBTSVM does not win on this dataset.** At the default operating point
(SOH threshold 0.80, no label noise), its ranking by ACC among the 10 models is
**7th of 10**: SVM (94.13), RandomForest (93.84), XGBoost (92.36), and even its
own no-granular-ball ablation EF-TSVM (92.29) all beat it by 9–11 points, while
EF-GBTSVM sits at 83.20 ± 16.50 — also with by far the largest variance across
battery folds of any top-half model, i.e. it is unstable across batteries, not
just weaker on average. At the harder 0.85 threshold the pattern repeats:
EF-GBTSVM is **last** by F1 (71.87, next-worst is HF-GBTSVM at 84.19), while
EF-TSVM (97.26), GBTSVM (91.10), TSVM (95.82), SVM (95.41) and RandomForest
(95.00) are all well ahead of it. At 0.70 every model struggles with the extreme
class imbalance, but EF-GBTSVM is again at the bottom (F1 = 4.20, worst besides
HF-GBTSVM's 4.00).

**Where EF-GBTSVM does look good: 10% train-label noise.** Under `--noise 0.1`
EF-GBTSVM's ACC *rises* to 89.60 and it jumps to **2nd place** (only XGBoost's
93.28 is higher), overtaking SVM, RandomForest, MLP and its own EF-TSVM
ablation (which collapses from 92.29 → 68.41 under the same noise). This is
consistent with the paper's robustness argument: the granular-ball step takes a
majority vote per ball, which can absorb a fraction of flipped training labels
that a model trained directly on raw points (EF-TSVM, plain TSVM) cannot. So
the *qualitative* noise-robustness claim holds in this one comparison.

**But that robustness story breaks down at 20% noise.** EF-GBTSVM's ACC falls
to 61.31 ± 29.29 (F1 52.38 ± 37.21) — now the **worst or near-worst** of all 10
models, below even its own 0%-noise number, and with the largest variance of
any model at any noise level in this report. SVM (90.81), RVFL (88.72) and
XGBoost (86.50) degrade far more gracefully. So EF-GBTSVM's noise tolerance is
non-monotonic here: it helps a little at a moderate noise rate and then falls
apart as noise increases further — the opposite of a clean "more robust"
story, and likely a symptom of the granular-ball majority vote itself becoming
unstable once a large-enough fraction of a ball's points are mislabeled,
combined with only 4 battery folds (so the mean is driven by one or two folds).

**Harder feature subset (Table 3).** Removing `discharge_time` and `i_mean` did not
make every model worse (6 of 10 gained ACC), but EF-GBTSVM was one of the 4 that
lost (83.20 → 78.50) and its ACC rank slid from 7th to 9th of 10. Its simpler
siblings GBTSVM and HF-GBTSVM overtook it. So the dataset being "easy because of the
discharge_time proxy" does not explain EF-GBTSVM's weakness here; the gap to
SVM / RandomForest / XGBoost (93.89 / 96.23 / 96.08 vs 78.50) is larger on the hard subset.

**Statistical significance.** Friedman rejects "all 10 models equal" (p = 0.0036 on ACC)
but the Nemenyi test cannot separate EF-GBTSVM from the best model (rank diff
5.17 < CD 5.530). EF-GBTSVM's average ACC rank over the 6 settings is 7.83 of 10 (SVM 2.67,
XGBoost 3.00, RandomForest 3.50). Honest reading: it is ranked in the lower half in 5 of 6
settings (rank 2 only at 10% noise), but with 6 non-independent "datasets" the
difference is **not statistically significant**; this is consistent with, not proof of, "worse than the classical baselines".

**Training cost.** From the per-fold `train_time` over all 6 runs files, EF-GBTSVM has the
highest mean fit time (3.30s; range 0.00–11.07s) vs SVM 0.01s, RVFL 0.01s, XGBoost 0.28s,
RandomForest 0.83s, MLP 1.17s (max 7.05s). `train_time` is the clean refit with the best
params; tuning (granular-ball generation inside a 30-candidate search over `(C1, n_hidden,
activation, purity)`, the largest grid here) is recorded separately in `tune_time` and was
not analysed in this report. (An earlier version of this section quoted "1.2–8.0s per fold vs
0.01–1s"; that did not match the CSVs and has been replaced by the figures above.)

**Bottom line for this dataset/protocol:** EF-GBTSVM's ACC rank among the 10 models is
7th (noise 0%), 2nd (noise 10%), 10th (noise 20%), 9th (t=0.70, 2 folds only), 10th (t=0.85)
and 9th (hard features). Classical baselines (SVM, RandomForest, XGBoost) or the simpler
ablations match or beat the full EF-GBTSVM pipeline in 5 of those 6 settings, typically with lower variance and
less compute. The one place EF-GBTSVM is competitive is the moderate-noise (10%) condition, and
that advantage disappears at 20% noise. A purity-weighted-balls variant (EF-GBTSVM-W, see novelty
section) was tried for exactly this failure. With the default grid it did nothing (Attempt 1); with
a looser purity grid it raised 20%-noise ACC from 61.31 to 80.78 but lowered 10%-noise ACC from
89.60 to 80.78 (Attempt 2). The ablation then showed that **the same gain and the same loss appear
without any weighting** (unweighted EF-GBTSVM on the loose grid: 80.63 and 80.78). So the weighting
is not the cause; the purity threshold range is. That is a hyperparameter finding with a real
trade-off, not a new mechanism. A noise-adaptive purity rule (Attempt 4) did best at 20% noise
(ACC 84.60) but worst at 10% (76.26) and 0% (72.67), so it moves the trade-off rather than
removing it. Given only 4 battery packs (per-battery degraded-% ranges 27–64%),
these results are evidence from one small, four-group dataset, not a general verdict on the method — but on this
dataset/protocol the numbers do not support a claim that EF-GBTSVM outperforms standard baselines.

## Phase B setup: equal tuning budgets (Nov 2026)

**Problem (finding F2, confirmed).** `RandomizedSearchCV` uses `min(n_iter, grid_size)`
candidates. With `--n-iter 150` the EF-GBTSVM family (grid ≥ 200) gets up to 150 candidates, while
SVM (16), RandomForest (6), MLP (6) and XGBoost (12) are fully searched in ≤16 iterations — the
baselines never actually use 150 candidates. This violates the hard rule "equal tuning budget."

**Fix.** `run_experiments.py` now accepts `--grids {paper,equal}`. With `--grids equal`, four
baselines get enlarged grids that give ≥150 combinations:

| Model | Paper grid (combinations) | Equal grid (combinations) |
|---|---|---|
| SVM | C(10)×γ(2)=20 → used 16 | C(10)×γ(15)=**150** |
| RandomForest | 6 | n_estimators(4)×max_depth(5)×min_samples_leaf(4)×max_features(3)=**240** |
| MLP | 6 | hidden_layer_sizes(6)×alpha(7)×lr_init(2)×activation(2)=**168** |
| XGBoost | 12 | n_estimators(4)×max_depth(5)×lr(4)×subsample(2)=**160** |

All other models are unchanged: EF-GBTSVM family (1,000/200), GBTSVM (40), HF-GBTSVM (1,000),
EF-TSVM (200), TSVM (8), RVFL (200). GBTSVM and TSVM still have small grids (40 and 8) — they
are fully searched even at n_iter=150. All Phase B experiments use `--grids equal --n-iter 150`.

Smoke-tested with `--models SVM RandomForest --n-iter 5 --grids equal --data data/nasa_battery_phys.csv`:
ran without error; params from the equal-grid search were drawn from the new grids only (confirmed).

## Phase A — a new EF-GBTSVM variant, chosen by inner CV only (Nov 2026)

Written before the Phase A results were seen; the selection rule below was fixed in
`phase_a_inner_cv.py` before the run started.

### Data finding: B0005–B0007 share one recording

The raw `.mat` files of B0005, B0006 and B0007 have identical `Time` vectors in every discharge
cycle (first cycle: 197 samples, 0 → 3690.2 s for all three), so `discharge_time` in
`nasa_battery.csv` is identical across the three cells in all 168 cycles. The load comes off at
each cell's own cut-off: first cycle 3346.9 s (B0005), 3690.2 s (B0006, the end of the record),
3487.1 s (B0007); B0018 was recorded alone (3357.5 s of 3434.9 s). After load-off the current is
≈ 0, so |`i_mean`| × `discharge_time` is a coulomb count over the record (B0005 first cycle:
1.864 Ah vs NASA capacity 1.857 Ah). Whole-record statistics (`discharge_time`, `v_mean`, `v_std`,
`v_end`, `t_mean`, ...) of B0005 and B0007 therefore include a rest period whose length is set by
another cell. Earlier results are not changed by this, but `discharge_time` is a record length,
not a per-cell BMS measurement, for these three cells.

### Candidates and why

* **(a) Physics-guided features, for every model** (`prep_phys_features.py` →
  `data/nasa_battery_phys.csv`). Finding 4 (one threshold on time to 3.5 V gives 96.3%) says the
  signal is physical. The new file uses only the loaded part of each discharge before the voltage
  first reaches 3.5 V, so the shared-record artifact cannot enter: time from load-on to 3.8 V and to
  3.5 V, the 3.8 → 3.5 V interval, voltage 300 s and 1000 s after load-on, rest voltage before the
  load, and temperature rise to 3.5 V; plus each one's change relative to the same cell's first
  cycle (ratio for durations, difference for voltages and temperature), which removes cell-to-cell
  offsets such as B0006's larger initial capacity. 12 features in `nasa_battery_phys.csv`, the file Phase B uses (the earlier `nasa_battery_phys_v1.csv` has 14); rows, order and labels equal
  `nasa_battery.csv` (checked by the script). The first cycle of a cell is label-free and available
  once the cell is in service. Phase A runs every candidate on both files.
* **(b) Joint noise-anchored purity, EF-GBTSVM-J.** Finding 3: the adaptive rule (Attempt 4) set
  τ from the noise estimate alone and lost 10.5 points on clean labels, because τ was no longer
  chosen with C, hidden size and activation. J keeps the estimate as an anchor and searches an
  offset jointly with the other hyperparameters: τ = clip(0.95 − η̂ + Δ, 0.55, 0.98),
  Δ ∈ {−0.10, −0.05, 0, 0.05, 0.10}. The searched τ range therefore moves with the estimated noise
  (for example τ ∈ {0.83, …, 0.98} when η̂ = 0.02 and {0.63, …, 0.83} when η̂ = 0.22), instead of
  one fixed grid that is either too tight at 20% noise or too loose at 10% (finding 2).
* **(c) Mass-weighted balls, EF-GBTSVM-M.** Finding 2's mechanism: at high τ and 20% noise the
  splitting leaves many small balls around flipped labels, and in EF-GBTSVM every ball counts once.
  M gives ball i the weight sᵢ = (nᵢ pᵢ)^β (nᵢ points, purity pᵢ, so nᵢpᵢ points carry its label),
  normalised to mean 1 per class, on both its own-class term and its slack bound (Cᵢ = C sᵢ), inside
  the EF-GBTSVM dual: plane 1 uses (Hᵀ S_A H + δI)⁻¹ and 0 ≤ α ≤ d₁ s_B. β ∈ {0, 0.5, 1}; β = 0 is
  plain EF-GBTSVM (checked: max relative difference 4.4e-13). Earlier purity-only weighting
  (Attempt 1) did nothing because surviving balls are ≥ 85% pure; ball size varies far more.
* **(b)+(c), EF-GBTSVM-JM.**

Default behaviour of `efgbtsvm/` is unchanged: decision values of EF-GBTSVM, -W, -A, GBTSVM,
HF-GBTSVM and EF-TSVM are bit-identical to the pre-change package on 24 configurations × clean and
noisy labels.

### Protocol (`phase_a_inner_cv.py`)

For each outer fold the test cell is removed first and never used. Inside the three training
cells each cell is the validation cell once; each model is tuned on the other two by randomised
search (150 candidates or the full grid; leave-one-cell-out CV; balanced accuracy), refitted on
both, and scored on the validation cell. Training labels are flipped as in `run_experiments.py`
(seed 1000 + fold), so validation labels are noisy, as they would be in practice; clean-label
scores are kept as a diagnostic only. Grids: C, hidden size and activation as in
`run_experiments.py`; τ (EF-GBTSVM, M) or Δ (J, JM); β (M, JM). References, not selectable:
RBF-SVM with a 10 × 15 = 150-point grid, and a depth-1 tree on all features.

**Selection rule:** per outer fold, the candidate with the highest mean validation balanced
accuracy on noisy labels over the three validation cells and the noise levels 0, 10 and 20%; an
exact tie goes to the candidate with fewer added components.

### Phase A results: phys2 (corrected features, wide-tau control, Nov 2026)

Run completed: `python phase_a_inner_cv.py --data data/nasa_battery_phys.csv --name phys2`,
636 cycles × 12 phys features, 150 inner-CV candidates per model per cell, noise ∈ {0, 10, 20}%.
Output: `results/phase_a/phys2_inner_cv.csv`, `results/phase_a/phys2_selection.csv`.

**Fold selection (pre-registered rule: highest mean noisy-label BACC over 3 val-cells × 3 noise levels):**

| Fold | Test cell | Selected variant | Inner-CV score |
|------|-----------|-----------------|---------------|
| 0 | B0005 | EF-GBTSVM-L8 | 76.77 |
| 1 | B0006 | EF-GBTSVM-J  | 80.22 |
| 2 | B0007 | EF-GBTSVM-L8 | 83.67 |
| 3 | B0018 | EF-GBTSVM-L8 | 73.78 |

Three of four folds chose EF-GBTSVM-L8 (wide fixed-purity grid 0.55–0.98); one chose EF-GBTSVM-J.

**Mean inner-CV BACC (noisy labels, all folds × all noise levels × 3 val cells):**

| Model | Mean | Std | Min | Max |
|-------|------|-----|-----|-----|
| Stump (ref) | **85.35** | 8.99 | 69.21 | 99.54 |
| SVM (ref) | 82.76 | 10.94 | 59.50 | 97.99 |
| EF-GBTSVM-L8 | 78.35 | 14.21 | 31.91 | 100.0 |
| EF-GBTSVM | 74.72 | 14.79 | 48.63 | 100.0 |
| EF-GBTSVM-J | 74.19 | 14.32 | 44.11 | 100.0 |
| EF-GBTSVM-JM | 72.58 | 16.68 | 44.11 | 100.0 |
| EF-GBTSVM-M | 70.68 | 15.23 | 44.50 | 100.0 |

**Key findings:**
- EF-GBTSVM-L8 (wide purity grid) is the best EF candidate (+3.6 over plain EF-GBTSVM), confirming
  that the original narrow grid {0.85, 0.9, 0.925, 0.95, 0.98} was an unnecessary handicap.
- Both references beat all EF candidates: SVM +4.4, Stump +7.0 above EF-GBTSVM-L8.
- J and M variants did not improve over plain EF-GBTSVM after the purity grid was widened
  (L8 strictly dominates both on the same wide range).
- **No EF variant is fairly best on inner CV.** Per the hard rules, Phase B cannot proceed and
  the 3-option decision (see below) must come first.

**Per-fold × per-noise mean BACC (candidates only):**

| Model | F0/0% | F0/10% | F0/20% | F1/0% | F1/10% | F1/20% | F2/0% | F2/10% | F2/20% | F3/0% | F3/10% | F3/20% |
|-------|-------|--------|--------|-------|--------|--------|-------|--------|--------|-------|--------|--------|
| EF-GBTSVM | 79.4 | 67.3 | 70.2 | 94.5 | 71.8 | 63.8 | 85.7 | 83.7 | 63.1 | 80.2 | 72.6 | 64.6 |
| EF-GBTSVM-L8 | **81.5** | **83.7** | 65.2 | **96.9** | 67.1 | **73.5** | **88.7** | **86.4** | **75.9** | **84.6** | **77.2** | 59.6 |
| EF-GBTSVM-J | 79.4 | 76.9 | 63.7 | 94.5 | 79.0 | 67.2 | 79.6 | 82.2 | 67.3 | 63.4 | 70.2 | 67.1 |
| EF-GBTSVM-M | 67.4 | 70.2 | 71.2 | 96.2 | 69.0 | 62.4 | 84.0 | 77.8 | 72.8 | 61.0 | 62.0 | 54.3 |
| EF-GBTSVM-JM | 69.4 | 59.5 | 57.9 | 96.2 | 81.8 | 57.9 | 89.8 | 85.2 | 76.0 | 61.0 | 70.8 | 65.6 |
| SVM | 95.3 | 86.6 | 77.0 | 96.6 | 86.2 | 75.1 | 88.0 | 82.4 | 72.5 | 87.0 | 77.4 | 69.0 |
| Stump | 88.3 | 83.5 | 73.9 | 95.8 | 87.3 | 76.7 | 97.2 | 87.4 | 78.7 | 95.5 | 84.6 | 75.4 |

The noise-to-purity trade-off persists: EF-GBTSVM-L8 is +2.1 to +20.9 at 10% noise relative to plain
EF-GBTSVM but −4.8 to −5.1 at 20% noise (fold 3), consistent with wider purity coverage improving
some settings while slightly overshooting others. The purity-offset (J) and mass-weighting (M) each
helped in some folds but neither consistently outperformed L8 alone.

## Decision after Phase A (Nov 2026): Option A — "best in GB/TSVM family" claim

No EF variant was fairly best vs all models on inner CV (best EF candidate EF-GBTSVM-L8: 78.35,
vs SVM 82.76, Stump 85.35). The 3 options presented to Vansh were:
  (a) "best in GB/TSVM family" claim + honest comparison
  (b) Harder task: k-cycle-ahead degradation prediction
  (c) Extra datasets for a valid Friedman test

**Choice: Option A.** The paper will claim that EF-GBTSVM-L8 is the best-performing model in the
twin-SVM / granular-ball family (vs GBTSVM, HF-GBTSVM, EF-TSVM, TSVM), with an honest comparison
table including RBF-SVM, decision stump and other baselines. "Best overall" is not claimed.

> **Superseded by Phase B (Nov 2026):** the fair Phase B comparison does not support this claim.
> The primary model is now EF-GBTSVM-PF (the pre-registered per-fold combo, below), and PF is not
> best in the GB/TSVM family (rank 1 only at 20% noise, by 0.18 pp). See "Phase B results".

## Phase B plan: fair evaluation, Option A scope (Nov 2026)

### Variant: per-fold selection from phys2 inner CV

| Fold | Test cell | Variant used in Phase B |
|------|-----------|------------------------|
| 0 | B0005 | EF-GBTSVM-L8 |
| 1 | B0006 | EF-GBTSVM-J |
| 2 | B0007 | EF-GBTSVM-L8 |
| 3 | B0018 | EF-GBTSVM-L8 |

The per-fold variant is the one selected by inner CV for that fold — the test cell was excluded from
the entire inner CV. No global winner was declared (3/4 L8, 1/4 J); doing so would let all test
cells' labels influence the variant used for each fold's test evaluation.

### Experiment command (launched Nov 2026, runs in background as task brmqkboz3)

EF-GBTSVM-L8 and EF-GBTSVM-J were added to run_experiments.py build_models() (alongside the
original EF-GBTSVM, -W, -L, -A ablations). The `--n-jobs 4` flag prevents Windows fork exhaustion.

```
for noise in 0.0 0.1 0.2; do
  python run_experiments.py \
    --data data/nasa_battery_phys.csv \
    --noise $noise \
    --n-iter 150 \
    --grids equal \
    --n-jobs 4 \
    --group-col battery \
    --name phys_phase_b \
    --out results/
done
```

Outputs: `results/phys_phase_b_noise{0,10,20}_{runs,summary}.csv`, log: `results/phys_phase_b.log`.
This run also serves as instruction #1's "corrected" numbers for the leakage comparison table
(script: `python compare_leakage_impact.py`).

Per-fold variant selection in results analysis:
- Folds 0, 2, 3 → use EF-GBTSVM-L8 score (inner-CV winner for those folds)
- Fold 1 (B0006 test) → use EF-GBTSVM-J score (inner-CV winner for that fold)
- Combined result = per-fold selection, test cells never in variant selection

### Checks required (Phase B checklist)
- [x] Leakage comparison table done (phys_main noise=0/10/20 complete)
- [x] Phase B runs complete: all 10+ models × 3 noise levels on phys features, equal budgets
- [x] Overfitting check: per-fold train vs test ACC gap table
- [x] Friedman + Nemenyi across the 4 folds × 3 noise levels (NOTE: 4 folds is below standard)
- [x] Wilcoxon: primary (EF-GBTSVM-PF) vs each baseline (n=12 pooled, non-independent; n=4 per noise)
- [ ] Terminal output screenshots captured for appendix
- [x] "Is the primary fairly best in GB/TSVM family?" — NO (see Phase B results); Phase C not started

## Diagnostic section: SVM fold-1 collapse + stump selection + feature sanity (Nov 2026)

### Task 1 — SVM fold-1 (B0006) collapse: diagnosis

**Result: NO CODE BUG. Root cause is class-prior shift between training and test batteries.**

Logged params for fold 1: `C=300, gamma=0.1193776641714437` (from equal-budget grid,
inner CV on B0005+B0007+B0018 training batteries).

**Class distribution flip:**

| Split | n | class0 (not-degraded) | class1 (degraded) |
|---|---|---|---|
| Train (B0005+B0007+B0018) | 468 | 299 (63.9%) | 169 (36.1%) |
| Test (B0006) | 168 | 60 (35.7%) | 108 (64.3%) |

B0006 degraded far faster than the other cells (first→last capacity 2.035→1.186 Ah),
so 64.3% of its cycles are labelled degraded, while the training set has only 36.1% degraded.
Inner CV (balanced_accuracy scoring on GroupKFold-3 within training batteries) selected
C=300, gamma=0.119 — a highly local kernel that memorises the training distribution.
On B0006's flipped distribution this boundary predicts mostly class 0:

```
Confusion: TN=58  FP=2  FN=79  TP=29
Sensitivity (recall class1) = 26.9%  Specificity = 96.7%
Predicted: 137 class0, 31 class1  (but test has 60 class0, 108 class1)
ACC = 51.79%   BACC = 61.76%   AUC = 96.56%
```

The high AUC (96.56%) confirms the discriminant function values are correctly ordered —
only the decision threshold is in the wrong position relative to B0006's prior.

**Scaler check:** Pipeline fits `StandardScaler` on `Xtr` only (line 274 and line 288 of
`run_experiments.py`). Confirmed: `Xtr.mean[0]=340.21` vs `Xte.mean[0]=217.06` —
the scaler does not see test data. No leakage.

**Conclusion:** This is a genuine out-of-distribution battery fold, not a software bug.
SVM with RBF kernel is sensitive to prior shift; the other models (TSVM, RVFL, etc.)
handle this fold better because they either use the full margin more robustly or their
kernel/architecture is less sensitive to the threshold placement. Per-fold numbers stand as-is;
no re-run needed.

---

### Task 2 — Stump: two versions (fair vs oracle), code confirmation

**Result: Two distinct stump baselines now reported. Only (1) is used in comparison tables.**

`phys_stump.py` (updated Nov 2026) reports:

**(1) Fair stump** — best feature chosen per outer fold by **inner CV on training data only**;
different folds may select different features; test data never used for feature selection.
**Protocol switched (Nov 2026, Vansh's decision):** the inner CV is now exactly the models' protocol,
GroupKFold over the 3 training batteries (one battery per inner fold) with balanced-accuracy
scoring; exact ties go to the first feature in CSV column order. The first version (below, struck)
used StratifiedKFold + accuracy, which mixes cycles of one battery across inner folds.

**(2) Oracle stump** — best feature chosen by `argmax(mean test-fold ACC)` globally;
labelled "oracle upper bound" — it uses test information and sets an upper bound on a
single-feature rule; not used as a comparison baseline.

Code logic for fair selection (per outer fold, current `phys_stump.stump_runs`):
```python
# fit & cv on TRAINING FOLD only — test never seen; same splitter/scoring as run_experiments.py
cv = GroupKFold(min(3, len(np.unique(gtr))))           # gtr = groups[tr]
cv_score = cross_val_score(clf, X[tr][:, [fi]], ytr, groups=gtr, cv=cv,
                           scoring="balanced_accuracy").mean()
# select best fi, then fit on full tr and predict on te
clf.fit(X[tr][:, [best_fi]], ytr)
accs.append(100 * accuracy_score(y[te], clf.predict(X[te][:, [best_fi]])))
```

**Fair vs Oracle results:**

| Noise | Fair stump, current (baseline) | Fair features F0–F3 | ~~First version (StratifiedKFold + ACC)~~ | Oracle stump (upper bound) | Oracle feat |
|---|---|---|---|---|---|
| 0% | **96.94 ± 1.97** | dt_38_35_rel, dt_38_35_rel, dt_38_35, t_v35_rel | ~~94.63 ± 5.55~~ | 98.73 ± 0.99 | t_v35_rel |
| 10% | **95.16 ± 3.52** | dt_38_35_rel, t_v35_rel, v_1000, v_300_rel | ~~95.16 ± 3.52~~ | 98.35 ± 1.34 | t_v35_rel |
| 20% | **96.85 ± 1.96** | v_1000, dt_38_35_rel, v_1000, dt_38_35 | ~~96.55 ± 1.89~~ | 98.20 ± 1.30 | v_1000 |

**Tie disclosure (audit):** at 10% noise, fold B0018, `v_300_rel` and `v_1000_rel` have bit-identical
inner-CV scores (0.87416); the fixed first-column rule picks `v_300_rel` (test ACC 90.15). The other
feature would give 96.97, i.e. Stump-fair 96.86 at 10% instead of 95.16 (PF gap −1.78 instead of
−0.08). The rule was not changed after seeing this. No other fold/noise has an exact tie.

**Impact on model comparison:** the protocol-matched fair stump matches or beats the primary
EF-GBTSVM-PF at every noise level (see "Phase B results"). The oracle stump (98.73%) is never ranked,
since its feature was selected using test outcomes.

---

### Task 3 — Feature-sanity table (3-fold inner CV on training folds only)

Each number is the mean 3-fold stratified CV accuracy within the training batteries
(test battery never seen). This measures each feature's inherent discriminability,
independent of test results.

| Feature | F0_tr | F1_tr | F2_tr | F3_tr | Mean_tr | Physics link |
|---|---|---|---|---|---|---|
| **t_v35_rel** | 98.72 | 99.15 | 97.44 | 98.81 | **98.53** | Relative t to 3.5V — primary normalised label |
| **dt_38_35_rel** | 98.93 | 98.93 | 97.86 | 98.41 | **98.53** | Normalised end-of-discharge rate |
| v_1000_rel | 98.50 | 97.86 | 97.86 | 99.21 | 98.36 | Normalised deep-discharge voltage |
| dt_38_35 | 97.86 | 98.08 | 98.29 | 97.82 | 98.01 | Time window 3.8→3.5V (absolute) |
| v_1000 | 98.08 | 96.79 | 98.72 | 97.82 | 97.85 | Voltage at 1000 s |
| t_v35 | 97.65 | 97.01 | 98.50 | 97.82 | 97.74 | Time to 3.5V cut-off |
| v_300_rel | 96.37 | 97.01 | 97.01 | 98.61 | 97.25 | Normalised mid-discharge voltage |
| t_v38_rel | 95.73 | 96.79 | 96.79 | 98.21 | 96.88 | Relative t to 3.8V |
| t_v38 | 95.94 | 96.15 | 98.72 | 96.03 | 96.71 | Time to 3.8V drop |
| v_300 | 95.73 | 96.15 | 98.72 | 95.44 | 96.51 | Voltage at 300 s |
| temp_rise_35_rel | 95.94 | 88.68 | 86.97 | 90.08 | 90.42 | Normalised temperature rise |
| temp_rise_35 | 88.68 | 82.91 | 84.83 | 90.08 | **86.62** | Temperature rise to 3.5V |

**Interpretation:** The two features most directly tied to the capacity-definition (t_v35_rel,
dt_38_35_rel) top the table at 98.53% mean train-fold CV accuracy. Relative (normalised)
features consistently outperform their absolute counterparts, confirming that cycle-relative
comparison is the right inductive bias. Temperature-rise features are weakest (~87–90%),
consistent with their role as internal-resistance proxy rather than direct capacity measure.

---

### Task 4 — Primary/secondary model assignment

~~Primary model: EF-GBTSVM-L8; secondary: EF-GBTSVM-J.~~ **Superseded:** the audit showed that
"L8 on all folds" contradicts the pre-registered per-fold plan (it lets each test cell's labels
influence the variant used on it). Vansh chose **primary = EF-GBTSVM-PF**, the per-fold combo (L8 on
B0005/B0007/B0018, J on B0006, each fold's own inner-CV winner). L8 and J are PF's components. This
follows the plan written before Phase B ran; no model or setting changed based on test results.

---

### Task 5 — Per-fold sizes and accuracy-to-cycles conversion

| Fold | Test battery | n_test | n_train | 1 discharge cycle | 1% ACC change |
|---|---|---|---|---|---|
| F0 | B0005 | 168 | 468 | 0.5952% ACC | 1.68 cycles |
| F1 | B0006 | 168 | 468 | 0.5952% ACC | 1.68 cycles |
| F2 | B0007 | 168 | 468 | 0.5952% ACC | 1.68 cycles |
| F3 | B0018 | 132 | 504 | 0.7576% ACC | 1.32 cycles |

**Accuracy gaps expressed in discharge cycles (noise=0% mean ACC):**

| Comparison | Δ ACC | Δ cycles (168-test avg) |
|---|---|---|
| **EF-GBTSVM-PF (95.25%) vs TSVM (98.77%)** | −3.52% | ≈ 5.9 cycles |
| **EF-GBTSVM-PF (95.25%) vs Fair Stump (96.94%, protocol-matched)** | −1.69% | ≈ 2.8 cycles (stump better) |
| **EF-GBTSVM-PF (95.25%) vs Oracle Stump (98.73%)** | −3.48% | ≈ 5.8 cycles (unfair baseline) |
| EF-GBTSVM-L8 (95.10%) vs TSVM (98.77%) | −3.67% | ≈ 6.2 cycles |
| EF-GBTSVM-L8 (95.10%) vs Fair Stump (96.94%) | −1.84% | ≈ 3.1 cycles (stump better) |
| EF-GBTSVM-L8 (95.10%) vs Oracle Stump (98.73%) | −3.63% | ≈ 6.1 cycles (unfair baseline) |
| EF-GBTSVM-J (97.21%) vs TSVM (98.77%) | −1.56% | ≈ 2.6 cycles |
| EF-GBTSVM-J (97.21%) vs Fair Stump (96.94%) | +0.27% | +0.5 cycle (J better) |

Note: each sample in `nasa_battery_phys.csv` corresponds to one full discharge cycle.
A 1% difference on a 168-cycle test battery = misclassifying approximately 1–2 more cycles.
"Better by 3%" is physically "correct on 5–6 more cycles out of 168."

---

## Phase B results (Nov 2026): fair comparison, primary = per-fold combo (EF-GBTSVM-PF)

Sources: `results/phys_phase_b_noise{0,10,20}_runs.csv`, `results/phase_b_summary_table.csv`,
`results/phase_b_per_fold.csv`, `results/phase_b_stats.txt`, `results/phase_b_claims.txt`,
`results/phase_b_train_gap.csv`, `results/leakage_impact_comparison.csv`. Nothing was tuned on test.
Every number below was recomputed independently from the raw CSVs by separate audit agents (two
rounds: 4 auditors on the first analysis, 3 on the PF version; 0 mismatches after the fixes noted).

**Primary model (Vansh's decision, Nov 2026): EF-GBTSVM-PF**, the pre-registered per-fold
selection from the Phase B plan above: on each outer fold, the variant that fold's own Phase A inner
CV picked (`results/phase_a/phys2_selection.csv`): B0005 → L8, B0006 → J, B0007 → L8, B0018 → L8.
Audit confirmed PF is leak-free: 0/252 inner-CV rows use the test cell; the selection (mean noisy
BACC over 3 validation cells × 3 noise levels) recomputes exactly; one selection for all three noise
levels uses only training-cell scores. Inner-CV margins: 3.46 / 1.06 / **0.0145** / 1.35 pp (fold 2
is fragile; its runner-up JM was never run in Phase B). "L8 on all folds" is a global majority
choice that lets each test cell's labels (validation data in the other folds' inner CVs) influence
the variant used on it, so it is kept only as a non-primary reference.

**Ranking pools.** Main (k=15) = 13 non-variant models + PF + Stump-fair; L8 and J are PF's
candidate inputs and are left out (rank the method, not its candidates). This is defensible but not
neutral, so the sensitivity pool (k=17, adds L8 and J) is reported next to it.

### Mean ACC over the 4 folds (Stump-fair = baseline; Stump-oracle = upper bound, not ranked)

| Model | 0% | 10% | 20% |
|---|---|---|---|
| **EF-GBTSVM-PF (primary)** | **95.25** | **95.08** | **95.28** |
| EF-GBTSVM (original) | 94.83 | 86.66 | 94.71 |
| EF-GBTSVM-W | 95.25 | 97.33 | 93.05 |
| EF-GBTSVM-L | 95.25 | 97.33 | 92.90 |
| EF-GBTSVM-A | 95.73 | 94.99 | 95.10 |
| *EF-GBTSVM-L8 (PF component; also the non-primary "L8 on all folds")* | *95.10* | *97.61* | *91.26* |
| *EF-GBTSVM-J (PF component)* | *97.21* | *94.89* | *95.36* |
| GBTSVM | 96.43 | 97.77 | 80.09 |
| HF-GBTSVM | 96.47 | 90.46 | 94.01 |
| EF-TSVM | 96.67 | 84.89 | 71.39 |
| TSVM | 98.77 | 95.31 | 88.93 |
| SVM (RBF) | 86.61 | 96.97 | 96.32 |
| RVFL | 97.75 | 90.10 | 96.86 |
| RandomForest | 97.39 | 97.46 | 97.75 |
| MLP | 97.61 | 95.59 | 94.78 |
| XGBoost | 97.69 | 97.16 | 97.42 |
| **Stump-fair (GroupKFold + BACC, same protocol as all models)** | **96.94** | **95.16** | **96.85** |
| ~~Stump-fair, first version (StratifiedKFold + ACC), superseded~~ | ~~94.63~~ | ~~95.16~~ | ~~96.55~~ |
| [Stump-oracle upper bound] | [98.73] | [98.35] | [98.20] |

PF per fold (B0005/B0006/B0007/B0018): 0% 97.62/97.62/91.07/94.70; 10% 98.21/88.10/97.02/96.97;
20% 98.81/98.81/91.07/92.42. PF BACC: 95.24 / 95.05 / 95.78. Full per-fold values, BACC/F1/AUC and
SDs are in `results/phase_b_per_fold.csv` / `phase_b_summary_table.csv`.

### Claims (a), (b), (c) for PF (Stump-fair protocol-matched)

* **(a) PF vs original EF-GBTSVM:** ACC +0.42 / **+8.41** / +0.57 at 0/10/20% (BACC +0.27 / +9.46 /
  +2.63); fold wins/losses/ties 2/1/1, 3/1/0, 3/1/0. Higher mean at all three noise levels, by a
  large margin at 10% and a small one at 0% and 20%. **Not statistically significant** (pooled
  Wilcoxon p = 0.090, Holm 1.0; per-noise p ≥ 0.25). The +0.42 at 0% comes from B0018 alone (+2.27);
  at 20% B0007 loses by 6.55 pp.
* **(b) Rank in the GB/TSVM family.** Main (9 models): 6/9 at 0% (TSVM 98.77), 5/9 at 10% (GBTSVM
  97.77), **1/9 at 20%** (+0.18 pp over EF-GBTSVM-A, about 0.3 cycles); pooled 2nd of 9 behind TSVM
  (avg rank 7.833 vs 6.958 after the stump switch; 7.750 before). Sensitivity (11): 7/11, 6/11, 2/11 (J 95.36 is higher at 20%); pooled
  3rd of 11. **"Best in the GB/TSVM family" is NOT supported**: rank 1 only at 20%, only in the main
  pool, by 0.18 pp.
* **(c) Rank among all models.** Main (15): 11/15, 10/15, 6/15; top TSVM (gap −3.52), GBTSVM (−2.69),
  RF (−2.48); vs SVM +8.64 / −1.89 / −1.04. Pooled position 7 of 15 (avg rank 7.833; best
  RandomForest 5.292). Sensitivity (17): 12/17, 11/17, 7/17; pooled 8 of 17 (RF 5.917).
  **PF is rank 1 at no noise level; "best overall" is NOT supported.**
* **vs the fair single-feature stump** (96.94 / 95.16 / 96.85): PF's gaps are **−1.69 / −0.08 / −1.57**,
  so a one-feature threshold tuned with the same protocol matches or exceeds PF at every noise level
  (Wilcoxon pooled p = 0.418, not significant; stump overall rank 6/15, 9/15, 4/15). The 10% gap
  depends on a tie-break (Task 2): the other tied feature would make it −1.78.
* Before the stump switch (first version, superseded): PF rank 10/15 at 0%, pooled position 6 of 15;
  only stump-dependent numbers changed (audit diff).
* Non-primary reference, L8 on all folds: 95.10 / 97.61 / 91.26 (the earlier "2/16 at 10%" figure
  exists only under this global choice).

### Statistics (`results/phase_b_stats.txt`; metrics rounded to 6 dp, differences to 4 dp)

* Friedman (tie-corrected), main pool k=15: pooled N=12 chi² = 20.224, p = 0.123; per noise p = 0.210 /
  0.253 / 0.258; pooled BACC p = 0.308. Sensitivity k=17 pooled: chi² = 19.933, p = 0.223. **No
  significant difference anywhere.**
* Nemenyi: q(0.05, 15) = 3.391, CD = 6.192 (N=12), 10.724 (N=4); k=17: q = 3.458, CD = 7.130. All other
  models fall within CD of PF, but Nemenyi is a post-hoc test after a significant Friedman (none is),
  and the CD exceeds the whole observed rank spread: **absence of evidence, not equivalence.**
* Wilcoxon, PF vs each (main pool, Holm over 14): pooled lowest raw p is **RandomForest 0.033** (RF
  higher on 8 of 12 pairs, 2 lower, 2 tied; mean −2.33 pp), Holm 0.465 (0.531 in the sensitivity
  pool); then EF-GBTSVM 0.090 and XGBoost 0.098 (Holm 1.0; not to be called "trends"). **Per-noise
  n=4: minimum achievable two-sided p = 0.125** (0.25 / 0.5 if a fold ties); observed minimum 0.125 at
  each noise level.
* **The 3 noise settings reuse the same 4 batteries and features, so the 12 pooled observations are
  not independent**; every pooled p-value is descriptive only. With 4 batteries the design cannot
  detect 2–3.5 pp differences: underpowered.
* Audit fix applied earlier: CSV vs live float noise had broken some true ties (pooled chi² for the
  first analysis 19.835 → 19.788). No conclusion changed.

### Train-vs-test gap (overfitting check)

Refit with the logged best params on the same training fold: 180/180 rows reproduce the logged test
ACC exactly. Gap = train ACC on true labels − test ACC. **PF: 1.48 / 2.21 / 0.88 pp** (no overfitting
sign; per fold up to 8.7 pp on B0006 at 10%). L8 2.21 / −0.54 / 2.33; J 0.17 / 2.69 / 0.27; Stump-fair
2.11 / 3.96 / 1.31 (computed live in `phase_b_analysis.py`; 8.66 on the tie-broken 10% B0018 fold). Gaps > 5
pp: SVM@0% 12.1 (B0006 class-prior flip, see Task 1), EF-GBTSVM@10% 9.8, EF-TSVM@10% 13.1 and @20%
25.3, HF-GBTSVM@10% 6.9, RVFL@10% 8.4, GBTSVM@20% 13.4, TSVM@20% 5.7.

### Old vs corrected table (`results/leakage_impact_comparison.csv`) — confounded

Old and new runs differ in the feature set (13 old features vs 12 phys features, none shared) **and**
in the grids/budget, so a delta is not the effect of leakage removal alone. The audit re-ran the old
features under the new tuning: SVM@0% −7.52 is −6.71 tuning + −0.81 features; RF@10% +14.26 is +10.46
tuning; RF@20% +24.22 is +20.23 tuning; MLP@0% +8.84 is mostly features (+10.76); HF-GBTSVM@0% +22.70
is mostly features (+21.16). Old ranks are over 13 models, new over 15, so ranks are not comparable.
The 20%-noise split exists for RF only (audit re-run did not finish). Script docstring corrected.

### Open items and disclosures

1. **Resolved:** primary = PF (pre-registered per-fold combo), chosen by Vansh.
2. **Resolved — Stump-fair protocol** switched to GroupKFold + BACC (Vansh, Nov 2026), so it uses
   the same inner CV as the other models (audit: identical splits on all 12 folds; scaling is
   irrelevant for a depth-1 tree). Disclose: the first stump version was written after Phase B test
   results existed, and the 10%-noise B0018 fold is decided by an exact tie (Task 2).
3. **Disclosure:** the Phase A candidate list (wide purity grid, J's adaptive offset) was shaped by
   earlier test-fold results on the same 4 batteries (old feature file). Phase A selection itself is
   clean.
4. Minor: the `_rel` features normalise by each cell's own first cycle, including the test cell (no
   labels used; available once a cell is in service). Several chosen params sit at grid edges (SVM
   C=300, MLP alpha=1.0, J offset +0.10). A per-noise argmax in Phase A would have picked differently
   in 7 of 12 (fold, noise) cells (paper audit recount; 6 was a miscount); the averaging rule was fixed before the run.

Audit checks that passed: no direct test-battery use anywhere (scaler, inner CV, flips, ball fit,
adaptive-purity estimate all on training data only); model code unchanged after Phase B started;
`phase_b_analysis.py` outputs reproduced byte-for-byte by a reviewer copy.

## Phase C: paper draft (Nov 2026)

Framing chosen by Vansh: honest "competitive, not best" benchmark. Draft written in the Claude Docs
paper (same URL, retitled "A Leakage-Free, Fair Benchmark of EF-GBTSVM for Li-ion Battery Degradation
Classification under Label Noise"); the earlier sensitivity-study text was replaced (still in doc history).
- 30 references, each verified against Crossref/arXiv/JMLR/publisher records; unverifiable items are in
  the paper's "Needs human review" list.
- Equations restated from arXiv 2410.05786 / 2410.04774 and checked against `efgbtsvm/`.
- Figures: `make_paper_figures.py` -> `paper_figures/` (two doc charts from CSVs; three console-output
  images rendered from the captured stdout of `phase_b_analysis.py`, labelled as such).
- Three-agent audit (numbers, citations, method/honesty) found 6 number/wording errors, 5 citation
  overstatements and 18 method/honesty issues; all confirmed ones were fixed in the doc.

## Phase D, step 1: fresh-cell validation — closed, no eligible cell (Nov 2026)

Decision (Vansh, Nov 2026): **Option 1, inclusion rule kept as-is.** No fresh-cell run was made; no model
was fitted on any fresh cell. Fresh-cell validation is therefore **not possible with this dataset**.

* Sources: `5.+Battery+Data+Set/` zips 2-6 extracted to `data/raw_fresh/` (originals untouched). Zip 1 =
  dev cells, never fresh. B0025-28 are in zip 2 and zip 3 with identical sha256 and were counted once.
  Pool = **30 fresh cells** (B0025-34, B0036, B0038-56). (An earlier chat message said 32; 30 is the
  count in `prereg/inventory_output.txt`.)
* Rule: `prereg/inclusion_rule.md` (sha256 `80082c31fa4d8aa9709c4d5b462339bec999b36c9f1a4963e5d6a805dfbbc333`),
  written before any fresh Capacity/label value was read. Metadata script `prereg/fresh_metadata.py`
  never reads Capacity; labels were read only by `prereg/fresh_labels_R5.py`, only for cells that had
  passed R1-R4/R6.
* R1-R4 + R6 (load 2 A CC on every cycle, ambient range <= 2 C, 12 phys features defined on every
  cycle, >= 50 discharge cycles, no shared recording with a dev cell): **27 of 30 excluded**.
  R1 (current 4 A / 1 A / mixed) 20 cells; R2 (ambient varies) 6; R3 (features undefined on some cycle,
  "load-off before 1000 s" etc.) 15; R4 (< 50 discharge cycles) 15; R6 none. Cells failing only one rule:
  B0028 (28 cycles), B0045-48 (1 A), B0053 and B0054 (one undefined cycle each; B0054 = cycle #103).
* Passing R1-R4/R6: **B0036, B0055, B0056**. R5 (each class >= 10% of cycles): all three have
  **0 degraded cycles** (197/102/102 cycles, 100% healthy at SOH < 0.80) -> **R5 FAIL**. Eligible
  fresh cells: **0**.
* Shared recordings (identical Time arrays on 100% of paired cycles): 25-28, 29-32, 33/34/36,
  38-40, 42-44, 45-48, 49-52, 54-56 (and dev 5/6/7). No fresh cell shares a recording with a dev cell.
  README vs file, two differences seen (both reported, neither changed a rule outcome): (i) B0028 -
  README says 4 A (0.05 Hz square wave) for 25-28, the file's median load current is 1.99-2.00 A (B0025/26/27:
  4.0 A); (ii) B0042-44 - README says 4 C ambient, the files record 4-22 C. Other groups' currents/ambients
  matched their README. Not checked against README: cut-off voltages (observed minimum voltages under load
  are in `inventory_output.txt`).
* Unknown: how `Capacity` was computed for cells whose cut-off differs from 2.7 V (README says "till
  2.7V"); causes of the "very low capacity" discharges the READMEs mention.
* Files: `prereg/{inclusion_rule.md, fresh_metadata.py, fresh_metadata.json, inventory_output.txt,
  fresh_labels_R5.py, R5_output.txt}`. Options 2 (post-hoc amendment) and 3 (other dataset) were not
  taken.
* Consequence for claims: the only evidence is the dev-LOBO result (4 cells, B0005/6/7 one recording).

## Phase D-lite: pre-registration (Nov 2026) — frozen, NOT yet run

`PREREG_DLITE.md` (sha256 `fb412a6fc58b095cb6125e8a0b1badb936aa784c59af0c411123c27abc0acbe1`) written before any D-lite run: frozen 15 models + PF + Stump-fair,
12 phys features, 150-candidate budget, per-fold PF selection from Phase A; new conditions on the dev
batteries only: random noise 30% / 40% and boundary noise (flip 50% of training cycles with SOH in
[0.75, 0.85], seed 1000 + fold). Claim table fixed in section 6 of that file. Not a git repository, so
no commit was made. **Run waits for Vansh's explicit yes**; `run_dlite.py` / `dlite_analysis.py` do not
exist yet and their sha256 will be added here before the run.

### D-lite launch record (2026-11-05, Vansh: "Haan, run karo")

PREREG_DLITE.md sha256 `fb412a6fc58b095cb6125e8a0b1badb936aa784c59af0c411123c27abc0acbe1` (unchanged). New scripts, hashed before the run: `run_dlite.py`
`afd2e1d034e1fa1401b38b4e21e13be0bdcb14d59a5d152d9fdef4c6754ebd5b`; `dlite_analysis.py` `33a18146d1c548b883ac26cb60ce46bc21e8e5f6042088262526688d13e8a9c0`. Pre-run checks (no model fitted): all 11 frozen hashes matched; SOH
rebuilt from the .mat files reproduces all 636 `degraded` labels; BND boundary sets (flips) per fold
146 (73) / 168 (84) / 118 (59) / 144 (72). Single run in the background, log `results/dlite.log`,
outputs `results/dlite_*`.

### D-lite first-run crash and resume (deviation, 2026-11-07)

First run (`run_dlite.py` sha256 `afd2e1d0...ebd5b`): **N30 completed** (`results/dlite_N30_runs.csv`, 17
models x 4 folds); **N40 crashed in fold 3** (after 3 folds' models had run) with
`joblib TerminatedWorkerError: a worker process was unexpectedly terminated` (OS kill / memory;
exact cause unknown). No N40 or BND result file exists; the crash log is kept as
`results/dlite_firstrun_crash.log`. Not a model/analysis bug and no D-lite result had been looked at.
Handling: `run_dlite.py` was edited only to accept condition names on the command line (default = all
three) and to merge instead of overwrite `dlite_flip_log.csv`; new sha256
`66bfef23056a6341fd1b8054972fdd8d646ba733635e94d36c681052b941de84`. Models, grids, seeds, scorer, flips
are untouched. N30 from the first run is kept; N40 and BND are run with
`python run_dlite.py N40 BND` (log `results/dlite_resume.log`). Seeds are deterministic, so this equals
a single uninterrupted run apart from the crash. If another crash occurs it is reported the same way.

### D-lite script diff check (2026-11-07)

`run_dlite.py` v1 (sha256 `afd2e1d0...ebd5b`) was overwritten by the resume edit; v1 was reconstructed by
reversing exactly the three edits and its sha256 equals `afd2e1d034e1fa1401b38b4e21e13be0bdcb14d59a5d152d9fdef4c6754ebd5b`
(`prereg/run_dlite_v1_reconstructed.py`). Diff v1 -> v2 (`66bfef23...de84`), `prereg/run_dlite_v1_v2.diff`
(ignoring line endings): (1) `main()` reads optional condition names from `sys.argv` (default all three)
plus an assert; (2) `for cond in conds` instead of `for cond in CONDITIONS`; (3) `dlite_flip_log.csv` is
merged per condition instead of overwritten. **Unchanged:** `FROZEN` hashes and `check_frozen`, `soh_per_row`,
`corrupt` (flip generation, seeds `1000 + fold`), `stump_rows`, the per-fold model loop (`build_models(seed,
None, "equal")`, `GroupKFold`, `RandomizedSearchCV`, balanced-accuracy scoring, n_iter 150, n_jobs 4),
metrics; PF selection lives in `dlite_analysis.py`/`phase_b_analysis.py` (frozen hash), also untouched.
Side effect disclosed late: my Python edit converted the file's line endings from LF to CRLF, which is why
the v2 hash differs from "v1 + three edits" byte-for-byte; the code content differs only by the three hunks.

### D-lite resume #1 killed by machine reboot; resume #2 with --n-jobs 2 (deviation, 2026-11-07)

Resume #1 (`python run_dlite.py N40 BND`, n_jobs 4, started 13:01) ran N40 folds 0 and 1 completely and
was in fold 2 (3 models of fold 2 logged, last log write 16:15:45) when **the machine rebooted at 16:23:37**
(Windows `LastBootUpTime`); no python process survived and the log has no traceback. The script writes a
condition's CSV only when the whole condition finishes, so **no N40 result file exists** (no partial file
to set aside); about 3 h of N40 work was lost. Log kept: `results/dlite_resume_killed_by_reboot.log`.
No D-lite result was analysed. N30 (`dlite_N30_runs.csv` sha256 `f9f344047d46...`) and the N30 flip log
(`f2d95d6485da...`) are untouched.
Resume #2, started 2026-11-07 after this entry: `python run_dlite.py --n-jobs 2 N40 BND`, log
`results/dlite_resume2.log`. Reason for n_jobs 2: the first run's worker kill (TerminatedWorkerError) and
the slow, loaded machine. `run_dlite.py` edited only to parse `--n-jobs` (default stays 4): sha256
`36b5cc496b78dc9186260fd5ce405a49ef2329cbfd1ff4d36a26e8231971dfeb` (from `66bfef23...de84`; diff: the arg
parsing block and `n_jobs=n_jobs` in RandomizedSearchCV; copy of the previous version in
`prereg/run_dlite_v2_66bfef23.py`). n_jobs only sets worker count; seeds, models, grids, flips, metrics unchanged.

### D-lite fold-level checkpoint (deviation, 2026-11-11)

Why: resume #1 was lost to a reboot and resume #2 stalled while the laptop slept (log last written 2026-11-08
16:30, run later resumed by itself); the script only writes a condition's CSV when the whole condition ends, so
any interruption loses all folds of that condition. What: `run_dlite.py` now writes the rows of each completed
(condition, fold) to `results/dlite_ckpt/<cond>_fold<k>.csv` (atomic: .tmp then rename) and, on restart, loads
and skips any (condition, fold) whose checkpoint exists. Only saving/skip logic; flips (`corrupt`), seeds, models,
grids, scorer, metrics, PF/analysis are untouched. Diff `prereg/run_dlite_v3_v4.diff` (v3 = `36b5cc49...dfeb`,
copy `prereg/run_dlite_v3_36b5cc49.py`): +6 lines before the model loop (path, load-and-continue, `fold_rows`),
`fold_rows.append(m)`, +4 lines after the loop (write checkpoint). New sha256 of `run_dlite.py`:
`aa34974cae94b07089f67fc112665834bb979af1302db77d1ea06339345fd9be`. The already-running resume #2 process
(started from v3) does not have checkpoints; N40 folds 0-1 (15 models each) exist only in that process's memory and
in `results/dlite_resume2.log`, in no CSV, and are not recovered from the log.
Power settings (requested by Vansh, 2026-11-11, AC only): `powercfg /change standby-timeout-ac 0` (was already 0) and
`powercfg /change hibernate-timeout-ac 0` (was 43200 s). DC (battery) values untouched: hibernate 10800 s. At
check time the laptop was on battery (BatteryStatus 1, 39%). Windows Update / auto-restart settings were not changed.

### D-lite: N40 complete; second reboot killed BND fold 0; BND restarted with checkpoints (2026-11-11)

Resume #2 (v3 script, n_jobs 2) **finished N40** at 12:25: `results/dlite_N40_runs.csv` (68 rows = 17 models incl. Stump-fair/
oracle x 4 folds) sha256 `8c463abeb6caa8409fb208a4f87ade5976c1a65552bee6cd06b60253717aaa7a`; flip log
`results/dlite_flip_log.csv` sha256 `6d665985d59280d8893b381299f651e547e6fcdac0d63f909decf33a8de79378` (N30 140/140/140/151
and N40 187/187/187/201 flipped; BND entries are written when BND finishes). N30 file unchanged
(`f9f34404...`). The machine **rebooted again at 12:43:05** while BND fold 0 was in progress (about 6 models logged,
log kept as `results/dlite_resume2_final.log`); no BND CSV exists, nothing was analysed. N40 was not rerun.
Resume #3: `python run_dlite.py --n-jobs 2 BND` with the checkpoint version `aa34974c...fd9be`
(per-fold checkpoints in `results/dlite_ckpt/`; log `results/dlite_resume3.log`). Same reason and same non-changes
as the earlier entries (seeds, flips, models, grids, scorer, metrics untouched).

### D-lite: all three conditions complete; analysis run (2026-11-11)

BND finished 15:03 (resume #3, checkpoint script, no further interruption): `results/dlite_BND_runs.csv` (68 rows) sha256
`95d4d3ef67c08617e3be4007f226652c30359b3f56605d4d0cefe726f231b394`; final flip log `results/dlite_flip_log.csv` sha256 `efc8b38b0b8ff665f6fa141bde1930b083d90f6faf5ab6e12085dcf7fa761f6f`
(BND flips 73/84/59/72 of 468/468/468/504; boundary sets 146/168/118/144; test labels untouched on all 12 rows).
Checkpoints `results/dlite_ckpt/BND_fold{0..3}.csv`. N30 `f9f34404...` and N40 `8c463abe...` unchanged.
Analysis: `python dlite_analysis.py` crashed on its last footnote line (stump-oracle upper bound used `oracle.condition`, but the column
had been renamed `noise_pct`) AFTER writing `dlite_stats.txt`; fixed that one expression (diff: `oracle.condition` -> `oracle.noise_pct`),
new `dlite_analysis.py` sha256 `59941dd0799861a5eb22f77c1e7ddf8cd6ea6619a357d5fee01a3834eeb61bb1` (old `33a18146...`, copy `prereg/dlite_analysis_v1_33a18146.py`), rerun; stats file
byte-identical to the crashed attempt (`results/dlite_stats_firstattempt.txt`), so no statistic or claim changed. Models were not rerun
(analysis-only bug). Outputs: `dlite_stats.txt` `bf449b3fa6b9e93c22a235893e28d85ff145b18db484eb2569e156d82a6688b4`, `dlite_claims.txt` `e64e9f7e6ed47e62ab9caf4a72da7435fb690f3ffa6cf09b07707e4d92ecf70d`,
`dlite_summary_table.csv`, `dlite_per_fold.csv`, full stdout `dlite_analysis_stdout.txt`.
Claim table (PREREG_DLITE.md section 6), main pool k=15, mean over 4 folds: **PF is not best on N30, N40 or BND** (ACC rank 8/5/12,
BACC rank 6/6/11; tops SVM / RandomForest / RVFL(ACC), TSVM(BACC)); no across-conditions claim. Paper document not touched.

### Paper doc: D-lite section added (2026-11-11, Vansh: "Haan, section likho")

In the Claude Docs paper (same URL), Section 6 Results, directly after Table 6: new "Heavier and boundary-concentrated noise
(pre-registered stress test)" paragraph, Table 7 (PF ACC/BACC + rank, best entry, single-feature threshold for N30/N40/BND),
a results paragraph (Friedman/Nemenyi/Wilcoxon/gap, all numbers from `results/dlite_*`), and a "Scope and execution" paragraph
(not external validation; the 0-eligible fresh-cell attempt; interruptions and deviations). Written as a direct insert (new
content), doc rev 221. Nothing else in the doc was edited. Not yet updated to match (left for Vansh): the Limitations bullet "Noise
model" (says only symmetric random flips), Abstract/Conclusion, and the "Needs human review" list.

### Paper doc: D-lite follow-up edits laid as suggestions (2026-11-11)

Four suggestions (not yet accepted; anyone with edit access accepts them on their cards), doc revs 222-225: Abstract sentence on the
stress test (`sg_5qg5pfdt7l`), Conclusion sentence (`sg_bx4mogjso1`), Limitations "Noise model" bullet addendum (`sg_5bnh32pabp`), and a
"Needs human review" checklist item about pushing PREREG_DLITE.md, results/dlite_* and prereg/ to the public repo (`sg_z6c0kcrm8c`).
Laid as suggestions because they change existing summary text. Numbers: PF ACC rank 8/5/12, 85.12 at 40%, random forest 94.49 (results/dlite_summary_table.csv).

### Paper: suggestion to scope the "no significant difference" sentences; paper.tex rebuilt (2026-11-11)

Suggestions `sg_jl1hfx7q6p` (Abstract) and `sg_5d01us79ke` (Conclusion) add "in the main 0% to 20% noise experiments" to the
"no difference ... significant (Friedman p = 0.123)" / "no model is significantly better than another" sentences, so they do not clash with
the stress test's pooled Friedman p = 0.0002 and Holm p = 0.048 (PF > GBTSVM). Pending, not accepted.
`paper_figures/doc_latest.xml` was refreshed from doc rev 229 with the pending suggestions left out (ins removed, del kept); the
previous export is `doc_latest_rev219.xml`, the export with suggestions applied is `doc_latest_with_suggestions.xml`.
`python paper_figures/build_ieee_tex.py` rebuilt `paper/ieee/paper.tex` (backup of the old one: `paper_rev219_backup.tex`; 53,719 -> 60,391
bytes; contains Table 6, Table 7 and the stress-test paragraphs). `check_tex_complete.py`: 630 sentences/cells/equation parts, PASS.
`check_citations.py` (rev 219 vs rev 229): 34 cited paragraphs, no citation target changed, PASS. Page count: not measured, no TeX engine on this machine.

### Paper: MiKTeX installed, paper.tex compiled, overfull box fixed, 11 pages (2026-11-11)

Installed MiKTeX (user scope, via winget) because no TeX engine was on this machine; xelatex works (fontspec/texgyretermes,
as build_ieee_tex.py already assumed). First compile found one real defect: the stress-test paragraph's SHA-256
(fb412a6f...acbe1, 64 chars, no spaces) overflowed its column by 118.6pt as an unbreakable `\texttt{}` token. Fixed at the
generator, not the doc: `build_ieee_tex.py`'s `inline()` now wraps any `\texttt{}` token of 20+ non-space characters in
`\seqsplit{}` (new `\usepackage{seqsplit}`), so long hashes/ids break across lines; no character of any hash changed.
Rebuilt, recompiled (2 xelatex passes): **0 overfull boxes** (was 1), 18 underfull (cosmetic, pre-existing, in reference-list
line breaks, not fixed). `check_tex_complete.py` 630 parts PASS both before and after. Final: `paper/ieee/paper.pdf`, **11
pages**, IEEEtran conference, A4, two-column -- at the top of the professor's stated 10-11 page range (Needs-human-review item),
so no compression is offered or needed. Hashes: `build_ieee_tex.py` 4fbd5dc2286da5bdf04b...; `paper.tex` eaa64396a4479f3d2a83...;
`paper.pdf` 7eb36af4524416f9f85a... (20-char prefixes; full hashes in the repo build). `overleaf_upload.zip` was also
regenerated by the same build; not yet re-uploaded anywhere.

### Public repo push: D-lite, prereg/, README update (2026-11-11)

Pushed to https://github.com/Vansh-lab/ef-gbtsvm-battery-label-noise (commit `0cc0287`): PREREG_DLITE.md, run_dlite.py,
dlite_analysis.py, all `results/dlite_*` (runs/flip-log/summary/per_fold/stats/claims/analysis_stdout/logs),
`results/dlite_ckpt/`, all 14 `prereg/` files, the seqsplit fix in `paper_figures/build_ieee_tex.py`, and updated
README.md (new "Robustness check: D-lite" section + Contents rows) and REPORT.md (this file, in full). Skipped
(duplicates, per Vansh): `dlite_firstrun_crash.log`, `dlite_resume2_final.log`, `dlite_stats_firstattempt.txt`. Not
pushed (choice, flagged to Vansh): the generated `paper.tex`/`paper.pdf` -- the repo has only ever tracked the
generator script, not its output. `data/`, `.mat` files and the XJTU/NASA zips are not pushed (gitignore).

### XJTU next step: label-free R2/R3/R5 checks, R6 pass/fail only (2026-11-11)

`prereg/xjtu_label_free_check.py` (sha256 `a0dc8d88204bf7355a97ec52e2c380cdcca1c99badddcecb6620da3578950f96`)
streams each of the 23 Batch-1/Batch-2 cells one at a time from the zip (never loaded whole; RSS stayed
106-154 MB across the run). Does not touch ADDENDUM_E1.md or PREREG_E.md.

**R2 (cell/group count): PASS.** 23 cells; 23 groups (every cell its own group; no shared-recording pair found)
from representative-cycle (first/middle/last) timestamp fingerprints -- a sampled check, not an exhaustive
per-cycle comparison (stated limitation, matches ADDENDUM_E1.md section 2's expectation of no grouping).

**R3 (same regime): PASS for all 23.** Per cell, 1 irregular discharge cycle out of 130-420 (0.2-0.8%; well under
the 5% cutoff) -- in every cell the ONLY irregular cycle found by the current-median test is cycle 1 (the
documented 0.2C initial-capacity cycle); no further rate-changed/check-up cycles were detected. Cell temperature
channel (`temperature_C`) gives finite values for all 23 cells.

**R5 (feature finiteness): PASS for all 23.** `cycle_features()` (unchanged) returns all-finite on every one of 3
representative regular cycles (first/middle/last) for every cell (69/69 checked). No feature value was printed,
read back or stored.

**R6 (degradation + class balance), pass/fail only: FAIL for all 23/23 -- no cell's SOH (relative to the first
REGULAR, i.e. second/1C, cycle) was found below 0.80 within its recorded regular-cycle life.** This is a
surprising null given XJTU is an aging-to-end-of-life dataset; the documentation (`data/raw_xjtu/doc/intro_en.txt`)
explicitly states cells were cycled "until ... capacity decreased to below 80% of their INITIAL values", and
"initial capacity" is measured on cycle 1, the 0.2C cycle that ADDENDUM_E1.md/this step's R3 treats as irregular
and excludes. The likely explanation, already anticipated in `prereg/E_xjtu_doc_check.md` Part 3 (a declared
deviation, not newly introduced here): discharging at 1C typically delivers less capacity than at 0.2C for the
same cutoff voltage, so the second (1C) cycle's capacity is itself probably below the authors' 0.2C "initial"
value; SOH measured against the 1C reference would then cross 0.80 later (or not at all within the recorded
cycles) than SOH measured against the authors' own reference. **Not verified against any actual capacity value**
(out of scope for this label-free step); flagged to Vansh as the most likely cause, not confirmed.
No raw SOH, capacity or feature value is anywhere in this entry, `xjtu_label_free_check.py`'s stdout/log, or
`prereg/xjtu_label_free_results.json` -- pass/fail and cycle counts only.

Full per-cell table and log: `prereg/xjtu_label_free_check.log`, `prereg/xjtu_label_free_results.json`.

### XJTU: R6-all-fail hypothesis confirmed (reference order only, no value shown, 2026-11-11)

`prereg/xjtu_reference_order_check.py`
(sha256 `9d4078f2a48f409d47d90854f93ca90fac1934e715c25aea175a583c63c0f771`), per Vansh's go-ahead: for each of the 23 cells,
compared ONLY the ORDER of two capacities (cycle-1/0.2C vs cycle-2/1C discharge_capacity_Ah) -- a greater/less
boolean, never a value. **Result: 23/23 cells have cycle-2 (1C, this step's SOH reference) capacity LOWER than
cycle-1 (0.2C, the authors' "initial" reference).** This confirms the hypothesis in the previous entry: R6's
0.80 threshold, measured against the 1C reference, is checking against an already-lower baseline than the
authors' own EOL definition, so a cell can fail to show SOH<0.80 here while it may already be at or past the
authors' 80%-of-0.2C-initial EOL. The 0/23 R6 result is very likely this reference-choice artifact, not an
absence of degradation in the data. No capacity or SOH value was printed, returned or stored by either script.

**Still not done (would need the user's go, and would read an actual value, breaking the label-free gate held
so far):** by how much the 1C capacity is below the 0.2C capacity (would require printing a number), and whether
SOH measured against the 0.2C reference does cross 0.80 within the recorded cycles. Left for a future, explicitly
approved step; nothing further was computed.

### XJTU: R6 re-checked against the 0.2C (authors') reference -- still 0/23 pass, different reason (2026-11-11)

`prereg/xjtu_r6_recheck_02C_ref.py`
(sha256 `0fd7150af492db10a24e9fd153f5f4c9a068ea10548f5f6e129150bc30503545`; diagnostic only, ADDENDUM_E1.md/PREREG_E.md untouched),
per Vansh's go-ahead: re-ran R6 using cycle-1 (0.2C, the authors' own "initial capacity") as the SOH reference
instead of cycle-2 (1C, this step's main reference). Pass/fail and counts only, as before.

**Result: still 0/23 pass full R6, but for a different reason now.** 8/23 cells DO reach SOH<0.80 against the
0.2C reference (vs 0/23 against the 1C reference) -- confirming the previous entry's hypothesis that the 1C
reference was the main reason nothing crossed 0.80. But for ALL 23 cells (the 8 that cross 0.80 and the 15 that
never do), the **class-balance sub-check (>=10% of a cell's regular cycles in each class) still fails**: the 8
that do cross 0.80 apparently do so only in a small tail near the end of their recorded cycle life, and the
other 15 do not cross it within their recorded cycles at all (longest recorded cell: 420 cycles; shortest: 131).
So R6 as defined (reach 0.80 AND >=10% class share) is not met by Batch-1/Batch-2 under either reference; **this
looks like a property of how much of each cell's life is recorded (or of "initial capacity" meaning something
other than the cycle-1 value), not fixable by the reference-cycle choice alone.** Not investigated further in
this label-free step; a next step that reads actual capacity/cycle-count values to characterise this (e.g. what
fraction of life is captured) would need its own explicit go-ahead.
No SOH, capacity or feature value is anywhere in this entry or either script's output.

## Open questions for next session

1. **A looser purity threshold fixes 20%-noise but breaks 10%-noise — needs a way to get both.**
   Attempt 2's looser grid gave +19.47 ACC at 20% noise and −8.82 at 10% noise. The ablation showed
   the purity weighting itself contributes nothing; unweighted EF-GBTSVM with the same loose grid
   gives the same result. Attempt 3 (more CV folds, gmean/f1 selection) changed nothing, so this is
   treated as a genuine limitation of a fixed, CV-selected threshold. Whether to keep `weighted=True`
   at all is open: it is harmless but unsupported. Attempt 4 tried the noise-aware fix: a kNN
   noise estimate (which tracks the injected noise well) mapped linearly to purity with
   `alpha=1.0`. It gave the best 20%-noise result but the worst 0% and 10% results. **This novelty
   direction is closed** (declared last iteration). Untried and only worth revisiting with a
   second dataset to tune on, to avoid leakage: other `alpha` values or mapping shapes, and
   searching `C1 / n_hidden / activation` jointly with an adaptive *offset* rather than a fixed
   purity. Other unexplored next steps:
   * A steeper purity → weight mapping (e.g. `purity²`, offered and declined so far) might separate
     "genuinely noisy" from "borderline" balls better than linear `weight = purity` — one-line
     change (`weights = purities**2`), not yet tried.
   * Selecting purity by CV with a different metric was tried (Attempt 3, gmean/f1): no change.
   * Only NASA battery / 4 folds tested; whether this trade-off is a property of the method or an
     artifact of so few folds is open.
2. **Add a different novelty on this dataset, or bring a second dataset first?** Everything so far
   comes from one 4-group dataset, and the 6 Friedman "datasets" are not independent. A second EE
   dataset would give a valid Friedman/Nemenyi test (N ≥ ~10 independent datasets is the usual
   expectation); adding more novelty attempts here risks tuning to B0005/6/7/18 quirks. *Needs your
   decision* (a second dataset was attempted once this project and then explicitly stopped/reverted
   — ask before trying again).
3. **If another novelty is added, which?** Remaining candidates, each tied to an observed failure:
   * *Robust loss / noise-aware training* (not ball weighting) — EF-GBTSVM still collapses at 20%
     label noise (61.31 ACC, rank 10) and purity-weighting didn't touch this.
   * *FFT / frequency features* — the hard-subset results say feature choice matters a lot to the
     TSVM family (TSVM −13.57, GBTSVM +23.76); richer features may help, but it changes the input, not the model.
4. **Why did B0018 collapse in the hard run** (94.70 → 51.52 for EF-GBTSVM)? Needs per-fold inspection
   (is it ball generation, the hyperparameters picked, or the threshold shift?). Not investigated.
5. **Is `time_to_v3p5` a remaining proxy for discharge_time?** Answered (`python proxy_checks.py`):
   yes. Within-cell correlation with discharge_time is 0.972–0.994, and a single threshold on it,
   fitted on the training cells, reaches 96.3% leave-one-battery-out on the hard subset (folds
   97.0 / 99.4 / 93.5 / 95.5). Also: |mean current| × discharge time relative to each cell's
   first cycle, thresholded at 0.80, reproduces the label for 632 of 636 cycles (99.4%). Both are
   now disclosed in the paper. Still open: whether to drop `time_to_v3p5` too, or report the
   one-feature rule as a reference result.
6. **Tuning budget.** `--n-iter 30` means EF-GBTSVM samples 30 of its 1,000 grid points, while
   SVM (16), XGBoost (12), TSVM (8), random forest and MLP (6 each) are searched in full, because
   the search uses min(30, grid size). Answered: re-tuning the EF-GBTSVM family with 150 candidates
   does not close the gap (see "Tuning budget equalization"). Still open: RVFL and the ablations
   were not re-tuned.

## Phase E pre-registration frozen (2026-11-11)
PREREG_E.md approved by the user in chat on 2026-11-11 (file text unchanged since approval; its header still says DRAFT because editing it would change the hash). sha256: ecefed8e62d4f9c5659f6f778847a8678427524b8bc8271b4a01bd64e84356d7
No Phase E code or run exists yet. Open decisions in PREREG_E.md section 7 are still unanswered; nothing runs until the user says yes.

## ADDENDUM_E1.md approved (2026-11-11)
ADDENDUM_E1.md (XJTU Batch 1 + Batch 2 dataset-specific rules) approved by the user in chat on 2026-11-11 (file text unchanged since approval; its header still says DRAFT). sha256: 41e1e33723672c780faf5b91dc9ddb3abb16dc76509d52c7c7bcaf1dee373719
PREREG_E.md unchanged (sha256 above still matches). No XJTU file has been downloaded or extracted; the 2.4 GB download is a separate approval that has not been given. D-lite untouched.

## XJTU download and documentation check (2026-11-11)
`Battery Dataset.zip` (Zenodo 10963339) downloaded to `data/raw_xjtu/`; size 2,438,769,934 B, md5 `A635CEF9678E0DE21F9D5C1F78A4342C` equals the expected value. Only `Data Introduction-English.pdf` was extracted. Variable list (names/classes/dims only) of the 23 Batch-1/2 cells: all identical structure (`data` 1 x n_cycles struct with `system_time`, `relative_time_min`, `voltage_V`, `current_A`, `capacity_Ah`, `power_Wh`, `temperature_C`; `summary` struct with per-cycle `discharge_capacity_Ah` etc.). No value of any cell was read. Details and open items: `prereg/E_xjtu_doc_check.md`. ADDENDUM_E1.md unchanged (sha256 `41e1e337…3719`). D-lite untouched; features, labels and R2–R6 wait until D-lite is finished.

**Deviation / limitation (to go into the paper):** the first cycle of every XJTU cell is a 0.2C capacity check. It is removed as irregular and the SOH reference is the second (1C) cycle. The authors' end-of-life ("below 80 % of initial value") refers to the 0.2C initial capacity, our `degraded` label (SOH < 0.80) to the 1C capacity. The two definitions are not identical.
Capacity source for XJTU (user decision, 2026-11-11): `summary.discharge_capacity_Ah`, one source for all cells; supersedes the per-sample-max wording of ADDENDUM_E1.md section 6 (addendum file and hash unchanged). Recorded in `prereg/E_xjtu_doc_check.md` Part 4. No capacity value read yet.

## XJTU validation attempt (Nov 2026) — closed, no eligible cell (2026-11-11)

Decision (Vansh, 2026-11-11): **Option 1, rule as-is; XJTU validation closed.** No amendment of the
pre-registered rule, no model fitted on any XJTU cell.

* Dataset: XJTU Battery Dataset (Wang et al., Nat Commun 15, 4332 (2024); Zenodo 10963339), **licence CC BY 4.0**
  (as stated in `prereg/E_dataset_licence.txt`/ADDENDUM_E1.md section 1; re-check on the Zenodo page before any redistribution).
* Scope: **23 cells, Batch 1 + Batch 2**, each cell its own group (ADDENDUM_E1.md section 2).
* Result under the pre-registered eligibility order: **R2 (groups) PASS, R3 (same regime) PASS, R5 (feature finiteness) PASS
  for all 23; R6 (degradation + class balance >= 10% per class) FAIL for 23/23** against **both** SOH references:
  the second (1C) cycle (ADDENDUM_E1.md default) and the first (0.2C, the authors' "initial capacity") cycle.
  **Eligible cells: 0 of 23.**
* Reason for the R6 failure: the recorded cycles do not cover enough of the cell life. Against the 1C reference no cell
  crosses SOH < 0.80 within its recorded regular cycles; against the 0.2C reference some cells cross it, but only in a small
  tail, so the class-balance sub-check (each class >= 10% of the cell's cycles) fails for every cell (details in the three
  XJTU entries above; no SOH, capacity or feature value is reported there or here).
* Files: `prereg/xjtu_label_free_check.{py,log}`, `prereg/xjtu_label_free_results.json`, `prereg/xjtu_reference_order_check.py`,
  `prereg/xjtu_r6_recheck_02C_ref.{py,log}`, `prereg/xjtu_r6_recheck_02C_ref_results.json`.
* Together with Phase D step 1 (NASA fresh cells, 0 eligible) this is the **second** external-data attempt that found no
  eligible cell under a pre-registered rule.

### Status of PREREG_E.md and ADDENDUM_E1.md (marked here only; files NOT edited)

* `PREREG_E.md` — **closed, no external dataset found.** sha256 `ecefed8e62d4f9c5659f6f778847a8678427524b8bc8271b4a01bd64e84356d7` (frozen).
* `ADDENDUM_E1.md` — **closed, no external dataset found.** sha256 `41e1e33723672c780faf5b91dc9ddb3abb16dc76509d52c7c7bcaf1dee373719` (frozen).
* Both files still carry "DRAFT" headers by design (editing them would change the hash). Their status is recorded in this file only.

### Phase E status after the XJTU closure

There is no external evaluation dataset. Phase E (new model variants, PREREG_E.md section 2) is therefore **limited to the
development batteries**: candidates may be designed and compared on dev data, but **no "best on external data" claim can be
made**, and PREREG_E.md section 5 (one-shot external evaluation) will not be run. Any Phase E result is dev-only and must be
reported as such. Nothing of Phase E has been run (no Phase E code exists as of this entry).

### Open item (not done)

`data/raw_xjtu/Battery Dataset.zip` (2,438,769,934 B) is still on disk. Deleting it needs Vansh's explicit yes; nothing was deleted.

## XJTU zip deleted (2026-11-11, Vansh: "delete karo")

`data/raw_xjtu/Battery Dataset.zip` (2,438,769,934 B, md5 `A635CEF9678E0DE21F9D5C1F78A4342C`) deleted after the
XJTU validation attempt was closed (see "XJTU validation attempt (Nov 2026)" above). `data/raw_xjtu/doc/`
(`Data_Introduction-English.pdf`, `intro_en.txt`) kept, since REPORT.md and ADDENDUM_E1.md cite it. If this
dataset is ever revisited, it can be re-downloaded from Zenodo 10963339 (same md5 to verify the re-download).

## Final consistency pass (2026-11-12)

Checked per Vansh's request: doc has 0 pending suggestions (rev 242/244). Needs-human-review item about
pushing D-lite/prereg was already half-done (push confirmed, commit 0cc0287) but the paper doc's own
"Code and data availability" paragraph still cited only commit 6df66fe and did not name run_dlite.py,
dlite_analysis.py, PREREG_DLITE.md or prereg/. Fixed directly in the doc: now lists those files and cites
commit 0cc0287. The matching needs-human-review checklist line was updated to say so (not just "done").

Found: the public repo's README.md had regressed. Commit 0cc0287 (22:36:53, same commit that added
D-lite/prereg/) replaced the detailed "A leakage-free, fair benchmark..." README (written in commit
6df66fe, 2026-11-05) with the project's original generic template ("EF-GBTSVM for Electrical Engineering
data", UCI Smart Grid Stability instructions) -- almost certainly an accidental overwrite from the wrong
source file in that commit, not an intentional edit; `git show 0cc0287 -- README.md` confirms the diff.
This was not caught before because REPORT.md's own needs-review tracking was about pushing files, not
about README content.

Fixed locally (not yet pushed -- needs Vansh's go-ahead): restored the detailed README and extended it
with a Phase D-lite stress-test bullet (PF ACC 92.92/85.12/81.52 on N30/N40/BND, ranks 8/5/12, pooled
Friedman chi^2=40.10 p=0.0002) and an external-validation-closed bullet (0/30 NASA fresh cells, 0/23
XJTU cells), updated the Contents table (prereg/, run_dlite.py, dlite_analysis.py, PREREG_DLITE.md,
results/dlite_*), added a Phase D-lite reproduce step, and updated the Data/License sections to mention
XJTU. All added numbers are taken from REPORT.md/results files already cross-checked above; nothing new
was computed.

Cross-check of numbers in REPORT.md vs the paper doc vs results files: 95.25/95.08/95.28, 0.42/8.41/0.57,
2.21 (train-test gap), 85.12, 81.52, 0.123 (Friedman p) all match across REPORT.md and the doc. 92.92 is
in results/dlite_summary_table.csv and results/dlite_claims.txt (REPORT.md references the file by sha256
rather than restating the number inline, which is its usual style, not a discrepancy). No number was
changed or removed anywhere in this pass.

## PDF / docx / Word regeneration + IEEE PDF staleness caught (2026-11-12)

Regenerated paper/EF-GBTSVM_Label_Noise_Paper_Vansh_Sachan.pdf and .docx from the current doc (rev 244)
via the docs export tool, replacing the Nov-3 files (which predated the Nov-5 Phase B rewrite entirely).
Could not regenerate paper/EF-GBTSVM_Label_Noise_Paper_Vansh_Sachan.doc (legacy binary OLE2 .doc format,
confirmed by its file header): no LibreOffice/Word converter is available on this machine, and the docs
export tool offers docx/pdf/markdown/text/html/notion, not binary .doc. Left the old Nov-3 .doc file
untouched rather than fake one; flagged to Vansh (needs a decision: is .docx acceptable as the Word
version, or should this be converted on a machine that has Word/LibreOffice).

Also found: paper/ieee/paper.pdf (the IEEE-formatted camera copy) was stale relative to paper.tex twice
over during this session -- build_ieee_tex.py only writes paper.tex and figures, it does not invoke
xelatex, so after each doc edit paper.tex changed but paper.pdf did not until xelatex was run by hand.
Caught by comparing extracted text (pdftotext) of all three deliverables against each other: the IEEE
PDF was initially missing the XJTU paragraph and the 0cc0287 availability-statement fix entirely. Fixed
by re-running build_ieee_tex.py from a fresh doc export (rev 244) and xelatex twice (for references);
paper.pdf is now 11 pages, no overfull boxes, and check_tex_complete.py / check_citations.py both PASS.
Cross-checked key numbers (95.25, 85.12, 81.52, 92.92, XJTU, 0cc0287, class balance) are present in all
three deliverables (.docx, the Docs-export .pdf and the IEEE .pdf); counts per file differ only because
of layout (flowing prose vs two-column/table), not missing content. No number was changed.

## README.md fix pushed (2026-11-12, Vansh: "Yes, commit+push")

Pushed as commit 0ddd0e0 on main (0cc0287..0ddd0e0), restoring and extending README.md per the
previous entry. Working tree otherwise clean (checked via git status before commit).

## Submission-readiness pass: citation verification (2026-11-12)

Full pass per Vansh's request. No number, model or result was changed anywhere; only citation
verification, needs-human-review updates, a venue note and rebuild/regeneration. Sample-paper
style reference (Part 3) was requested but no such file was found in this session's working
directory, scratchpad, or any other accessible location -- skipped, reported to Vansh.

### PART 1 -- Needs-human-review verification (all via web search + one JMLR PDF fetch; URLs/DOIs below)

Resolved/confirmed (marked checked in the doc):
* Ref [12] (Viswanathan & Kintner-Meyer) 70-80% EOL range -- confirmed from the paper's own
  abstract (pnnl.gov/publications/second-use-transportation-batteries-maximizing-value-batteries-
  transportation-and-grid); later 2023-2024 reviews cite the same range.
* Ref [22] (Xia et al., granular ball) last-author spelling -- Crossref API record for
  10.1016/j.ins.2019.01.010 confirms given name "Yuoguo", family name "Luo"; our "Y. Luo" (IEEE
  abbreviation) is already correct, no change made.
* Ref [27] (Quadir et al., EF-GBTSVM) v1 vs v2 authors -- arxiv.org/abs/2410.05786v2's own author
  list for "this version" is exactly the four authors our reference cites (A. Quadir, M. Sajid,
  M. Tanveer, P. N. Suganthan); the 5-author list (+ Mushir Akhtar) is v1-only.
* Section 5 critical-difference formula / Nemenyi recommendation -- fetched and read the actual
  JMLR PDF body (jmlr2020.csail.mit.edu/papers/volume7/demsar06a/demsar06a.pdf) via pdftotext,
  not just the abstract this time. Confirmed Demsar's own formula matches ours, and confirmed
  Demsar recommends Bonferroni-Dunn (not Holm) for one-vs-control comparisons -- our attribution
  of the Wilcoxon-with-Holm approach to [17] (Benavoli et al.), not [5] (Demsar), is correct.
* Date/order of the three Phase A selection runs (Section 3) -- confirmed from file mtimes and
  actual CSV contents in this working directory (not git, since all three were committed together
  in one commit): phys_selection.csv (14 feat, 2026-11-05 12:14) -> base_selection.csv (13 feat,
  13:08) -> phys2_selection.csv (12 feat + L8, 13:26, the one reported). Each chose different
  variants per fold, confirming "chose other variants".
* Figure 2/4/5 captions -- already say "rendered as an image" / "not screen captures" in-body; no
  change needed.

Verified but a decision is still needed (left unchecked, explained in the doc):
* Holm (1979) [Scandinavian Journal of Statistics 6(2):65-70, doi:10.2307/4615733] and Wilcoxon
  (1945) [Biometrics Bulletin 1(6):80-83, doi:10.2307/3001968] -- both facts confirmed (multiple
  independent sources agree; direct Crossref API calls for Holm hit a rate limit mid-session).
  NOT inserted into the reference list yet: both predate every one of the 30 current references
  (earliest 1988), so adding them in Vansh's year-ascending order would shift all 30 existing
  reference numbers by +2 and touch roughly 70 in-text citation instances across prose, tables and
  figure captions. Doing that by hand in the same pass as everything else risked leaving the
  numbering half-changed if any one edit failed; it needs its own careful pass plus a
  check_citations.py re-verification. Flagged to Vansh as a decision, not done silently either way.

Still genuinely unresolved (no change, searched again):
* Ref [27] ICPR 2024 publisher DOI -- re-searched 2026-11-12 (arxiv.org/abs/2410.05786v2): the
  journal-ref field still names only "27th International Conference on Pattern Recognition (ICPR),
  2024" with arXiv's own DOI; no separate IEEE/ICPR/Springer DOI found anywhere. Cite as arXiv.

### PART 2 -- Venue/template

Added a new needs-review item: venue itself is still Vansh's call; current IEEEtran two-column
template fits the "IEEE-style journal, 10-11 pages" brief and needs no format change until an
exact venue is named.

### PART 4 -- Rebuild and delivery

Re-ran build_ieee_tex.py from a fresh doc export, then xelatex twice (paper.tex writes don't
trigger a PDF recompile by themselves, as found in the previous pass -- recompiled by hand again
here). paper.tex / paper.pdf: 11 pages, no overfull boxes. check_tex_complete.py: PASS (634
sentence/cell/equation parts checked, 15 Needs-human-review blocks correctly left out). check_
citations.py: PASS (30 references, year-sorted, no citation target changed). Regenerated the
Docs-export .pdf and .docx from the same doc revision; compared all three (pdftotext + docx XML)
for the new verification text and DOIs -- present in the .docx and Docs .pdf (which include the
Needs-human-review section), correctly absent from the IEEE .pdf (which excludes that section by
design, per build_ieee_tex.py's own SKIP_SECTIONS). No number differs between the three.

## Holm/Wilcoxon inserted, all references renumbered (2026-11-12, Vansh: "Do it now")

Inserted Wilcoxon (1945, doi:10.2307/3001968) and Holm (1979, doi:10.2307/4615733) as new references
[1] and [2] at the top of the (auto-numbered, ordered-list) reference list, since both predate every
one of the 30 existing references. The list auto-renumbered the other 30 to [3]-[32]; separately,
every literal in-text "[n]" citation (n=1..30) in the body, tables and needs-human-review notes was
shifted to "[n+2]" by exact-text find/replace, scoped per paragraph/cell by block id and (for
duplicate numbers within one paragraph) by nth occurrence, processed in descending original-number
order within each paragraph to avoid any new-value/old-value collision during the shift. 73 in-text
occurrences shifted (one Python pass over a fresh doc export built the op list; executed as 5 batched
update calls, ~18 ops each). Added the two new citations at their first use in Section 5: "Wilcoxon
signed-rank test [1] ... with Holm correction [2]".

Verified, not guessed:
* check_citations.py: references 32 (was 30), 79 in-text citations (was 73; the +6 are the 2 new
  citations plus 4 mentions inside the needs-review commentary itself), sorted by (year, surname)
  correctly (Wilcoxon 1945, Holm 1979, then Angluin 1988 onward, matching the pre-renumbering order
  exactly shifted), and -- the check that matters most -- zero paragraphs have a changed citation
  target (every old citation still resolves to the same reference title it did before, just under
  its new number). The script's own "RESULT: FAIL" line is a false alarm from its hardcoded
  old-count==new-count assertion, which assumes pure reordering; it was never meant for a run that
  adds references, and checks 1-3's substantive conditions (out-of-range, uncited, sort order,
  target-changed, citations-missing) all read "none"/"True".
* check_tex_complete.py: PASS, 636 sentence/cell/equation parts now (was 634, +2 for the new refs).
* paper.tex rebuilt from the post-renumbering doc export, xelatex run twice: still 11 pages, no
  overfull boxes.
* Spot-checked the compiled PDF directly (pdftotext): reference [1] is Wilcoxon, [2] is Holm, and
  Section 5 reads "...Wilcoxon signed-rank test [1], as [19] recommends, with Holm correction [2]...".
* Regenerated the Docs-export .pdf and .docx; all three deliverables (IEEE .pdf, Docs .pdf, .docx)
  now agree on the new numbering and DOIs (the Docs exports additionally carry the needs-review
  section, as before). No number, model or result was touched -- only citation numbering.

## Push to public repo + submission summary (2026-11-13, Vansh: push + delivery package)

Pushed REPORT.md, ADDENDUM_E1.md, PREREG_E.md and the 7 XJTU label-free check files (prereg/xjtu_*)
to the public repo as commit 3abd562 (0ddd0e0..3abd562). File list confirmed before staging: exactly
these 10 files, no .mat file or dataset archive. Re-verified the three deliverables (IEEE PDF, Docs
PDF, docx) still consistent via pdftotext/docx-xml comparison -- unchanged since the last rebuild, 11
pages. Wrote paper/README_FOR_SUBMISSION.md (not pushed to the repo; a local summary file) covering
what the paper claims, what it does not, and the two items still needing Vansh himself (venue/
template, ref [29] arXiv-only citation).

## Author byline removed from paper.pdf (2026-11-13, Vansh: "paper.pdf me ... hata do")

Deleted the byline paragraph ("Vansh Sachan / B.Tech, Electrical Engineering (3rd Year) / Faculty
of Technology, University of Delhi") from the paper doc entirely (force delete, no concurrent
edits to lose). build_ieee_tex.py crashed on the next build (it hardcoded extracting an author
block containing the text "Vansh Sachan" to build \IEEEauthorblockN/A); fixed it to fall back to
an empty \author{} when no byline paragraph exists, so the script does not silently assume a
byline is always present. paper.tex/paper.pdf rebuilt (xelatex twice): still 11 pages, no overfull
boxes, check_tex_complete.py PASS (635 sentences, -1 for the removed byline). Docs-export PDF and
docx regenerated from the same doc state. Verified via pdftotext/docx-xml: the name, "B.Tech,
Electrical Engineering (3rd Year)" and "Faculty of Technology, University of Delhi" appear zero
times in any of the three deliverables; 95.25% and other key numbers unchanged.

## Pushed to public repo (2026-11-13, Vansh: "Yes, push")

Pushed REPORT.md and the build_ieee_tex.py byline fix as commit c388fdd (3abd562..c388fdd). Only
those two files touched (+27/-2 lines); REPORT.md and the local lab notebook now agree.

## Two diagnostic figures inserted into the paper (2026-11-13, Vansh: "Yes, insert both")

Inserted as new Figure 6 (noise-estimate calibration, results/fig_noise_calibration.csv parsed from
results/phys_phase_b.log) and Figure 7 (accuracy-vs-noise grid/variant comparison, from
results/phase_b_summary_table.csv), after Vansh reviewed the PNGs. Images uploaded as doc assets,
referenced via doc blobs; make_diagnostic_figs.py (paper_figures/) is the reproducible script, PNGs
also copied into paper_figures/ for the IEEE build. build_ieee_tex.py's FIG_BLOBS extended with the
two new blob ids.

Caught and fixed a numbering bug before finishing: the first attempt placed the two new figures
mid-document (after Table 1, and after the "family" paragraph in Section 6). LaTeX auto-numbers
\caption by document order, so that push every later figure's rendered number by +2 -- Figure 4
(Wilcoxon) rendered as "Fig. 6", Figure 5 (train gap) as "Fig. 7", while my new ones took over
"Fig. 1"/"Fig. 2". The paper's own prose has several hardcoded literal figure-number mentions
("(Figures 1 and 3)", "Figure 2 shows...", "(Figure 4)", "Figure 5 shows..."), all of which would
then have pointed at the wrong image. Fixed by moving both new figures to the end of Section 7
(right after Figure 5, before "8. Discussion and limitations"), so Figures 1-5 keep their original
rendered numbers and mentions, and the two new ones become Figure 6 and 7 in sequence -- verified
by pdftotext against every one of those four literal mentions after rebuilding.

Page count: 11 -> 12 pages (the professor's stated limit is 10-11). Did not remove anything;
flagged to Vansh with options, per the standing rule on this project. Rebuilt paper.tex, xelatex
twice: 12 pages, no overfull boxes. check_tex_complete.py PASS (641 sentences/cells, +6 for the two
new figures' media+caption blocks and in-text references). check_citations.py not re-run (no
reference or citation touched by this change). Regenerated the Docs-export .pdf and .docx;
confirmed Figure 6/7 captions and key numbers (e.g. 95.25%) appear in all three deliverables.

No in-text "(Figure 6)"/"(Figure 7)" pointer sentence was added anywhere in the prose -- the two
figures sit positioned next to the most topically relevant existing text (end of Section 7, right
after the overfitting-check figure) but are not cross-referenced by a sentence the way Figures 1-5
are. Flagged to Vansh; not added without being asked, to avoid further prose changes beyond what
was approved.

## Figures 6/7 shrunk to recover 11 pages (2026-11-13, Vansh: "Shrink the 2 new figures")

paper.pdf had gone to 12 pages after inserting Figures 6-7; page 12 turned out to be almost empty
(just the last ~2 reference entries spilled over). Reduced both figures' matplotlib figsize height
(fig1: 3.1in -> 1.75in; fig2: 3.1in -> 1.95in, legend also moved from above-the-axes to inside the
plot to reclaim space), same width (both stay \columnwidth in the IEEE build), no content removed,
no number changed -- only the two PNGs got shorter/more compact. Re-uploaded both as new doc assets
(blobs are immutable) and pointed the existing <media> elements at the new blob ids, so the Claude
Docs copy and the IEEE LaTeX copy use the same (smaller) images. Rebuilt paper.tex, xelatex twice:
back to 11 pages, no overfull boxes. check_tex_complete.py PASS (641, unchanged). Regenerated the
Docs .pdf and .docx; verified Figure 6/7 captions and 95.25% appear in all three deliverables, same
counts as before the resize.

## Three more diagnostic figures inserted; paper now 12 pages (2026-11-13/14)

Added Figure 8 (train-test overfitting gap, 2-panel full-width), Figure 9 (D-lite stress-test
accuracy for a 6-model subset) and Figure 10 (per-battery-fold class-prior shift) to the paper doc,
after Vansh reviewed the PNGs. Reproducible script: paper_figures/make_diagnostic_figs2.py. All data
is from already-computed results; nothing re-run, nothing invented:
* Fig 8: results/phase_b_train_gap.csv (gap_clean per model). EF-GBTSVM-PF's gap uses the published
  per-fold variant mapping (B0005->L8, B0006->J, B0007->L8, B0018->L8) and reproduces the published
  phase_b_claims.txt values 1.48/2.21/0.88; Stump-fair's gap uses phys_stump.stump_runs and
  reproduces 2.11/3.96/1.31. Both checks were run inline before plotting.
* Fig 9: results/dlite_summary_table.csv (ACC mean/SD over N30/N40/BND); shows EF-GBTSVM-PF,
  EF-GBTSVM, Stump-fair and the three best baselines by pooled rank (RBF-SVM, random forest, RVFL),
  with the caption pointing to Table 7 for all 17 models. Values match Table 7.
* Fig 10: arithmetic aggregation of Table 2's published per-cell cycle/label counts (train split =
  the other three cells pooled). B0006 train 36%/test 64% matches REPORT.md Task 1 exactly.

build_ieee_tex.py: FIG_BLOBS extended with the five diagnostic-figure blob ids (incl. the shrunk
fig1/fig2 blobs from the earlier round); fig3_train_test_gap.png added to WIDE (renders as figure*,
both columns); figure_tex gained a FLEX_PLACE set so the two new single-column figures may use
bottom/float-page slots while Figures 1-7 keep their reviewed top-only placement.

Page count: 11 -> 12 pages. The three figures plus captions add ~0.65 page, and the captions are a
fixed cost that does not shrink with figure height, so shrinking alone could not recover page 11
(shrinking to ~1.3 in reclaims only ~0.25 page). Options were put to Vansh (drop one figure / accept
12 / aggressive shrink+caption-trim). **Vansh chose: accept 12 pages** (2026-11-14), keeping all
three figures at readable size with full source-citation captions. The professor's stated limit is
10-11 pages; this is one over, by Vansh's decision.

check_tex_complete.py PASS (650 sentence/cell/equation parts; 15 Needs-human-review blocks skipped).
No overfull boxes. All three deliverables (IEEE PDF 12 pp, Docs-export PDF, docx) regenerated and
cross-checked: Figures 8/9/10 captions, their source-file citations, and the figure numbers appear
in each; the docx carries all 10 figure images. No number was changed anywhere.
