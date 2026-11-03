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
