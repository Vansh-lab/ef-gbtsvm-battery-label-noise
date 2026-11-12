# PREREG_E.md — Phase E: candidate extensions of EF-GBTSVM, dev-only selection, one-shot external evaluation

Written 2026-11-11 (DRAFT, awaiting the user's approval). Status when written: **no Phase E model, code or
result exists.** Only existing code (`efgbtsvm/model.py`, `efgbtsvm/twsvm.py`, `run_experiments.py`) and the text of
REPORT.md / PREREG_DLITE.md were read to write this. **No D-lite result has been looked at** and none is used
anywhere in this document (D-lite is running; its outputs are off limits for design).

## 0. Disclosure (read first)

* The failures that motivate the candidates were **observed in earlier outer (leave-one-battery-out) results** on
  B0005/6/7/18: (i) the 20 % random-noise collapse of EF-GBTSVM (REPORT.md, "Open questions" item 3), (ii) the
  purity trade-off: a looser purity fixes 20 % noise but breaks 10 % noise, so one CV-selected purity cannot serve
  both (REPORT.md, Attempts 2–4), (iii) the B0006 fold: training 36.1 % degraded vs test 64.3 %, SVM AUC 96.56 %
  but ACC 51.79 %, i.e. the decision threshold sits in the wrong place under class-prior shift (REPORT.md,
  "Task 1"). The *choice of which failures to attack* is therefore not independent of old test folds.
* The base grid used below (EF-GBTSVM-L8, 8 purity values) was itself picked in Phase A from inner-CV on these
  batteries after earlier outer results shaped the candidate list (PREREG_DLITE.md section 0). It is inherited.
* From here on, **design decisions use dev batteries only (section 4)**. No outer-test label or score of any
  dev battery, and no D-lite output, enters a decision.
* No hypothesis below is known to be true. Each may fail; a null result will be reported as such.

## 1. Frozen (nothing here is edited)

`efgbtsvm/model.py`, `efgbtsvm/twsvm.py`, `granular_ball.py`, `noise.py`, `rvfl_features.py`, `run_experiments.py`,
`phys_stump.py`, `data/nasa_battery_phys.csv` keep the sha256 values recorded in PREREG_DLITE.md section 1. All
Phase E code goes into **new files** (`e_methods.py`, `run_phase_e_dev.py`, `run_phase_e_eval.py`,
`phase_e_analysis.py`); their sha256 is written to REPORT.md before they are run. Metrics, scorer (balanced
accuracy), seeds rule (`random_state = fold`), `RandomizedSearchCV(n_iter=150)` and the 15 baseline model
definitions/grids (`build_models(seed, grids="equal")`) are unchanged.

Unit tests (synthetic data and dev *training* data only, no outer labels): with its new knob switched off each
candidate must reproduce EF-GBTSVM-L8 predictions exactly (tau = 0; one level; no calibration). A test that fails
stops the run.

## 2. Candidates (at most 4; defined before any run)

Common base for all four: EF-GBTSVM-L8 = enhanced features + granular balls + TWSVM, grid C1 in
{1e-5..100} (8 values), n_hidden in {23,63,103,143,183}, activation in {1,2,3,4,10}, purity in the 8 values
{0.55,0.65,0.75,0.85,0.90,0.925,0.95,0.98} => **1600 points**. Notation: ball set from purity p, TWSVM plane
scores s(x) = d2(x) − d1(x) (`TWSVM.score`), prediction = [s(x) > 0].

### (a) EF-GBTSVM-MG — multi-granularity ensemble
* **New:** M members trained on the same noisy training fold, each with its own purity p_m from a level set S
  (so each member sees a different ball granularity); C1, n_hidden, activation are shared. Member score is
  scale-normalised using **training data only**: z_m(x) = s_m(x) / sd_{x in train}(s_m(x)) (no centring, so the
  sign is kept). Ensemble score F(x) = (1/M) * sum_m z_m(x); prediction [F(x) > 0]. Uniform weights, no learned
  weights.
* **Failure targeted:** purity trade-off (ii): instead of selecting one purity, average across coarse and fine
  balls so no single choice has to fit both 10 % and 20 % noise.
* **Hyperparameters:** C1 (8) × n_hidden (5) × activation (5) × level set S ∈ {S1 = {0.65,0.85,0.95},
  S2 = {0.55,0.75,0.95}, S3 = {0.75,0.90,0.98}, S4 = all 8 L8 values} (4) = **800 points**.
* **Cost note:** one candidate costs M model fits (3, 3, 3, 8), so compute is *not* equal to L8 even though the
  candidate count is.

### (b) EF-GBTSVM-P — pinball loss in the TWSVM
* **New:** the hinge slack in each TWSVM plane is replaced by the pinball loss L_tau(z) = z if z ≥ 0, −tau·z if
  z < 0, with z = e + G u1 (plane 1; plane 2 symmetric). With multipliers alpha (for xi ≥ z) and beta
  (for xi ≥ −tau·z), alpha + beta = C, and gamma = alpha − tau·beta, the dual (derived here; to be checked
  numerically by the tau = 0 test and by a primal-vs-dual check on synthetic data) is the **existing** dual
  max e'gamma − ½ gamma' G (H'H + delta I)^-1 G' gamma but with the box **−tau·C·w_i ≤ gamma_i ≤ C·w_i**
  instead of 0 ≤ alpha_i ≤ C·w_i. u1 = −(H'H + delta I)^-1 G' gamma as before. tau = 0 is exactly the current
  model. Only `_box_qp`'s lower bound changes, in a new subclass (frozen `twsvm.py` untouched).
* **Failure targeted:** (i) 20 % noise collapse: hinge loss lets mislabelled points far on the wrong side keep
  large influence; pinball with tau > 0 makes the loss two-sided and is reported in the literature as less
  sensitive to feature/label noise near the margin. **Citation not yet verified: goes to "Needs human review".**
  Honest expectation: label noise here is flipped *cycle* labels, which is not the noise model of that
  literature, so it may not help.
* **Hyperparameters:** L8 grid (1600) × tau ∈ {0, 0.1, 0.25, 0.5, 1.0} (5) = **8000 points**.

### (c) EF-GBTSVM-C — training-only threshold calibration
* **New:** after hyperparameters are chosen exactly as for L8 (inner CV at threshold 0), the decision
  threshold t replaces 0. t is chosen from **out-of-fold training scores only**: for the 3 training batteries
  run leave-one-battery-out (3 refits at the chosen params), normalise each fold's scores by its own training
  score sd, pool the OOF scores, and take t = argmax over candidate cut points (midpoints of sorted pooled
  scores) of balanced accuracy on the pooled OOF labels (ties: the t closest to 0). Final model is the refit on
  the whole training fold; prediction [s(x)/sd_train > t]. No test label, no test feature and no test score is
  used. This equalises the two classes' weight, i.e. removes the dependence on the training class prior.
* **Failure targeted:** (iii) prior-shift fold (B0006): high AUC but threshold misplaced.
* **Limit stated in advance:** it cannot know a *test* prior; it removes the training prior only. If a held-out
  battery's prior differs in a way that a balanced threshold does not fix, it will not help. Transductive
  variants (using unlabelled test features to estimate the prior) are **excluded** from Phase E.
* **Hyperparameters:** none new; grid = L8 (1600 points). Extra cost: 3 refits per final model; **0 extra
  candidates**. AUC is unchanged by construction.

### (d) EF-GBTSVM-X — combination of two of the above
* **Definition:** the product of the two single candidates ranked highest in section 4 (ties broken in the
  order c, b, a). Allowed pairs and grids: a+b: 800 × 5 = 4000 points; a+c: 800 points; b+c: 8000 points. For
  the pair with b, the MG members all share tau. Combination order for a+c: calibrate on the ensemble score F.
* **Failure targeted:** both of its parts. Nothing beyond the two parts is added.

### Search budget for all four
`RandomizedSearchCV(..., n_iter=150, random_state=fold)`, so every candidate (and each of the 15 baselines)
evaluates **150 hyperparameter candidates** inner-CV'd with GroupKFold(3) and balanced accuracy, same as Phase B.
Equal *candidate count* does **not** mean equal search coverage: 150/1600 for L8/C, 150/800 for MG, 150/8000 for P
(a smaller fraction of a larger space). This is a stated limitation, not tuned.

## 3. Pools

New dataset main pool (k = 17): the 15 models of PREREG_DLITE.md section 1 (EF-GBTSVM, -W, -L, -A, -L8, -J,
GBTSVM, HF-GBTSVM, EF-TSVM, TSVM, SVM, RVFL, RF, MLP, XGBoost) + Stump-fair + **E\*** (the adopted candidate,
section 4). PF is *not* in this pool: its per-fold selection (Phase A) was defined on the dev batteries and does
not transfer. Sensitivity pool (k = 15): without L8 and J. L8 stays in the main pool because E\* is built on it
(direct comparator).

## 4. Development rules (dev batteries only)

Allowed data: B0005, B0006, B0007, B0018 (`nasa_battery_phys.csv`). For each outer fold k (test battery held
out, **never opened in this section**) and each inner split j of GroupKFold(3) over the 3 training batteries:
1. Training part = 2 batteries; flip its labels at rate r ∈ {0, 0.10, 0.20}
   (`flip_labels`, `RandomState(2000 + 100·k + 10·j + index(r))`; different from the 1000+k outer seeds).
2. Each method (baseline L8, and a, b, c) runs its full pipeline (150-candidate search, inner GroupKFold(2) over
   the 2 batteries, balanced accuracy; (c) calibrates on the 2-battery leave-one-out OOF scores).
3. Score = balanced accuracy on the **clean (true) labels of the held-out inner battery j**.
This gives 4 × 3 × 3 = 36 paired values per method. (Using true labels of *training* batteries in this
simulated-noise study is allowed; the outer test battery is not touched.)

Decision rule, fixed now:
1. S(m) = mean of the 36 values. Candidate (d) is built from the two highest S among a, b, c, then evaluated
   in the same 36 cells.
2. E\* = argmax S over {a, b, c, d}. Ties (|ΔS| < 1e-6): simpler first, c < b < a < d.
3. E\* is **adopted** only if S(E\*) − S(L8) ≥ 0.01 (1 percentage point of BACC) **and** E\* beats L8 in at
   least 24 of the 36 paired cells. The thresholds guard against picking noise; they are stated before any value
   exists.
4. If not adopted: "no Phase E candidate adopted" is the result, the new dataset is **not** used (it stays
   unburned), and the dev table is reported anyway.
All 36 × 5 values are reported whatever the outcome. No candidate, noise level, fold or inner split is dropped
after seeing them. Compute: dev cost (4·3·3 searches × 5 methods, MG about 3–8× per candidate) is estimated by
a timing check on dev data **before** the dev run, and the run only starts after the user's yes (RAM is
limited).

## 5. Evaluation rules (new dataset, once)

* **One run.** The new dataset (name to be supplied by the user) is run **once**, after E\* is adopted and its
  code hash is recorded. Models: the 17 of section 3, same protocol as Phase B/D-lite: leave-one-group-out
  (group = cell; cells sharing a recording form one group, detected as in `prereg/inclusion_rule.md`
  "Grouping"), `RandomizedSearchCV`, n_iter = 150, `GroupKFold(3)` inner, balanced accuracy,
  `random_state = fold`, `--grids equal`, scaler → classifier pipeline, the 12 physics features
  computed with `prep_phys_features.cycle_features` **unchanged**, label `degraded` = SOH < 0.80 (SOH =
  capacity / the cell's first-cycle capacity).
* **Conditions (training labels only; definitions are those of PREREG_DLITE.md section 2):** N00, N10, N20
  (random flips, `RandomState(1000 + k)`), N30, N40, BND (boundary noise, band 0.75 ≤ SOH ≤ 0.85, half flipped).
  Every condition is reported.
* **Metrics:** ACC, BACC, F1, AUC per (condition, fold, model) as `metrics()`; rank = rank of the mean over
  folds, rounded to 6 decimals, descending.
* **No reruns** for performance reasons. A bug → fix, rerun all models and conditions equally, keep first-run
  files, document the deviation. Crash/timeout reported as it happened. New files `results/phaseE_*` only.
* No dev-battery results are pooled with the new-dataset results in any test.

### Statistics
* Friedman (tie-corrected) per condition, blocks = independent groups (G); Nemenyi descriptive only.
* **One confirmatory test:** E\* vs EF-GBTSVM-L8, balanced accuracy per group on condition **N20**, two-sided
  Wilcoxon signed-rank, α = 0.05, no adjustment (a single pre-specified test). Needs G ≥ 6 (minimum two-sided
  p is 0.03125 at G = 6; at G = 5 it is 0.0625 and the test cannot be significant). If G < 6 the confirmatory
  test is declared "not testable".
* All other comparisons (E\* vs the other 15, other conditions): Wilcoxon with Holm over 16, **descriptive**.
  Even if every group favours E\*, the minimum Holm-adjusted p at G = 8 is 0.0078 × 16 = 0.125, so these cannot
  be significant at small G; this is stated, not hidden.
* Cells/groups are the independent units; observations are not independent across conditions (same cells and
  features). Absence of significance is not equivalence.

### Claim table (fixed now; mirrors PREREG_DLITE.md section 6)
Rank 1 = strictly highest mean over groups at 6 decimals in the main pool (k = 17). A tie for the top is not rank 1.

| Result on a condition (evaluated separately) | Wording |
|---|---|
| E\* is rank 1 in mean ACC **and** mean BACC | "E\* has the highest mean ACC and BACC on <condition> (main pool, G groups, one external dataset); not a statistical claim unless the confirmatory test row applies" |
| Anything else (one of two, not rank 1, tie, incomplete folds) | "E\* is not best on <condition>"; its rank, mean and gap to the top model are reported |
| Confirmatory test on N20 significant **and** E\* mean BACC > L8 | "E\* improves BACC over its base EF-GBTSVM-L8 under 20 % random label noise on this dataset (Wilcoxon p = …, G = …)" — no wording about being best overall |
| Confirmatory test not significant / not testable | "no demonstrated improvement over L8" |

A claim "best across conditions" needs row 1 on all six; otherwise per-condition statements only. Sensitivity-pool
(k = 15) ranks are shown beside the main pool; if the claim fails there, the report says so. A p-value never
upgrades a rank claim.

## 6. Dataset eligibility rule (written without looking at any label, capacity or SOH of the new dataset)

Checks run in this order; the first failure excludes the dataset ("excluded, reason"). R1–R5 use metadata and
voltage/current/time/temperature signals only; R6 is evaluated **last** and reads capacity only to test existence
of SOH < 0.80, not to inspect values.

* **R1 Licence/terms.** The licence or terms of use are located, the exact text and URL are copied into
  `prereg/E_dataset_licence.txt`, and shown to the user. Eligible only if they permit research use. "Not
  specified"/unclear → **not auto-eligible**: asked to the user, listed under "Needs human review". Redistribution
  terms are recorded; if unclear, neither raw data nor processed CSVs are redistributed (as for NASA PCoE).
* **R2 Cell count.** ≥ 8 cells after removing cells that fail R3–R5, and ≥ 6 independent groups after merging
  cells with identical recording (same rule as `prereg/inclusion_rule.md` grouping: ≥ 50 % of shared discharge
  cycle indices have exactly equal time arrays). Fewer → excluded (the confirmatory test would be untestable).
* **R3 Same regime.** One discharge protocol and one temperature for all kept cells: on every discharge cycle the
  median of |I| over samples with |I| > 0.5 A lies within ±5 % of the dataset-wide median of those medians;
  ambient/chamber temperature range within a cell ≤ 2 °C and cell medians within ±2 °C of each other; charge
  protocol recorded and identical in type. A cell outside the regime is dropped (listed); datasets mixing
  regimes without ≥ 8 cells in one regime fail R2.
* **R4 Not dev.** No cell is B0005/6/7/18 or has a time vector identical to them (NASA PCoE cells B0005–B0056
  already used or screened are not re-used as "new").
* **R5 Features.** All 12 physics features defined and finite on every discharge cycle with
  `cycle_features()` unchanged, ≥ 50 discharge cycles per cell. A cell that raises or gives NaN is dropped; the
  feature definitions (3.8 V / 3.5 V crossings etc.) are **not** changed to fit a new chemistry.
* **R6 Degradation (last).** Every kept cell reaches SOH < 0.80 (SOH = capacity / first-cycle capacity, finite
  and > 0 on all cycles). Proposed addition (remove if you disagree): each class has ≥ 10 % of the cell's cycles,
  as in `prereg/inclusion_rule.md` R5. A cell failing R6 is dropped, then R2 is re-checked.

All exclusions are listed with reasons; no cell is removed for any reason not in R1–R6.

## 7. Open decisions (need the user; nothing runs until answered)

1. Name of the new dataset (supplied by the user), and whether "licence not specified" is acceptable (R1).
2. Keep conditions N30, N40, BND on the new dataset (6 conditions, more compute) or only N00/N10/N20?
3. Adoption thresholds (1 pp, 24/36) and the dev noise levels {0, 10, 20 %}: fixed here by judgment, not derived.
4. Whether the extra "≥ 10 % per class" in R6 stays.
5. Dev compute budget approval and RAM/timing, after D-lite finishes.

## 8. Freeze

The sha256 of this file is recorded in REPORT.md once the user approves. The directory is not a git repository
(as of 2026-11-05). Nothing is run until the user says yes.
