# ADDENDUM_E1.md — Phase E: dataset-specific rules for the XJTU Battery Dataset (Batch 1 + Batch 2)

Written 2026-11-11 (DRAFT, awaiting the user's approval). Status when written: **no XJTU file has been downloaded,
opened or extracted; no capacity, SOH, label, time vector or feature of XJTU has been seen.** Facts about the data
below come from the user's instructions or from the web-page text recorded in `prereg/E_dataset_licence.txt`; anything
else is marked *unverified*. `PREREG_E.md` is **not modified** (its sha256 stays frozen); this addendum only fixes
the items PREREG_E.md section 7 left open, for this one dataset. Where the addendum differs from PREREG_E.md the
difference is listed in section 9. D-lite files are not touched and no D-lite output is used.

The addendum's own sha256 is **not** written to REPORT.md until the user approves it.

## 0. Gate

Nothing is downloaded, listed from the network, extracted or run until the user says yes. This addendum applies only
if a candidate E\* is adopted under PREREG_E.md section 4; if none is adopted the dataset stays unburned.

## 1. Dataset and licence (R1)

* Dataset: XJTU Battery Dataset (Wang et al., Nat Commun 15, 4332 (2024)). **Only Batch 1 (8 cells) and Batch 2
  (15 cells) are used.** Batches 3–6 (variable discharge) are excluded in advance because they cannot satisfy R3
  (one discharge protocol).
* Source: Zenodo record 10963339, file `Battery Dataset.zip` (2.4 GB). Zenodo rights: Creative Commons Attribution 4.0
  International (CC BY 4.0), access "Open".
* Author site (https://wang-fujin.github.io/) footer, quoted from `prereg/E_dataset_licence.txt`:
  "Copyright (c) 2023. Xi'an Jiaotong University All rights reserved." It conflicts with CC BY 4.0 and neither page says
  which governs. **Recorded, not resolved; listed under "Needs human review".** Consequence adopted in advance: raw
  files and processed CSVs of XJTU are **not redistributed** (same treatment as NASA PCoE); only code, hashes and
  aggregate result tables are shared.
* Citation: Zenodo asks to cite Wang, F., Zhai, Z., Zhao, Z. et al., Nat Commun 15, 4332 (2024),
  https://doi.org/10.1038/s41467-024-48779-z. The user mentions a request to cite **three** papers; only this one is in
  `prereg/E_dataset_licence.txt`. The other two are **not known to me and are not filled in** — to be copied
  verbatim from the Zenodo page / `Data Introduction.pdf` after approval (Needs human review).
* `prereg/E_dataset_licence.txt` already contains the CC BY 4.0 text and the "All rights reserved" footer line. It does
  not yet contain the three-paper citation request. I have not edited it in this step.
* R1 outcome (decision of the user needed): research use permitted by CC BY 4.0; footer conflict acknowledged.

## 2. Groups

* **Default: every cell is its own group.** Batch is **not** a group (batch is a design factor, see section 4).
* Two cells are merged into one group only if the files give evidence of a shared recording, defined as: in **every**
  discharge cycle index that both cells contain, the **absolute timestamp array is exactly equal (same values)
  and the sample count is equal**. "Every cycle" means all common cycle indices, not ≥ 50 % as in
  `prereg/inclusion_rule.md`; this is stricter and is the rule for this dataset. Groups are connected components.
* If a file has only relative time and no absolute timestamp (*unverified which*), no merge can be shown and every
  cell stays its own group; this is stated in the report rather than replaced by a looser test.
* The comparison is done from per-cycle fingerprints (sample count and a sha256 of the timestamp array), saved while
  each cell is streamed (section 7), so no two cells have to be in RAM together.
* G = number of groups after R2–R6. Statistics treat groups as the independent units (PREREG_E.md section 5).

## 3. Features

* The 12 physics features of `prep_phys_features.cycle_features` are used **unchanged**, with one pre-declared
  exception.
* **Temperature rule:** if the files contain a cell-temperature channel (a channel that varies in time within a
  discharge), the 12 features run as is. If **no temperature channel is found**, `temp_rise_35` and `temp_rise_35_rel`
  are removed **for all models** — E\*, the 15 baselines, and Stump-fair — and everything runs on the remaining **10
  features**. No temperature channel is imputed or substituted, and the other 10 feature definitions are unchanged.
* The decision is taken once, from the file's channel list (names and shapes only), before any feature is computed,
  and written to REPORT.md. It is the same for all cells: a cell whose temperature channel is present while others'
  is absent is a mixed case → the 10-feature variant is used for all kept cells (no per-cell feature sets).
* Ambient/chamber temperature for R3 is read from whatever the files provide; if absent, R3's temperature clause is
  recorded as "not testable from the files" and the temperature regime is taken from the dataset documentation
  (*unverified*) and listed under Needs human review.

## 4. Charge-rate difference accepted (2C vs 3C)

Batch 1 is charged at 2C and Batch 2 at 3C (per the user's instruction; *unverified in the files*). PREREG_E.md R3 asks
that the charge protocol be "recorded and identical in type". Interpretation fixed here: **constant-current charge in
both batches counts as the same type**, the rate differs (2C vs 3C), and the difference is **accepted**, not corrected.
Limitations stated in advance:

* Batch is confounded with charge rate (and with cell-to-cell design factors I cannot see). A held-out cell from one
  batch is partly predicted from cells of the other batch, so results mix cell-to-cell and charge-rate shift.
* Charge rate may change ageing rate and the shape of the discharge curve; the physics features are discharge
  features and are not adjusted for it.
* As an additional **descriptive** table (not used in any claim or test), per-fold results are labelled by batch.
* No claim is made that results transfer to other charge rates or to variable discharge.

## 5. Conditions

* Conditions: **N00, N10, N20** only (random label flips on the training cells, `RandomState(1000 + k)`, as in
  PREREG_E.md section 5). N30, N40 and BND are not run (PREREG_E.md open decision 2, answered by the user).
* **N20 is the confirmatory condition**: E\* vs EF-GBTSVM-L8, balanced accuracy per group, two-sided Wilcoxon
  signed-rank, α = 0.05, no adjustment. It **needs G ≥ 6**; otherwise it is declared "not testable".
* The claim "best across conditions" now needs rank 1 in mean ACC and BACC on **all three** conditions; otherwise
  per-condition statements only. Other comparisons remain descriptive (Holm over 16), with the same stated limit that
  they may not reach significance at small G.

## 6. Capacity, SOH and label (read only in R6)

* **Source of per-cycle capacity:** the file's discharge-capacity column (the "Q" / discharge-capacity channel as named
  in the file; *name unverified*). Per-cycle capacity = **max over the samples of that discharge cycle** of that
  column. (If the column is cumulative within a cycle, its max is the cycle's capacity; if it already holds a per-cycle
  scalar, that scalar is used.) If the file offers **only one** way to obtain capacity, that way is used and the
  report states which. **No integration of current is substituted** and no second method is mixed in for some cells.
* **SOH(c) = capacity(c) / capacity of the first regular discharge cycle** of that cell. Label `degraded` =
  SOH < 0.80, as in dev.
* **Irregular cycles (rate-adjusted, check-up, partial or reference cycles):** a discharge cycle is *irregular* if it
  fails the R3 current test (median of |I| over samples with |I| > 0.5 A not within ±5 % of the dataset-wide median
  of those medians) or is flagged as such by the file/`Data Introduction.pdf`. Handling, fixed now:
  1. Irregular cycles are **removed entirely** (no feature row, no label, not used as the first-cycle reference);
     cycles are not relabelled or interpolated.
  2. The reference capacity is that of the first *regular* cycle.
  3. A cell with **more than 5 % irregular cycles** is outside the regime and dropped under R3 (the 5 % cutoff is a
     proposal by judgment, not derived; user may change it before approval).
  4. The number of removed cycles per cell is reported.
* A cell with non-finite or non-positive capacity on any *regular* cycle fails R6. Capacity is never inspected for
  values other than existence of SOH < 0.80 and class balance, and only at R6.
* Limitation: a discharge that stops before cut-off (partial) is not detected unless it fails the current test; its
  max-Q would understate capacity. Not corrected.

## 7. Memory rule

* The 2.4 GB zip is **never loaded into RAM**. It stays on disk and is opened with a streaming/random-access reader.
* **One cell at a time:** read one cell's member file(s), compute the per-cycle features, capacity, R3 statistics and
  the timestamp fingerprints (sections 2, 3, 6), write a small per-cell summary to disk, release the arrays
  (`del` + `gc.collect()`), then take the next cell. No two cells are in RAM together. The model runs read only the
  small per-cell summaries.
* If a cell must be extracted to a temp directory to be read (e.g. HDF5-based `.mat`, *format unknown*), extract that
  one cell only, process it, delete the temp copy. The extraction directory is new and empty, scripts live elsewhere,
  and any Python that reads the files runs with `-I` (downloaded files are untrusted data).
* Peak RSS is logged per cell and reported. If one cell alone does not fit in RAM, the run stops and the problem is
  reported; the cell is not subsampled.
* Download of the zip needs ~2.4 GB of free disk, plus temp space of one cell; free disk is checked before and the
  download is a separate approval (filename, source, size stated again then).

## 8. Eligibility order (unchanged from PREREG_E.md section 6)

R1 licence → R2 cell/group count → R3 same regime → R4 not dev → R5 features → R6 degradation (last).

Dataset-specific readings, no new rules:

* R1: section 1.
* R2: ≥ 8 cells and **≥ 6 groups** after removing cells failing R3–R6 (re-checked after each drop). With 23 cells
  available and every cell its own group this is expected to hold, but it is a check, not an assumption.
* R3: regime = Batch 1 + Batch 2 discharge protocol; irregular-cycle handling and the 5 % cutoff from section 6;
  charge-type reading from section 4.
* R4: no cell equals B0005/6/7/18 (time-vector fingerprints compared with the dev batteries); R4 is cheap because the
  fingerprints exist from section 2.
* R5: finite features on every regular discharge cycle, ≥ 50 discharge cycles per cell, with the feature set decided
  in section 3. Definitions (3.8 V / 3.5 V crossings etc.) are not changed to fit this chemistry; a cell that raises
  or gives NaN is dropped and listed.
* R6 (last): every kept cell reaches SOH < 0.80, and **each class ≥ 10 % of the cell's regular cycles** (the user kept
  this). A cell failing R6 is dropped, then R2 is re-checked.

All exclusions are listed with reasons; no cell is removed for a reason outside R1–R6 and section 6.3.

## 9. Differences from PREREG_E.md (all declared before any XJTU data is seen)

| Item | PREREG_E.md | This addendum |
|---|---|---|
| Grouping rule | ≥ 50 % of shared cycle indices have equal time arrays | every common cycle index: equal absolute timestamps **and** equal sample count; default one cell = one group |
| Features | 12 features | 10 features (no `temp_rise_35`, `temp_rise_35_rel`) for all models if no temperature channel exists |
| Charge protocol (R3) | identical in type | 2C and 3C accepted as the same type |
| Conditions | N00, N10, N20, N30, N40, BND | N00, N10, N20 |
| "Best across conditions" | all six | all three |
| Irregular cycles | not specified | removed; reference = first regular cycle; > 5 % irregular drops the cell |
| R6 class share | proposed ≥ 10 % | kept ≥ 10 % |

Unchanged: the 17-model main pool and 15-model sensitivity pool, metrics, `RandomizedSearchCV(n_iter=150)`,
`GroupKFold(3)`, balanced accuracy scorer, `random_state = fold`, `--grids equal`, one run only, no reruns for
performance reasons, new files `results/phaseE_*` only, the confirmatory test and the claim table (restricted as in
section 5).

## 10. Needs human review

1. Licence conflict: CC BY 4.0 (Zenodo) vs "All rights reserved" (author site footer); neither says which governs.
2. The two further papers to cite (user mentioned three); only the Nat Commun paper is recorded.
3. Charge rate (2C / 3C), discharge protocol, ambient temperature, cell counts per batch and capacity column name are
   **taken from the user's message / unread documentation**, not verified; to be checked in `Data Introduction.pdf` and
   the file channel list after approval.
4. 5 % irregular-cycle cutoff (judgment, not derived).
5. Download approval (2.4 GB) and free-disk / RAM check.

## 11. Freeze

The sha256 of this file is recorded in REPORT.md only after the user says it is approved. Until then nothing is
downloaded, extracted or run, `PREREG_E.md` and D-lite are untouched.
