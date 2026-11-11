# E_xjtu_doc_check.md — verification note for ADDENDUM_E1.md "Needs human review" items

Written 2026-11-11. ADDENDUM_E1.md is **not edited** (its sha256 `41e1e337…3719` stays valid); this file is the separate
verification note. Scope of this step: download + md5, zip listing (names, sizes), and reading **one documentation file**.
**No cell file was opened. No Q, voltage, capacity, SOH, label or feature was read or computed.** D-lite untouched.

## 1. Download and integrity
* Source: Zenodo record 10963339, `Battery Dataset.zip`, saved as `data/raw_xjtu/Battery Dataset.zip`.
* Size: 2,438,769,934 bytes (equals the size Zenodo reported in the response header).
* md5 computed: `A635CEF9678E0DE21F9D5C1F78A4342C`; expected (from the user): `a635cef9678e0de21f9d5c1f78a4342c` → **match**.
* Free disk before download: 5.89 GB.

## 2. Zip listing (names/sizes only)
65 entries. Relevant to Batch 1 + Batch 2:
* `Batch-1/2C_battery-1.mat … 2C_battery-8.mat` — 8 files, 44.4–47.9 MB each.
* `Batch-2/3C_battery-1.mat … 3C_battery-15.mat` — 15 files, 13.9–33.7 MB each.
  → 8 + 15 = 23 cells, as in ADDENDUM_E1.md. One file per cell, format `.mat` (version, v7 vs v7.3/HDF5, **not yet determined**).
* Not used (outside the addendum): Batch-3 (R2.5), Batch-4 (R3), Batch-5 (RW), Batch-6 (Sim_satellite), 8 files each.
* Also in the zip: `Data Introduction-English.pdf` (234,734 B), `Data Introduction-Chinese.pdf` (289,047 B, **not read**),
  `Temperature_Compensation_Data.mat` (16.7 MB, **not opened**).
* Only `Data Introduction-English.pdf` was extracted, to `data/raw_xjtu/doc/Data_Introduction-English.pdf`; its text is in
  `data/raw_xjtu/doc/intro_en.txt` (pdftotext). Nothing else was extracted.

## 3. What the English documentation says (Batch 1/2 and general)
* Cells: 18650 NCM (LiNi0.5Co0.2Mn0.3O2), "LISHEN", nominal 2000 mAh, nominal 3.6 V; charge cut-off 4.2 V, discharge cut-off 2.5 V.
* "The whole experiment was conducted at room temperature." Sampling 1 Hz. 55 cells in 6 batches (8+15+8+8+8+8 = 55).
* Batch-1 (8 cells): first cycle = initial-capacity measurement: CC-CV charge 0.5C (1 A) to 4.2 V, cut-off 0.02C (40 mA), rest
  5 min, discharge **0.2C (0.4 A)** to 2.5 V. Other cycles: **CC-CV charge 2.0C (4 A)** to 4.2 V, cut-off 0.05C (0.1 A), rest 5 min,
  **discharge 1.0C (2 A) to 2.5 V**, rest 5 min. "Fully charged and discharged."
* Batch-2 (15 cells): same, except charge **3.0C (6 A)**.

## 4. Items of ADDENDUM_E1.md section 10 — match / mismatch
| # | Addendum item | Documentation says | Result |
|---|---|---|---|
| 1 | Licence conflict CC BY 4.0 vs "All rights reserved" | The PDF contains **no licence or copyright text** (read in full, 91 lines). | **Unresolved**; still needs the user's decision. |
| 2 | Two further papers to cite | The PDF contains **no citation of any paper**. Only the Zenodo/Nat Commun citation is known. | **Not found here**; the other two are still unknown, not invented. Needs the user or the Zenodo record's full text. |
| 3a | Batch 1 charge 2C, Batch 2 charge 3C | Yes: 2.0C (4 A) and 3.0C (6 A). | **Match** |
| 3b | Charge is constant-current in both (same "type") | Both are **CC-CV** (CV until 0.05C), identical but for the CC rate. | **Match** (supports the section-4 reading, stronger than "constant current") |
| 3c | Discharge protocol one regime for B1+B2 | 1.0C (2 A) to 2.5 V for all non-first cycles in both batches. | **Match** |
| 3d | Cell counts 8 and 15 | 8 and 15. | **Match** |
| 3e | Ambient temperature | "room temperature", no number given; fluctuations acknowledged; a separate **ambient-only** channel is stored in `Temperature_Compensation_Data.mat` ("without charging or discharging"). | **Partly**: no °C value; R3's "≤ 2 °C within a cell / ±2 °C between cells" **cannot be tested from the documentation**. Whether the cell files carry their own temperature channel is **unknown** (cell files not opened). |
| 3f | Capacity column name | **Not given** in the PDF (no variable names). | **Unresolved**; needs a look at the variable list of one cell file (names/shapes only) — a separate step. |
| 4 | 5 % irregular-cycle cutoff | n/a (judgment). | unchanged |
| 5 | Disk/RAM | Disk 5.89 GB free before; zip now ~2.44 GB on disk. | done for disk |

## 5. New facts the documentation adds (not anticipated in the addendum)
1. **The first cycle of every cell is a different protocol: discharge at 0.2C (0.4 A) after a 0.5C charge.** Under ADDENDUM_E1.md
   section 6 it fails the R3 current test (|I| ≈ 0.4 A, not within ±5 % of ≈ 2 A) and is an *irregular* cycle: it is **removed**, and
   the SOH reference is the first *regular* (1C) cycle, i.e. cycle 2 in the documentation's numbering. This was already the
   addendum's rule, so **no change is needed**, but the consequence is stated here: the SOH denominator is the 1C capacity, not
   the 0.2C initial capacity the authors define as "initial". The "80 % of initial value" end-of-life in the documentation (authors'
   0.2C basis) is therefore **not identical** to SOH < 0.80 here. Each cell loses exactly 1 cycle to this rule (≤ 2 % at ≥ 50 cycles,
   below the 5 % cutoff). To be reported as a deviation from the authors' EOL definition.
2. **Temperature channel:** the documentation describes only a *separate* ambient measurement, not a per-cell temperature in
   the cell files. The addendum's section-3 rule (10 features if the cell files have no temperature channel) stays the decision
   rule; the ambient file is **not** used as a substitute (it is not time-aligned per discharge by any statement in the PDF).
   Whether to use it for R3's temperature clause is for the user to decide.
3. **Shared recordings:** the PDF says experiments "had a few pauses … our publicly available data has been properly
   consolidated", and says nothing about two cells being recorded together. No evidence for grouping; the addendum's default
   (one cell = one group unless timestamp arrays and sample counts match) stands.
4. **Partial discharges:** Batch 1/2 discharges are full (to 2.5 V) per the documentation, which lowers the partial-discharge
   limitation of addendum section 6 for these two batches (not verified in the files).

## 6. Still open (needs the user, nothing run)
* The two other papers to cite; licence conflict.
* Variable names/shapes in one cell `.mat` (file format, capacity column, temperature channel, timestamp type): needs a step
  that opens a cell file's *variable list only*. Not done; awaiting an explicit go and, per the user, features/eligibility
  R2–R6 only after D-lite finishes.

---
# Part 2 — variable-list check of the 23 cell files (added 2026-11-11, after the user's go)

**Method:** header-only walk of each Batch-1/Batch-2 `.mat` directly inside the zip (no extraction). `scipy.io.whosmat` gave the
top-level variables; to get the struct field names a small script (kept in the scratchpad, not in the project) parsed the MATLAB v5
element headers (class, dimensions, name) and skipped over every array's bytes without interpreting them. **No array value was
read or printed** (no voltage, current, capacity, temperature, time, nor the `description` strings). Dimensions below are the
only numbers seen. Nothing was deleted, nothing was extracted. D-lite untouched.

## A. Variable list (one cell: `Batch-1/2C_battery-1.mat`; format MATLAB v5, readable with `scipy.io`)
```
data     : struct [1, 390]      (1 x n_cycles; one element per cycle; field names are common to all elements)
  system_time        : cell   [24444, 1]    (per-sample timestamps, stored as a cell array)
  relative_time_min  : double [24444, 1]
  voltage_V          : double [24444, 1]
  current_A          : double [24444, 1]
  capacity_Ah        : double [24444, 1]
  power_Wh           : double [24444, 1]
  temperature_C      : double [24444, 1]
  description        : char   [1, 46]
summary  : struct [1, 1]
  charge_capacity_Ah, discharge_capacity_Ah, charge_power_Wh, discharge_power_Wh,
  charge_median_voltage, discharge_median_voltage, charge_mean_voltage, discharge_mean_voltage : double [390, 1]
  cycle_life : double [1, 1]
  description : char [1, 20]
```
(Dimensions in this listing are those of element 1 of `data`, i.e. the first cycle, of this cell.)

## B. Other 22 cells
All 23 files (8 in Batch-1, 15 in Batch-2) have the **same variables, the same field names and the same classes** (1 distinct
signature over names+classes). Only the number of cycles differs: from `whosmat`, `data` is 1 x n with n between 131 and 420
(all >= 50). Per-cycle sample counts and per-field dimensions of cycles other than element 1 were **not** read; "same structure"
is therefore established for names/classes, not for per-cycle lengths.

## C. The two questions
1. **Capacity:** it exists, in **two** places, both double:
   * `data(c).capacity_Ah` — per-sample column, one array per cycle (the "Q" column of addendum section 6);
   * `summary.discharge_capacity_Ah` — one value per cycle (plus `summary.charge_capacity_Ah`).
   Consequence: ADDENDUM_E1.md section 6 said "if the file offers only one way, that way is used". Here there are **two**, and the
   addendum text names the per-sample column (max over the discharge samples of the cycle). **Not decided here — needs the user:**
   keep the addendum's per-sample rule, or use `summary.discharge_capacity_Ah` as the single source. Either way only one source is
   to be used for all cells, with no mixing, chosen before any capacity value is looked at. What `data(c).capacity_Ah` holds
   within a cycle (e.g. whether it also runs during charge) is **unknown** — values not read. Note: each element of `data` spans a
   whole cycle (24444 samples at 1 Hz for cycle 1 of cell 1, i.e. charge+rest+discharge), not only the discharge, so "discharge
   segment" must be identified by current sign when features are built (a later step).
2. **Temperature:** a per-sample **cell temperature channel exists by name**: `data(c).temperature_C`, same length as voltage.
   Whether it holds finite, time-varying values (addendum section 3 requires "varies in time within a discharge") is **not known**
   (no values read). Under the addendum rule the 12-feature branch is expected if it is finite and varies, otherwise the 10-feature
   branch; this will be decided once, at the R5 step after D-lite finishes, for all cells together. The ambient file
   `Temperature_Compensation_Data.mat` is still not opened and not used.
3. **Grouping:** `system_time` is present as a per-sample cell array (absolute timestamps, if its entries are absolute time —
   content not read), and sample counts are available from `voltage_V` etc. So the addendum's section-2 merge test is applicable.

# Part 3 — first-cycle (0.2C) deviation, to go into the paper's limitations
* By the documentation (Part 1, section 5) the first cycle of every cell is a 0.2C capacity check; ADDENDUM_E1.md section 6 removes
  it as irregular and sets the SOH reference to the first regular 1C cycle (the second cycle).
* The authors' "capacity below 80 % of initial value" refers to the 0.2C initial capacity; our label `degraded` = SOH < 0.80 is on the
  **1C** capacity of the second cycle. These definitions are **not identical**; a cell reaches our threshold at a different cycle than
  on the authors' basis, in a direction not known (no values read). **Deviation to be stated as a limitation in the paper.**
* Which capacity source is used (item C.1) does not change this deviation.

# Part 4 — decision on the capacity source (user, in chat, 2026-11-11; no capacity value read before or after)
**Capacity source = `summary.discharge_capacity_Ah` (one value per cycle), for all cells, no mixing.**
This replaces the "max of the per-sample column" wording of ADDENDUM_E1.md section 6 for this dataset (the addendum text is not edited;
its sha256 is unchanged). `data(c).capacity_Ah` is **not** used for capacity. SOH(c) = `summary.discharge_capacity_Ah[c]` / its value at
the first regular cycle (the second cycle, see Part 3). Cycle alignment: element c of `summary.discharge_capacity_Ah` corresponds to
element c of `data` (both have n_cycles entries, per the listed dimensions); irregular-cycle removal (addendum section 6) is applied
by cycle index to both. Non-finite or non-positive entries on a regular cycle fail R6. Values are first read at R6, after D-lite ends.
