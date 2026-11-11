# Fresh-cell inclusion rule (fixed 2026-11-05, BEFORE any fresh label / capacity / SOH value was read)

State of knowledge when written: I (the assistant) had read the six README files (design: currents,
ambient temperatures, cut-off voltages per group) and file names/sizes only. No fresh Capacity value,
SOH value, label or class balance had been computed or seen. Metadata that the rule uses (current,
ambient, time vectors, feature definability) is computed by prereg/fresh_metadata.py, which does NOT
read the Capacity field.

Candidate pool: B0025..B0056 cells present in data/raw_fresh/ (zip 2-6). B0025-28 appear in zip 2 and
zip 3 with identical sha256 and are counted once. Dev cells (B0005/6/7/18, zip 1) are never fresh.

A fresh cell is INCLUDED only if ALL of the following hold; otherwise it is listed "excluded, reason".

R1 Load profile (matches dev cells = 2 A constant current): on EVERY discharge cycle, the median of
   |Current_measured| over samples with |I| > 0.5 A lies in [1.90, 2.10] A.
R2 Ambient: ambient_temperature is recorded on every discharge cycle and its range within the cell is
   <= 2 C. Ambient value itself is NOT an inclusion criterion (dev = 24 C); each included cell is tagged
   with a stratum: T24 (24 +/- 2 C), T4 (4 +/- 2 C) or "other"; results are reported per stratum as well
   as pooled.
R3 Features: all 12 physics features of prep_phys_features.py (t_v38, t_v35, dt_38_35, v_300, v_1000,
   temp_rise_35 and their _rel versions) are defined and finite on EVERY discharge cycle, using that
   file's cycle_features() unchanged (so: load-on preceded by a sample, voltage reaches 3.5 V under
   load, >= 1000 s of load). A cycle that raises or gives NaN excludes the cell (no cycle dropping).
   Discharge cut-off voltage is not a separate criterion: cut-offs differ even among dev cells
   (README: 2.7/2.5/2.2/2.5 V) and the features use only samples above 3.5 V.
R4 Size: >= 50 discharge cycles.
R5 Label: Capacity field present, finite and > 0 on every discharge cycle; label =
   (Capacity / first-cycle Capacity < 0.80), exactly as dev. Each class (SOH<0.80 and SOH>=0.80) must
   contain >= 10% of the cell's cycles. This step is evaluated LAST and only for cells that passed R1-R4.
R6 Independence from dev: the cell's discharge time vectors must not be identical to those of a dev
   cell (see grouping below); if they are, excluded.

Grouping (shared-recording check, as B0005/6/7): two cells are in the same group if, over the discharge
cycle indices both have, at least 50% have exactly equal Time arrays (same length, same values). Groups
are the connected components. Same-group cells are dependent: they are split together (group-aware),
and the statistics treat a group as one independent unit wherever noted in PREREG.md. README
co-listing is reported for information only; the file comparison decides.

Known caveats written in advance (not rule inputs): the README says Capacity is "for discharge till
2.7V" while cut-offs differ per cell, so how Capacity was computed for cells cut off at other voltages
is unknown; READMEs of groups 41-56 say some discharges have very low capacity with unanalysed causes.
