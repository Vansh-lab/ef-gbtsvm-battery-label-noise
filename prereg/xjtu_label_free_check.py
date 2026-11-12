"""XJTU Phase E next step: label-free R2/R3/R5 checks, then R6 pass/fail only.

    python -I prereg/xjtu_label_free_check.py

Scope (per Vansh, 2026-11-11): R2 (cell/group count) and R3 (same regime) are checked from
discharge current, temperature and timing signals ONLY -- no capacity, SOH or label is read for
them. R5 (feature finiteness) runs prep_phys_features.cycle_features() unchanged on a few
representative cycles per cell and reports finite/not-finite only; no feature VALUE is printed or
written anywhere, even though none of the 12 features is itself a SOH/capacity quantity. R6 is
run LAST, reads summary.discharge_capacity_Ah only for that one check, and reports PASS/FAIL (and
a cycle count) only -- no SOH, capacity or ratio value is ever printed, logged or returned.

Does not touch ADDENDUM_E1.md or PREREG_E.md (their hashes are unaffected by this script).
Capacity source decided earlier (prereg/E_xjtu_doc_check.md Part 4): summary.discharge_capacity_Ah,
one value per cycle, no mixing. First (0.2C) cycle of every cell is irregular by the documentation
(prereg/E_xjtu_doc_check.md Part 3); SOH reference = first REGULAR (1C) cycle.

Memory rule (ADDENDUM_E1.md section 7): the 2.4 GB zip is never loaded whole; one cell's member
.mat is read into a BytesIO (tens of MB), processed, then deleted + gc.collect() before the next
cell. Peak RSS is logged per cell (psutil). Untrusted downloaded data: run with -I.
"""
import gc
import hashlib
import io
import json
import os
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import psutil
from scipy.io import loadmat

sys.path.insert(0, os.getcwd())
from prep_phys_features import cycle_features  # unchanged, per PREREG_E.md section 0

ZIP = "data/raw_xjtu/Battery Dataset.zip"
PREFIX = "Battery Dataset/"
CELLS = ([f"Batch-1/2C_battery-{i}.mat" for i in range(1, 9)]
         + [f"Batch-2/3C_battery-{i}.mat" for i in range(1, 16)])
assert len(CELLS) == 23
LOAD_ON_A = 0.5                    # same threshold as prep_phys_features.LOAD_ON_A
IRREGULAR_TOL = 0.05               # +/-5% of the dataset median (PREREG_E.md R3 wording)
MAX_IRREGULAR_FRAC = 0.05          # R3 cutoff (ADDENDUM_E1.md section 6.3)
N_REPRESENTATIVE = 3               # R5: first/middle/last regular cycle
PROC = psutil.Process(os.getpid())


def rss_mb():
    return PROC.memory_info().rss / 1e6


def last_run(mask):
    """Index range (start, end) inclusive of the LAST contiguous True run in mask, or None."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    splits = np.where(np.diff(idx) > 1)[0]
    run = np.split(idx, splits + 1)[-1]
    return int(run[0]), int(run[-1])


def discharge_record(cyc):
    """One data(c) element (whole cycle: charge+rest+discharge) -> a NASA-style discharge-only
    record (Time in seconds, Voltage_measured, Current_measured, Temperature_measured), by taking
    the LAST contiguous |current|>0.5A run (discharge is always last in the documented protocol)
    plus one sample before it. Returns None if unparseable (no load found, or it starts at sample 0
    so there is no pre-sample)."""
    cur = np.atleast_1d(np.asarray(cyc.current_A, dtype=float))
    r = last_run(np.abs(cur) > LOAD_ON_A)
    if r is None or r[0] == 0:
        return None
    start, end = r[0] - 1, r[1]
    t = np.atleast_1d(np.asarray(cyc.relative_time_min, dtype=float))[start:end + 1] * 60.0
    v = np.atleast_1d(np.asarray(cyc.voltage_V, dtype=float))[start:end + 1]
    c = cur[start:end + 1]
    tmp = np.atleast_1d(np.asarray(cyc.temperature_C, dtype=float))[start:end + 1]
    return SimpleNamespace(Time=t, Voltage_measured=v, Current_measured=c, Temperature_measured=tmp), \
        float(np.median(np.abs(cur[r[0]:r[1] + 1])))


def cell_timestamp_fingerprint(cyc):
    """Structural fingerprint of one cycle's timing, for the R2 shared-recording check. Treats
    system_time as opaque (no attempt to interpret it as a real time): only equality matters."""
    st = np.asarray(cyc.system_time).ravel()
    n = len(st)
    if n == 0:
        sample = ""
    else:
        idxs = sorted(set([0, n // 2, n - 1]))
        sample = repr([str(st[i]) for i in idxs])
    return hashlib.sha256(f"{n}|{sample}".encode()).hexdigest()


def process_cell(name, zf):
    raw = zf.read(PREFIX + name)
    m = loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)
    del raw
    data = np.atleast_1d(m["data"])
    n_cycles = len(data)

    cur_medians = np.full(n_cycles, np.nan)       # discharge |I| median per cycle (NOT a label)
    temp_min = np.full(n_cycles, np.nan)           # whole-cycle temperature_C min/max (NOT a label)
    temp_max = np.full(n_cycles, np.nan)
    fail_parse = []                                # cycles where no discharge segment was found
    fp = {}                                        # representative-cycle fingerprints for R2

    rep_idx = sorted(set([0, n_cycles // 2, n_cycles - 1]))
    for c in range(n_cycles):
        cyc = data[c]
        tmp = np.atleast_1d(np.asarray(cyc.temperature_C, dtype=float))
        if tmp.size and np.isfinite(tmp).any():
            finite = tmp[np.isfinite(tmp)]
            temp_min[c], temp_max[c] = finite.min(), finite.max()
        rec = discharge_record(cyc)
        if rec is None:
            fail_parse.append(c)
            continue
        _, cur_medians[c] = rec
        if c in rep_idx:
            fp[c] = cell_timestamp_fingerprint(cyc)

    out = dict(name=name, n_cycles=n_cycles, cur_medians=cur_medians.tolist(),
               temp_min=float(np.nanmin(temp_min)) if np.isfinite(temp_min).any() else None,
               temp_max=float(np.nanmax(temp_max)) if np.isfinite(temp_max).any() else None,
               temp_finite=bool(np.isfinite(temp_min).any()),
               fail_parse=fail_parse, fingerprints=fp)
    del data, m
    gc.collect()
    return out


def representative_regular_cycles(n_cycles, irregular_mask, k=N_REPRESENTATIVE):
    regular = [c for c in range(n_cycles) if not irregular_mask[c]]
    if not regular:
        return []
    if len(regular) <= k:
        return regular
    picks = [regular[0], regular[len(regular) // 2], regular[-1]]
    return sorted(set(picks))


def r5_r6_check(name, zf, irregular_mask):
    """One reload (not two) for both checks: R5 (cycle_features finiteness on representative
    regular cycles -- values never kept/printed) and R6 (reads summary.discharge_capacity_Ah
    only for this, returns booleans/counts only, no capacity or SOH value kept)."""
    raw = zf.read(PREFIX + name)
    m = loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)
    del raw
    data = np.atleast_1d(m["data"])
    picks = representative_regular_cycles(len(data), irregular_mask)
    n_finite = 0
    for c in picks:
        rec = discharge_record(data[c])
        if rec is None:
            continue
        try:
            f = cycle_features(rec[0])
            if all(np.isfinite(list(f.values()))):
                n_finite += 1
        except Exception:
            pass
    n_cycles_data = len(data)
    del data

    cap = np.atleast_1d(np.asarray(m["summary"].discharge_capacity_Ah, dtype=float))
    del m
    n = len(cap)
    regular = np.array([not irregular_mask[c] for c in range(min(n, len(irregular_mask)))])
    cap_reg = cap[regular]
    finite_pos = bool(np.isfinite(cap_reg).all() and (cap_reg > 0).all())
    r6 = dict(finite_positive=finite_pos, n_regular=int(regular.sum()))
    if finite_pos and len(cap_reg) > 1:
        soh = cap_reg / cap_reg[0]
        reaches_080 = bool((soh < 0.80).any())
        frac_degraded = float((soh < 0.80).mean())
        class_ok = bool(0.10 <= frac_degraded <= 0.90)
        r6.update(reaches_080=reaches_080, class_balance_ok=class_ok)
    else:
        r6.update(reaches_080=False, class_balance_ok=False)
    del cap, cap_reg
    gc.collect()
    return len(picks), n_finite, r6


def main():
    zf = zipfile.ZipFile(ZIP)
    print(f"R2: {len(CELLS)} candidate cells (Batch-1: 8, Batch-2: 15)", flush=True)

    # ---- pass 1: per-cell current medians, temp range, representative fingerprints ----
    cells = {}
    for name in CELLS:
        rss0 = rss_mb()
        cells[name] = process_cell(name, zf)
        print(f"  read {name}: {cells[name]['n_cycles']} cycles, "
              f"unparsed={len(cells[name]['fail_parse'])}, RSS {rss0:.0f}->{rss_mb():.0f} MB", flush=True)

    # dataset-wide median of NON-FIRST-cycle discharge current medians (pooled), for the R3 test.
    # cycle 0 is excluded a priori: the documentation says it is always a different (0.2C) protocol.
    pooled = []
    for c in cells.values():
        pooled += [x for i, x in enumerate(c["cur_medians"]) if i > 0 and np.isfinite(x)]
    M = float(np.median(pooled))
    print(f"\nR3: dataset-wide median of non-first-cycle discharge |I| medians over {len(pooled)} "
          f"cycles computed (value not printed; used only as a tolerance anchor).", flush=True)

    # ---- R2: grouping from representative-cycle fingerprints ----
    names = list(cells)
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = cells[names[i]]["fingerprints"], cells[names[j]]["fingerprints"]
            common = set(a) & set(b)
            if common and all(a[k] == b[k] for k in common):
                parent[find(names[i])] = find(names[j])
    groups = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    n_groups = len(groups)

    # ---- per-cell irregular-cycle classification + R3/R2 verdicts ----
    rows = []
    for name, c in cells.items():
        med = np.array(c["cur_medians"])
        irregular = np.zeros(c["n_cycles"], dtype=bool)
        irregular[0] = True                                             # doc-known 0.2C cycle
        valid = np.isfinite(med)
        irregular[valid] |= (np.abs(med[valid] - M) / M > IRREGULAR_TOL)
        irregular[0] = True                                             # re-assert after the mask op
        for i in c["fail_parse"]:
            irregular[i] = True                                         # unparsed = can't confirm regular
        frac_irr = float(irregular.mean())
        r3_pass = frac_irr <= MAX_IRREGULAR_FRAC and c["temp_finite"]
        rows.append(dict(name=name, n_cycles=c["n_cycles"], n_groups_total=n_groups,
                         group=find(name), frac_irregular=round(frac_irr, 4),
                         n_irregular=int(irregular.sum()), r3_pass=r3_pass,
                         temp_finite=c["temp_finite"], _irregular=irregular))
        print(f"  {name}: n_cycles={c['n_cycles']} irregular={int(irregular.sum())} "
              f"({100*frac_irr:.1f}%) temp_finite={c['temp_finite']} -> R3 {'PASS' if r3_pass else 'FAIL'}",
              flush=True)

    r2_pass_dataset = len(CELLS) >= 8 and n_groups >= 6
    print(f"\nR2: {len(CELLS)} cells, {n_groups} groups after fingerprint matching -> "
          f"{'PASS' if r2_pass_dataset else 'FAIL'} (needs >=8 cells, >=6 groups)", flush=True)

    # ---- R5 (feature finiteness) + R6 (degradation/class balance), one reload per cell ----
    print("\nR5 + R6 (R6 reads summary.discharge_capacity_Ah only now; no value printed):", flush=True)
    for row in rows:
        n_checked, n_finite, r6 = r5_r6_check(row["name"], zf, row["_irregular"])
        row["r5_checked"], row["r5_finite"] = n_checked, n_finite
        row["r5_pass"] = n_checked > 0 and n_finite == n_checked
        row["r6_pass"] = bool(r6["finite_positive"] and r6["reaches_080"] and r6["class_balance_ok"])
        row["r6_n_regular"] = r6["n_regular"]
        print(f"  {row['name']}: R5 {n_finite}/{n_checked} finite -> "
              f"{'PASS' if row['r5_pass'] else 'FAIL'}; R6 {'PASS' if row['r6_pass'] else 'FAIL'} "
              f"(n_regular_cycles={r6['n_regular']})", flush=True)

    # ---- summary table (pass/fail + counts only; no SOH/capacity value anywhere) ----
    print("\n=== Summary: cell x (R2, R3, R5, R6) ===")
    print(f"{'cell':26s} {'group':6s} {'R2':5s} {'R3':5s} {'R5':5s} {'R6':5s}")
    for row in rows:
        gid = list(groups).index(row["group"]) + 1
        print(f"{row['name']:26s} g{gid:<5d} "
              f"{'PASS' if r2_pass_dataset else 'FAIL':5s} "
              f"{'PASS' if row['r3_pass'] else 'FAIL':5s} "
              f"{'PASS' if row['r5_pass'] else 'FAIL':5s} "
              f"{'PASS' if row['r6_pass'] else 'FAIL':5s}")

    n_all_pass = sum(row["r3_pass"] and row["r5_pass"] and row["r6_pass"] for row in rows)
    print(f"\nCells passing R2(dataset)+R3+R5+R6 all four: {n_all_pass} / {len(rows)}")
    print(f"Groups among the cells that pass R3+R5+R6: "
          f"{len({row['group'] for row in rows if row['r3_pass'] and row['r5_pass'] and row['r6_pass']})}")

    for row in rows:
        row.pop("_irregular")
    json.dump(dict(n_cells=len(CELLS), n_groups=n_groups, r2_pass=r2_pass_dataset, rows=rows),
              open("prereg/xjtu_label_free_results.json", "w"), indent=1)
    print("\nwrote prereg/xjtu_label_free_results.json (pass/fail and counts only)")


if __name__ == "__main__":
    main()
