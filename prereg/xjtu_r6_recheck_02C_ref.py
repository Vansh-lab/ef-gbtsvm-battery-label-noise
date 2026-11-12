"""XJTU: R6 re-check using the authors' own reference (cycle 1, 0.2C "initial" capacity) instead
of this step's cycle-2 (1C) reference. Diagnostic only, per Vansh's explicit go-ahead
(2026-11-11) -- does NOT change ADDENDUM_E1.md (its capacity-source decision, PREREG_E.md section
0, stays as written; this is a separate, clearly-labelled recheck, not an edit to either file).
Same discipline as the main R6 step: pass/fail and counts only, no SOH or capacity value is
printed, returned or stored anywhere.

    python -I prereg/xjtu_r6_recheck_02C_ref.py
"""
import gc
import io
import json
import sys
import zipfile

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, ".")
from prereg.xjtu_label_free_check import (CELLS, IRREGULAR_TOL, PREFIX, ZIP, process_cell)


def main():
    zf = zipfile.ZipFile(ZIP)
    cells = {name: process_cell(name, zf) for name in CELLS}
    pooled = [x for c in cells.values() for i, x in enumerate(c["cur_medians"])
              if i > 0 and np.isfinite(x)]
    M = float(np.median(pooled))

    rows = []
    for name, c in cells.items():
        med = np.array(c["cur_medians"])
        irregular = np.zeros(c["n_cycles"], dtype=bool)
        irregular[0] = True
        valid = np.isfinite(med)
        irregular[valid] |= (np.abs(med[valid] - M) / M > IRREGULAR_TOL)
        irregular[0] = True
        for i in c["fail_parse"]:
            irregular[i] = True

        raw = zf.read(PREFIX + name)
        m = loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)
        del raw
        cap = np.atleast_1d(np.asarray(m["summary"].discharge_capacity_Ah, dtype=float))
        del m
        n = len(cap)
        ref_ok = np.isfinite(cap[0]) and cap[0] > 0
        # cycles used for the SOH series: all REGULAR cycles after cycle 0 (cycle 0 itself is the
        # reference, not a labelled cycle, same as the main step's convention)
        regular_after0 = np.array([not irregular[c] for c in range(1, min(n, len(irregular)))])
        cap_series = cap[1:][regular_after0]
        finite_pos = bool(ref_ok and np.isfinite(cap_series).all() and (cap_series > 0).all())
        if finite_pos and len(cap_series) > 0:
            soh = cap_series / cap[0]
            reaches_080 = bool((soh < 0.80).any())
            frac_degraded = float((soh < 0.80).mean())
            class_ok = bool(0.10 <= frac_degraded <= 0.90)
        else:
            reaches_080, class_ok = False, False
        r6_pass = bool(finite_pos and reaches_080 and class_ok)
        rows.append(dict(name=name, n_regular_after0=int(regular_after0.sum()),
                         r6_02C_ref_pass=r6_pass, reaches_080=reaches_080,
                         class_balance_ok=class_ok))
        print(f"  {name}: n_regular={int(regular_after0.sum())} reaches_080={reaches_080} "
              f"class_balance_ok={class_ok} -> R6(0.2C ref) {'PASS' if r6_pass else 'FAIL'}", flush=True)
        del cap, cap_series
        gc.collect()

    n_pass = sum(r["r6_02C_ref_pass"] for r in rows)
    n_reach = sum(r["reaches_080"] for r in rows)
    print(f"\nR6 with 0.2C (cycle-1) reference: {n_pass}/{len(rows)} cells PASS "
          f"(reach SOH<0.80 AND class balance ok); {n_reach}/{len(rows)} reach SOH<0.80 at all "
          "(regardless of class balance).")
    print("No SOH or capacity value was printed.")
    json.dump(dict(reference="cycle-1 (0.2C, authors' initial)", rows=rows),
              open("prereg/xjtu_r6_recheck_02C_ref_results.json", "w"), indent=1)
    print("wrote prereg/xjtu_r6_recheck_02C_ref_results.json (pass/fail and counts only)")


if __name__ == "__main__":
    main()
