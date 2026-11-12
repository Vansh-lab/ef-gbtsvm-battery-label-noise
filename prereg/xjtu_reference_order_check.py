"""XJTU: does the authors' "initial" (cycle-1, 0.2C) capacity exceed the cycle-2 (1C) capacity
this step uses as the SOH reference? A single comparison per cell (greater/not-greater), NOT a
value: tests the hypothesis in REPORT.md "XJTU next step" for why R6 failed on all 23 cells.
No capacity number is printed, returned or stored anywhere.

    python -I prereg/xjtu_reference_order_check.py
"""
import io
import zipfile

import numpy as np
from scipy.io import loadmat

ZIP = "data/raw_xjtu/Battery Dataset.zip"
PREFIX = "Battery Dataset/"
CELLS = ([f"Batch-1/2C_battery-{i}.mat" for i in range(1, 9)]
         + [f"Batch-2/3C_battery-{i}.mat" for i in range(1, 16)])


def main():
    zf = zipfile.ZipFile(ZIP)
    n_cycle1_lower, n_checked = 0, 0
    for name in CELLS:
        raw = zf.read(PREFIX + name)
        m = loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)
        cap = np.atleast_1d(np.asarray(m["summary"].discharge_capacity_Ah, dtype=float))
        c0, c1 = cap[0], cap[1]
        ok = np.isfinite(c0) and np.isfinite(c1) and c0 > 0 and c1 > 0
        lower = bool(ok and c1 < c0)          # cycle-2 (1C) capacity < cycle-1 (0.2C) capacity
        n_checked += int(ok)
        n_cycle1_lower += int(lower)
        print(f"  {name}: cycle-2(1C) capacity {'<' if lower else '>=' if ok else '?'} "
              f"cycle-1(0.2C) capacity -> {'supports hypothesis' if lower else 'does NOT support'}",
              flush=True)
        del raw, m, cap
    print(f"\n{n_cycle1_lower}/{n_checked} cells: 1C (reference) capacity < 0.2C ('initial') capacity.")
    print("No capacity value was printed.")


if __name__ == "__main__":
    main()
