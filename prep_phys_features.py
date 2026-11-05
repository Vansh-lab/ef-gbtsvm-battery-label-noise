"""Physics-guided, partial-discharge features for the NASA PCoE cells (Phase A, candidate a).

    python prep_phys_features.py data/raw/B0005.mat data/raw/B0006.mat \
           data/raw/B0007.mat data/raw/B0018.mat --out data/nasa_battery_phys.csv

Why a second feature file: B0005, B0006 and B0007 were recorded together, so each
discharge record has the same time vector for all three cells, and the record ends
after the last of them reaches cut-off. Every cell's record has a rest period with the
load off after its cut-off; for B0005-B0007 its length depends on the OTHER cells.
Whole-record statistics in nasa_battery.csv (discharge_time, v_mean, v_std, v_end,
t_mean, ...) therefore carry information from co-recorded cells, including a held-out
test cell. The features below use only samples under load, before the cell's own
cut-off:

    t_v38, t_v35   time from load-on to 3.8 V and to 3.5 V (linear interpolation)
    dt_38_35       t_v35 - t_v38, the time to drop from 3.8 V to 3.5 V
    v_300, v_1000  voltage 300 s and 1000 s after load-on (v_1000 can be below 3.5 V
                   in aged cells; it is still under load)
    temp_rise_35   temperature rise from load-on to 3.5 V

and, for each of these, its change relative to the same cell's first discharge cycle
(ratio for durations, difference for voltages and temperature), suffix _rel. The first
cycle is available for any cell in service and uses no labels.

Load-on time is the midpoint between the last sample without load and the first
sample with load: the switch happens inside that interval, whose length changes with
the logging rate (about 19 s or 10 s), so anchoring on either sample would shift every
time-based feature by the logging rate. The rest voltage before the load is not used:
it steps with the logging protocol at cycle 31 of B0005-B0007 by more than its ageing
trend (Phase A review).

Label: identical to prep_nasa_battery.py (SOH = capacity / first-cycle capacity <
--soh-threshold). Rows, order and labels match nasa_battery.csv; main() checks this.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat

from prep_nasa_battery import cycle_features as base_cycle_features

LOAD_ON_A = 0.5            # |current| above this = load applied (discharge is ~2 A)
DURATIONS = ["t_v38", "t_v35", "dt_38_35"]
LEVELS = ["v_300", "v_1000", "temp_rise_35"]


def _crossing_time(t, v, level):
    """First time v falls to `level`, linearly interpolated between samples."""
    k = int(np.argmax(v <= level))
    if v[k] > level:
        return np.nan
    if k == 0:
        return t[0]
    t0, t1, v0, v1 = t[k - 1], t[k], v[k - 1], v[k]
    return t0 + (v0 - level) * (t1 - t0) / (v0 - v1)


def cycle_features(d):
    t = np.atleast_1d(np.asarray(d.Time, dtype=float))
    v = np.atleast_1d(np.asarray(d.Voltage_measured, dtype=float))
    cur = np.atleast_1d(np.asarray(d.Current_measured, dtype=float))
    tmp = np.atleast_1d(np.asarray(d.Temperature_measured, dtype=float))
    on = np.where(np.abs(cur) > LOAD_ON_A)[0]
    k_on, k_off = on[0], on[-1]
    if k_on == 0:
        raise ValueError("no sample before load-on")
    t_on = 0.5 * (t[k_on - 1] + t[k_on])
    tl, vl, Tl = t[k_on:k_off + 1], v[k_on:k_off + 1], tmp[k_on:k_off + 1]
    t38 = _crossing_time(tl, vl, 3.8)
    t35 = _crossing_time(tl, vl, 3.5)
    if not (t_on + 1000 < t[k_off]):          # fixed-time voltages must be under load
        raise ValueError("load-off before 1000 s")
    return dict(
        t_v38=t38 - t_on,
        t_v35=t35 - t_on,
        dt_38_35=t35 - t38,
        v_300=float(np.interp(t_on + 300, tl, vl)),
        v_1000=float(np.interp(t_on + 1000, tl, vl)),
        temp_rise_35=float(np.interp(t35, tl, Tl)) - Tl[0],
    )


def load_battery(path):
    """Same cycle walk and skip rule as prep_nasa_battery.load_battery."""
    name = os.path.splitext(os.path.basename(path))[0]
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    key = name if name in m else [k for k in m if not k.startswith("__")][0]
    rows, k = [], 0
    for c in np.atleast_1d(m[key].cycle):
        if str(c.type) != "discharge":
            continue
        k += 1
        try:
            base_cycle_features(c.data, c.ambient_temperature)   # same skip rule as before
            f = cycle_features(c.data)
            f["capacity"] = float(np.asarray(c.data.Capacity).ravel()[0])
        except Exception as e:
            print(f"  {name}: skipping discharge #{k} ({e})")
            continue
        f.update(battery=name, cycle=k)
        rows.append(f)
    return pd.DataFrame(rows)


def add_relative(b):
    """Change relative to the cell's first cycle: ratio (durations), difference (levels)."""
    for c in DURATIONS:
        b[c + "_rel"] = b[c] / b[c].iloc[0]
    for c in LEVELS:
        b[c + "_rel"] = b[c] - b[c].iloc[0]
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mats", nargs="+")
    ap.add_argument("--out", default="data/nasa_battery_phys.csv")
    ap.add_argument("--soh-threshold", type=float, default=0.80)
    ap.add_argument("--check-against", default="data/nasa_battery.csv",
                    help="existing CSV whose battery order and labels must match ('' to skip)")
    a = ap.parse_args()

    frames = []
    for p in a.mats:
        b = load_battery(p)
        b["soh"] = b["capacity"] / b["capacity"].iloc[0]
        b["degraded"] = (b["soh"] < a.soh_threshold).astype(int)
        frames.append(add_relative(b))
        print(f"{b['battery'].iloc[0]}: {len(b)} discharge cycles, "
              f"degraded = {b['degraded'].sum()} ({100 * b['degraded'].mean():.0f}%)")
    df = pd.concat(frames, ignore_index=True)
    feats = DURATIONS + LEVELS + [c + "_rel" for c in DURATIONS + LEVELS]
    out = df[["battery"] + feats + ["degraded"]]
    if out.isna().any().any():
        raise SystemExit(f"NaN in features: {out.columns[out.isna().any()].tolist()}")
    if a.check_against:
        ref = pd.read_csv(a.check_against)
        same = (len(ref) == len(out)
                and (ref["battery"].values == out["battery"].values).all()
                and (ref["degraded"].values == out["degraded"].values).all())
        if not same:
            raise SystemExit(f"rows/labels differ from {a.check_against}")
        print(f"rows, battery order and labels match {a.check_against}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}: {out.shape}, features = {feats}")


if __name__ == "__main__":
    main()
