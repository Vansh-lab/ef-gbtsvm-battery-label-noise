"""Turn the NASA PCoE Li-ion battery .mat files (B0005, B0006, B0007, B0018, ...)
into ONE binary-classification CSV: one row per DISCHARGE cycle.

    python prep_nasa_battery.py data/raw/B0005.mat data/raw/B0006.mat \
           data/raw/B0007.mat data/raw/B0018.mat --out data/nasa_battery.csv

Label  : degraded = 1 if SOH = capacity / first-cycle capacity < --soh-threshold
         (default 0.80), else healthy = 0.
Features (only things a BMS can measure during a discharge; NO capacity / SOH):
         discharge_time, v_mean, v_min, v_std, v_start, v_end, t_mean, t_max,
         t_rise, time_to_tmax, i_mean, time_to_v3p5, ambient_temp
         (+ cycle index only if --keep-cycle; it is a near-proxy for age)
Column `battery` is the group id for leave-one-battery-out evaluation:
    python run_experiments.py --data data/nasa_battery.csv --label-col degraded \
           --group-col battery ...

Note: discharge_time correlates strongly with capacity (capacity = integral of
current over time), so this task is easy for good models. That is physically
legitimate, but state it in the paper, and report SOH-threshold sensitivity.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat


def cycle_features(d, ambient):
    t = np.atleast_1d(np.asarray(d.Time, dtype=float))
    v = np.atleast_1d(np.asarray(d.Voltage_measured, dtype=float))
    cur = np.atleast_1d(np.asarray(d.Current_measured, dtype=float))
    tmp = np.atleast_1d(np.asarray(d.Temperature_measured, dtype=float))
    below = np.where(v <= 3.5)[0]
    return dict(
        discharge_time=t[-1] - t[0],
        v_mean=v.mean(), v_min=v.min(), v_std=v.std(), v_start=v[0], v_end=v[-1],
        t_mean=tmp.mean(), t_max=tmp.max(), t_rise=tmp.max() - tmp[0],
        time_to_tmax=t[tmp.argmax()] - t[0],
        i_mean=cur.mean(),
        time_to_v3p5=(t[below[0]] - t[0]) if len(below) else (t[-1] - t[0]),
        ambient_temp=float(ambient),
    )


def load_battery(path):
    name = os.path.splitext(os.path.basename(path))[0]
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    key = name if name in m else [k for k in m if not k.startswith("__")][0]
    cycles = np.atleast_1d(m[key].cycle)
    rows, k = [], 0
    for c in cycles:
        if str(c.type) != "discharge":
            continue
        k += 1
        try:
            f = cycle_features(c.data, c.ambient_temperature)
            f["capacity"] = float(np.asarray(c.data.Capacity).ravel()[0])
        except Exception as e:                      # corrupt / empty cycle
            print(f"  {name}: skipping discharge #{k} ({e})")
            continue
        f.update(battery=name, cycle=k)
        rows.append(f)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mats", nargs="+")
    ap.add_argument("--out", default="data/nasa_battery.csv")
    ap.add_argument("--soh-threshold", type=float, default=0.80)
    ap.add_argument("--keep-cycle", action="store_true")
    ap.add_argument("--drop-feature", default="",
                    help="comma-separated feature names to remove, e.g. discharge_time,i_mean")
    a = ap.parse_args()

    frames = []
    for p in a.mats:
        b = load_battery(p)
        b["soh"] = b["capacity"] / b["capacity"].iloc[0]
        b["degraded"] = (b["soh"] < a.soh_threshold).astype(int)
        print(f"{b['battery'].iloc[0]}: {len(b)} discharge cycles, capacity "
              f"{b['capacity'].iloc[0]:.3f} -> {b['capacity'].iloc[-1]:.3f} Ah, "
              f"degraded = {b['degraded'].sum()} ({100 * b['degraded'].mean():.0f}%)")
        frames.append(b)
    df = pd.concat(frames, ignore_index=True)
    drop = ["capacity", "soh"] + ([] if a.keep_cycle else ["cycle"])
    extra = [s.strip() for s in a.drop_feature.split(",") if s.strip()]
    unknown = [c for c in extra if c not in df.columns]
    if unknown:
        raise SystemExit(f"--drop-feature: unknown column(s) {unknown}")
    feats = [c for c in df.columns if c not in drop + extra + ["battery", "degraded"]]
    out = df[["battery"] + feats + ["degraded"]].dropna()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}: {out.shape}, features = {feats}")


if __name__ == "__main__":
    main()
