"""Metadata + rule R1-R4, R6 for fresh cells. Never reads the Capacity field.
    python -I prereg/fresh_metadata.py
"""
import glob, hashlib, itertools, json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from scipy.io import loadmat
from prep_phys_features import cycle_features

def load(path):
    name = os.path.splitext(os.path.basename(path))[0]
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    return name, np.atleast_1d(m[name].cycle)

def scan(path):
    name, cyc = load(path)
    types = [str(c.type) for c in cyc]
    dis = [c for c in cyc if str(c.type) == "discharge"]
    med, amb, vmin, tv, fail = [], [], [], [], []
    for k, c in enumerate(dis, 1):
        d = c.data
        t = np.atleast_1d(np.asarray(d.Time, float)); cur = np.atleast_1d(np.asarray(d.Current_measured, float))
        v = np.atleast_1d(np.asarray(d.Voltage_measured, float))
        on = np.abs(cur) > 0.5
        med.append(float(np.median(np.abs(cur[on]))) if on.any() else np.nan)
        amb.append(float(c.ambient_temperature)); vmin.append(float(v[on].min()) if on.any() else np.nan)
        tv.append(hashlib.sha1(t.tobytes()).hexdigest())
        try:
            f = cycle_features(d)
            if not all(np.isfinite(list(f.values()))): raise ValueError("nan")
        except Exception as e:
            fail.append((k, str(e)))
    return dict(name=name, n_total=len(cyc), n_dis=len(dis),
                n_charge=types.count("charge"), n_imp=types.count("impedance"),
                med_I=(np.nanmin(med), np.nanmax(med)), amb=(min(amb), max(amb)),
                vmin=(np.nanmin(vmin), np.nanmax(vmin)), tv=tv, feat_fail=fail,
                first_time=[float(x) for x in np.atleast_1d(cyc[0].time)] if hasattr(cyc[0], "time") else None)

files = {}
for p in sorted(glob.glob("data/raw/B00*.mat")): files[os.path.basename(p)[:5]] = ("dev", p)
for p in sorted(glob.glob("data/raw_fresh/*/B00*.mat")):
    n = os.path.basename(p)[:5]
    if n in files and files[n][0] == "fresh":
        a = hashlib.sha256(open(p, "rb").read()).hexdigest(); b = hashlib.sha256(open(files[n][1], "rb").read()).hexdigest()
        assert a == b, f"{n} duplicate differs"; continue
    files[n] = ("fresh", p)
info = {n: dict(scan(p), role=r, path=p) for n, (r, p) in files.items()}

# grouping
def share(a, b):
    k = min(len(a["tv"]), len(b["tv"]))
    return sum(x == y for x, y in zip(a["tv"][:k], b["tv"][:k])) / k if k else 0.0
names = list(info); parent = {n: n for n in names}
def find(x):
    while parent[x] != x: x = parent[x]
    return x
pairs = {}
for a, b in itertools.combinations(names, 2):
    s = share(info[a], info[b])
    if s > 0:
        pairs[f"{a}-{b}"] = round(s, 3)
    if s >= 0.5: parent[find(a)] = find(b)
groups = {}
for n in names: groups.setdefault(find(n), []).append(n)

rows = []
for n in names:
    i = info[n]; reasons = []
    if i["role"] == "fresh":
        if not (1.90 <= i["med_I"][0] and i["med_I"][1] <= 2.10): reasons.append(f"R1 load current median |I| range {i['med_I'][0]:.2f}-{i['med_I'][1]:.2f} A")
        if i["amb"][1] - i["amb"][0] > 2: reasons.append(f"R2 ambient varies {i['amb'][0]:.0f}-{i['amb'][1]:.0f} C")
        if i["feat_fail"]: reasons.append(f"R3 features undefined on {len(i['feat_fail'])} cycles (first: #{i['feat_fail'][0][0]} {i['feat_fail'][0][1]})")
        if i["n_dis"] < 50: reasons.append(f"R4 only {i['n_dis']} discharge cycles")
        g = next(v for v in groups.values() if n in v)
        if any(info[x]["role"] == "dev" for x in g): reasons.append("R6 shares recording with dev cell")
    i["reasons"] = reasons
    stratum = "T24" if abs(i["amb"][0]-24) <= 2 and i["amb"][1]-i["amb"][0] <= 2 else "T4" if abs(i["amb"][0]-4) <= 2 and i["amb"][1]-i["amb"][0] <= 2 else "other"
    i["stratum"] = stratum
    print(f"{n} {i['role']:5s} cycles={i['n_total']:4d} discharge={i['n_dis']:4d} chg={i['n_charge']} imp={i['n_imp']} "
          f"medI={i['med_I'][0]:.2f}-{i['med_I'][1]:.2f}A amb={i['amb'][0]:.1f}-{i['amb'][1]:.1f}C stratum={stratum} "
          f"Vmin_under_load={i['vmin'][0]:.2f}-{i['vmin'][1]:.2f} featfail={len(i['feat_fail'])} "
          f"{'PASS R1-R4,R6' if not reasons and i['role']=='fresh' else ('EXCL: ' + '; '.join(reasons) if reasons else '')}")
print("\nshared-time-vector pairs (fraction of paired discharge cycles with identical Time):")
for k, v in pairs.items(): print("  ", k, v)
print("\ngroups:", {k: v for k, v in groups.items() if len(v) > 1})
json.dump(dict(groups=list(groups.values()), pairs=pairs, passing=[n for n in names if info[n]["role"]=="fresh" and not info[n]["reasons"]],
               stratum={n: info[n]["stratum"] for n in names},
               excluded={n: info[n]["reasons"] for n in names if info[n]["reasons"]}),
          open("prereg/fresh_metadata.json", "w"), indent=1)
