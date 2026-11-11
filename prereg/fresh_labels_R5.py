"""Rule R5 only: class balance of cells that passed R1-R4,R6. Prints nothing but counts. No model."""
import json, glob, os, numpy as np
from scipy.io import loadmat
meta = json.load(open("prereg/fresh_metadata.json"))
for n in meta["passing"]:
    p = glob.glob(f"data/raw_fresh/*/{n}.mat")[0]
    cyc = [c for c in np.atleast_1d(loadmat(p, squeeze_me=True, struct_as_record=False)[n].cycle) if str(c.type) == "discharge"]
    cap = np.array([float(np.asarray(c.data.Capacity).ravel()[0]) for c in cyc])
    ok = np.isfinite(cap).all() and (cap > 0).all()
    soh = cap / cap[0]; lab = soh < 0.80
    print(n, "n=", len(cap), "capacity finite>0:", ok, "degraded:", int(lab.sum()), f"({100*lab.mean():.1f}%)",
          "healthy:", int((~lab).sum()), f"({100*(~lab).mean():.1f}%)", "R5", "PASS" if ok and 0.10 <= lab.mean() <= 0.90 else "FAIL")
