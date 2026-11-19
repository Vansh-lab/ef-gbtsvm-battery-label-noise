"""Final checks on the compiled manuscript (three reported conditions: 0, 6, 6.8 %)."""
import os
import re
import pandas as pd
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")
doc = pymupdf.open(os.path.join(HERE, "main.pdf"))
text = "\n".join(p.get_text() for p in doc)
flat = re.sub(r"\s+", " ", text).replace("±", "±")
ok = lambda b: "PASS" if b else "**FAIL**"

print(f"Pages: {len(doc)}  (limit 18)  -> {ok(len(doc) <= 18)}")

names = ["Vansh", "Sachan", "University of Delhi", "Faculty of Technology"]
print("Anonymity (no author name or affiliation):",
      ok(not any(n.lower() in text.lower() for n in names)),
      "| PDF author metadata:", repr(doc.metadata.get("author")))

heads = {p.get_text().split("\n")[0].strip() for p in doc[1:]}
print("Page headers carry no author name:", ok(all("Sachan" not in h for h in heads)))

MODELS = ["EF-GBTSVM", "EF-GBTSVM-A", "EF-GBTSVM-L", "EF-GBTSVM-W", "EF-GBTSVM-L8",
          "EF-GBTSVM-J", "EF-GBTSVM-PF", "GBTSVM", "HF-GBTSVM", "EF-TSVM", "TSVM",
          "RBF-SVM", "RVFL", "Random forest", "MLP", "XGBoost",
          "Single-feature threshold"]
print("All 17 models present:", ok(all(m in flat for m in MODELS)))

# --- every value for the three reported conditions must be printed verbatim
COMB = pd.read_csv(os.path.join(RES, "finegrid_plus_phaseb_0_10_combined.csv"))
FG2 = pd.read_csv(os.path.join(RES, "finegrid2_summary_table.csv"))
CONDITIONS = [("0%", COMB, 0), ("6%", COMB, 6), ("6.8%", FG2, 68)]
SRC = {"EF-GBTSVM", "EF-GBTSVM-A", "EF-GBTSVM-L", "EF-GBTSVM-W", "EF-GBTSVM-L8",
       "EF-GBTSVM-J", "EF-GBTSVM-PF", "GBTSVM", "HF-GBTSVM", "EF-TSVM", "TSVM",
       "SVM", "RVFL", "RandomForest", "MLP", "XGBoost", "Stump-fair"}
total = missing = 0
misses = []
for label, df, key in CONDITIONS:
    sub = df[df["noise_pct"] == key]
    for _, r in sub.iterrows():
        if r["model"] not in SRC:
            continue
        for m in ["ACC", "BACC", "F1", "AUC"]:
            total += 1
            s = f"{r[m+'_mean']:.2f}±{r[m+'_std']:.2f}"
            if s not in flat:
                missing += 1
                misses.append(f"{label} {r['model']} {m} {s}")
print(f"Reported-condition values printed verbatim: {total-missing}/{total} -> {ok(missing == 0)}")
for x in misses[:8]:
    print("   missing:", x)

# --- no value from a condition the paper no longer reports may appear
PHB = pd.read_csv(os.path.join(RES, "phase_b_summary_table.csv"))
DL = pd.read_csv(os.path.join(RES, "dlite_summary_table.csv"))
stale, checked = [], 0
reported = set()
for label, df, key in CONDITIONS:
    for _, r in df[df["noise_pct"] == key].iterrows():
        for m in ["ACC", "BACC", "F1", "AUC"]:
            reported.add(f"{r[m+'_mean']:.2f}±{r[m+'_std']:.2f}")
for df, keys, tag in [(COMB, [2, 4, 8, 10], "dropped sweep level"),
                      (PHB, [20], "20% level"),
                      (FG2, [62, 64, 66, 70, 72, 74, 76, 78], "exploratory level"),
                      (DL, ["N30", "N40", "BND"], "stress condition")]:
    for key in keys:
        for _, r in df[df["noise_pct"] == key].iterrows():
            for m in ["ACC", "BACC"]:
                s = f"{r[m+'_mean']:.2f}±{r[m+'_std']:.2f}"
                checked += 1
                if s in flat and s not in reported:
                    stale.append(f"{tag} {key} {r['model']} {m} {s}")
print(f"No value from a removed condition leaks in ({checked} checked): {ok(not stale)}")
for x in stale[:8]:
    print("   leaked:", x)

# --- the three condition labels, and only those, head the results tables
print("Condition labels present:",
      ok(all(c in flat for c in ["0%", "6%", "6.8%"])))

refs = sorted({int(n) for n in re.findall(r"\[(\d+)\] [A-Z]", flat)})
links = [l["uri"] for p in doc for l in p.get_links() if l.get("uri")]
dois = sorted({u for u in links if "doi.org" in u})
print(f"Reference entries: {len(refs)} -> {ok(len(refs) == 40)} | "
      f"distinct DOI links: {len(dois)} -> {ok(len(dois) == 40)}")

body = flat.split("References [1]")[0]
cites = [int(n) for g in re.findall(r"\[([\d,–\-]+)\]", body)
         for n in re.findall(r"\d+", g)]
seen, order = set(), []
for c in cites:
    if c not in seen:
        seen.add(c)
        order.append(c)
print("First citation in the body is [1]:", ok(order[:1] == [1]))

print("Proposed row labelled '(proposed)':", ok("EF-GBTSVM-PF (proposed)" in flat))

# --- unresolved cross-references
print("No unresolved references ('??'):", ok("??" not in text))
