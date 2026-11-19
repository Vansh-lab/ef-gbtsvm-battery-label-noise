"""Prepare raster assets for the Word build.

Word embeds PNG, not PDF, so every figure is rasterised here, and every display
equation is typeset by LaTeX into its own small PNG so the maths stays correct.
"""
import os
import re
import shutil
import subprocess
import sys

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "docx_assets")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- figures
tex = open(os.path.join(HERE, "main.tex"), encoding="utf-8").read()
used = sorted(set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)))
sizes = {}
for name in used:
    src = os.path.join(FIG, name)
    if not os.path.exists(src):
        print("MISSING FIGURE:", name)
        continue
    doc = pymupdf.open(src)
    page = doc[0]
    # 200 dpi keeps small lettering legible on screen and in print
    pm = page.get_pixmap(dpi=200)
    png = name.replace(".pdf", ".png")
    pm.save(os.path.join(OUT, png))
    sizes[png] = (page.rect.width / 72.0, page.rect.height / 72.0)  # inches
    doc.close()
print(f"rasterised {len(sizes)} figures")

# ---------------------------------------------------------------- equations
# Pull the display-maths blocks straight out of main.tex, in order.
EQ_ENVS = ("equation", "align")
blocks = []
for m in re.finditer(r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}", tex, re.S):
    blocks.append((m.group(1), m.group(2)))
print(f"found {len(blocks)} display-maths blocks")

PREAMBLE = r"""\documentclass[12pt,border=2pt,varwidth=15cm]{standalone}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{newtxtext,newtxmath}
\begin{document}
"""
work = os.path.join(OUT, "_eqwork")
os.makedirs(work, exist_ok=True)
eq_meta = []
for i, (env, body) in enumerate(blocks, 1):
    # standalone cannot number equations; strip alignment-only markers it dislikes
    src = PREAMBLE + "\\begin{" + env + "*}" + body + "\\end{" + env + "*}\n\\end{document}\n"
    stem = f"eq{i:02d}"
    with open(os.path.join(work, stem + ".tex"), "w", encoding="utf-8") as f:
        f.write(src)
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", stem + ".tex"],
                       cwd=work, capture_output=True, text=True)
    pdf = os.path.join(work, stem + ".pdf")
    if not os.path.exists(pdf):
        print(f"  {stem}: FAILED to typeset")
        eq_meta.append(None)
        continue
    doc = pymupdf.open(pdf)
    page = doc[0]
    doc_pm = page.get_pixmap(dpi=300)
    doc_pm.save(os.path.join(OUT, stem + ".png"))
    eq_meta.append({"png": stem + ".png",
                    "w_in": page.rect.width / 72.0,
                    "h_in": page.rect.height / 72.0})
    doc.close()
ok = sum(1 for e in eq_meta if e)
print(f"typeset {ok}/{len(blocks)} equation images")

import json
json.dump({"figures": sizes, "equations": eq_meta},
          open(os.path.join(OUT, "assets.json"), "w"), indent=1)
print("wrote docx_assets/assets.json")
