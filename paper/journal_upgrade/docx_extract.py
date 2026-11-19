"""Turn main.tex into a JSON content model for the Word build.

Cross-references are resolved from main.aux, so the numbers in the Word file are the
same numbers the PDF prints. Anything this parser does not recognise is reported at the
end rather than dropped silently.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = json.load(open(os.path.join(HERE, "docx_assets", "assets.json")))
unhandled = []

# ----------------------------------------------------------------- label map
labels = {}
aux = open(os.path.join(HERE, "main.aux"), encoding="utf-8", errors="replace").read()
for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux):
    labels[m.group(1)] = m.group(2)

# ------------------------------------------------------------- bibliography
bib_src = open(os.path.join(HERE, "references_40.tex"), encoding="utf-8").read()
bib_items = re.findall(
    r"\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem|\\end\{thebibliography\})", bib_src, re.S)
bib_num = {k: i for i, (k, _) in enumerate(bib_items, 1)}

GREEK = {
    r"\tau": "τ", r"\pi": "π", r"\rho": "ρ", r"\chi": "χ", r"\delta": "δ",
    r"\alpha": "α", r"\gamma": "γ", r"\xi": "ξ", r"\psi": "ψ", r"\phi": "φ",
    r"\Delta": "Δ", r"\mathcal{G}": "G", r"\lfloor": "⌊", r"\rfloor": "⌋",
    r"\times": "×", r"\le": "≤", r"\ge": "≥", r"\pm": "±", r"\dagger": "†",
    r"\ldots": "…", r"\cdot": "·", r"\in": "∈", r"\{": "{", r"\}": "}",
}
SUP = {"2": "²", "3": "³", "-5": "⁻⁵", "\\top": "ᵀ"}


def demath(s):
    """Best-effort conversion of inline maths to Unicode text."""
    s = s.strip()
    for a, b in GREEK.items():
        s = s.replace(a, b)
    s = re.sub(r"\^\{?([-\d\\a-z]+)\}?", lambda m: SUP.get(m.group(1), "^" + m.group(1)), s)
    s = re.sub(r"_\{([^}]*)\}", lambda m: m.group(1), s)
    s = re.sub(r"_(\w)", r"\1", s)
    s = s.replace("\\,", " ").replace("\\;", " ").replace("\\ ", " ")
    s = re.sub(r"\\mathbb\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:text|mathrm)\{([^}]*)\}", r"\1", s)
    s = s.replace("$", "").replace("{", "").replace("}", "")
    return s


def cite(keys):
    nums = []
    for k in keys.split(","):
        k = k.strip()
        if k in bib_num:
            nums.append(bib_num[k])
        else:
            unhandled.append(f"unknown cite key {k}")
    nums.sort()
    return "[" + ", ".join(str(n) for n in nums) + "]"


def inline(s):
    """LaTeX inline markup -> list of runs [{text, bold, italic, url}]."""
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = s.replace("\\proposed", "EF-GBTSVM-PF").replace("\\nz", "%")
    s = re.sub(r"\\cite\{([^}]*)\}", lambda m: cite(m.group(1)), s)
    s = re.sub(r"\\ref\{([^}]*)\}", lambda m: labels.get(m.group(1), "?"), s)
    s = re.sub(r"\\footnote\{((?:[^{}]|\{[^{}]*\})*)\}", r" (\1)", s)
    s = re.sub(r"\\href\{([^}]*)\}\{((?:[^{}]|\{[^{}]*\})*)\}", r"\2 <\1>", s)
    s = re.sub(r"\\url\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\$([^$]*)\$", lambda m: demath(m.group(1)), s)
    s = s.replace("---", "\u2014").replace("--", "\u2013")
    s = re.sub(r"``([^']*)''", "\u201c\\1\u201d", s)
    for a, b in [("\\%", "%"), ("\\&", "&"), ("\\_", "_"), ("\\#", "#"),
                 ("\\ ", " "), ("\\,", " "),
                 ("~", " "), ("\\par", " "), ("\\smallskip", " "),
                 ("\\noindent", ""), ("\\textbullet", "\u2022"), ("\\/", "")]:
        s = s.replace(a, b)

    runs, pos = [], 0
    for m in re.finditer(r"\\(textbf|emph|textit)\{((?:[^{}]|\{[^{}]*\})*)\}", s):
        if m.start() > pos:
            runs.append({"text": s[pos:m.start()]})
        runs.append({"text": m.group(2),
                     "bold": m.group(1) == "textbf",
                     "italic": m.group(1) in ("emph", "textit")})
        pos = m.end()
    if pos < len(s):
        runs.append({"text": s[pos:]})
    out = []
    for r in runs:
        t = re.sub(r"\s+", " ", r["text"])
        t = re.sub(r"\\[a-zA-Z]+", "", t)
        if t:
            out.append({**r, "text": t})
    return out


# ------------------------------------------------------------------- tables
def _skip_group(s, i):
    """Advance past one balanced {...} or [...] group starting at or after i."""
    while i < len(s) and s[i] in " \n\t":
        i += 1
    if i >= len(s) or s[i] not in "{[":
        return i
    close = "}" if s[i] == "{" else "]"
    depth = 0
    while i < len(s):
        if s[i] in "{[":
            depth += 1
        elif s[i] in "}]":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return i


def tabular_body(chunk):
    """Return just the rows of the first tabular/tabularx in chunk.

    The column spec can itself contain braces (e.g. {l*{6}{c}}), so it is skipped by
    brace matching rather than by a regular expression.
    """
    m = re.search(r"\\begin\{tabular[xX]?\}", chunk)
    if not m:
        return chunk
    i = m.end()
    i = _skip_group(chunk, i)      # optional [pos] or the width of tabularx
    i = _skip_group(chunk, i)      # column spec (or width then spec)
    end = chunk.find("\\end{tabular", i)
    return chunk[i:end if end > 0 else len(chunk)]


def parse_tabular(chunk):
    """A LaTeX tabular -> {header:[row,...], rows:[row,...], shaded:[idx]}."""
    body = tabular_body(chunk)
    body = re.sub(r"\\(?:toprule|midrule|bottomrule)", "", body)
    body = re.sub(r"\\cmidrule(\(.*?\))?\{[^}]*\}", "", body)
    header, rows, shaded = [], [], []
    for raw in body.split("\\\\"):
        raw = re.sub(r"^\s*\[[-\d.a-z]*\]", "", raw).strip()   # drop \\[2pt] spacing
        if not raw:
            continue
        is_shaded = "\\cellcolor" in raw or "\\rowcolor" in raw
        raw = raw.replace("\\rowcolor{proposedrow}", "")
        cells, spans = [], []
        for c in raw.split("&"):
            c = c.replace("\\cellcolor{proposedrow}", "")
            mc = re.match(r"\s*\\multicolumn\{(\d+)\}\{[^}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}",
                          c)
            if mc:
                span, c = int(mc.group(1)), mc.group(2)
            else:
                span = 1
            txt = "".join(r["text"] for r in inline(c)) if c.strip() else ""
            cells.append(txt.strip())
            spans.append(span)
        if not any(cells):
            continue
        rows.append({"cells": cells, "spans": spans})
        if is_shaded:
            shaded.append(len(rows) - 1)
    # a leading row whose first cell is empty is a spanning condition header
    nhdr = 2 if (rows and rows[0]["cells"] and rows[0]["cells"][0] == "") else 1
    header, data = rows[:nhdr], rows[nhdr:]
    shaded = [i - nhdr for i in shaded if i >= nhdr]
    return {"header": header, "rows": [r["cells"] for r in data], "shaded": shaded}


# ------------------------------------------------------------------ walking
src = open(os.path.join(HERE, "main.tex"), encoding="utf-8").read()
body = src.split("\\begin{document}", 1)[1].split("\\input{references_40.tex}")[0]

blocks = []
eq_i = 0

# frontmatter
title = re.search(r"\\title\{\\textbf\{(.*?)\}\}", src, re.S).group(1)
blocks.append({"type": "title", "text": re.sub(r"\s+", " ", title)})

hl = re.search(r"\\textbf\{HIGHLIGHTS\}(.*?)\}\}", body, re.S).group(1)
items = [inline(x) for x in re.findall(r"\\textbullet\\?\s*(.*?)\\par", hl + "\\par")]
blocks.append({"type": "heading", "level": 2, "text": "Highlights"})
blocks.append({"type": "bullets", "items": [i for i in items if i]})

kw = re.search(r"\\textbf\{KEYWORDS\}\\par\\smallskip(.*?)\\end\{minipage\}", body, re.S).group(1)
kws = [re.sub(r"\s+", " ", k).strip() for k in kw.split("\\par") if k.strip()]
blocks.append({"type": "heading", "level": 2, "text": "Keywords"})
blocks.append({"type": "para", "runs": [{"text": " · ".join(kws)}]})

ab = re.search(r"\\textbf\{ABSTRACT\}\\par\\smallskip(.*?)\\end\{minipage\}", body, re.S).group(1)
blocks.append({"type": "heading", "level": 2, "text": "Abstract"})
blocks.append({"type": "para", "runs": inline(ab)})

main_body = body.split("\\end{frontmatter}", 1)[1]

# Pull floats and equations out first, leaving placeholders, so the remaining text
# can be split on blank lines without tripping over environments.
placeholders = {}


def stash(kind, payload):
    key = f"@@{kind}{len(placeholders)}@@"
    placeholders[key] = payload
    return "\n\n" + key + "\n\n"


def take_float(m):
    chunk = m.group(0)
    cap = re.search(r"\\caption\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", chunk)
    lab = re.search(r"\\label\{([^}]+)\}", chunk)
    num = labels.get(lab.group(1), "") if lab else ""
    imgs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", chunk)
    sub = re.findall(r"\\caption\{([^}]*)\}\\end\{subfigure\}", chunk)
    sub = ["".join(r["text"] for r in inline(s)) for s in sub]
    return stash("FIG", {"type": "figure", "number": num,
                         "caption": inline(cap.group(1)) if cap else [],
                         "images": [i.replace(".pdf", ".png") for i in imgs],
                         "subcaptions": sub})


def take_table(m):
    chunk = m.group(0)
    cap = re.search(r"\\caption\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", chunk)
    lab = re.search(r"\\label\{([^}]+)\}", chunk)
    num = labels.get(lab.group(1), "") if lab else ""
    inp = re.search(r"\\input\{tables/([^}]+)\}", chunk)
    if inp:
        tb = open(os.path.join(HERE, "tables", inp.group(1)), encoding="utf-8").read()
    else:
        tb = chunk
    grid = parse_tabular(tb)
    note = re.search(r"\\par\\smallskip\{\\scriptsize ([^}]*)\}", chunk)
    return stash("TAB", {"type": "table", "number": num,
                         "caption": inline(cap.group(1)) if cap else [],
                         "note": inline(note.group(1)) if note else [],
                         **grid})


def take_eq(m):
    global eq_i
    meta = ASSETS["equations"][eq_i] if eq_i < len(ASSETS["equations"]) else None
    eq_i += 1
    if not meta:
        unhandled.append("equation image missing")
        return ""
    return stash("EQ", {"type": "equation", **meta})


def take_algo(m):
    txt = m.group(0)
    txt = re.sub(r"\\noindent\\fbox\{\\begin\{minipage\}\{[^}]*\}", "", txt)
    txt = txt.replace("\\end{minipage}}", "")
    parts = [inline(p) for p in re.split(r"\\par", txt) if p.strip()]
    return stash("ALG", {"type": "callout", "paras": [p for p in parts if p]})


main_body = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", take_float, main_body, flags=re.S)
main_body = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", take_table, main_body, flags=re.S)
main_body = re.sub(r"\\begin\{(equation|align)\}.*?\\end\{\1\}", take_eq, main_body, flags=re.S)
main_body = re.sub(r"\\noindent\\fbox\{\\begin\{minipage\}.*?\\end\{minipage\}\}", take_algo,
                   main_body, flags=re.S)


def take_enum(m):
    items = re.findall(r"\\item\s*(.*?)(?=\\item|\Z)", m.group(1), re.S)
    return stash("ENUM", {"type": "numbered", "items": [inline(i) for i in items]})


main_body = re.sub(r"\\begin\{enumerate\}(?:\[[^\]]*\])?(.*?)\\end\{enumerate\}",
                   take_enum, main_body, flags=re.S)

# force every heading onto its own chunk, wherever it sits in the source
main_body = re.sub(r"(\\(?:sub)?section\*?\{[^}]*\}(?:\\label\{[^}]*\})?)",
                   r"\n\n\1\n\n", main_body)

for chunk in re.split(r"\n\s*\n", main_body):
    chunk = chunk.strip()
    if not chunk:
        continue
    if chunk in placeholders:
        blocks.append(placeholders[chunk])
        continue
    m = re.match(r"\\section\*?\{(.*?)\}(\\label\{[^}]*\})?$", chunk, re.S)
    if m:
        blocks.append({"type": "heading", "level": 1, "text": inline(m.group(1))[0]["text"]})
        continue
    m = re.match(r"\\subsection\*?\{(.*?)\}(\\label\{[^}]*\})?$", chunk, re.S)
    if m:
        blocks.append({"type": "heading", "level": 2, "text": inline(m.group(1))[0]["text"]})
        continue
    if chunk.startswith("\\begin{enumerate}"):
        items = re.findall(r"\\item\s*(.*?)(?=\\item|\\end\{enumerate\})", chunk, re.S)
        blocks.append({"type": "numbered", "items": [inline(i) for i in items]})
        continue
    if chunk.startswith("\\FloatBarrier") or chunk.startswith("\\appendix"):
        continue
    # a chunk may open with one or more headings before its first paragraph
    while True:
        lead = re.match(r"\\(sub)?section\*?\{(.*?)\}(\\label\{[^}]*\})?\s*(.*)",
                        chunk, re.S)
        if not lead:
            break
        blocks.append({"type": "heading", "level": 2 if lead.group(1) else 1,
                       "text": inline(lead.group(2))[0]["text"]})
        chunk = lead.group(4).strip()
    if not chunk:
        continue
    runs = inline(chunk)
    if runs:
        blocks.append({"type": "para", "runs": runs})
    if "\\begin{" in chunk:
        unhandled.append("possible unparsed env: " + chunk[:70])

# references
blocks.append({"type": "heading", "level": 1, "text": "References"})
refs = []
for n, (key, raw) in enumerate(bib_items, 1):
    url = re.search(r"\\href\{([^}]*)\}", raw)
    txt = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", raw)
    txt = "".join(r["text"] for r in inline(txt))
    txt = re.sub(r"\s*doi:\S+\s*$", "", txt).strip()
    refs.append({"n": n, "text": txt, "url": url.group(1) if url else ""})
blocks.append({"type": "refs", "items": refs})

json.dump({"blocks": blocks, "figsizes": ASSETS["figures"]},
          open(os.path.join(HERE, "docx_assets", "content.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

kinds = {}
for b in blocks:
    kinds[b["type"]] = kinds.get(b["type"], 0) + 1
print("blocks:", kinds)
print("equations placed:", eq_i, "of", len(ASSETS["equations"]))
if unhandled:
    print("\nNEEDS ATTENTION:")
    for u in unhandled[:15]:
        print("  -", u)
else:
    print("no unhandled constructs")
