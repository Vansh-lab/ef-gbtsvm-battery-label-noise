"""Build an IEEEtran (conference, two-column, A4) LaTeX version of the paper from the Claude Docs
export, plus static PDF versions of the two doc charts. Changes no number and no text: the prose,
tables, equations and references are taken from the doc as they stand.

    python paper_figures/build_ieee_tex.py            (reads paper_figures/doc_latest.xml)

Writes paper/ieee/: paper.tex, fig_acc_by_noise.pdf, fig_ranks.pdf, console_*.png
Left out of the PDF: the "Needs human review" checklist (an internal list, not paper content).
Display equations are the doc's own LaTeX, with each top-level \\qquad turned into a line break so
that they fit one column.
"""
import html
import os
import re
import shutil
from html.parser import HTMLParser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SRC = "paper_figures/doc_latest.xml"
OUT = "paper/ieee"
FIG_BLOBS = {"blob/c9f63a1a-5af7": "console_claims_main.png",
             "blob/0bab1c29-1b6e": "console_wilcoxon_main.png",
             "blob/4f33e175-cf82": "console_train_gap.png"}
EMBED_FIGS = {"node/d118714c-1932": "fig_acc_by_noise.pdf", "node/c3fabf0f-db18": "fig_ranks.pdf"}
WIDE = {"console_claims_main.png", "console_train_gap.png",
        "console_wilcoxon_main.png"}                                # figure* (both columns)
SKIP_SECTIONS = {"Needs human review"}

# ---- text -> LaTeX ---------------------------------------------------------------------------
MATH_RULES = [  # applied to raw text before escaping; each becomes a protected math token
    (r"q\u221a\(k\(k \+ 1\)/\(6N\)\)", lambda m: "$q\\sqrt{k(k+1)/(6N)}$"),
    (r"R\^\(([^)]+)\)", lambda m: "$\\mathbb{R}^{" + m.group(1).replace("\u00d7", "\\times ") + "}$"),
    (r"R\^([a-z])\b", lambda m: "$\\mathbb{R}^{" + m.group(1) + "}$"),
    (r"\bGB_j\b", lambda m: "$GB_j$"),
    (r"\(n_i \u03c0_i\)\^\u03b2", lambda m: "$(n_i\\pi_i)^{\\beta}$"),
    (r"\u03c6\(xW \+ b\u1d40\)", lambda m: "$\\phi(xW+b^{\\top})$"),
    (r"\u230a\u03c1n\u230b", lambda m: "$\\lfloor\\rho n\\rfloor$"),
    (r"(?<![A-Za-z])([A-Za-z\u0370-\u03ff])_([a-z])(?![A-Za-z])",
     lambda m: "$" + SYM.get(m.group(1), m.group(1)).strip("$") + "_" + m.group(2) + "$"),
    (r"([0-9]+)([\u207b\u2070\u00b9\u00b2\u00b3\u2074-\u2079]+)",
     lambda m: "$" + m.group(1) + "^{" + m.group(2).translate(SUP) + "}$"),
    (r"([A-Za-z\u0370-\u03ff\u2080-\u2089]+?)([\u2080-\u2089]+)",
     lambda m: "$" + "".join(SYM.get(c, c).strip("$") for c in m.group(1)) + "_{" + m.group(2).translate(SUB) + "}$"),
]
SUP = str.maketrans("\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079", "-0123456789")
SUB = str.maketrans("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089", "0123456789")
SYM = {"\u03c4": "$\\tau$", "\u03c0": "$\\pi$", "\u03b7\u0302": "$\\hat{\\eta}$", "\u03b7": "$\\eta$",
       "\u0394": "$\\Delta$", "\u03b4": "$\\delta$", "\u03c6": "$\\phi$", "\u03c8": "$\\psi$",
       "\u03d1": "$\\vartheta$", "\u03b3": "$\\gamma$", "\u03b1": "$\\alpha$", "\u03b2": "$\\beta$",
       "\u03c1": "$\\rho$", "\u03be": "$\\xi$", "\u03c7\u00b2": "$\\chi^2$", "\u03c7": "$\\chi$",
       "\u2208": "$\\in$", "\u2264": "$\\le$", "\u2265": "$\\ge$", "\u00d7": "$\\times$",
       "\u00b1": "$\\pm$", "\u2212": "$-$", "\u2192": "$\\rightarrow$", "\u221a": "$\\surd$",
       "\u2026": "\\ldots{}", "\u2013": "--", "\u2014": "---", "\u201c": "``", "\u201d": "''",
       "\u2018": "`", "\u2019": "'", "\u00a0": "~", "\u1d40": "$^{\\top}$", "\u00b2": "$^2$",
       "\u00b7": "$\\cdot$", "\u0161": "\\v{s}", "\u00e9": "\\'e", "\u00fc": "\\\"u", "\u00f6": "\\\"o"}
ESC = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#",
       "^": r"\^{}", "_": r"\_", "%": r"\%", "~": r"\~{}"}


def tex(s, cite=True):
    tokens = []

    def keep(latex):
        tokens.append(latex)
        return f"\x00{len(tokens) - 1}\x00"

    for pat, rep in MATH_RULES:
        s = re.sub(pat, lambda m: keep(rep(m)), s)
    s = re.sub(r'"([^"]*)"', lambda m: "“" + m.group(1) + "”", s)   # TeX-style quotes
    if cite:
        s = re.sub(r"(?<![A-Za-z])\[(\d+)\]", lambda m: keep(f"\\cite{{r{m.group(1)}}}"), s)
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        two = s[i:i + 2]
        if c == "\x00":
            j = s.index("\x00", i + 1)
            out.append(tokens[int(s[i + 1:j])])
            i = j + 1
            continue
        if two in SYM:
            out.append(SYM[two]); i += 2; continue
        out.append(SYM.get(c) or ESC.get(c) or c)
        i += 1
    return re.sub(r"\$\$", "", "".join(out))      # merge adjacent math tokens


class Inline(HTMLParser):
    """paragraph inner XML -> LaTeX, tracking bold/italic/code/link."""
    def __init__(self, cite=True):
        super().__init__(convert_charrefs=True)
        self.parts, self.stack, self.cite, self.buf = [], [], cite, []

    def flush(self):
        # the doc splits one run of text into several <text> nodes (e.g. "[" and "13]" after an
        # edit), so text is converted only at formatting boundaries
        if self.buf:
            self.parts.append(tex("".join(self.buf), self.cite))
            self.buf = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("bold", "italic", "code", "link", "hardbreak"):
            self.flush()
        if tag in ("bold", "italic", "code"):
            self.parts.append({"bold": r"\textbf{", "italic": r"\emph{", "code": r"\texttt{"}[tag]); self.stack.append(tag)
        elif tag == "link":
            self.parts.append(r"\href{" + a.get("href", "").replace("%", r"\%").replace("#", r"\#") + "}{"); self.stack.append(tag)
        elif tag == "hardbreak":
            self.parts.append(r"\\ ")

    def handle_endtag(self, tag):
        if tag in ("bold", "italic", "code", "link"):
            self.flush()
            self.parts.append("}")

    def handle_data(self, data):
        self.buf.append(data)


def inline(xml, cite=True):
    xml = re.sub(r"<(date|mention)[^>]*/>", "", xml)
    xml = xml.replace("<hardBreak", "<hardbreak")
    xml = re.sub(r"</?text\b[^>]*>", "", xml)
    xml = re.sub(r"</(bold|italic|code)><\1>", "", xml)     # one bold run split into several
    p = Inline(cite)
    p.feed(xml)
    p.flush()
    return "".join(p.parts).strip()


def plain(xml):
    return html.unescape(re.sub(r"<[^>]+>", "", xml))


# ---- block walk ------------------------------------------------------------------------------
def top_blocks(xml):
    body = re.search(r"<doc[^>]*>(.*)</doc>", xml, re.S).group(1)
    pos, out = 0, []
    pat = re.compile(r"<(paragraph|list|table|codeBlock|embed)\b")
    while True:
        m = pat.search(body, pos)
        if not m:
            break
        tag = m.group(1)
        if body[m.start():].startswith("<embed"):
            end = body.index("/>", m.start()) + 2
        else:
            depth, j = 0, m.start()
            open_re = re.compile(rf"<{tag}\b|</{tag}>")
            for mm in open_re.finditer(body, m.start()):
                depth += 1 if mm.group(0).startswith(f"<{tag}") else -1
                if depth == 0:
                    end = mm.end(); break
        out.append((tag, body[m.start():end]))
        pos = end
    return out


def split_display(eq):
    eq = eq.strip()
    parts = [p.strip(" ,") for p in re.split(r"\\qquad", eq) if p.strip(" ,")]
    lines, carry = [], ""
    for p in parts:
        if p == r"\text{s.t.}":
            carry = r"\text{s.t.}\ "; continue
        lines.append(carry + p); carry = ""
    body = r" \\ ".join(lines)
    return "\\begin{equation}\n\\begin{gathered}\n" + body + "\n\\end{gathered}\n\\end{equation}"


def table_tex(xml, caption, label):
    rows = re.findall(r"<row\b[^>]*>(.*?)</row>", xml, re.S)
    cells = [[inline(c) for c in re.findall(r"<cell\b[^>]*>(.*?)</cell>", r, re.S)] for r in rows]
    n = len(cells[0])
    longest = max(len(plain(c)) for r in rows for c in re.findall(r"<cell\b[^>]*>(.*?)</cell>", r, re.S))
    wide = n >= 7 or longest > 45
    if n >= 7:
        spec = "l" + "c" * (n - 1)
        tab = rf"\begin{{tabular}}{{{spec}}}"
        end = r"\end{tabular}"
    elif not wide:                       # small table: shrink to the column if it is too wide
        spec = "l" + "c" * (n - 1)
        tab = r"\resizebox{\columnwidth}{!}{" + rf"\begin{{tabular}}{{{spec}}}"
        end = r"\end{tabular}}"
    else:                                # the column with the longest text takes the free width
        per_col = [sum(len(plain(c)) for c in col) for col in
                   zip(*[re.findall(r"<cell\b[^>]*>(.*?)</cell>", r, re.S) for r in rows])]
        spec = "".join("X" if k == per_col.index(max(per_col)) else "l" for k in range(n))
        tab = rf"\begin{{tabularx}}{{\textwidth}}{{{spec}}}"
        end = r"\end{tabularx}"
    env = "table*" if wide else "table"
    lines = [rf"\begin{{{env}}}[!t]", r"\centering", r"\footnotesize" if n >= 7 else r"\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\setlength{\tabcolsep}{3pt}" if n >= 7 else "",
             tab, r"\toprule", " & ".join(cells[0]) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in cells[1:]]
    lines += [r"\bottomrule", end, rf"\end{{{env}}}"]
    return "\n".join(l for l in lines if l)


def figure_tex(fname, caption, label):
    env = "figure*" if fname in WIDE else "figure"
    width = r"\textwidth" if env == "figure*" else r"\columnwidth"
    return (rf"\begin{{{env}}}[!t]" "\n" r"\centering" "\n"
            rf"\includegraphics[width={width}]{{{fname}}}" "\n"
            rf"\caption{{{caption}}}" "\n" rf"\label{{{label}}}" "\n" rf"\end{{{env}}}")


def list_tex(xml):
    kind = re.search(r"kind='(\w+)'", xml).group(1)
    env = "enumerate" if kind == "ordered" else "itemize"
    items = re.findall(r"<listItem\b[^>]*>(.*?)</listItem>", xml, re.S)
    body = "\n".join(r"\item " + inline(re.sub(r"</?paragraph[^>]*>", " ", it)) for it in items)
    return f"\\begin{{{env}}}\n{body}\n\\end{{{env}}}"


def build():
    x = open(SRC, encoding="utf-8").read()
    blocks = top_blocks(x)
    title = authors = None
    out, abstract, keywords, refs = [], [], "", []
    section, pending_caption, fig_n = None, None, 0
    i = 0
    while i < len(blocks):
        tag, b = blocks[i]
        inner = re.sub(r"^<\w+[^>]*>|</\w+>$", "", b)
        h = re.search(r"heading='(\d)'", b[:200])
        if tag == "paragraph" and h:
            text = plain(inner).strip()
            if h.group(1) == "1":
                title = tex(text)
            else:
                section = re.sub(r"^\d+\.\s*", "", text)
                if section not in ("Abstract", "References") and section not in SKIP_SECTIONS:
                    star = "*" if not re.match(r"^\d+\.", text) else ""
                    out.append(f"\\section{star}{{{tex(section)}}}")
            i += 1; continue
        if section in SKIP_SECTIONS:
            i += 1; continue
        if title and section is None:              # byline / author block
            if "Vansh Sachan" in plain(inner):
                parts = [tex(p) for p in re.split(r"<hardBreak[^>]*/>", inner) for p in [plain(p)]]
                authors = parts
            i += 1; continue
        if section == "Abstract":
            t = plain(inner)
            if t.startswith("Keywords:"):
                keywords = tex(t.split(":", 1)[1].strip().rstrip("."))
            else:
                abstract.append(inline(inner))
            i += 1; continue
        if section == "References":
            refs = [inline(re.sub(r"</?paragraph[^>]*>", " ", it), cite=False)
                    for it in re.findall(r"<listItem\b[^>]*>(.*?)</listItem>", b, re.S)]
            i += 1; continue
        if tag == "paragraph":
            t = plain(inner).strip()
            mcap = re.match(r"^Table (\d+)\.\s*(.*)", t, re.S)
            media = re.search(r"<media[^>]*blob='([^']+)'", b)
            if mcap and i + 1 < len(blocks) and blocks[i + 1][0] == "table":
                cap = inline(inner)
                cap = re.sub(r"^\\textbf\{Table \d+\.\}\s*", "", cap)
                out.append(table_tex(blocks[i + 1][1], cap, f"tab:{mcap.group(1)}"))
                i += 2; continue
            if media:
                fname = FIG_BLOBS[media.group(1)]
                cap_xml = re.sub(r"^<\w+[^>]*>|</\w+>$", "", blocks[i + 1][1])
                cap = re.sub(r"^\\textbf\{Figure (\d+)\.\}\s*", "", inline(cap_xml))
                num = re.search(r"Figure (\d+)\.", plain(cap_xml)).group(1)
                out.append(figure_tex(fname, cap, f"fig:{num}"))
                i += 2; continue
            if t:
                out.append(inline(inner))
            i += 1; continue
        if tag == "embed":
            ref = re.search(r"ref='([^']+)'", b).group(1)
            cap_raw = html.unescape(re.search(r"caption='([^']*)'", b).group(1))
            num = re.match(r"Figure (\d+)\.", cap_raw).group(1)
            cap = tex(re.sub(r"^Figure \d+\.\s*", "", cap_raw))
            out.append(figure_tex(EMBED_FIGS[ref], cap, f"fig:{num}"))
            i += 1; continue
        if tag == "codeBlock":
            eq = html.unescape(re.sub(r"<!\[CDATA\[|\]\]>|<[^>]+>", "", inner))
            out.append(split_display(eq))
            i += 1; continue
        if tag == "list":
            out.append(list_tex(b))
            i += 1; continue
        if tag == "table":
            out.append(table_tex(b, "", "tab:x"))
            i += 1; continue
        i += 1

    bib = "\n".join(f"\\bibitem{{r{k}}} {r}" for k, r in enumerate(refs, 1))
    auth = r"\IEEEauthorblockN{" + authors[0].replace(r"\textbf{", "").rstrip("}") + "}" \
        + r"\IEEEauthorblockA{" + r" \\ ".join(authors[1:]) + "}"
    doc = rf"""\documentclass[conference,a4paper]{{IEEEtran}}
\usepackage{{amsmath,amssymb}}
\usepackage{{fontspec}}           % Tectonic is XeTeX-based: IEEEtran's Times (ptm) has no TU shapes
\setmainfont{{texgyretermes}}[Extension=.otf, UprightFont=*-regular, BoldFont=*-bold,
  ItalicFont=*-italic, BoldItalicFont=*-bolditalic]   % Times clone, so the page count matches IEEE's Times layout
\setmonofont{{texgyrecursor}}[Extension=.otf, UprightFont=*-regular, BoldFont=*-bold, Scale=0.95]
\usepackage{{graphicx}}
\usepackage{{booktabs,tabularx}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{url}}
\begin{{document}}
\title{{{title}}}
\author{{{auth}}}
\maketitle
\begin{{abstract}}
{chr(10).join(abstract)}
\end{{abstract}}
\begin{{IEEEkeywords}}
{keywords}
\end{{IEEEkeywords}}

{chr(10).join(chr(10) + o for o in out)}

\begin{{thebibliography}}{{30}}
{bib}
\end{{thebibliography}}
\end{{document}}
"""
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "paper.tex"), "w", encoding="utf-8").write(doc)
    for f in FIG_BLOBS.values():
        shutil.copy(os.path.join("paper_figures", f), os.path.join(OUT, f))
    return doc


# ---- static charts (same CSVs as the doc charts) ----------------------------------------------
def charts():
    plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False})
    d = pd.read_csv("paper_figures/chart_acc_by_noise.csv")
    order = d.groupby("model")["acc_mean"].mean().sort_values(ascending=True).index
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    shades = {0: 0.45, 10: 0.72, 20: 1.0}
    for k, m in enumerate(order):
        r = d[d.model == m]
        pf = m == "EF-GBTSVM-PF"
        col = "#1f5fbf" if pf else "#6b6b6b"
        ax.plot([r.acc_mean.min(), r.acc_mean.max()], [k, k], color=col, lw=1, alpha=0.5)
        for _, row in r.iterrows():
            ax.scatter(row.acc_mean, k, s=14, color=col, alpha=shades[row.noise_pct], zorder=3,
                       label=f"{row.noise_pct}% noise" if (k == 0) else None)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    for t in ax.get_yticklabels():
        if t.get_text() == "EF-GBTSVM-PF":
            t.set_fontweight("bold")
    ax.set_xlim(70, 100)
    ax.set_xlabel("mean test accuracy (%)")
    ax.grid(axis="x", color="#dddddd", lw=0.5)
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, frameon=False, fontsize=6.5)
    for h in leg.legend_handles:
        h.set_color("#6b6b6b")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_acc_by_noise.pdf"))
    plt.close(fig)

    r = pd.read_csv("paper_figures/chart_pooled_ranks.csv").sort_values("avg_rank", ascending=False)
    best, cd = r.avg_rank.min(), r.cd.iloc[0]
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    ax.axvspan(best, best + cd, color="#999999", alpha=0.15, lw=0)
    ax.axvline(best + cd, color="#777777", ls="--", lw=0.8)
    ax.text(best + cd - 0.1, len(r) - 0.3, f"best + CD ({cd:.2f})", ha="right", va="bottom", fontsize=6.5)
    for k, (_, row) in enumerate(r.iterrows()):
        pf = row.model == "EF-GBTSVM-PF"
        ax.scatter(row.avg_rank, k, s=14, color="#1f5fbf" if pf else "#6b6b6b", zorder=3)
    ax.set_yticks(range(len(r)))
    ax.set_yticklabels(r.model)
    for t in ax.get_yticklabels():
        if t.get_text() == "EF-GBTSVM-PF":
            t.set_fontweight("bold")
    ax.set_xlim(1, 15)
    ax.set_xticks(range(1, 16, 2))
    ax.set_xlabel("Friedman average rank (lower is better)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ranks.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    build()
    charts()
    print("wrote", sorted(os.listdir(OUT)))
