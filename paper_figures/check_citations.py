"""Citation integrity check for the paper doc (run after any reference renumbering).

    python paper_figures/check_citations.py <old_doc_xml> <new_doc_read_json>

old_doc_xml       : doc XML before renumbering (rev 204)
new_doc_read_json : saved output of the docs `read` (kind "view") after renumbering

Checks
  1. every in-text [n] in the body is 1..N, and every reference 1..N is cited at least once
  2. references are sorted by (year, first-author surname, title)
  3. no citation changed target: for every paragraph, the references its citations point to
     (identified by title) are the same before and after
"""
import html
import json
import re
import sys
import unicodedata

CIT = re.compile(r"(?<![A-Za-z])\[(\d+)\]")


def paragraphs(xml):
    out, cur = [], None
    for pid, body in re.findall(r"<paragraph id='([^']+)'[^>]*>(.*?)</paragraph>", xml, re.S):
        if not pid.startswith("."):
            cur = pid.split(".")[0]
        else:
            pid = cur + pid
        out.append((pid, html.unescape(re.sub(r"<[^>]+>", "", body))))
    return out


def norm(t):
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def split(paras):
    i = next(k for k, p in enumerate(paras) if p[1].strip() == "References")
    return paras[:i], [p[1] for p in paras[i + 1:]]


def key(ref):
    return norm(re.search(r"\"([^\"]+)\"", ref).group(1))


def year(ref):
    return int(re.findall(r"\b(19\d\d|20\d\d)\b", re.split(r"\bdoi:|Available:", ref)[0])[-1])


def surname(ref):
    return norm(ref.split(",")[0].split(" and ")[0].split()[-1])


def main(old_xml_path, new_json_path):
    old = paragraphs(open(old_xml_path, encoding="utf-8").read())
    d = json.load(open(new_json_path, encoding="utf-8"))
    new_xml = json.loads([e["text"] for e in d if e["type"] == "text"][-1])["data"]["xml"]
    new = paragraphs(new_xml)
    old_body, old_refs = split(old)
    new_body, new_refs = split(new)
    n = len(new_refs)
    ok = True

    cited = [int(c) for _, t in new_body for c in CIT.findall(t)]
    bad = sorted({c for c in cited if not 1 <= c <= n})
    uncited = [k for k in range(1, n + 1) if k not in cited]
    print(f"references: {n} (before: {len(old_refs)}); in-text citations: {len(cited)}")
    print(f"[check 1] out-of-range citations: {bad or 'none'}; uncited references: {uncited or 'none'}")
    ok &= not bad and not uncited and n == len(old_refs)

    sort_key = [(year(r), surname(r), key(r)) for r in new_refs]
    is_sorted = sort_key == sorted(sort_key)
    print(f"[check 2] sorted by (year, first-author surname, title): {is_sorted}")
    for k, (y, s, _) in enumerate(sort_key, 1):
        print(f"    [{k:2d}] {y} {s}")
    ok &= is_sorted

    old_by_id = dict(old_body)
    changed, compared = [], 0
    for pid, t in new_body:
        if pid not in old_by_id:
            continue
        a = [key(old_refs[int(c) - 1]) for c in CIT.findall(old_by_id[pid])]
        b = [key(new_refs[int(c) - 1]) for c in CIT.findall(t)]
        if a or b:
            compared += 1
            if a != b:
                changed.append(pid)
    lost = [pid for pid, t in old_body if CIT.search(t) and pid not in dict(new_body)]
    print(f"[check 3] paragraphs with citations compared: {compared}; target changed: {changed or 'none'}; "
          f"paragraphs with citations missing after the edit: {lost or 'none'}")
    ok &= not changed and not lost
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
