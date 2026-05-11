"""
build_thebibliography.py
========================
Parse sn-bibliography.bib and the .tex files, emit a Nature-Communications-style
numbered \\begin{thebibliography} block in citation order.

This replaces the BibTeX workflow for the final NC submission, since NC requires
all references to be inlined in the manuscript .tex file (no separate .bib).

Output: bibliography.tex  (paste this into sn-article.tex in place of
                           \\bibliography{sn-bibliography})
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX_FILES = ["sn-article.tex", "result1.tex", "result2.tex", "result3.tex",
             "result4.tex", "result5.tex", "result6.tex",
             "methods.tex", "discussion.tex"]
BIB_FILE = HERE / "sn-bibliography.bib"
OUT_FILE = HERE / "bibliography.tex"


# --- Parse .bib --------------------------------------------------------

def parse_bib(path: Path) -> dict:
    """Minimal BibTeX parser: returns {key: {field: value, ...}}.

    Handles braced and quoted values; concatenates multi-line; ignores @comment
    and @preamble; preserves order of appearance.
    """
    text = path.read_text(encoding="utf-8")
    # Remove % comments (BibTeX doesn't have them officially but allow LaTeX-style)
    text = re.sub(r"(?m)^\s*%.*$", "", text)

    entries = {}
    i = 0
    n = len(text)
    while i < n:
        # find '@type{key,'
        m = re.search(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text[i:])
        if not m:
            break
        etype = m.group(1).lower()
        key = m.group(2)
        i += m.end()
        if etype in ("comment", "preamble", "string"):
            # skip to matching brace
            depth = 1
            while i < n and depth:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            continue
        fields = {}
        # Read fields up to balancing '}'
        depth = 1
        buf = ""
        while i < n and depth:
            c = text[i]
            if c == "{":
                depth += 1
                buf += c
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
                buf += c
            else:
                buf += c
            i += 1
        # Parse field=value pairs with proper brace-balanced value extraction
        j = 0
        L = len(buf)
        while j < L:
            # find 'fieldname='
            nm = re.search(r"(\w+)\s*=\s*", buf[j:])
            if not nm:
                break
            fname = nm.group(1).lower()
            j += nm.end()
            if j >= L:
                break
            c = buf[j]
            if c == "{":
                # find matching closing brace
                depth_v = 1
                j += 1
                start = j
                while j < L and depth_v:
                    if buf[j] == "{":
                        depth_v += 1
                    elif buf[j] == "}":
                        depth_v -= 1
                        if depth_v == 0:
                            break
                    j += 1
                fval = buf[start:j]
                j += 1  # skip closing brace
            elif c == '"':
                j += 1
                start = j
                while j < L and buf[j] != '"':
                    j += 1
                fval = buf[start:j]
                j += 1
            else:
                # number / unquoted
                nm2 = re.match(r"(\d+|\w+)", buf[j:])
                if not nm2:
                    j += 1
                    continue
                fval = nm2.group(1)
                j += nm2.end()
            # consume optional comma + whitespace
            while j < L and buf[j] in " \t\n\r,":
                j += 1
            fval = re.sub(r"\s+", " ", fval).strip()
            fields[fname] = fval
        fields["__type__"] = etype
        entries[key] = fields
        i += 1  # skip closing }
    return entries


# --- Find citation order ----------------------------------------------

CITE_RE = re.compile(r"\\(?:cite|citep|citet|citeauthor|citeyear|nocite)[a-z]*\*?\{([^}]+)\}")

def citation_order(tex_files: list[str]) -> list[str]:
    """Return list of unique citation keys in order of first appearance.

    Order across files = order in TEX_FILES list (which roughly mirrors
    the order in sn-article.tex via \\input).
    """
    seen = []
    for f in tex_files:
        try:
            text = (HERE / f).read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        for m in CITE_RE.finditer(text):
            for key in (k.strip() for k in m.group(1).split(",")):
                if key and key not in seen:
                    seen.append(key)
    return seen


# --- Format an entry in Nature style ----------------------------------

def format_authors(authors_field: str) -> str:
    """Convert 'Last, First and Last, First' -> 'Last, F. & Last, F.' (Nature)."""
    if not authors_field:
        return ""
    authors = [a.strip() for a in re.split(r"\s+and\s+", authors_field)]
    fmt = []
    for a in authors:
        if "," in a:
            last, first = a.split(",", 1)
            initials = "".join(
                f"{p[0]}." for p in first.replace(".", " ").split() if p
            )
            fmt.append(f"{last.strip()}, {initials}")
        else:
            # 'First Last' -> 'Last, F.'
            parts = a.split()
            if len(parts) >= 2:
                last = parts[-1]
                initials = "".join(f"{p[0]}." for p in parts[:-1] if p)
                fmt.append(f"{last}, {initials}")
            else:
                fmt.append(a)
    if len(fmt) == 1:
        return fmt[0]
    if len(fmt) == 2:
        return f"{fmt[0]} \\& {fmt[1]}"
    return ", ".join(fmt[:-1]) + f" \\& {fmt[-1]}"


def format_entry(key: str, e: dict) -> str:
    """Emit one \\bibitem{key} Author. Title. Journal Vol, Pages (Year)."""
    etype = e.get("__type__", "article")
    authors = format_authors(e.get("author", ""))
    title = e.get("title", "").strip().rstrip(".")
    title = re.sub(r"\s*\{(.*?)\}\s*", r"\1", title)  # strip braces
    year = e.get("year", "")
    parts = [authors] if authors else []
    if title:
        parts.append(f"{title}.")
    if etype == "article":
        journal = e.get("journal", "")
        volume = e.get("volume", "")
        pages = e.get("pages", "").replace("--", "--")
        bits = []
        if journal:
            bits.append(f"\\emph{{{journal}}}")
        if volume:
            bits.append(f"\\textbf{{{volume}}}")
        if pages:
            bits.append(pages)
        if bits:
            parts.append(", ".join(bits) + (f" ({year})." if year else "."))
        elif year:
            parts.append(f"({year}).")
    elif etype in ("inproceedings", "conference"):
        booktitle = e.get("booktitle", "")
        pages = e.get("pages", "")
        bits = []
        if booktitle:
            bits.append(f"In \\emph{{{booktitle}}}")
        if pages:
            bits.append(pages)
        if bits:
            parts.append(", ".join(bits) + (f" ({year})." if year else "."))
    elif etype in ("book", "incollection"):
        publisher = e.get("publisher", "")
        bits = []
        if publisher:
            bits.append(publisher)
        if bits:
            parts.append(", ".join(bits) + (f", {year}." if year else "."))
        elif year:
            parts.append(f"({year}).")
    elif etype == "misc":
        howp = e.get("howpublished", "") or e.get("note", "")
        if howp:
            parts.append(f"{howp}.")
        if year:
            parts.append(f"({year}).")
    else:
        if year:
            parts.append(f"({year}).")
    body = " ".join(s for s in parts if s).strip()
    return f"\\bibitem{{{key}}} {body}"


# --- Main --------------------------------------------------------------

def main():
    bib = parse_bib(BIB_FILE)
    print(f"[parse] {len(bib)} entries in {BIB_FILE.name}")

    order = citation_order(TEX_FILES)
    print(f"[order] {len(order)} unique cited keys")

    missing = [k for k in order if k not in bib]
    if missing:
        print(f"[warn] {len(missing)} cited keys NOT in bib: {missing}", file=sys.stderr)

    lines = ["\\begin{thebibliography}{99}"]
    for key in order:
        if key not in bib:
            lines.append(f"\\bibitem{{{key}}} [MISSING BIB ENTRY: {key}]")
            continue
        lines.append(format_entry(key, bib[key]))
    lines.append("\\end{thebibliography}")

    OUT_FILE.write_text("\n\n".join(lines), encoding="utf-8")
    print(f"[save] {OUT_FILE}  ({len(order)} \\bibitem entries)")


if __name__ == "__main__":
    main()
