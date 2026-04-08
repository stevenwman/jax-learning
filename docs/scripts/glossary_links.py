"""MkDocs hook: auto-link first occurrence of glossary terms on each page.

Reads glossary.md headings, builds a term->anchor map, and replaces the first
occurrence of each term with a subtle link. Skips code blocks, headings, and
existing links. Adds {.gl} attribute for CSS styling.
"""

import re
from pathlib import Path

# Built once at startup
_TERMS: list[tuple[str, str, re.Pattern]] = []


def on_startup(**kwargs):
    """Parse glossary.md and build term->anchor lookup."""
    glossary_path = Path(__file__).parent.parent / "glossary.md"
    if not glossary_path.exists():
        return

    _TERMS.clear()
    heading_re = re.compile(r"^### (.+)$", re.MULTILINE)

    for match in heading_re.finditer(glossary_path.read_text()):
        raw_heading = match.group(1)

        # MkDocs anchor: lowercase, spaces to hyphens, strip non-alphanum
        anchor = raw_heading.lower()
        anchor = re.sub(r"[^a-z0-9 -]", "", anchor)
        anchor = re.sub(r"\s+", "-", anchor).strip("-")

        # Display term: text before parenthetical
        display = re.split(r"\s*\(", raw_heading)[0].strip()

        # Collect match terms
        terms_to_match = [display]
        if "(" in raw_heading:
            paren_content = re.search(r"\(([^)]+)\)", raw_heading)
            if paren_content:
                abbr = paren_content.group(1).split(",")[0].strip()
                if len(abbr) <= 6 and abbr.isupper():
                    terms_to_match.append(abbr)

        for term in terms_to_match:
            if len(term) <= 5 and term.isupper():
                pattern = re.compile(r"(?<![`\[/\w])" + re.escape(term) + r"(?![`\]\w\(])")
            else:
                pattern = re.compile(
                    r"(?<![`\[/\w])" + re.escape(term) + r"(?![`\]\w\(])", re.IGNORECASE
                )
            _TERMS.append((term, anchor, pattern))

    # Sort longest first so "MuJoCo Warp" matches before "MuJoCo"
    _TERMS.sort(key=lambda t: -len(t[0]))


def on_page_markdown(markdown: str, page, config, files, **kwargs):
    """Replace first occurrence of each glossary term with a link."""
    if page.file.src_path == "glossary.md":
        return markdown

    # Compute relative path from this page to glossary.md
    depth = page.file.src_path.count("/")
    glossary_rel = "../" * depth + "glossary.md"

    # Split: even parts = prose, odd parts = code (fenced + inline)
    code_re = re.compile(r"(```[\s\S]*?```|`[^`\n]+`)")
    parts = code_re.split(markdown)

    # First pass: find positions of first occurrence of each term
    replacements = []  # (part_idx, start, end, term, anchor)
    matched_anchors = set()

    for i in range(0, len(parts), 2):  # only prose parts
        part = parts[i]
        for term, anchor, pattern in _TERMS:
            if anchor in matched_anchors:
                continue
            for m in pattern.finditer(part):
                start = m.start()
                # Skip heading lines
                line_start = part.rfind("\n", 0, start) + 1
                line_end = part.find("\n", start)
                if line_end == -1:
                    line_end = len(part)
                line = part[line_start:line_end]
                if line.lstrip().startswith("#"):
                    continue
                # Skip if inside existing markdown link
                before = part[max(0, start - 80):start]
                if re.search(r"\[[^\]]*$", before):
                    continue
                # Skip if inside HTML tag
                if re.search(r"<[^>]*$", before):
                    continue
                # Skip if inside bold markers
                if before.endswith("**"):
                    continue

                replacements.append((i, m.start(), m.end(), m.group(0), anchor, glossary_rel))
                matched_anchors.add(anchor)
                break

    # Second pass: apply replacements in reverse order (so offsets stay valid)
    # Group by part index
    by_part: dict[int, list] = {}
    for r in replacements:
        by_part.setdefault(r[0], []).append(r)

    for idx, reps in by_part.items():
        part = parts[idx]
        # Apply in reverse offset order
        for _, start, end, text, anchor, grel in sorted(reps, key=lambda r: -r[1]):
            link = f"[{text}]({grel}#{anchor}){{.gl}}"
            part = part[:start] + link + part[end:]
        parts[idx] = part

    return "".join(parts)
