"""MkDocs hook: auto-link first occurrence of terms on each page.

Two sources:
  1. glossary.md headings -> links to glossary.md#anchor
  2. includes/term_links.yml -> links to arbitrary pages (e.g. api/algos.md#ppo)

Skips code blocks, headings, existing links, and the target page itself.
Adds {.gl} attribute for subtle CSS styling.
"""

import re
from pathlib import Path

import yaml

# (term_text, target_path, pattern)
_TERMS: list[tuple[str, str, re.Pattern]] = []


def _add_term(term: str, target: str):
    """Add a term with appropriate case-sensitivity."""
    if len(term) <= 6 and term.isupper():
        pattern = re.compile(r"(?<![`\[/\w])" + re.escape(term) + r"(?![`\]\w\(])")
    else:
        pattern = re.compile(
            r"(?<![`\[/\w])" + re.escape(term) + r"(?![`\]\w\(])", re.IGNORECASE
        )
    _TERMS.append((term, target, pattern))


def on_startup(**kwargs):
    """Parse glossary.md headings and term_links.yml."""
    docs_dir = Path(__file__).parent.parent
    _TERMS.clear()

    # Source 1: glossary.md headings
    glossary_path = docs_dir / "glossary.md"
    if glossary_path.exists():
        heading_re = re.compile(r"^### (.+)$", re.MULTILINE)
        for match in heading_re.finditer(glossary_path.read_text()):
            raw_heading = match.group(1)

            # MkDocs anchor
            anchor = raw_heading.lower()
            anchor = re.sub(r"[^a-z0-9 -]", "", anchor)
            anchor = re.sub(r"\s+", "-", anchor).strip("-")

            target = f"glossary.md#{anchor}"

            # Display term: text before parenthetical
            display = re.split(r"\s*\(", raw_heading)[0].strip()
            _add_term(display, target)

            # Also match short abbreviation from parens
            if "(" in raw_heading:
                paren_content = re.search(r"\(([^)]+)\)", raw_heading)
                if paren_content:
                    abbr = paren_content.group(1).split(",")[0].strip()
                    if len(abbr) <= 6 and abbr.isupper():
                        _add_term(abbr, target)

    # Source 2: term_links.yml for non-glossary terms
    extra_path = docs_dir / "includes" / "term_links.yml"
    if extra_path.exists():
        extra = yaml.safe_load(extra_path.read_text()) or {}
        for term, target in extra.items():
            _add_term(str(term), str(target))

    # Sort longest first so "MuJoCo Warp" matches before "MuJoCo"
    _TERMS.sort(key=lambda t: -len(t[0]))


def on_page_markdown(markdown: str, page, config, files, **kwargs):
    """Replace first occurrence of each term with a link."""
    src = page.file.src_path

    # Compute relative prefix from this page to docs root
    depth = src.count("/")
    prefix = "../" * depth

    # Split: even parts = prose, odd parts = code (fenced + inline)
    code_re = re.compile(r"(```[\s\S]*?```|`[^`\n]+`)")
    parts = code_re.split(markdown)

    # First pass: find positions of first occurrence of each term
    replacements = []  # (part_idx, start, end, text, rel_target)
    matched_targets = set()

    for i in range(0, len(parts), 2):  # only prose parts
        part = parts[i]
        for term, target, pattern in _TERMS:
            if target in matched_targets:
                continue

            # Skip if this page IS the target page
            target_page = target.split("#")[0]
            if src == target_page:
                matched_targets.add(target)
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

                rel_target = prefix + target
                replacements.append((i, m.start(), m.end(), m.group(0), rel_target))
                matched_targets.add(target)
                break

    # Second pass: apply replacements in reverse order
    by_part: dict[int, list] = {}
    for r in replacements:
        by_part.setdefault(r[0], []).append(r)

    for idx, reps in by_part.items():
        part = parts[idx]
        for _, start, end, text, rel_target in sorted(reps, key=lambda r: -r[1]):
            link = f"[{text}]({rel_target}){{.gl}}"
            part = part[:start] + link + part[end:]
        parts[idx] = part

    return "".join(parts)
