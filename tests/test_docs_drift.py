"""Docs drift auditor — cross-reference checks.

Each function catches a specific class of drift that's been reported in
multiple doc-review rounds:

1. ``test_no_reverted_obs_dims``            — 51d / 125d obs ghosts
2. ``test_no_archived_script_refs``         — ``train_offpolicy.py`` ghosts
3. ``test_algo_constructor_kwargs_match_docs`` — docs claim a kwarg the class
                                               doesn't have (``handle_truncation``)
4. ``test_docs_references_resolve``         — full-qualified ``jax_rl.x.y``
                                               names that don't resolve
5. ``test_cli_flags_in_reference_match_scripts`` — CLI flag drift between
                                               argparse and ``cli-flags.md``
6. ``test_arxiv_ids_resolve``               — every arXiv ID cited in docs/
                                               must return HTTP 200 (slow,
                                               needs network, @pytest.mark.slow)

These are deliberately cheap (regex + import). Tests flag specific
file:line locations so the next fix pass is surgical.
"""
from __future__ import annotations

import importlib
import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs"
CONTEXT_ROOT = REPO_ROOT / ".context"

sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_md_files(*roots: Path, exclude_names=("archive", "tmp", "journals")):
    """Yield all .md files under roots, skipping historical subtrees."""
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.md"):
            parts = set(p.parts)
            if parts & set(exclude_names):
                continue
            yield p


def _grep_files(pattern: str, files):
    """Return [(path, line_no, line)] matching `pattern`."""
    rx = re.compile(pattern)
    hits = []
    for p in files:
        try:
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if rx.search(line):
                    hits.append((p, i, line))
        except Exception:
            pass
    return hits


# ---------------------------------------------------------------------------
# 1. Reverted obs dims
# ---------------------------------------------------------------------------


def test_no_reverted_obs_dims():
    """Catch the 51d / 125d obs-dim ghost from the 2026-04-10 experiment.

    The reverted values were 51d (state) and 125d (privileged_state). Real
    shapes are 48d / 122d. If these strings show up in docs/ or .context/
    (excluding archives and journals), docs have re-drifted.
    """
    pattern = r"\b(?:51d|125d|51-dim|125-dim)\b|\(51,\)|\(125,\)"
    # Historical / discussion subtrees where mentioning the reverted dims is
    # legitimate (lessons document the drift pattern, plans reference
    # proposed future dims, references capture cross-project notes).
    files = list(_iter_md_files(
        DOCS_ROOT, CONTEXT_ROOT,
        exclude_names=("archive", "tmp", "journals", "lessons", "plans",
                       "references", "go2", "tutorials-dev"),
    ))
    hits = _grep_files(pattern, files)
    if hits:
        msg = "\n".join(
            f"  {p.relative_to(REPO_ROOT)}:{ln} — {line.strip()}"
            for p, ln, line in hits
        )
        pytest.fail(
            f"Reverted 51d/125d obs-dim refs found (real dims are 48/122):\n{msg}"
        )


# ---------------------------------------------------------------------------
# 2. Archived script refs
# ---------------------------------------------------------------------------


def test_no_archived_script_refs():
    """`train_offpolicy.py` was split into per-algo scripts. Docs must not
    cite it anymore (archives and journals are allowed)."""
    hits = _grep_files(r"train_offpolicy\.py", list(_iter_md_files(DOCS_ROOT)))
    if hits:
        msg = "\n".join(
            f"  {p.relative_to(REPO_ROOT)}:{ln} — {line.strip()}"
            for p, ln, line in hits
        )
        pytest.fail(f"archived `train_offpolicy.py` referenced in docs:\n{msg}")


# ---------------------------------------------------------------------------
# 3. Algo constructor kwarg drift
# ---------------------------------------------------------------------------

ALGO_MODULES = {
    "SAC": "jax_rl.algos.sac",
    "TD3": "jax_rl.algos.td3",
    "FastSAC": "jax_rl.algos.fast_sac",
    "FastTD3": "jax_rl.algos.fast_td3",
    "FlashSAC": "jax_rl.algos.flash_sac",
    "PPO": "jax_rl.algos.ppo",
}


def _extract_signature_block(text: str, class_name: str) -> str | None:
    """Find the ```python fence that begins with ``ClassName(``."""
    pattern = re.compile(
        r"```python\n(" + re.escape(class_name) + r"\(.*?)\n```",
        re.DOTALL,
    )
    m = pattern.search(text)
    return m.group(1) if m else None


def _parse_kwargs_from_sig_block(block: str) -> list[str]:
    """Pick out ``name: Type`` or ``name =`` arg lines."""
    return re.findall(r"^\s*([a-z_][a-z_0-9]*)\s*[:=]", block, re.MULTILINE)


@pytest.mark.parametrize("cls_name,module_name", sorted(ALGO_MODULES.items()))
def test_algo_constructor_kwargs_match_docs(cls_name, module_name):
    algos_md = DOCS_ROOT / "api" / "algos.md"
    if not algos_md.exists():
        pytest.skip("docs/api/algos.md missing")
    text = algos_md.read_text(encoding="utf-8")
    block = _extract_signature_block(text, cls_name)
    if block is None:
        pytest.skip(f"no signature block for {cls_name} in algos.md")
    doc_kwargs = _parse_kwargs_from_sig_block(block)

    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    real_params = set(inspect.signature(cls.__init__).parameters) - {"self"}

    phantom = [k for k in doc_kwargs if k not in real_params]
    if phantom:
        pytest.fail(
            f"{cls_name}: docs claim kwargs {phantom} that don't exist on "
            f"{module_name}.{cls_name}.__init__. Real params: {sorted(real_params)}"
        )


# ---------------------------------------------------------------------------
# 4. Full-qualified refs must resolve
# ---------------------------------------------------------------------------


def test_docs_references_resolve():
    """Any ``jax_rl.module.attribute`` chain in docs must import cleanly."""
    # Match `jax_rl.foo.bar` or longer. Stop at word-boundaries / punct.
    rx = re.compile(r"\bjax_rl(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
    failures = []
    for md in _iter_md_files(DOCS_ROOT):
        text = md.read_text(encoding="utf-8")
        for m in rx.finditer(text):
            ref = m.group(0)
            # Don't check refs inside obvious prose links like
            # `jax_rl/training/offpolicy_loop.py` — those use '/' not '.'.
            # (regex above already uses `.` so we're fine.)
            if not _dotted_ref_resolves(ref):
                # Locate line
                line_no = text.count("\n", 0, m.start()) + 1
                failures.append((md, line_no, ref))
    if failures:
        msg = "\n".join(
            f"  {p.relative_to(REPO_ROOT)}:{ln} — {ref}"
            for p, ln, ref in failures
        )
        pytest.fail(f"dotted jax_rl refs that don't resolve:\n{msg}")


def _dotted_ref_resolves(ref: str) -> bool:
    """Try to import the longest possible prefix, then getattr the rest."""
    parts = ref.split(".")
    # Walk longest-module-prefix: try importing shorter and shorter.
    for split in range(len(parts), 0, -1):
        mod_path = ".".join(parts[:split])
        try:
            module = importlib.import_module(mod_path)
        except Exception:
            continue
        # Walk remaining parts as attrs
        obj = module
        for attr in parts[split:]:
            if not hasattr(obj, attr):
                return False
            obj = getattr(obj, attr)
        return True
    return False


# ---------------------------------------------------------------------------
# 5. CLI flags match argparse
# ---------------------------------------------------------------------------

TRAIN_SCRIPTS = [
    "train_sac.py",
    "train_td3.py",
    "train_fast_sac.py",
    "train_fast_td3.py",
    "train_flashsac.py",
    "train_ppo_fast.py",
]


def _flags_from_help(script: str) -> set[str]:
    script_path = REPO_ROOT / script
    if not script_path.exists():
        return set()
    res = subprocess.run(
        ["uv", "run", "python", str(script_path), "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    if res.returncode != 0:
        return set()
    # Pull --flag-name tokens from --help output
    return set(re.findall(r"--[a-z][a-z0-9-]+", res.stdout)) - {"--help"}


def _flags_from_cli_md(script: str) -> set[str]:
    md = DOCS_ROOT / "reference" / "cli-flags.md"
    if not md.exists():
        return set()
    text = md.read_text(encoding="utf-8")
    # Split on `## \`script.py\`` headers
    parts = re.split(r"^## `([^`]+)`", text, flags=re.MULTILINE)
    # parts = [preamble, name1, body1, name2, body2, ...]
    for i in range(1, len(parts), 2):
        if parts[i] == script:
            section = parts[i + 1]
            return set(re.findall(r"--[a-z][a-z0-9-]+", section))
    return set()


@pytest.mark.slow
@pytest.mark.parametrize("script", TRAIN_SCRIPTS)
def test_cli_flags_in_reference_match_scripts(script):
    real = _flags_from_help(script)
    if not real:
        pytest.skip(f"could not run `{script} --help`")
    documented = _flags_from_cli_md(script)
    if not documented:
        pytest.fail(f"{script} has no section in docs/reference/cli-flags.md")
    missing_in_docs = real - documented
    phantom_in_docs = documented - real
    if missing_in_docs or phantom_in_docs:
        pytest.fail(
            f"{script} flag drift:\n"
            f"  missing in docs: {sorted(missing_in_docs)}\n"
            f"  phantom in docs: {sorted(phantom_in_docs)}"
        )


# ---------------------------------------------------------------------------
# 6. arXiv ID resolution
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_arxiv_ids_resolve():
    """Every arXiv ID cited in docs/ must resolve to a real paper (HTTP 200).

    Catches fabricated citations before they reach readers. Gated @pytest.mark.slow
    because it makes network requests. Run with:
        uv run python -m pytest tests/test_docs_drift.py::test_arxiv_ids_resolve -v -m slow

    Patterns matched:
        arXiv:2512.01996
        arxiv.org/abs/2512.01996
    """
    import urllib.request
    import urllib.error

    ARXIV_URL = "https://arxiv.org/abs/{id}"
    TIMEOUT_S = 10

    # Collect all unique IDs and their first source location
    id_pattern = re.compile(
        r"(?:arXiv:|arxiv\.org/abs/)(\d{4}\.\d{4,5})"
    )
    id_sources: dict[str, tuple[Path, int]] = {}
    for md in _iter_md_files(DOCS_ROOT):
        text = md.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            for m in id_pattern.finditer(line):
                arxiv_id = m.group(1)
                if arxiv_id not in id_sources:
                    id_sources[arxiv_id] = (md, i)

    if not id_sources:
        pytest.skip("no arXiv IDs found in docs/")

    failures = []
    for arxiv_id, (source_path, line_no) in sorted(id_sources.items()):
        url = ARXIV_URL.format(id=arxiv_id)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                status = resp.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        except Exception as exc:
            status = f"error: {exc}"

        if status != 200:
            failures.append(
                f"  {source_path.relative_to(REPO_ROOT)}:{line_no} — "
                f"arXiv:{arxiv_id} returned {status} ({url})"
            )

    if failures:
        pytest.fail(
            "arXiv IDs that did not resolve (fabricated or mistyped citations):\n"
            + "\n".join(failures)
        )
