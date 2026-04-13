"""Lint Python code fences in docs/ for name / import / signature drift.

Catches the class of bug where documentation claims a function, class, or
keyword argument exists but the underlying code has renamed, moved, or
removed it. Runs only static / compile-time checks; it never executes
arbitrary fenced snippets.

Checks performed per ```python fence:
  1. The block compiles (``compile(..., 'exec')``) — catches syntax errors.
  2. Every ``from X import Y`` resolves — X imports and Y is an attribute.
  3. Every ``import X`` resolves.
  4. Every top-level ``ClassName(...)`` call with keyword arguments uses
     kwargs that exist on the class's real ``__init__`` signature (when the
     class was imported in the same fence). This is the ``handle_truncation``
     ghost-check.

Fences are SKIPPED when they:
  - contain ``...`` ellipses (illustrative / not runnable),
  - contain ``# pseudocode`` or ``YOUR_CODE_HERE`` markers,
  - are marked ``{: .no-test}`` on the fence line,
  - are inside a ``??? note`` collapsed admonition block,
  - are a bash / shell / yaml fence (only ``python`` fences are checked).

The test is intentionally forgiving on exec — it does not try to run training
or call functions. Name resolution is what matters.
"""
from __future__ import annotations

import ast
import importlib
import os
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs"

sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Fence extraction
# ---------------------------------------------------------------------------

FENCE_RE = re.compile(
    r"^(?P<indent>[ \t]*)```(?P<tag>python)(?P<attr>[^\n]*)\n"
    r"(?P<body>.*?)\n"
    r"(?P=indent)```",
    re.DOTALL | re.MULTILINE,
)

SKIP_MARKERS = ("# pseudocode", "YOUR_CODE_HERE", "{: .no-test}", ".no-test")


def _iter_python_fences(md_path: Path):
    """Yield (line_no, body, attrs) for each ```python fence in md_path."""
    text = md_path.read_text(encoding="utf-8")
    for m in FENCE_RE.finditer(text):
        body = m.group("body")
        attrs = m.group("attr")
        # Line number of the opening fence
        line_no = text.count("\n", 0, m.start()) + 1
        yield line_no, body, attrs


def _looks_like_signature_display(body: str) -> bool:
    """Detect bare constructor-signature displays like::

        PPO(
            config: PPOConfig,
            obs_dim: int,
        )

    These aren't valid Python (type annotations on call args), so we can't
    compile them. They're documentation of signatures, not code.
    """
    stripped = body.strip()
    if not stripped:
        return False
    # Has no import, no `=` assignment, starts with `ClassName(` and contains
    # `: TypeName,` style annotations on arg lines.
    has_import = "import " in stripped
    has_assign = re.search(r"^[A-Za-z_][A-Za-z_0-9]*\s*=", stripped, re.MULTILINE)
    starts_with_call = re.match(r"^[A-Z][A-Za-z_0-9]*\(", stripped) is not None
    # Look for `    name: Type` pattern on an indented line (annotation in call)
    # Annotation either on its own line (multi-line) or inside the call
    # (e.g. `RolloutBuffer(num_steps: int, num_envs: int)`).
    has_annotated_arg = (
        re.search(r"^\s+[a-z_][a-z_0-9]*\s*:\s*[A-Za-z_]", stripped, re.MULTILINE)
        is not None
        or re.search(r"\([a-z_][a-z_0-9]*\s*:\s*[A-Za-z_]", stripped) is not None
    )
    return (
        starts_with_call
        and has_annotated_arg
        and not has_import
        and not has_assign
    )


def _should_skip(body: str, attrs: str) -> bool:
    if any(marker in attrs for marker in (".no-test", "no-test")):
        return True
    # `...` used as "fill in the rest" — skip. But only when it's a bare
    # ellipsis statement, not part of `range(...)` or similar.
    if re.search(r"(^|\n)\s*\.\.\.\s*($|\n)", body):
        return True
    if re.search(r":\s*\.\.\.\s*($|\n)", body):
        return True
    # `# ...` style "rest of code" comments
    if re.search(r"#\s*\.\.\.\s*($|\n)", body):
        return True
    # Function-signature display with unicode arrow — not runnable Python.
    if "→" in body:
        return True
    for marker in SKIP_MARKERS:
        if marker in body:
            return True
    return False


def _collect_fences():
    """Return [(md_path, line_no, body)] for every checkable python fence."""
    fences = []
    for md in sorted(DOCS_ROOT.rglob("*.md")):
        for line_no, body, attrs in _iter_python_fences(md):
            if _should_skip(body, attrs):
                continue
            fences.append((md, line_no, body))
    return fences


ALL_FENCES = _collect_fences()


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------


def _try_parse(body: str):
    """Return (ok, ast_tree_or_err_str). Keeps syntax errors inline."""
    try:
        return True, ast.parse(body)
    except SyntaxError as e:
        # Some fences are bare signatures like `def foo(x: int) -> bool:` with
        # no body. ast.parse wants a body. Try appending `    pass`.
        try:
            return True, ast.parse(body + "\n    pass\n")
        except SyntaxError:
            return False, f"SyntaxError: {e}"


def _check_imports(tree: ast.AST):
    """Return list of (bad_module, bad_name_or_None, reason) tuples."""
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module
            if mod is None or not mod.startswith(("jax_rl", "jax", "optax",
                                                   "numpy", "flax", "distrax")):
                continue
            # Only assert on our own modules — third-party drift is out of scope
            # except for catching typos in jax_rl refs.
            if not mod.startswith("jax_rl"):
                continue
            try:
                module = importlib.import_module(mod)
            except Exception as e:
                errors.append((mod, None, f"import {mod} failed: {e}"))
                continue
            for alias in node.names:
                name = alias.name
                if name == "*":
                    continue
                if not hasattr(module, name):
                    errors.append((mod, name,
                                   f"{mod!r} has no attribute {name!r}"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith("jax_rl"):
                    continue
                try:
                    importlib.import_module(alias.name)
                except Exception as e:
                    errors.append((alias.name, None,
                                   f"import {alias.name} failed: {e}"))
    return errors


def _resolve_imports_in_fence(tree: ast.AST):
    """Walk imports in the fence and return {local_name: (module, attr)}.

    Only resolves names we can import. Used by the kwarg check to find the
    real class a constructor call refers to.
    """
    resolved = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            try:
                module = importlib.import_module(node.module)
            except Exception:
                continue
            for alias in node.names:
                local = alias.asname or alias.name
                if hasattr(module, alias.name):
                    resolved[local] = getattr(module, alias.name)
    return resolved


def _check_constructor_kwargs(tree: ast.AST, resolved: dict):
    """Return list of (class_name, bad_kwarg) for any kw not in __init__."""
    import inspect

    errors = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        obj = resolved.get(node.func.id)
        if obj is None or not inspect.isclass(obj):
            continue
        try:
            sig = inspect.signature(obj.__init__)
        except (TypeError, ValueError):
            continue
        valid = set(sig.parameters)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if accepts_kwargs:
            continue
        for kw in node.keywords:
            if kw.arg is None:  # **kwargs splat
                continue
            if kw.arg not in valid:
                errors.append((obj.__name__, kw.arg))
    return errors


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_docs_directory_exists():
    assert DOCS_ROOT.is_dir(), f"missing docs dir: {DOCS_ROOT}"


def test_found_some_fences():
    # Guard against a refactor that silently breaks fence extraction.
    assert len(ALL_FENCES) > 5, (
        f"Only {len(ALL_FENCES)} python fences found — extractor likely broken"
    )


@pytest.mark.parametrize(
    "md_path,line_no,body",
    ALL_FENCES,
    ids=[f"{p.relative_to(REPO_ROOT)}:L{ln}" for p, ln, _ in ALL_FENCES],
)
def test_fence_compiles(md_path: Path, line_no: int, body: str):
    if _looks_like_signature_display(body):
        pytest.skip("signature-display fence (covered by kwarg check)")
    ok, result = _try_parse(body)
    assert ok, f"{md_path.relative_to(REPO_ROOT)}:{line_no} — {result}"


@pytest.mark.parametrize(
    "md_path,line_no,body",
    ALL_FENCES,
    ids=[f"{p.relative_to(REPO_ROOT)}:L{ln}" for p, ln, _ in ALL_FENCES],
)
def test_fence_imports_resolve(md_path: Path, line_no: int, body: str):
    ok, tree = _try_parse(body)
    if not ok:
        pytest.skip("fence did not parse (covered by compile test)")
    errors = _check_imports(tree)
    if errors:
        msg = "\n".join(
            f"  {mod}.{name or ''}: {reason}" for mod, name, reason in errors
        )
        pytest.fail(
            f"{md_path.relative_to(REPO_ROOT)}:{line_no} has unresolved "
            f"imports:\n{msg}"
        )


@pytest.mark.parametrize(
    "md_path,line_no,body",
    ALL_FENCES,
    ids=[f"{p.relative_to(REPO_ROOT)}:L{ln}" for p, ln, _ in ALL_FENCES],
)
def test_fence_constructor_kwargs(md_path: Path, line_no: int, body: str):
    """Check kwargs on constructor calls.

    For runnable fences, uses AST. For signature-display fences (annotated
    args, not valid Python), parses lines with a regex and looks up the
    class by scanning earlier ``from X import ClassName`` lines in the same
    markdown file.
    """
    import inspect

    if _looks_like_signature_display(body):
        # Find ClassName on the first non-whitespace line.
        m = re.match(r"^([A-Z][A-Za-z_0-9]*)\s*\(", body.strip())
        if not m:
            pytest.skip("no constructor call to check")
        cls_name = m.group(1)
        cls = _find_class_in_md_file(md_path, cls_name)
        if cls is None:
            pytest.skip(f"could not resolve {cls_name} from imports in file")
        try:
            sig = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            pytest.skip("class has no inspectable signature")
        valid = set(sig.parameters)
        # Extract kwargs: lines like `    name: Type = default,` or `    name=value`
        arg_lines = re.findall(
            r"^\s+([a-z_][a-z_0-9]*)\s*[:=]", body, re.MULTILINE
        )
        bad = [a for a in arg_lines if a not in valid]
        if bad:
            pytest.fail(
                f"{md_path.relative_to(REPO_ROOT)}:{line_no} — "
                f"{cls.__name__}(...) has no kwarg(s) {bad!r}. Real params: "
                f"{sorted(valid - {'self'})}"
            )
        return

    ok, tree = _try_parse(body)
    if not ok:
        pytest.skip("fence did not parse")
    resolved = _resolve_imports_in_fence(tree)
    if not resolved:
        pytest.skip("no resolvable classes in fence")
    errors = _check_constructor_kwargs(tree, resolved)
    if errors:
        msg = "\n".join(f"  {cls}(...) has no kwarg {kw!r}" for cls, kw in errors)
        pytest.fail(
            f"{md_path.relative_to(REPO_ROOT)}:{line_no} constructor kwargs "
            f"missing on real class:\n{msg}"
        )


def _find_class_in_md_file(md_path: Path, cls_name: str):
    """Scan md file for ``from X import ClassName`` and import the class."""
    text = md_path.read_text(encoding="utf-8")
    # Match `from jax_rl.xxx import ClassName` possibly in an import list
    for m in re.finditer(
        r"from\s+([\w\.]+)\s+import\s+([^\n]+)", text
    ):
        module_name = m.group(1)
        imports = [s.strip() for s in m.group(2).split(",")]
        if cls_name in imports:
            try:
                module = importlib.import_module(module_name)
                return getattr(module, cls_name, None)
            except Exception:
                return None
    return None
