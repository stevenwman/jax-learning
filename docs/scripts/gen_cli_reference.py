"""Generate CLI flags reference table from argparse definitions.

Run: uv run python docs/scripts/gen_cli_reference.py

Imports each entry script's `build_parser()` and reflects on
`parser._actions` to render the markdown. Single source of truth: the
script's parser is what `--help` shows AND what these docs render.
Adding a flag in a script + re-running this generator is a one-step update.

Pre-2026-04-25 this file held a 471-line hand-mirror of every parser; now
~110 lines of reflection. See B5.9 lessons in
.context/lessons/infrastructure.md.
"""

import argparse
import importlib
import sys
from pathlib import Path

# Add repo root to path so we can import scripts/<name> as `scripts.<name>`.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# Order matters — this is the rendering order in cli-flags.md.
SCRIPTS: list[tuple[str, str]] = [
    ("train_ppo_fast.py",        "scripts.train_ppo_fast"),
    ("train_ppo_contraction.py", "scripts.train_ppo_contraction"),
    ("train_sac.py",             "scripts.train_sac"),
    ("train_td3.py",             "scripts.train_td3"),
    ("train_fast_sac.py",        "scripts.train_fast_sac"),
    ("train_fast_td3.py",        "scripts.train_fast_td3"),
    ("train_flashsac.py",        "scripts.train_flashsac"),
    ("train_tdmpc2.py",          "scripts.train_tdmpc2"),
    ("train_pusht.py",           "scripts.train_pusht"),
    ("record_video.py",          "scripts.record_video"),
    ("eval_tdmpc2.py",           "scripts.eval_tdmpc2"),
    ("check_tdmpc2_determinism.py", "scripts.check_tdmpc2_determinism"),
    ("record_video_tdmpc2.py",   "scripts.record_video_tdmpc2"),
]


def _type_name(action: argparse.Action) -> str:
    """Human-readable type string for an argparse action."""
    if isinstance(action, argparse._StoreTrueAction):
        return "flag"
    if action.type is None:
        return "str"
    name = getattr(action.type, "__name__", str(action.type))
    if action.nargs == "+":
        return f"{name}+"
    if action.nargs == 2:
        return f"{name} {name}"
    return name


def _default_str(action: argparse.Action) -> str:
    """Human-readable default value."""
    if isinstance(action, argparse._StoreTrueAction):
        return "off"
    if action.default is None:
        return "from preset" if "preset" in (action.help or "").lower() or "config" in (action.help or "").lower() else "-"
    if action.required:
        return "**required**"
    return f"`{action.default}`"


def render_parser(name: str, parser: argparse.ArgumentParser) -> str:
    """Render a parser's arguments as a markdown table."""
    lines = [
        f"## `{name}`",
        "",
        "| Flag | Type | Default | Description |",
        "|------|------|---------|-------------|",
    ]
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        flag = ", ".join(f"`{o}`" for o in action.option_strings) or f"`{action.dest}`"
        lines.append(
            f"| {flag} | {_type_name(action)} | {_default_str(action)} | {action.help or ''} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    header = """# CLI Flags

Auto-generated from argparse definitions. Regenerate with:

```bash
uv run python docs/scripts/gen_cli_reference.py
```

---

"""
    sections = []
    for display_name, module_path in SCRIPTS:
        module = importlib.import_module(module_path)
        if not hasattr(module, "build_parser"):
            raise RuntimeError(
                f"{module_path} does not expose build_parser(). "
                f"Add `def build_parser() -> argparse.ArgumentParser` at module "
                f"level so this generator can reflect on its flags."
            )
        sections.append(render_parser(display_name, module.build_parser()))

    output = header + "\n---\n\n".join(sections)

    # Write directly to the docs file
    out_path = Path(__file__).resolve().parents[1] / "reference" / "cli-flags.md"
    out_path.write_text(output)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
