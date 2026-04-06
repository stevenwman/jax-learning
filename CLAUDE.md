# Project Instructions

## Superpowers Output
Superpowers specs go in `.superpowers/specs/`, plans in `.superpowers/plans/` (NOT `docs/superpowers/`).

## Doc Sync Checkpoint

After completing a logical chunk of work (debugging session, feature, config change, training run), do a doc sweep before moving on. Use all three to find what's stale:
1. **Your context** — you know what you just worked on
2. **`git diff --name-only` + `git status`** — catches committed and uncommitted changes
3. **Grep key values** (Kd, Kp, eval scores, etc.) across docs to find mismatches

### Always check these docs:
- `.context/TODO.md` — mark completed items, add new ones
- `.context/journals/YYYY-MM-DD.md` — what happened and results
- `.context/LESSONS.md` + `lessons/*.md` — anything reusable learned
- `.context/AGENT_HANDOFF.md` — if project state, benchmarks, or workflows changed
- `.context/NEW_AGENT_PROMPT.md` — if onboarding-relevant info changed
- `deploy/README.md` — if deploy code, deps, or PD gains changed
- Any other `README.md` in the repo

### Cross-reference table (code → docs):
| Code area | Also update |
|---|---|
| `jax_rl/envs/locomotion/` (env config, physics, rewards) | `deploy/go2_constants.py`, `.context/go2/mjcf_comparison.md` |
| `deploy/` (deploy code, constants, interface) | `deploy/README.md` |
| `train_*.py` (training scripts, CLI flags) | `.context/AGENT_HANDOFF.md` (Quick Reference), run `docs/scripts/gen_cli_reference.py` |
| `jax_rl/configs/env_presets.py` | Run `docs/scripts/gen_env_presets.py` |
| `jax_rl/algos/*.py` docstrings | API docs auto-update on `mkdocs build` |
| Training results (new eval scores, benchmarks) | `.context/AGENT_HANDOFF.md` (benchmarks), `.context/TODO.md`, `docs/index.md` |
