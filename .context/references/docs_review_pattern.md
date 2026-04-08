---
name: Multi-persona docs review pattern
description: Dispatch 4 parallel subagents (high schooler, undergrad, PhD, frontend engineer) to review docs site — catches accessibility, accuracy, rigor, and visual design issues
type: feedback
---

When Steven asks for a docs review, dispatch 4 parallel Opus subagents with these personas:

1. **High schooler** — knows Python basics, no RL/JAX/MuJoCo. Tests accessibility, jargon, can they follow from zero. Rates pages 1-5.
2. **Undergrad CS** — knows ML/PyTorch, intro RL, no JAX. Tests learning curve, JAX-isms, factual accuracy, cross-references code. Rates pages 1-5.
3. **PhD researcher** — knows JAX/MuJoCo/RL deeply. Tests code correctness, benchmark rigor, architecture decisions, adoption readiness. Rates code/docs/adoption 1-10.
4. **Frontend engineer** — UI/UX focus. Reviews information density, typography, navigation, color contrast, responsive layout, theme config. Reads rendered HTML in `site/` and `mkdocs.yml`. Gives concrete CSS/config fixes.

**Why:** Each catches different issues. High schooler finds jargon. Undergrad finds factual errors. PhD finds code smells and credibility gaps. Frontend catches visual/layout problems that content reviewers miss. Overlap between reviewers = high-confidence issues.

**How to apply:**
- Run all 4 in background with `run_in_background: true`
- Give content reviewers (1-3) access to both `docs/` and the full codebase — tell them to cross-reference
- Give frontend reviewer (4) access to `docs/`, `mkdocs.yml`, `docs/stylesheets/`, and `site/` (built output)
- Use `Explore` subagent type for frontend (read-only), `general-purpose` for others
- Compile results into a single prioritized action list grouped by effort (quick fix / medium / large)

**Key lessons from first run (2026-04-08):**
- Frontend reviewer caught that `show_source: true` made API pages 6,300+ lines — content reviewers didn't notice
- Frontend caught `grid cards` requires Material Insiders — content reviewers saw the broken rendering but didn't diagnose it
- PhD reviewer caught single-seed benchmark claims — high schooler/undergrad didn't question the numbers
- All 3 content reviewers independently flagged the same monitoring command bug (`/tmp/claude-*`) — high confidence
- Undergrad found the most factual errors (5) by systematically cross-referencing docs against code
