---
name: Multi-persona docs review pattern
description: Dispatch parallel Opus subagents to review docs site, with anti-hallucination protocol and drift-test preflight
type: feedback
---

When Steven asks for a docs review, dispatch parallel Opus subagents with these personas:

1. **High schooler** — Python basics, no RL/JAX/MuJoCo. Accessibility, jargon, can-they-follow-from-zero.
2. **Undergrad CS** — ML/PyTorch + intro RL, no JAX. Factual accuracy via cross-referencing docs vs code.
3. **PhD researcher** — JAX/MuJoCo/RL deep. Code correctness, benchmark rigor, architecture, adoption readiness.
4. **Frontend engineer** — UI/UX. Typography, navigation, CSS, theme config. Reads rendered `site/` + `mkdocs.yml`.

## When to run each persona

- **Run all 4 for the first review** of a docs site or after major content changes
- **Skip high schooler from round 2+** — their feedback is vibes-level and repeats across rounds. Their #1 ask (always) is "one-paragraph intro" for terms; after you hear it once, you know
- **Undergrad + PhD + Frontend** is the standard 3-persona follow-up
- **Stop doing rounds when:** a round produces <3 verified findings. Drift-test infrastructure (see below) protects most structural issues automatically; reviewers should catch diminishing-returns semantic/pedagogy issues

## Anti-hallucination protocol (REQUIRED after round 1)

Prior rounds surfaced real hallucinations: undergrad falsely flagging real arxiv IDs as fake (date priors), content agents making up import names, reviewers citing line numbers from memory. The protocol below cut hallucinations to near-zero on round 5 (undergrad 97% verified, PhD 81%, frontend 79%).

### Rule 1: Run drift tests FIRST
Every reviewer must start with:
```
uv run python -m pytest tests/test_docs_drift.py tests/test_docs_code_blocks.py -v --no-header -q 2>&1 | tail -15
```
If a class of drift (phantom kwargs, reverted obs dims, archived script refs, CLI flag drift, arxiv 404s, unresolvable imports) is already protected by a passing test, the reviewer should NOT re-flag it. Focus on what tests can't catch.

### Rule 2: Every factual claim must be tool-backed
No citing line numbers, function names, benchmark numbers, or dates from memory. Reviewer must paste grep/read/WebFetch output inline with each claim. Prompt template enforces this via `[VERIFIED: <source>]` tagging.

### Rule 3: Flag knowledge-cutoff traps
Inline in every reviewer prompt: "Today is YYYY-MM-DD. Your training cutoff is earlier. Post-cutoff dates (papers, GPU models, tool versions) MUST be verified, never assumed fake."

Prior false-positive: round 4 undergrad flagged arXiv:2512.01996 and 2604.04539 as fabricated (operating on pre-2026 priors). Both are real papers from Dec 2025 / April 2026.

### Rule 4: Self-audit pass at end
Reviewer re-reads their own report before submitting, marks each claim `[VERIFIED]` or `[OPINION]`, drops any claim that can't be backed. Measurable: every round-5 reviewer dropped 0-2 claims during self-audit that were genuine false positives.

### Prompt template (paste into reviewer prompts)

```
## ANTI-HALLUCINATION PROTOCOL (read first, follow strictly)

**Today is <DATE>. Your training cutoff is earlier. Assume the world has moved on.**

### Rule 1: Run drift tests FIRST
Run: `uv run python -m pytest tests/test_docs_drift.py tests/test_docs_code_blocks.py -v --no-header -q 2>&1 | tail -15`
Read the output. These tests automatically catch: phantom constructor kwargs,
reverted symbols (obs dims, archived scripts), CLI flag drift, arxiv 404s,
unresolvable code-fence imports. Do NOT re-flag issues these tests cover.

### Rule 2: Every factual claim MUST be tool-backed
Citing line numbers, function names, benchmark numbers, dates → MUST have
grep/read/WebFetch output to back it. No memory citations.

### Rule 3: Traps from prior rounds
- arXiv IDs with dates >= current-year are LIKELY REAL. Verify with WebFetch before claiming fake.
- Function/class names in docs → grep the codebase before claiming doesn't exist
- Line numbers → Read the exact line you cite, don't approximate

### Rule 4: Self-audit pass
Before submitting, re-read your own report. Mark each claim [VERIFIED (source)]
or [OPINION]. Drop claims you can't back.

### Output: require self-audit stats
- Total claims: N
- Verified: N (X%)
- Opinions (explicitly labeled): N
- Dropped during audit: N (list them briefly)
```

## Dispatch mechanics

- Run all reviewers in background with `run_in_background: true`
- Use `opus` model for all 4 (reasoning-heavy; the cost saved by Sonnet isn't worth hallucination risk)
- Content reviewers (1-3): `subagent_type: general-purpose`
- Frontend reviewer (4): `subagent_type: Explore` (read-only is appropriate)
- Compile results into a prioritized action list: 🔴 critical / 🟠 high / 🟡 medium / 🟢 polish

## Typical output rates (for calibration)

Based on 5 rounds on jax-learning docs (2026-04-12):

| Round | Findings | Hallucinations (false positives) | Notes |
|---|---|---|---|
| 1 | ~30 across 4 reviewers | 2-3 | Baseline; drift tests didn't exist |
| 2 | ~20 | 2 (build_env_bundle hallucination, 18k sps ghost) | Truncation bug found! |
| 3 | ~15 | 1 (handle_truncation docs drift) | Drift tests added after this |
| 4 | ~10 | 2 (arxiv IDs flagged as fake, both real) | Post-drift-tests |
| 5 | **11 total, 0 hallucinations** | 0 | Anti-hallucination protocol + drift tests |

**Diminishing returns curve:** Each round catches ~50% fewer issues than the prior. Stop when a round nets <3 actionable items.

## Key lessons from multiple runs (2026-04-08 through 2026-04-12)

- **Frontend reviewer catches what content reviewers miss.** `show_source: true` making API pages 6,300+ lines, `grid cards` requiring Material Insiders, `.md-grid` width override stretching nav — no content reviewer noticed any of these.
- **PhD reviewer catches the real code bugs.** Truncation handling bug on round 2 (systematic Q underestimation on long-horizon tasks) was found only by the PhD persona cross-referencing algorithm files against each other. Worth the reviewer budget.
- **Undergrad catches drift — but only with cross-reference discipline.** Round 4 undergrad hallucinated arxiv IDs; round 5 undergrad with protocol caught a real broken pointer (`jax_rl/training/ppo_loop.py` doesn't exist).
- **High schooler's persistent ask is "one-paragraph intro".** Same finding every round. Acknowledge once in the docs or accept it's not fixable without a restructure.
- **"Cleanup of round N creates ghost refs for round N+1" is systemic.** Response: build drift tests (`tests/test_docs_drift.py` + `tests/test_docs_code_blocks.py`). They cost ~600 LOC once and protect against regression forever.
- **Prompts MUST re-state the current date.** Otherwise reviewers flag real post-cutoff papers/tools/GPU-models as fabricated based on prior knowledge.

## Integration with drift tests

The review pattern is HALF of the docs quality system. The other half is `tests/test_docs_drift.py` + `tests/test_docs_code_blocks.py` (commit `2d2c807` on jax-learning). Together:

- **Drift tests** (runs every commit): catches structural drift mechanically — phantom kwargs, reverted symbols, CLI flag mismatches, unresolvable imports, arxiv 404s
- **Review pattern** (run quarterly or after major content changes): catches semantic drift — wrong benchmark interpretations, missing algorithm features in descriptions, contradictory pages, accessibility problems

Neither replaces the other. Drift tests are fast (run on every commit) but limited. Reviews are slow + expensive but catch things tests can't describe.
