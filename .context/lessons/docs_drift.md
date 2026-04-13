# Docs drift — catch it with tests, not review rounds

## Pattern

Three consecutive 4-persona doc review rounds surfaced the same class of bug:
the docs claim a function / class / kwarg / CLI flag exists, but the code
doesn't have it (or has it under a different name). Fixers clean up one
occurrence; reviewers keep finding stragglers one round later.

Concrete ghosts caught historically:

| Ghost | Where the drift was | Real name / value |
|---|---|---|
| `build_env_bundle` | `docs/reference/training-loop.md` | `make_env_bundle` |
| `handle_truncation=True` | `docs/api/algos.md` (all 5 off-policy algos) | Removed from constructors; moved to `TrainConfig.handle_truncation` |
| `train_offpolicy.py` | many docs pages | Split into `train_sac.py` / `train_td3.py` / etc. |
| `51d state`, `(125,)` privileged_state | 10+ doc pages | Real dims are 48 / 122 (experiment reverted, docs weren't) |
| `--reset-mode` missing | `docs/reference/cli-flags.md` | Flag exists on all off-policy scripts |
| `--batch-size`, `--grad-updates-per-step`, `--buffer-size` | `docs/reference/cli-flags.md` (SAC/TD3/FastSAC/FastTD3) | Not actually defined in argparse — phantom docs |

## Fix: automated drift tests

`tests/test_docs_code_blocks.py` + `tests/test_docs_drift.py` run on every
commit and catch the full class:

1. **Every `from jax_rl.X import Y` in a docs fence must resolve.** Uses
   `importlib` + `getattr`. This catches `build_env_bundle` the moment it's
   typed.
2. **Every `ClassName(kwarg=...)` fence is matched against
   `inspect.signature(ClassName.__init__)`.** This catches `handle_truncation`
   and similar phantom kwargs.
3. **Every `train_offpolicy.py` mention in `docs/` fails the test.** Direct
   regex.
4. **Every `51d` / `125d` / `(51,)` / `(125,)` in `docs/` or `.context/`
   (excluding `archive/`, `journals/`, `lessons/`, `plans/`) fails.**
5. **CLI flag sets are compared between `<script>.py --help` and
   `docs/reference/cli-flags.md` section for that script.**

## Takeaways

- **Name-resolution is the highest ROI doc test.** We don't try to exec
  arbitrary snippets — just verify that every imported / referenced name
  exists. Runs in ~3 seconds. Catches ~80% of real drift.
- **Signature-display fences (annotated args, not runnable Python) are
  common in API docs.** The test skips them for compile but still regex-
  parses kwargs and checks them against the real `__init__`. Without this
  we miss the `handle_truncation` class of bug.
- **Excluding historical subtrees matters.** `archive/`, `journals/`,
  `lessons/`, `plans/` legitimately discuss reverted or future values.
  The test should flag only live docs that still claim drift as current.
- **Test failure IS the finding.** When you add this test suite, expect
  it to fail on current state. Each failure is a real doc fix to make.
  Commit the tests in the broken state, then fix the docs in a follow-up
  PR so the two commits are clearly separated (infra vs content).
