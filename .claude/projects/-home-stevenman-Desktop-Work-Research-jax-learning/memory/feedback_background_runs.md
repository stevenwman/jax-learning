---
name: feedback_background_runs
description: Run long benchmarks/training in background so user can keep talking
type: feedback
---

When running training scripts or benchmarks that take more than ~10 seconds, always run them in the background with output to a temp file. Don't block the conversation waiting for results.

**Why:** When the conversation is blocked by a long-running command, the user can't interact. They want to keep discussing while things run.

**How to apply:** Use `run_in_background: true` on Bash tool, or `&` with output redirect to a file. Then read the output file later when checking results.
