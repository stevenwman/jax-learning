# HP Tuning Agent — Design Spec

**Date:** 2026-03-26
**Status:** Approved (implementation deferred — spec ready when needed)
**Scope:** Autonomous/collaborative hyperparameter tuning via Claude Code session with file-based state

---

## Problem

HP tuning is currently manual: launch a run, wait 30-90 min, read the metrics, decide what to change, launch again. The feedback loop is slow and requires constant attention. We need an agent that can run this loop autonomously for known failure patterns, and collaborate with the human for harder decisions (reward shaping, novel failures).

## Solution

A Claude Code session that acts as a **file-based state machine**. All state lives in a tuning log on disk — the agent reads it fresh each iteration, diagnoses the current run, decides the next HP config, and writes its reasoning back. Context stays minimal because the log is the memory, not the conversation.

## Architecture

### Execution model

```
Claude Code session
  └─ Loop:
      1. Read tuning log (.context/tuning/<session>.md)
      2. Poll current run (wandb.Api or grep EVAL)
      3. Diagnose (rules first, LLM reasoning if no rule matches)
      4. Write diagnosis + next config to tuning log
      5. Act (launch next run / surface to human / stop)
      6. Sleep until run progresses (~5 min polls)
```

Single GPU (RTX 5080, 16GB) — one run at a time. Agent checks `nvidia-smi` before launching.

### Context optimization

The primary design constraint: minimize token usage and context accumulation across a multi-hour tuning session.

- **No training stdout in context** — agent polls via `wandb.Api()` or `grep EVAL <output_file> | tail -5`, never streams training output
- **Tuning log is the single source of truth** — agent reads it fresh each iteration instead of accumulating run history in conversation context
- **Sleep between polls** — `sleep 300` (5 min) between checks, not active waiting
- **Compact run entries** — each run is ~5-8 lines in the log. 10 runs = ~60 lines to read on restart
- **Session recovery** — if context compresses or session restarts, the agent re-reads the tuning log and picks up exactly where it left off

### Two modes

**Autonomous (tier 1-2 HPs):**
Agent runs the full loop: diagnose → decide → launch → repeat. Stops when a termination condition is met.

Invocation:
```
/tune --env Go2JoystickFlat --algo sac --target 200 --budget 10 --hours 8
```

**Collaborative (tier 3 / reward shaping):**
Agent runs the analysis loop but surfaces findings with plots/breakdowns and waits for human input before launching the next run.

Invocation:
```
/tune --env Go2JoystickFlat --algo sac --mode collaborative
```

The human can switch modes mid-session by editing the tuning log header or telling the agent.

---

## Tuning Log Format

File: `.context/tuning/<session_name>.md`

```markdown
# Tuning Session: Go2 SAC Phase B
Started: 2026-03-27 14:00
Target: eval > 200 | Budget: 10 runs / 8 hrs | Mode: autonomous
Algo: sac | Env: Go2JoystickFlat | Base seed: 0

## HP Search Space
Tier 1-2 (autonomous):
- lr: [3e-4, 1e-3, 3e-3]
- gamma: [0.97, 0.99]
- tau: [0.005, 0.01]
- alpha_init: [0.1, 0.5, 1.0]
- target_entropy_scale: [0.3, 0.5, 0.7]
- hidden_dim: [(256,256), (512,256,128)]
- policy_delay: [1, 4, 8]
- grad_updates_per_step: [4, 8, 16]
- obs_norm: [true, false]

Tier 3 (collaborative — needs human):
- reward weights (tracking_lin_vel, tracking_ang_vel, etc.)
- action_scale
- height_termination threshold

## Run 1: baseline
- Config: lr=1e-3, gamma=0.97, tau=0.005, hidden=(256,256), alpha_init=1.0
- Command: uv run python train_offpolicy.py --algo sac --env Go2JoystickFlat --obs-norm --wandb --seed 0
- wandb: sman2/jax-rl/runs/abc123
- Result: eval=142 @ 16M, Q bias=15.4, entropy=-0.9 @ 5M steps
- Diagnosis: ENTROPY_COLLAPSE — entropy hit -0.9 by 5M steps, alpha decayed too fast
- Next action: reduce alpha_init to 0.1, lower target_entropy_scale to 0.3

## Run 2: lower_alpha
- Config: lr=1e-3, alpha_init=0.1, target_entropy_scale=0.3 (rest same as run 1)
- Command: uv run python train_offpolicy.py --algo sac --env Go2JoystickFlat --obs-norm --wandb --seed 0 --alpha-init 0.1
- wandb: sman2/jax-rl/runs/def456
- Status: RUNNING (8M steps, eval=95, entropy=1.2 — healthy)
```

---

## Diagnostic Rules

Rule-based heuristics for tier 1-2 autonomous tuning. Evaluated in order — first match wins.

| ID | Pattern | Signal | Suggested Action |
|----|---------|--------|------------------|
| `ENTROPY_COLLAPSE` | entropy < -0.5 * action_dim before 2M steps | Alpha decaying too aggressively | Lower alpha_init, reduce target_entropy_scale |
| `Q_DIVERGENCE` | q_mean > 5x mc_mean | Critic overestimating | Reduce lr, increase tau, enable Q LayerNorm if off |
| `Q_UNDERFIT` | q_corr < 0.5 after 5M steps | Critic not tracking returns | Increase critic capacity (hidden_dim) or grad_updates_per_step |
| `REWARD_PLATEAU` | eval unchanged +/-5% for >3M steps, Q stable | Exploration insufficient | Increase entropy target, try larger network, surface to human if tier 1-2 exhausted |
| `SLOW_CONVERGENCE` | eval < 50% of target at budget midpoint | Learning rate or batch too conservative | Increase lr or batch_size |
| `SPS_DEGRADING` | sps dropped >30% from peak | GPU memory pressure | Check nvidia-smi, possible OOM from MJX recompilation |
| `TRAINING_DIVERGED` | eval dropped >50% from best and not recovering for 2M steps | Catastrophic forgetting or instability | Reduce lr, increase tau, try from best checkpoint |
| `NAN_DETECTED` | NaN in Q values or loss | Numerical instability | Enable NaN guard if off, reduce lr, check reward_scaling |
| `ALPHA_COLLAPSE` | alpha < 0.001 and still falling | Entropy term vanishing | Increase alpha_init, clip alpha_lr, or raise target_entropy |
| `OOM_CRASH` | Run exits with OOM/cuSolver error | GPU memory exhausted | Reduce batch_size, hidden_dim, or num_envs |

**When no rule matches:** Escalate to LLM reasoning. Claude reads the run's wandb curves, the tuning log history, and LESSONS.md, and reasons about what to try next. This costs more tokens but handles novel situations.

**Rollback rule:** If eval drops >20% below best for 2 consecutive runs with different HPs, revert to the best config and try a different dimension of the search space.

**Autonomous → collaborative escalation:** If in autonomous mode and the agent hits `REWARD_PLATEAU` with tier 1-2 HPs exhausted, the agent writes `Mode: collaborative (escalated — tier 1-2 exhausted)` to the log header, prints a diagnostic summary, and waits for human input before continuing.

**Rule evaluation requires:** The agent pulls these metrics from wandb at poll time:
- `actor/entropy` — for entropy collapse detection
- `critic/q1_mean` — for Q divergence
- `critic/q_corr`, `critic/q_bias` — for Q diagnostics (only available at eval points)
- `perf/eval_mean` — for plateau/convergence detection
- `infra/sps` — for SPS degradation

---

## Stopping Conditions (Autonomous Mode)

The agent stops when the **first** of these triggers:

1. **Target reached:** `eval_mean >= target` for 2 consecutive evals (not a fluke)
2. **Budget exhausted:** `num_runs >= max_runs` or `wall_clock >= max_hours`
3. **Plateau detected:** Last 3 runs with different HPs all within +/-5% eval of each other
4. **Human interrupt:** User tells the agent to stop or switches to collaborative mode

On stop, the agent writes a summary to the tuning log:
```markdown
## Session Summary
Best run: Run 4 (eval=198, lr=3e-4, gamma=0.97, alpha_init=0.1)
Runs completed: 6/10
Reason stopped: plateau detected (runs 4-6 all eval 195-200)
Key findings:
- alpha_init=0.1 >> 1.0 (entropy collapse with default)
- gamma=0.97 >> 0.99 (consistent with CheetahRun findings)
- hidden_dim=(256,256) sufficient, (512,256,128) no improvement
Recommended next steps: tier 3 reward shaping (tracking weights) — needs human
```

---

## Run Launch Protocol

Before launching each run:

1. **GPU check:** `nvidia-smi | grep python` — if anything running, wait and re-poll
2. **Build CLI command** from tuning log config, including `--wandb` flag
3. **Launch with `run_in_background=true`** — don't pipe stdout
4. **Discover wandb run ID** — query `wandb.Api().runs("sman2/jax-rl", order="-created_at", per_page=1)` filtered by run name, or parse from `wandb/latest-run` symlink. Record in tuning log.
5. **Poll loop:** every 5 min, pull latest metrics from wandb API. Fallback: `grep EVAL <output_file> | tail -5` if wandb API is stale.
6. **Run completion:** detect via wandb run state (`finished` or `failed`), confirmed by `nvidia-smi` (process gone)
7. **Stuck run timeout:** if no new metrics for 20 min and process still alive, log as stuck. Agent can kill and retry or surface to human.

If a run fails (crash, NaN, OOM):
- Log the failure, error message, and checkpoint path in the tuning log
- Diagnose using rules (NAN_DETECTED, OOM_CRASH, etc.)
- Count it against the budget
- Launch the next run with the fix

---

## CLI / Invocation

The agent is invoked as a Claude Code slash command or natural language request:

```
# Autonomous
"tune SAC on Go2, target eval 200, budget 10 runs"

# Collaborative
"tune SAC on Go2, collaborative mode — I want to see reward breakdowns"

# Resume
"resume the Go2 tuning session"  (reads .context/tuning/go2_sac_phase_b.md)
```

The agent needs these CLI args exposed on `train_offpolicy.py` to control HPs from the command line. Current state:
- **Already exist:** `--lr`, `--seed`, `--total-timesteps`, `--num-envs`, `--obs-norm`, `--wandb`, `--batch-size`, `--grad-updates-per-step`, `--buffer-size`, `--reward-scaling`
- **Need to be added:** `--alpha-init`, `--tau`, `--gamma`, `--hidden-dim`, `--policy-delay`

Notes:
- `--gamma` routes to `TrainConfig` (shared config), not algo config. Use `dataclasses.replace(cfg, gamma=...)`.
- `--hidden-dim` needs a custom type converter: `type=lambda s: tuple(int(x) for x in s.split(","))` to parse `256,256` or `512,256,128`.
- All other new flags route to algo config via `dataclasses.replace(algo_cfg, ...)`.

---

## Collaborative Mode Details

For tier 3 (reward shaping) or when rules don't match:

1. Agent pulls reward breakdown from wandb or `_traj.npz` (record_video saves per-term rewards)
2. Presents findings:
   - "Tracking reward is only 22% of total — pose dominates. Similar to the Go2 PPO debugging (see go2/lessons.md)"
   - "Q correlation is 0.95 — critic is accurate, the policy just isn't exploring high-reward regions"
3. Waits for human input before launching next run
4. Human can:
   - Suggest specific HP changes ("try tracking_lin_vel=15")
   - Ask for more diagnostics ("show me the action distribution")
   - Switch back to autonomous ("ok just sweep lr and gamma from here")

---

## Files Involved

### New files
- `.context/tuning/<session>.md` — tuning log (created per session)

### Modified files
- `train_offpolicy.py` — add CLI flags for HP overrides (alpha_init, tau, gamma, hidden_dim, policy_delay)

### Read-only dependencies
- `jax_rl/training/metrics_logger.py` — uses existing W&B prefixed keys for diagnosis
- `.context/LESSONS.md` — LLM reasoning reads this for known failure patterns
- `.context/go2/lessons.md` — Go2-specific context for reward shaping decisions

---

## What This Is NOT

- **Not W&B Sweeps** — we control the loop, one sequential run at a time, with LLM reasoning in the loop
- **Not a Python script** — it's a Claude Code session with access to LLM reasoning for novel situations
- **Not population-based training** — sequential, each run informed by all previous runs
- **Not an RL agent tuning RL** — rule-based heuristics + LLM reasoning, not learned optimization

---

## Implementation Phases

### Phase 1: Foundation
- Add missing CLI flags to `train_offpolicy.py`
- Implement tuning log read/write helpers
- Implement wandb metric polling
- Implement diagnostic rules

### Phase 2: Autonomous loop
- Run launch + GPU gating
- Poll → diagnose → decide → launch cycle
- Stopping conditions
- Session summary generation

### Phase 3: Collaborative mode
- Reward breakdown analysis
- Human-readable diagnostic reports
- Interactive mode with human input gates

### Phase 4: Hardening
- Session resume after disconnect
- Failed run recovery
- Edge cases (run stuck, wandb sync lag, etc.)
