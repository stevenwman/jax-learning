# Agent Handoff — JAX RL Framework

**Last updated:** 2026-03-26
**Branch:** `new_slate_linen`
**Status:** Active development — Go2 Phase A (PPO 244, motor actuators) and Phase B (FastSAC 226) COMPLETE. Sim2sim pipeline built, contact physics gap remaining.

> **Context budget:** This doc is your overview — skim structure, read details on demand. Other `.context/` docs are reference material. Don't pre-load them. When you hit a topic (Go2 rewards, PPO debugging, vision RL), grep or read the specific file. Treat `.context/` as a wiki, not a textbook.

---

## Part 1: Who You're Working With

Steven is a researcher building a JAX-based RL framework for robot learning. The end goal is skill discovery (DIAYN, METRA, USD) on real robots, but we're building the foundation first — algorithms, benchmarks, infrastructure.

### His expectations of you
- **Push back when warranted.** He doesn't want a yes-machine. If an approach has problems, say so directly. "Of course!" followed by implementing a bad idea helps no one.
- **Surface assumptions before acting.** Before implementing anything non-trivial, list your assumptions and ask him to correct. The most common failure mode is making wrong assumptions and running with them unchecked.
- **Be honest about uncertainty.** Say "I don't know" or "I'm not sure" rather than guessing confidently. He'd rather you investigate than hallucinate.
- **Explain mechanisms, not just fixes.** He wants to understand *why* something works or fails, not just the patch. This is a learning project.
- **Get to the root cause, not just a workaround.** He does NOT accept vague explanations. "It works at 8K batch but not 32K" is a correlation, not a cause. You must isolate the exact mechanism. If you can't solve it, that's OK, but the cause must be crystal clear.
- **Document everything.** Negative results, debugging trails, design decisions — all go in journals and lessons. He reviews these across sessions.
- **Don't over-engineer.** The simplest solution that works is the right one. He'll tell you if he wants more.

### Communication style
- Direct, informal. "aight", "yuh", "rip", "lol" are normal.
- He'll interrupt you mid-task if priorities shift — be ready to context-switch.
- He reads code diffs and terminal output directly — don't summarize what he can see.
- He'll leave for breaks (sometimes hours) and come back expecting you to have continued autonomously if possible.
- When he says "document this" or "update lessons" — do it immediately, don't batch.

### Hard rules
- Always use `uv run python` (never `python` or `python3`)
- No `Co-Authored-By` lines in git commits — he explicitly removed these and doesn't want them
- Off-policy algos must NOT normalize obs before storing in replay buffer (normalize at sample time with `--obs-norm` instead)
- Check memory files at the path in MEMORY.md before starting work

---

## Part 2: How We Work Together

### The brainstorming pattern
1. Steven gives a direction ("let's do vision RL", "fix the NaN", "refactor the code")
2. You present options with tradeoffs and your recommendation
3. He picks or redirects
4. You draft a plan (in `.context/` or a plan file)
5. He reviews, adjusts
6. You execute

**Don't skip to implementation.** Even if it seems simple, present the approach first.

### The debugging pattern
1. **State the symptom clearly** — what failed, at what step, what error
2. **Form hypotheses** — ranked by likelihood
3. **Test each systematically** — one at a time, with concrete evidence
4. **Don't guess-and-fix** — stress test edge cases directly (inject NaN/Inf) instead of running full training to reproduce
5. **Document the trail** — even wrong hypotheses go in the journal

### The paper audit pattern
When implementing from a paper:
1. **Read the paper text** — get the high-level algorithm
2. **Find the source code** — paper text omits critical details (tau, network arch, activation, etc.)
3. **Dispatch agents to audit** — compare source code configs against our implementation
4. **Fix configs to match paper exactly** before benchmarking

**Rule:** ALWAYS get the source code before writing the loss function. The paper text is insufficient. We lost 2 days on FastDSAC because "Gaussian distributional critic" actually uses Huber loss, not Gaussian NLL.

### Accessing external resources
WebFetch gets blocked by many sites. Workarounds:
1. **Google Drive:** `uv run gdown --folder <url> -O /tmp/output`
2. **GitHub private repos:** Ask user to clone locally
3. **Anonymous review sites:** Usually blocked. Ask user to download and place in `/tmp/`
4. **ArXiv HTML:** Usually works with WebFetch
5. **Local files are always readable**

### The training run pattern
1. **Smoke test first** — 200k-500k steps to verify no crashes
2. **Launch full run in background** — `run_in_background=true`
3. **Check periodically** — `grep "EVAL" <output_file> | tail -10`
4. **Kill zombie GPU processes before new runs** — `nvidia-smi | grep python`, then `kill <pid>`
5. **Log results immediately** — update journal with eval scores, step counts, wall-clock time

**Critical:** Do NOT pipe background tasks through `| head -N` or `| tail -N` — the pipe kills the process.

### The documentation pattern

| Doc | What goes in it | When to update |
|---|---|---|
| `.context/journals/YYYY-MM-DD.md` | What happened today | After every significant event |
| `.context/LESSONS.md` → `.context/lessons/*.md` | Reusable debugging lessons | When you learn something future sessions need |
| `.context/TODO.md` | Prioritized task list | When tasks complete or priorities shift |
| `deploy/README.md` | Deploy setup, usage, troubleshooting | When deploy code, deps, or PD gains change |
| `.context/go2/sac_phase_b.md` | SAC experiment plan + research | When SAC config or findings change |
| `.context/go2/mjcf_comparison.md` | Training vs deploy physics diff | When env physics overrides change |

**Don't forget non-.context docs.** `deploy/README.md` and `deploy/go2_constants.py` must stay in sync with training env changes (PD gains, default pose, action scale). If you change `go2_warp_joystick.py` or `go2_warp_base.py`, check whether deploy constants need updating too.

### The refactor philosophy
Brax-style shared utilities. No Trainer base class, no BaseAlgorithm ABC. Envs are self-contained black boxes, algos own their math, training scripts mediate. See `.context/archive/refactor_idea.md` for the full reasoning.

---

## Part 3: How to Find Answers

The lessons system is your search engine for "has this been solved before?" Read the index BEFORE investigating any bug, implementing any algorithm, or tuning any hyperparameter.

### The retrieval pattern

**Step 1: Read the index.** `.context/LESSONS.md` (~87 lines). Every lesson is a one-liner grouped by topic.

**Step 2: Drill into the relevant topic file.** Only read what's relevant:

| Topic file | When to read it |
|------------|----------------|
| `.context/lessons/ppo.md` | Debugging PPO (entropy, GAE, VLoss, tanh squashing, Brax parity) |
| `.context/lessons/offpolicy.md` | SAC/TD3 (obs norm, replay ratio, exploration, target entropy) |
| `.context/lessons/distributional.md` | C51/FastTD3/FastSAC/FastDSAC (V_min/V_max, paper-vs-code) |
| `.context/lessons/jax_performance.md` | Slow training or JIT issues (lax.scan, recompilation, carry cost) |
| `.context/lessons/infrastructure.md` | Checkpoint/eval/recording bugs (orbax, preprocessing mismatch) |
| `.context/lessons/mjx.md` | NaN/Inf crashes, GPU OOM, MJX recompilation |
| `.context/lessons/go2.md` | Go2 env (reward balance, actuator limits, contact physics) |

**Step 3: Don't read what you don't need.** The index tells you exactly which file has what.

### When to check lessons
- **Before debugging** — the answer might already exist
- **Before implementing from a paper** — check the "paper says X but code does Y" pattern
- **Before tuning HPs** — entropy_coef, gamma, tau, target_entropy all have documented conclusions
- **After solving a bug** — add a lesson to the topic file AND a one-liner to the index

### Other docs

| Doc | Path | When |
|-----|------|------|
| TODO list | `.context/TODO.md` | What to work on |
| Latest journal | `.context/journals/` (highest date) | What happened last session |
| Memory files | Path in `MEMORY.md` | Behavioral preferences, project references |
| Go2 PPO debugging | `.context/go2/ppo_debugging.md` | Full hypothesis log, run table |
| Go2 SAC Phase B | `.context/go2/sac_phase_b.md` | SAC plan, replay ratio research |
| Lit mismatch audit | `.context/archive/FAST_ALGOS_LIT_MISMATCH.md` | Paper vs code config audit |
| Framework plan | `.context/references/rl_framework_plan.md` | North star architecture (Phases 1-6) |
| Vision design | `.context/references/vision_rl_design.md` | CNN encoder, MJWarp, ManiSkill |
| Docs site | `.context/references/docs_site.md` | MkDocs site: what auto-updates, what's manual, generators, deployment |

---

## Part 4: Codebase Overview

### The three layers
```
Environment (MuJoCo Playground)
    ↓ obs, reward, done
Training Script (train_*.py) — glue layer, owns the loop
    ↓ batch
Algorithm (jax_rl/algos/*.py) — pure math, no env knowledge
```

The algo **never** knows about the env. The training script decides how to collect data (on-policy rollouts for PPO, single steps for off-policy).

### Key entry points
```
jax-learning/
├── train_ppo.py              # PPO (Python loop, ~32k sps, all envs)
├── train_ppo_fast.py         # PPO (lax.scan, ~110k sps, JIT-able envs only)
├── train_offpolicy.py        # SAC/TD3/FastTD3/FastSAC via --algo flag
├── record_video.py           # Loads any checkpoint, renders rollout + _traj.npz
├── jax_rl/algos/             # ppo.py, sac.py, td3.py, fast_td3.py, fast_sac.py
├── jax_rl/envs/locomotion/   # go2_warp_base.py, go2_warp_joystick.py, go2_bongo_handstand.py (MJX files archived in archive/)
├── jax_rl/configs/           # train_config.py, *_config.py, env_presets.py
├── jax_rl/training/          # checkpointing, eval_runner, env_setup, metrics_logger
├── jax_rl/buffers/           # jax_replay_buffer.py, rollout_buffer.py
├── jax_rl/envs/wrappers/     # FrameStackWrapper, vendored training wrappers (Vmap, Episode, AutoReset, DR)
├── tests/                    # 221 tests (uv run python -m pytest tests/ -v)
└── tools/brax_baselines/     # Brax PPO A/B test scripts
```

### Env framework coupling
Training wrappers (Vmap, Episode, AutoReset, DR) are vendored in `jax_rl/envs/wrappers/training.py` — no Brax training wrapper dependency. `env_setup.py` still uses Playground's registry (`pg_registry.load()`) for env loading and `mjx_env.MjxEnv` as the env type. All other core infra (algos, networks, configs, buffers, utils) is pure JAX/Flax/Optax with zero env framework dependencies. To add ManiSkill/HumanoidBench, extract an env factory interface from env_setup.py — everything downstream works unchanged.

### Config system
Each algo has its own config dataclass. Presets in `env_presets.py` return `(TrainConfig, AlgoConfig)` tuples. PPO-specific fields live in `PPOConfig`, not `TrainConfig`. CLI overrides via `dataclasses.replace(cfg, lr=args.lr)`.

### Checkpoint format
Every checkpoint contains: `meta.json` (full config), `metrics.csv` (training curve), `actor_params.npy` (inference), `orbax/` (training resume). `load_actor_for_inference()` loads just actor_params.npy — no orbax needed.

### Go2 env key facts
- **Dict obs**: `{"state": (48,), "privileged_state": (116-122,)}`
- **PPO**: asymmetric AC — actor sees "state", critic sees "privileged_state"
- **SAC/TD3**: both actor AND critic see "state" (48d) — no asymmetric
- **Actuator model**: `motor` (direct torque) + external PD per substep. Matches unitree_mujoco and real robot. (Was `general` with built-in PD — switched 2026-03-26.)
- **Working PPO config**: tracking_lin_vel=10.0, tracking_ang_vel=5.0, height_termination=True, Kp=35, Kd=0.1, calf_torque=45.43Nm
- **Best PPO**: eval 244 @ 50M steps (seed 4000, motor actuators)
- **Warp env**: `Go2WarpJoystickFlat` — uses unitree_mujoco's go2.xml (full cylinder collision geometry) via MuJoCo Warp backend. Eliminates sim2sim gap. `contact_mode` flag: `"training"` (firm contacts) / `"deploy"` (unitree-native physics). Kp=20, Kd=0.5 (unitree RL gains). Best result: **FastSAC eval 276.5 @ 18M steps** — surpasses MJX PPO 244. Sim2sim to CPU MuJoCo validated (walks 20s+ with random commands).
- **CRITICAL:** Warp env has joint→actuator remapping (`_act_to_joint`). Unitree XML has different qpos vs ctrl ordering. Without remap, PD applies torques to wrong legs.

---

## Part 5: Current State

### Benchmark results (as of 2026-03-26)

**CheetahRun** (6-dim actions):
| Algo | Eval | Steps | Notes |
|------|------|-------|-------|
| PPO | 826 | 20M | 2048 envs |
| Vanilla SAC | 771 | 5M | 128 envs, 8 min |
| **FastTD3** | **880** | 86M | 1024 envs |

**HumanoidRun** (21-dim actions):
| Algo | Eval | Steps | Notes |
|------|------|-------|-------|
| PPO | ~10 | 60M | Algorithm limit, not bug |
| Vanilla SAC | 426 | 20M | 128 envs |
| **FastSAC** | **892** | 100M | 1024 envs, SOTA |

**Go2 Joystick — MJX (archived)** (Menagerie go2_mjx.xml, 12-dim actions; env files in `jax_rl/envs/locomotion/archive/`):
| Algo | Eval | Steps | Notes |
|------|------|-------|-------|
| Our PPO (motor) | **244** | 50M | Seed 4000, motor actuators + external PD |
| Our PPO (general) | 233 | 50M | Seed 2100, old general actuators (deprecated) |
| FastSAC | 226 | 16M | Off-policy validated |
| Brax PPO | 17.9 | 50M | A/B baseline |

**Go2 Joystick — Warp** (unitree go2.xml, full collision geometry):
| Algo | Eval | Steps | Notes |
|------|------|-------|-------|
| **FastSAC (asym critic)** | **279.2** | 20M | Actor 48d, critic 122d privileged. ~2x faster to 270+ |
| FastSAC (symmetric) | 276.5 | 18M | Kp=20/Kd=0.5, sim2sim to CPU validated |
| PPO | 132 | 50M | Kp=20/Kd=0.5, entropy collapsed to squat |

**Key takeaways:** Low-dim → FastTD3. High-dim → FastSAC. gamma=0.97 for locomotion. C51 helps at scale. Vanilla algos at 128 envs are competitive for sample efficiency. Use `motor` actuators for sim2sim/sim2real transfer. **Warp + unitree MJCF eliminates sim2sim gap** — FastSAC on Warp surpasses MJX PPO.

### Roadmap
See `TODO.md` for full prioritized list. Summary:
- **Done:** Warp env (FastSAC 276.5), MJX→CPU transfer, DR v1, sim2sim validated.
- **Short-term:** DIAYN (north star), Kp/Kd DR, W&B HP tuning agent
- **Mid-term:** Vision RL (CNN encoder, DrQ), real robot deployment
- **Long-term:** DIAYN → METRA → USD (skill discovery on real Go2)

### Strategic note: Warp is the sole Go2 backend
**MJX Go2 env archived** (`jax_rl/envs/locomotion/archive/` — go2_base.py, go2_joystick.py, go2_cpu.py). `Go2WarpJoystickFlat` is the sole active Go2 locomotion env. `Go2BongoHandstand` (Warp, bongo board task) is also active. Warp is strictly better for our use case: supports cylinder collisions (MJX can't), trains on the exact unitree MJCF (zero sim2sim gap), faster on complex scenes, and we only use NVIDIA GPUs. New envs (other robots, terrains) should be built on Warp from the start. DIAYN, Kp/Kd DR, frame stacking — all Warp-only.

---

## Part 6: Quick Reference

### Commands
```bash
# Training
uv run python train_ppo_fast.py --env Go2WarpJoystickFlat --num-envs 1024 --total-timesteps 50000000  # Warp backend (unitree MJCF)
uv run python train_offpolicy.py --algo fast_sac --env Go2WarpJoystickFlat --num-envs 1024 --total-timesteps 20000000 --domain-rand  # FastSAC + DR on Warp

# Monitoring
nvidia-smi | grep python                    # Is it running?
grep "EVAL" /tmp/claude-*/tasks/*.output     # Eval scores

# Recording
MUJOCO_GL=egl uv run python record_video.py --checkpoint checkpoints/<dir>

# Sim2sim validation (CPU MuJoCo, same unitree MJCF as Warp training)
MUJOCO_GL=egl uv run python deploy/sim2sim_direct.py --checkpoint checkpoints/<dir> --vx 0.5 --record /tmp/sim2sim.mp4

# Testing
uv run python -m pytest tests/ -v

# Committing
git add <files> && git commit -m "fix: description"  # No Co-Authored-By
```

### Gotchas (quick-reference)
1. **Kill zombie GPU processes before launching** — `nvidia-smi | grep python`
2. **JIT compilation takes 1-3 min** — empty output is normal at start
3. **Can't run two 1024-env trainings** on 16GB — queue them
4. **`| tail -N` pipe kills background processes** — never pipe background output
5. **`@flax.struct.dataclass`** for training state, not regular `@dataclass`
6. **`jnp.where` for conditional updates** — no Python `if` inside JIT'd functions
7. **`XLA_CLIENT_MEM_FRACTION=0.7`** set in all train scripts — prevents OOM from MJX recompilation
8. **Check lessons before debugging** — the answer might already be there
