# Agent Handoff — JAX RL Framework

**Last updated:** 2026-03-21
**Branch:** `new_slate_linen`
**Status:** Active development — off-policy RL algorithms, approaching vision RL phase

---

## Part 1: Who You're Working With

Steven is a researcher building a JAX-based RL framework for robot learning. The end goal is skill discovery (DIAYN, METRA, USD) on real robots, but we're building the foundation first — algorithms, benchmarks, infrastructure.

### His expectations of you
- **Push back when warranted.** He doesn't want a yes-machine. If an approach has problems, say so directly. "Of course!" followed by implementing a bad idea helps no one.
- **Surface assumptions before acting.** Before implementing anything non-trivial, list your assumptions and ask him to correct. The most common failure mode is making wrong assumptions and running with them unchecked.
- **Be honest about uncertainty.** Say "I don't know" or "I'm not sure" rather than guessing confidently. He'd rather you investigate than hallucinate.
- **Explain mechanisms, not just fixes.** He wants to understand *why* something works or fails, not just the patch. This is a learning project.
- **Get to the root cause, not just a workaround.** He does NOT accept vague explanations. "It works at 8K batch but not 32K" is a correlation, not a cause. You must isolate the exact mechanism — which computation produces the first NaN, which gradient explodes, which parameter collapses. If you can't solve it, that's OK, but the cause must be crystal clear.
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
- Off-policy algos must NOT normalize obs before storing in replay buffer (see LESSONS.md for the full disaster story)
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

**Don't skip to implementation.** Even if it seems simple, present the approach first. He caught several wrong assumptions this way (e.g., we almost built a pixel wrapper before discovering Playground has `vision=True` built in).

### The debugging pattern
When something breaks:
1. **State the symptom clearly** — what failed, at what step, what error
2. **Form hypotheses** — ranked by likelihood
3. **Test each systematically** — one at a time, with concrete evidence
4. **Don't guess-and-fix** — the NaN investigation went through 3 wrong hypotheses before finding the real cause (MJX physics NaN, not algo instability)
5. **Document the trail** — even wrong hypotheses go in the journal. Future sessions need to know what was tried.

Example debugging trail (FastTD3 HumanoidRun NaN):
- H1: tau=0.125 too aggressive → wrong (debug script survived 2M steps)
- H2: obs normalization causing NaN → wrong (NaN'd with and without)
- H3: v_min/v_max mismatch → wrong (Q values stable before NaN)
- H4: MJX physics NaN (stochastic) → **correct** (debug sync points masked it, different step counts each run)
- Fix: NaN guard on env output + C51 log_prob clamp + action NaN guard

### The paper audit pattern
When implementing from a paper:
1. **Read the paper text** — get the high-level algorithm
2. **Find the source code** — paper text omits critical details (tau, network arch, activation, etc.)
3. **Dispatch agents to audit** — compare source code configs against our implementation
4. **Document ALL disparities** in `.context/archive/FAST_ALGOS_LIT_MISMATCH.md`
5. **Fix configs to match paper exactly** before benchmarking

We learned this the hard way — our FastTD3/FastSAC had 11+ critical config mismatches vs the paper's holosoma source code. tau was 25x wrong, network architecture was completely different, activation function was wrong. Always check the source.

### The "paper says X but code does Y" lesson (FastDSAC saga)
The most expensive debugging lesson from this project: FastDSAC's paper describes a "Gaussian distributional critic" with "Gaussian NLL loss." We implemented exactly that. It NaN'd. We spent two days trying variance clamping, log-variance parameterization, batch size reduction — all treating symptoms. Then we downloaded the paper's source code and discovered **the actual loss is Huber-based, not Gaussian NLL.** No `1/variance` anywhere in the real code. The name "Gaussian distributional" describes the output parameterization (mean, std), not the loss function.

**Rule:** When implementing from a paper, ALWAYS get the source code before writing the loss function. The paper text is insufficient. If source code is behind a paywall or anonymous link:
1. Try `uv run gdown --folder <google_drive_url>` for Google Drive links
2. Try `git clone` for anonymous review repos (4open.science, openreview)
3. Ask the user to download and place it locally — you can read local files
4. WebFetch often gets blocked by Cloudflare on anonymous review sites

### Accessing external resources
WebFetch gets blocked by many sites (Cloudflare challenges, bot detection, auth-required pages). When you hit a block:
1. **Google Drive:** `uv add gdown --dev && uv run gdown --folder <url> -O /tmp/output`
2. **GitHub private repos:** Ask user to clone locally
3. **Anonymous review sites (4open.science, openreview):** Usually blocked. Ask user to download ZIP and place in `/tmp/`
4. **ArXiv HTML:** Usually works with WebFetch
5. **PyPI packages:** Check if already installed (`uv run python -c "import X"`) before adding
6. **Local files are always readable** — if user can download it anywhere on the filesystem, you can read it

### Algorithm implementation tips
- **Q head outputs (mean, std) via softplus** — NOT (mean, variance). The loss formulation depends on this distinction.
- **Huber loss > MSE for distributional RL** — Huber caps large TD errors at linear growth, MSE lets them explode quadratically.
- **Per-sample gradient weighting with clamped ratios** — `clamp(weight, 0.1, 10)` prevents any single sample from dominating the batch gradient.
- **`z.clamp(-3, 3)` on target sampling** — prevents extreme tail samples from the target distribution.
- **EMA of batch std (`mean_std`)** — used for ratio computation, updated with tau_b=0.005 (much slower than Polyak tau).
- **`num_updates=2` for distributional critics** — more updates per step amplifies any instability. The paper uses 2, not 8.
- **`reward_scale=0.2`** — smaller rewards → smaller TD errors → more stable variance learning.

### The training run pattern
1. **Smoke test first** — 200k-500k steps to verify no crashes
2. **Launch full run in background** — `run_in_background=true`
3. **Check periodically** — `grep "EVAL" <output_file> | tail -10`
4. **Kill zombie GPU processes before new runs** — `nvidia-smi | grep python`, then `kill <pid>`
5. **Log results immediately** — update journal with eval scores, step counts, wall-clock time
6. **Queue next runs** — GPU can only handle one 1024-env run at a time

**Critical:** When launching background tasks, do NOT pipe through `| head -N` or `| tail -N` — the pipe kills the process after N lines of output. Use `run_in_background=true` and check the output file separately.

### The documentation pattern
Three docs serve different purposes:

| Doc | What goes in it | When to update |
|---|---|---|
| `.context/journals/YYYY-MM-DD.md` | What happened today — runs, results, decisions, failures | After every significant event |
| `.context/LESSONS.md` | Reusable debugging lessons — root causes, gotchas, patterns | When you learn something future sessions need |
| `.context/TODO.md` | Prioritized task list with status | When tasks complete or priorities shift |

**Rule:** Steven will ask "did you update lessons and journal?" after significant findings. If you haven't, do it before moving on. Don't batch documentation — it gets forgotten.

### The refactor philosophy
We chose **Brax-style shared utilities** over SB3-style class hierarchy or Tianshou-style Trainer class. The reasoning:

- **Algos own their training loops** — each `train_*.py` is self-contained
- **Shared utilities handle infra** — checkpointing, episode tracking, eval, env setup, logging
- **No Trainer base class** — too many layers to trace, PPO doesn't fit the off-policy template
- **No BaseAlgorithm ABC** — premature, interface is informal (`init`, `update`, `select_action`)

This was a deliberate decision after researching Brax, SB3, Tianshou, CleanRL, Isaac Lab, and MuJoCo Playground architectures. The full analysis is referenced in the 03-19 journal.

---

## Part 3: Repository Deep Dive

### Directory structure
```
jax-learning/
├── train_ppo.py              # PPO (on-policy, fundamentally different loop)
├── train_sac.py          # Vanilla SAC (128 envs, 5M steps default)
├── train_td3.py          # Vanilla TD3
├── train_fast_td3.py     # FastTD3: TD3 + C51 + large batch (1024 envs)
├── train_fast_sac.py     # FastSAC: SAC + C51 + paper recipe
├── train_fast_dsac.py    # FastDSAC: Gaussian distributional + DEM
├── record_video.py       # Algo-agnostic: loads any checkpoint, renders rollout
│
├── jax_rl/
│   ├── algos/            # Each algo is a standalone class, no base class
│   │   ├── ppo.py        # On-policy, uses RolloutBuffer + GAE
│   │   ├── sac.py        # Twin Q, entropy + alpha auto-tuning
│   │   ├── td3.py        # Twin Q, deterministic policy, delayed updates
│   │   ├── fast_td3.py   # + C51 distributional critic, Q averaging
│   │   ├── fast_sac.py   # + C51, policy delay=4, AdamW
│   │   └── fast_dsac.py  # + Gaussian critic, DEM, population diversity
│   │
│   ├── buffers/
│   │   ├── replay_buffer.py      # Numpy circular FIFO (CPU, default for vanilla)
│   │   ├── jax_replay_buffer.py  # GPU-resident, JIT'd add/sample (default for Fast)
│   │   └── rollout_buffer.py     # PPO on-policy buffer with GAE
│   │
│   ├── configs/
│   │   ├── train_config.py       # Shared: env_name, lr, gamma, num_envs, etc.
│   │   ├── ppo_config.py         # PPO-specific (clip_eps, gae_lambda, etc.)
│   │   ├── sac_config.py         # SAC + obs_normalization toggle
│   │   ├── td3_config.py         # TD3 + exploration noise
│   │   ├── fast_td3_config.py    # + C51 atoms, v_min/v_max, tapered dims
│   │   ├── fast_dsac_config.py   # + DEM temperature, beta range, variance eps
│   │   ├── env_presets.py        # Per-env config tuples: get_sac_preset("CheetahRun")
│   │   └── networks_config.py   # EncoderConfig, PolicyHeadConfig, etc.
│   │
│   ├── networks/
│   │   ├── encoders/mlp.py       # MlpEncoder (CNN planned)
│   │   ├── heads/
│   │   │   ├── gaussian.py       # GaussianHead (SAC actor, with optional DEM logits)
│   │   │   ├── deterministic.py  # DeterministicHead (TD3 actor)
│   │   │   ├── q_head.py         # Monolithic QHead (obs+action → scalar)
│   │   │   ├── q_distributional.py  # C51 QHead (obs+action → atom logits)
│   │   │   ├── q_gaussian.py     # Gaussian QHead (obs+action → mean+variance)
│   │   │   └── value.py          # ValueHead (PPO critic)
│   │   ├── builders.py           # Actor/Critic builders (PPO only currently)
│   │   └── distributions.py      # TanhNormal sampling via distrax
│   │
│   ├── training/                 # Shared utilities (the refactor result)
│   │   ├── checkpointing.py      # save_checkpoint, load_checkpoint, load_actor_for_inference
│   │   ├── episode_tracker.py    # EpisodeTracker class
│   │   ├── env_setup.py          # make_envs (with NaN guard), make_identity_norm_state
│   │   ├── metrics_logger.py     # log_training_step, make_metrics_row
│   │   └── eval_runner.py        # maybe_eval_and_checkpoint, final_eval_and_checkpoint
│   │
│   └── utils/
│       ├── normalization.py      # NormalizationState, update, normalize (with eps param)
│       ├── eval.py               # evaluate() — deterministic rollout with NaN guard
│       └── distributional.py     # C51: make_support, project_distribution, logits_to_q
│
├── tests/                        # 48 tests, all passing
│   ├── test_algo_configs.py      # All algos: init, update, optimizer compat, NaN guard
│   ├── test_checkpoint.py        # Save/load, meta.json, metrics CSV, orbax
│   ├── test_determinism.py       # Env + PPO training determinism
│   ├── test_normalization.py     # Normalization utilities
│   ├── test_ppo_setup.py         # PPO init, action, update, buffer/GAE
│   └── debug_fasttd3_nan.py      # Diagnostic script (not a pytest test)
│
├── checkpoints/                  # Training outputs (gitignored, ~299MB, needs purge)
│
└── .context/                     # Project documentation hub
    ├── AGENT_HANDOFF.md          # This file
    ├── TODO.md                   # Prioritized task list
    ├── LESSONS.md                # Accumulated debugging lessons (~800 lines)
    ├── journals/                 # Daily work logs
    │   ├── 2026-03-12.md through 2026-03-20.md
    ├── rl_framework_plan.md      # North star architecture (Phases 1-6)
    ├── refactor_idea.md          # Shared utilities design (Brax-style, not Trainer class)
    ├── vision_rl_design.md       # Vision RL: CNN encoder, MJWarp, ManiSkill
    ├── oom_investigation.md      # GPU OOM root cause (MJX recompilation)
    ├── FAST_ALGOS_LIT_MISMATCH.md # Paper vs implementation config audit
    └── builders_unification_plan.md # Encoder swappability for CNN/ViT
```

### The three layers of the codebase
```
Environment (MuJoCo Playground)
    ↓ obs, reward, done
Training Script (train_*.py) — glue layer, owns the loop
    ↓ batch
Algorithm (jax_rl/algos/*.py) — pure math, no env knowledge
```

- **Env** produces obs, consumes actions. We don't own this layer.
- **Algo** owns the math: `init()`, `update()`, `select_action()`. Never touches the env directly.
- **Training script** mediates: env step → buffer add → buffer sample → algo update → log → eval → checkpoint.

The algo **never** knows about the env. The training script decides how to collect data (on-policy rollouts for PPO, single steps for off-policy).

### Config system in detail
Each algo has its own config dataclass. Presets in `env_presets.py` return `(TrainConfig, AlgoConfig)` tuples:

```python
cfg, sac_cfg = get_sac_preset("CheetahRun")
# cfg: TrainConfig(env_name="CheetahRun", num_envs=128, lr=1e-3, gamma=0.99, ...)
# sac_cfg: SACConfig(tau=0.005, hidden_dim=(256,256), batch_size=512, ...)
```

**Note:** PPO-specific fields (`num_steps`, `policy_hidden_dim`, `value_hidden_dim`, `squash`, etc.) live in `PPOConfig`, not `TrainConfig`. `train_ppo.py` reads from `cfg.ppo` for these fields. `TrainConfig` has only shared fields (env, scale, optimizer, eval).

**Config override pattern** in train scripts:
```python
parser.add_argument("--lr", type=float, default=None)
if args.lr is not None:
    cfg = dataclasses.replace(cfg, lr=args.lr)
```

### Checkpoint format
Every checkpoint directory contains:
- `meta.json` — full config (TrainConfig + AlgoConfig), obs_dim, action_dim, algo name
- `metrics.csv` — full training curve (step, return, Q1, loss, sps, eval scores)
- `actor_params.npy` — actor params + norm state (for inference/recording)
- `orbax/` — full training state for resume (actor + critic + optimizer + target nets)

`load_actor_for_inference()` loads just actor_params.npy — no orbax, no algo import needed. This is what `record_video.py` uses.

---

## Part 4: Benchmark Results and What They Mean

### Full comparison table (as of 2026-03-21)

**CheetahRun** (6-dim actions — low-dim, deterministic policies work well):
| Algo | Eval | Steps | Envs | sps | Wall-clock |
|------|------|-------|------|-----|------------|
| PPO | 826 | 20M | 4096 | 65k | ~5 min |
| Vanilla SAC | 771 | 5M | 128 | 10k | ~8 min |
| Vanilla TD3 | 749 | 5M | 128 | 10k | ~7 min |
| **FastTD3** | **880** | 86M | 1024 | 18k | ~78 min |
| FastSAC (best) | 582 | 100M | 1024 | 15k | ~108 min |
| FastDSAC (best) | 567 | 30M | 1024 | 18k | ~28 min |

**HumanoidRun** (21-dim actions — high-dim, entropy exploration critical):
| Algo | Eval | Steps | Envs | sps | Wall-clock |
|------|------|-------|------|-----|------------|
| PPO | ~10 | 60M | 4096 | 65k | ~15 min |
| Vanilla TD3 | 4.3 | 5M | 128 | 10k | ~7 min |
| Vanilla SAC | 426 | 20M | 128 | 4.4k | ~76 min |
| FastTD3 | 665 | 100M | 1024 | 8k | ~3.4 hr |
| **FastSAC** | **892** | 100M | 1024 | 12.3k | ~2.3 hr |
| FastDSAC (128 envs) | 490 peak | 5M | 128 | 305 | ~4.5 hr |
| FastDSAC (1024 envs) | 316 peak (Inf'd @ 53M, now guarded) | 53M | 1024 | 2.3k | *running with Inf fix* |

### What the benchmarks tell us
- **Low-dim (CheetahRun):** Deterministic TD3 + C51 wins. SAC's entropy overhead isn't worth it when simple Gaussian noise suffices for 6-dim exploration.
- **High-dim (HumanoidRun):** SAC's entropy-based exploration is decisive. TD3 can't explore 21-dim action spaces with additive noise. FastSAC 892 vs FastTD3 665.
- **C51 distributional critic helps at scale** — both FastTD3 (880 vs 749) and FastSAC (892 vs 426) benefit from distributional representation.
- **Vanilla algos at 128 envs / 5M steps are competitive** — SAC 771 on CheetahRun beats FastSAC 582 at 100M steps. More envs ≠ better if the algorithm recipe is wrong.
- **gamma=0.97 is critical** — FastSAC went from 375 (gamma=0.99) to 582 (gamma=0.97) on CheetahRun. Same algo, same everything else.

---

## Part 5: Known Issues and Active Investigations

### MJX Physics NaN (SOLVED)
MuJoCo's MJX backend produces NaN obs stochastically on humanoid envs at 1024 parallel worlds. Contact solver failures or singular mass matrices.

**Fix:** Three-layer NaN guard in `env_setup.py`:
1. Zero NaN actions before `env.step()` (prevents NaN→actor→NaN action→crash chain)
2. Zero NaN obs after `env.step()`, force `done=True` (auto-resets crashed envs)
3. C51 log_prob clamp (`jnp.maximum(log_softmax, -30)`) prevents `-inf * 0 = NaN`

### GPU OOM at 67-85M Steps (MITIGATED, NOT FULLY SOLVED)
MJX recompiles `jit(while)` and `jit(scan)` with identical signatures ~2x/min. Over 2 hours, this accumulates ~480 CUDA command buffers that exhaust GPU driver memory.

**Mitigation:** `XLA_CLIENT_MEM_FRACTION=0.7` leaves headroom. Set in all train scripts.
**Root cause unknown:** Why does JAX recompile identical functions? Is it MJX, Playground, or JAX itself? Investigation pending — see `oom_investigation.md`.

### FastDSAC: Fully Rewritten to Match Paper Source Code (TESTING)
The paper says "Gaussian NLL" but the actual source code uses **Huber loss** (delta=50) with bounded ratio weighting. Our Gaussian NLL implementation NaN'd; the Huber rewrite is stable. See LESSONS.md for the full diagnostic trail.

**Two 1024-env scaling issues found and fixed:**
1. **Buffer too small** (51K at 1024 envs → 2M NaN). Fix: scale buffer to 400K.
2. **`Inf` not guarded** (MJX velocity overflow → 53M NaN). Fix: add `isinf()` to env step guard.

**Current status:** Running at 1024 envs with both fixes. If it survives past 53M steps, both fixes are confirmed. FastDSAC at 128 envs reached 490 peak eval on HumanoidRun.

### Debugging workflow: stress test edge cases directly
Don't wait for a full training run to reproduce a crash. Inject the suspected failure condition directly:
```python
# Instead of running 53M steps to see if it NaN's:
batch['obs'] = batch['obs'].at[0].set(float('inf'))
state, metrics = dsac.update(state, batch)
# If NaN → confirmed. If fine → wrong hypothesis. 30 seconds, not 7 hours.
```
This found the Inf root cause in one test after multiple failed full runs.

### Paper Config Disparities (DOCUMENTED, MOSTLY FIXED)
See `.context/archive/FAST_ALGOS_LIT_MISMATCH.md` for the full audit. Key items still not matching paper:
- Obs normalization: paper uses it, we have it as opt-in toggle
- Separate actor/critic normalizers: paper has two, we have one
- Some per-task hyperparameters not tuned (DEM temperature, beta range)

---

## Part 6: Upcoming Work

### Short-term (no GPU needed)
1. ~~TrainConfig cleanup~~ — **DONE** (PPO fields in PPOConfig, 48 tests pass)
2. ~~Q diagnostics~~ — **DONE**. All off-policy algos have `get_q_value(state, obs, action)`. Eval prints Q bias, RMSE, correlation vs MC returns automatically.
3. **Builders unification** — `make_encoder()` factory. Next immediate task. Prerequisite for SAC on Go2 and vision RL.
4. **Go2 sim-to-real** — comprehensive plan in `.context/go2_sim_to_real_plan.md`. Go2 MJCF from Menagerie, MjxEnv subclass, legged_gym rewards, deploy via ONNX on Jetson Orin Nano.
2. **Checkpoints purge** — delete orbax weights from failed runs, keep meta.json + metrics.csv
3. **Builders unification** — `make_encoder()` factory, prerequisite for CNN encoder

### Mid-term: Vision RL
Design doc: `.context/vision_rl_design.md`

**Key discovery:** Playground has built-in `vision=True` via Madrona MJX. No custom pixel wrapper needed:
```python
config_overrides = {"vision": True, "vision_config.render_batch_size": num_envs}
env = dm_control_suite.load(env_name, config_overrides=config_overrides)
env = wrapper.wrap_for_brax_training(env, vision=True, num_vision_envs=num_envs, ...)
```

**MJWarp rendering also verified working** on our hardware (RTX 5080):
```python
rc = mjx.create_render_context(model, nworld=4)  # works
result = mjx.render(mx, data, rc.pytree())        # produces (nworld, H*W) uint32 packed RGBA
```

Implementation order:
1. Install `madrona_mjx`, verify vision env loads
2. Builders unification
3. CNN encoder (`NatureCNN` + optional MLP layers, `CnnEncoderConfig`)
4. `--vision` flag on train scripts
5. DrQ augmentation (random image shifts)
6. Benchmark CartpoleBalance from pixels
7. ManiSkill integration (Gymnasium adapter + DLPack torch→JAX bridge)

### Long-term: Phase 6 (North Star)
- DIAYN — skill discovery wrapping SAC
- METRA — contrastive + metric-aware skills
- Goal-conditioned RL — encoder `context_dim` + `context_fusion`
- USD — Unified Skill Discovery

---

## Part 7: Common Gotchas

### GPU management
1. **Always check for zombie processes before launching:** `nvidia-smi | grep python`
2. **Kill zombies:** `kill <pid>`, wait 3 seconds, verify with `nvidia-smi`
3. **cuSolver errors** ("gpusolverDnCreate failed") = zombie hogging GPU memory
4. **Can't run two 1024-env trainings simultaneously** on 16GB — queue them
5. **JIT compilation takes 1-3 min** for 1024-env humanoid. Empty output is normal during this time.

### JAX/Flax specifics
6. **`jax.jit` caches by (function identity + input shapes + shardings)** — same shapes always reuse the cached compilation (except MJX's while/scan bug)
7. **`jax.lax.scan` > Python loops** for repeated computation — 542x speedup measured
8. **`@flax.struct.dataclass`** for training state, not regular `@dataclass` — needed for JAX tree operations
9. **`jnp.where` for conditional updates** — no Python `if` inside JIT'd functions
10. **DLPack for torch→JAX conversion** — zero-copy on same GPU, but has sync point

### Training pitfalls
11. **`| tail -N` pipe kills background processes** — never pipe background training output
12. **Obs normalization before buffer storage = disaster** — stale stats → ±2 billion values → Q divergence
13. **C51 `log_softmax` can produce `-inf`** — always clamp: `jnp.maximum(log_softmax(...), -30)`
14. **Softplus variance → 0 for large negative inputs** — use log-variance with clamping instead
15. **`tau=0.125` is not a typo** — Fast variants need it for high UTD ratio (8-12 gradient steps per env step)
16. **gamma=0.97 for locomotion** — paper uses it, verified massive performance difference
17. **FastDSAC batch size is 32K** (paper spec) — 4x larger than other algos

### Documentation pitfalls
18. **Don't batch documentation** — update journal/lessons immediately after discoveries
19. **Check LESSONS.md before debugging** — the answer might already be there from a previous session
20. **Journal ≠ Lessons** — journal is "what happened today", lessons are "what to remember forever"

---

## Part 8: Testing

```bash
uv run python -m pytest tests/ -v              # Full suite, 48 tests, ~2.5 min (needs GPU)
uv run python -m pytest tests/ -v -k "not ppo" # Skip PPO tests (less GPU)
uv run python -m pytest tests/test_normalization.py -v  # No GPU needed
```

### What the tests cover
- **test_algo_configs.py** (28 tests) — every algo: init + 1 update step with Adam/AdamW, critic_hidden_dim, policy delay, alpha init, obs norm config, NaN guard
- **test_checkpoint.py** (7 tests) — save/load round-trip, meta.json structure, metrics CSV, orbax restore
- **test_determinism.py** (2 tests) — env determinism, full PPO training determinism
- **test_normalization.py** (6 tests) — normalization utilities
- **test_ppo_setup.py** (5 tests) — PPO init, action selection, deterministic, update, buffer/GAE

### Test gaps (known)
- No integration test for sample-time obs normalization in training loop
- No test for record_video.py (needs display or EGL)
- No test for vision pipeline (not built yet)

---

## Part 9: Quick Reference

### Launch a training run
```bash
# Vanilla SAC on CheetahRun (quick, 128 envs)
uv run python train_sac.py --env CheetahRun --jax-buffer

# FastTD3 on HumanoidRun (long, 1024 envs)
uv run python train_fast_td3.py --env HumanoidRun --obs-norm

# With custom params
uv run python train_sac.py --env WalkerWalk --total-timesteps 10000000 --lr 3e-4 --obs-norm
```

### Check a running training
```bash
nvidia-smi | grep python                    # Is it running?
grep "EVAL" /tmp/claude-*/tasks/*.output     # Eval scores
tail -3 /tmp/claude-*/tasks/*.output         # Latest metrics
```

### Record a video from checkpoint
```bash
MUJOCO_GL=egl uv run python record_video.py --checkpoint checkpoints/<dir>
```

### Run tests
```bash
uv run python -m pytest tests/ -v
```

### Commit
```bash
git add <files>
git commit -m "fix: description of what and why"
# No Co-Authored-By lines
```

---

## Part 10: Document Navigation Guide

### Always read first
| Doc | Path | When |
|-----|------|------|
| This handoff | `.context/AGENT_HANDOFF.md` | Start of every session |
| TODO list | `.context/TODO.md` | To know what to work on |
| Memory files | Path in `MEMORY.md` (in the memory directory) | For behavioral preferences, project references |
| Latest journal | `.context/journals/` (highest date) | To know what happened last session |

### Read when debugging
| Doc | Path | What it tells you |
|-----|------|-------------------|
| Lessons | `.context/LESSONS.md` | Every debugging victory — **check here before investigating**, the answer might already exist |
| OOM investigation | `.context/archive/oom_investigation.md` | Full trail of the GPU memory investigation: hypotheses, tests, root cause (MJX recompilation) |

### Read when implementing algorithms
| Doc | Path | What it tells you |
|-----|------|-------------------|
| Lit mismatch audit | `.context/archive/FAST_ALGOS_LIT_MISMATCH.md` | Every config where our code differs from the paper, with severity ratings and source code citations |
| Framework plan | `.context/rl_framework_plan.md` | North star architecture (Phases 1-6), with documented deviations explaining where/why we diverged |
| Builders plan | `.context/builders_unification_plan.md` | How to make encoders swappable (MLP→CNN→ViT). Prerequisite for vision RL |

### Read when working on vision
| Doc | Path | What it tells you |
|-----|------|-------------------|
| Vision design | `.context/vision_rl_design.md` | Full design: CNN encoder, MJWarp verification, Madrona MJX `vision=True`, ManiSkill integration, DrQ augmentation, memory budget |

### Read when refactoring
| Doc | Path | What it tells you |
|-----|------|-------------------|
| Refactor idea | `.context/refactor_idea.md` | Brax-style shared utilities approach, framework comparison table (Brax vs SB3 vs Tianshou vs CleanRL), why we chose utilities over Trainer class |

### Read when adding a new algo
| Doc | Path | What it tells you |
|-----|------|-------------------|
| Any existing `jax_rl/algos/*.py` | Codebase | Pattern to follow: `__init__` builds networks, `init()` creates TrainingState, `update()` is JIT'd, `select_action()` for inference |
| Any existing `train_*.py` | Codebase | Training loop pattern: env step → buffer → sample → normalize → update → log → eval → checkpoint |
| `jax_rl/configs/env_presets.py` | Codebase | How to add per-env hyperparameter presets |

### Cross-references within docs
- **Journal mentions a lesson** → full lesson in LESSONS.md
- **LESSONS.md mentions a config mismatch** → full audit in FAST_ALGOS_LIT_MISMATCH.md
- **TODO.md references a design doc** → full design in the linked `.context/*.md`
- **Framework plan mentions a deviation** → deviation reason inline, implementation details in the relevant journal entry
- **Lit mismatch cites source code** → holosoma repo (`github.com/amazon-far/holosoma`) for FastTD3/FastSAC, arXiv 2603.12612 for FastDSAC
