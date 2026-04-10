# Off-Policy Loop Extraction — Spec

**Date:** 2026-04-10
**Related:** `.superpowers/specs/2026-04-10-offpolicy-training-scripts-review.md` (the review that motivated this)
**Status:** Draft → Plan

---

## Problem

The 4 non-FlashSAC per-algo training scripts (`train_sac.py`, `train_td3.py`, `train_fast_sac.py`, `train_fast_td3.py`) share **85-90% identical code** — roughly 200 of 250 lines per `train()` function is boilerplate (env setup, wandb, pipe/buffer, resume, tracker, ctx, ckpt_mgr, training loop, eval, checkpoint). The 10-20 lines that are genuinely algo-specific (optimizer, explore closure, log fields) are buried in noise.

**The disease is signal-to-noise, not duplication.** When you open `train_sac.py` to understand what SAC does, you wade through 200 lines of generic training plumbing to find the SAC-specific bits. The review's first instinct ("re-unify into one script with `if family == "sac"` branches") was wrong — it trades one kind of noise (duplication) for another (a monster dispatcher).

## Goal

Extract the shared off-policy training loop into a **single function-level utility** (no class, no inheritance, no hooks soup) in `jax_rl/training/offpolicy_loop.py` so that each per-algo script becomes a thin ~60-line file showing only what makes that algo distinct: optimizer choice, explore closure, log fields.

**Success criteria:**
1. `train_sac.py`, `train_td3.py`, `train_fast_sac.py`, `train_fast_td3.py` each have a `train()` function of ≤ 60 lines of non-import, non-blank code.
2. The extracted `run_offpolicy_loop` helper is ≤ 150 lines of non-comment code and reads top-to-bottom like a textbook off-policy loop.
3. All 4 thinned scripts parse their CLI, apply overrides, and dispatch to `run_offpolicy_loop` without functional regression (smoke tests — deferred to GPU availability).
4. `train_flashsac.py` is **untouched**. It stays standalone. Its Zeta noise, adaptive reward scaling, and BatchNorm coordination don't fit the shared loop shape.
5. The 14-line dict-obs detection block disappears from all per-algo scripts (folded into `make_envs` or a returned `EnvBundle` dataclass).

## Non-goals

- **No Trainer base class or ABC.** Project philosophy explicitly rejects this (AGENT_HANDOFF.md: "No Trainer base class, no BaseAlgorithm ABC"). The extraction is a function that takes the pieces.
- **No algo plugin registry.** The per-algo scripts still exist as explicit top-level entry points. Nothing is dynamically dispatched.
- **Not touching FlashSAC.** It has genuine algo-specific coupling (BN state, reward norm, Zeta noise). Forcing it into the shared helper would require hook-soup parameters that defeat the point.
- **Not touching `train_offpolicy.py`.** The old unified dispatcher already exists and was recently modernized (commit `3b00700`). It's not part of this refactor's scope. Separately, review item #2 (delete/archive it) is a later cleanup.
- **Not rewriting the algo files.** Any `jax_rl/algos/*.py` cleanup from the review (items on sac/td3/flash_sac) is deferred.
- **Not solving the `metrics_logger.py` fmt-string bug (#3).** Folded in as a separate bite-sized task because it's one line and the helper will pass `log_extra_fields` through to this function anyway.
- **Not fixing FlashSAC's unsafe truncation access (#8) or BN mutation (#7).** Deferred.

## Design

### Two helpers, one dataclass

#### 1. `EnvBundle` dataclass + `make_env_bundle` wrapper

Currently `make_envs` returns a 7-tuple `(env, env_step, env_state, eval_env, obs_dim, action_dim, key)` and each training script then does a 14-line block detecting `dict_obs`, `has_privileged`, `critic_obs_dim`. Add a new wrapper function alongside `make_envs` that returns a single dataclass:

```python
@dataclass
class EnvBundle:
    env: Any
    env_step: Callable
    env_state: Any
    eval_env: Any
    obs_dim: int
    action_dim: int
    critic_obs_dim: int | None  # None if symmetric
    has_privileged: bool
    dict_obs: bool
    key: jax.Array

def make_env_bundle(cfg, seed) -> EnvBundle:
    """make_envs + dict obs detection. Used by the 4 thinned off-policy scripts."""
    env, env_step, env_state, eval_env, obs_dim, action_dim, key = make_envs(cfg, seed)
    dict_obs = isinstance(env_state.obs, dict)
    has_privileged = dict_obs and "privileged_state" in env_state.obs
    critic_obs_dim = env_state.obs["privileged_state"].shape[-1] if has_privileged else None
    if dict_obs:
        obs_dim = env_state.obs["state"].shape[-1]
        if has_privileged:
            print(f"  Dict obs detected: actor={obs_dim}d, critic={critic_obs_dim}d (asymmetric)")
        else:
            print(f"  Dict obs detected: using 'state' key ({obs_dim}d) for off-policy")
    return EnvBundle(env, env_step, env_state, eval_env, obs_dim, action_dim,
                     critic_obs_dim, has_privileged, dict_obs, key)
```

**Why wrapping instead of modifying `make_envs`:** `make_envs` has 8 source callers (including `train_ppo.py`, `train_ppo_fast.py`, `train_offpolicy.py`, `train_flashsac.py`) and 3 test callers. Changing its signature atomically would touch 11 files for a refactor scoped to 4. The wrapping approach leaves every untouched caller untouched and keeps this refactor surgical. PPO and FlashSAC can adopt `make_env_bundle` later if they want.

#### 2. `run_offpolicy_loop()` function

Signature:

```python
def run_offpolicy_loop(
    cfg: TrainConfig,
    algo_cfg,
    algo,                       # already-constructed algo with optimizer bound
    algo_name: str,             # "sac" | "td3" | "fast_sac" | "fast_td3"
    env_bundle: EnvBundle,
    explore_fn: Callable,       # (actor_params, obs, key) -> action
    log_extra_fields: list,     # [(label, key, fmt), ...] for log_training_step
    log_extra_keys: list,       # keys to pull from metrics dict into wandb row
    seed: int = 0,
    resume: str | None = None,
    use_wandb: bool = False,
    wandb_project: str = "jax-rl",
) -> None:
    ...
```

The function:
1. Prints banner (algo_name + env + config)
2. Initializes W&B if requested
3. Builds `ObsPipeline`, buffer, `norm_state` from `env_bundle` + `algo_cfg`
4. Loads checkpoint if `resume` is set
5. Builds `EpisodeTracker`, `CheckpointManager`, `TrainContext`
6. Runs the main training loop (obs extract → normalize → explore → env step → buffer add → gradient updates → log → eval/checkpoint)
7. Runs final eval + checkpoint
8. Finishes W&B

The loop body is copy-pasted from one of the current scripts (they're identical). No new logic.

**What the helper does NOT take as parameters** (deliberately, to resist hook-soup):
- Optimizer construction — done in the caller before passing `algo`
- Algo instantiation — done in the caller
- A custom action-sampling-before-min-buffer — the uniform random action phase is universal
- Custom reward scaling — `cfg.reward_scaling` already exists
- Custom obs preprocessing — `ObsPipeline` handles it

If a future algo needs something the helper doesn't expose, that algo stays standalone (like FlashSAC).

### What each thinned script looks like

`train_sac.py` becomes approximately:

```python
"""SAC training script."""
import os, sys
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", "0.7")
sys.stdout.reconfigure(line_buffering=True)

import argparse
import optax

from jax_rl.algos.sac import SAC
from jax_rl.configs.env_presets import get_sac_preset
from jax_rl.training import make_envs, apply_cli_overrides, run_offpolicy_loop


def train(cfg, algo_cfg, seed=0, resume=None, use_wandb=False, wandb_project="jax-rl"):
    env_bundle = make_envs(cfg, seed)

    # SAC-specific: optimizer with optional grad clipping
    if algo_cfg.grad_clip_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(algo_cfg.grad_clip_norm),
            optax.adam(cfg.lr),
        )
    else:
        optimizer = optax.adam(cfg.lr)
    alpha_optimizer = optax.adam(algo_cfg.alpha_lr)

    # SAC-specific: algo
    algo = SAC(
        config=algo_cfg,
        obs_dim=env_bundle.obs_dim,
        action_dim=env_bundle.action_dim,
        optimizer=optimizer,
        alpha_optimizer=alpha_optimizer,
        gamma=cfg.gamma,
        handle_truncation=cfg.handle_truncation,
        critic_obs_dim=env_bundle.critic_obs_dim,
    )

    # SAC-specific: explore is pass-through (stochastic policy)
    def explore(actor_params, obs, key):
        return algo.select_action(actor_params, obs, key)

    run_offpolicy_loop(
        cfg, algo_cfg, algo, "sac", env_bundle, explore,
        log_extra_fields=[("Ent", "entropy", ".3f"), ("Alpha", "alpha", ".4f")],
        log_extra_keys=["entropy", "alpha", "alpha_loss"],
        seed=seed, resume=resume,
        use_wandb=use_wandb, wandb_project=wandb_project,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ... CLI args (kept per-script because some are algo-specific)
    args = parser.parse_args()
    cfg, algo_cfg = get_sac_preset(args.env)
    cfg, algo_cfg = apply_cli_overrides(args, cfg, algo_cfg)
    train(cfg, algo_cfg, seed=args.seed, resume=args.resume,
          use_wandb=args.wandb, wandb_project=args.wandb_project)
```

`train_td3.py` is identical structure with: TD3 algo, noise-injecting `explore()` closure, empty log fields. `train_fast_sac.py` adds the cosine LR schedule before the optimizer. `train_fast_td3.py` combines FastSAC's LR schedule with TD3's explore closure.

## Risks

1. **The `q_fn` lambda in eval_and_checkpoint.** Currently each script builds a lambda that calls `algo.get_q_value(ts, pipe.get_obs(obs), action, critic_obs=...)`. This assumes all 4 algos have a compatible `get_q_value` signature. **Mitigation:** unit test the helper against all 4 algos and fail loudly if signatures diverge. If they do diverge, the lambda moves into the helper and takes `algo` directly.

2. **The `_ts = training_state` closure workaround.** Currently there's a `_ts = training_state` line before the eval lambda to avoid capturing the loop variable by reference. This wart moves into the helper with a comment explaining why.

3. **Scope creep from changing `make_envs`.** Avoided by the wrapping approach (`make_env_bundle` wraps `make_envs`, nothing else changes). PPO scripts, FlashSAC, and 3 tests continue to call `make_envs` directly. No ripple.

4. **Debug-trace friction.** A bug in the loop now lives in `offpolicy_loop.py` instead of the script the user is running. **Mitigation:** keep the helper short (target ≤ 150 non-comment lines), include section comments matching the current structure (`# ── Env step ──`, `# ── Gradient updates ──`, etc.), and put a docstring at the top that lists the 4 variation points (algo, explore_fn, log_extra_fields, log_extra_keys).

5. **GPU is currently in use** (10.3 / 16.3 GiB, 86% util, two python processes). Smoke tests (`train_sac.py --total-timesteps 100000`) can't run today. **Mitigation:** unit tests (CPU-OK, use a trivial fake env + fake algo) are the primary validation this session. Per-script smoke tests get deferred to a "Tomorrow morning TODO" section in the plan.

## Open questions

1. **Where does `run_offpolicy_loop` live in `jax_rl/training/__init__.py`'s export surface?** Current exports: `make_envs, EpisodeTracker, load_checkpoint, log_training_step, make_metrics_row, maybe_eval_and_checkpoint, final_eval_and_checkpoint, ObsPipeline, TrainContext, apply_cli_overrides`. Add `run_offpolicy_loop` and `EnvBundle` to this list. No other change.

2. **Does `make_envs` still return a tuple for legacy callers, or does everything migrate atomically?** **Plan decision:** no migration. `make_envs` stays as-is. New `make_env_bundle` wrapper lives alongside it. Only the 4 thinned scripts use `make_env_bundle`. PPO/FlashSAC/tests untouched.

3. **Should `log_extra_fields` be a list of tuples or a dict?** Current code uses list of tuples `[(label, key, fmt)]`. Keep that shape to avoid touching `log_training_step`'s signature in this refactor. (The fmt-string bug in `metrics_logger.py` line 99 gets fixed in a separate task within this plan because it's one line and coupled to how the helper passes fields through.)

## Scope of this plan

Included:
- Create `EnvBundle` + refactor `make_envs`
- Create `run_offpolicy_loop` helper with unit tests
- Thin `train_sac.py`, `train_td3.py`, `train_fast_sac.py`, `train_fast_td3.py`
- Fix `metrics_logger.py` line 99 fmt-string bug (review item #3)
- Merge `ObsPipeline.make_buffer` / `make_buffer_with_critic` (review item #5)
- Update `AGENT_HANDOFF.md` to reflect thinned scripts (review item #9)

Deferred (added to plan's "Tomorrow morning TODO" section):
- Per-script GPU smoke tests (GPU busy, come back tomorrow)
- FlashSAC improvements (items #4, #7, #8)
- `train_offpolicy.py` deletion (item #2)
- Algo file clean-code nits (sac.py:98-103, td3.py:168-170, flash_sac.py:492)
