"""OOD belt-speed sweep for a trained PoseDR (or any splitbelt) checkpoint.

Builds fresh splitbelt envs with `tied(v)` schedule for each v in the sweep
list, runs N deterministic rollouts per setting via the standard
`jax_rl.utils.eval.evaluate` loop, and reports return mean/std.

Usage:
  XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python scripts/eval_splitbelt_ood.py \\
      --checkpoint checkpoints/.../best \\
      --speeds 0.3 0.5 1.0 1.5 2.0 2.5 \\
      --num-episodes 16 --episode-length 1000

Notes:
- PoseDR was trained with random_per_episode v∈[0.3, 1.5]. v∈{2.0, 2.5} is OOD.
- Reuses record_video._build_select_action for algo-agnostic actor reconstruction.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict

from jax_rl.training.checkpointing import load_actor_for_inference
from jax_rl.utils.eval import evaluate
from jax_rl.utils.normalization import normalize as norm_normalize, init as norm_init
import jax_rl.training.env_setup  # noqa: F401 — registers custom envs

# Reuse algo construction from record_video to stay algo-agnostic.
from scripts.record_video import _build_select_action  # type: ignore


def make_env(schedule_kind, schedule_params, base_env_name="Go2WarpSplitbeltPoseDR"):
    """Build splitbelt env with arbitrary schedule override."""
    from jax_rl.envs.locomotion.go2_warp_splitbelt import Go2WarpSplitbeltEnv
    from mujoco_playground import registry as pg_registry

    template = pg_registry.load(base_env_name)
    cfg = config_dict.ConfigDict(template._config.to_dict())
    cfg.unlock()
    cfg.schedule_kind = schedule_kind
    cfg.schedule_params = config_dict.create(**schedule_params)

    task_map = {
        "Go2WarpSplitbelt":       "splitbelt",
        "Go2WarpSplitbeltDR":     "splitbelt_dr",
        "Go2WarpSplitbeltPoseDR": "splitbelt_pose_dr",
    }
    return Go2WarpSplitbeltEnv(task=task_map[base_env_name], config=cfg)


def make_ood_env(speed: float, base_env_name: str = "Go2WarpSplitbeltPoseDR"):
    return make_env("tied", {"v": float(speed)}, base_env_name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--speeds", nargs="+", type=float, default=[0.3, 0.5, 1.0, 1.5, 2.0, 2.5],
                   help="Tied speed sweep")
    p.add_argument("--pairs", nargs="+", default=None,
                   help="Differential sweep: list of vL,vR pairs (e.g. 0.5,1.5 0.3,1.5 0.5,2.0)")
    p.add_argument("--num-episodes", type=int, default=16)
    p.add_argument("--episode-length", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-env", default="Go2WarpSplitbeltPoseDR")
    args = p.parse_args()

    meta, actor_params, norm_state, batch_stats = load_actor_for_inference(args.checkpoint)
    algo_name = meta.get("algo", "fast_sac")
    print(f"Loaded {algo_name} from {args.checkpoint}")

    # Probe one env to get obs_dim + action_dim.
    probe_env = make_ood_env(args.speeds[0], args.base_env)
    state = probe_env.reset(jax.random.PRNGKey(0))
    obs = state.obs["state"] if isinstance(state.obs, dict) else state.obs
    obs_dim = int(obs.shape[-1])
    action_dim = int(probe_env.action_size)
    print(f"  obs_dim={obs_dim} action_dim={action_dim}")

    algo, _kind = _build_select_action(meta, obs_dim, action_dim)

    init_key = jax.random.PRNGKey(args.seed)
    training_state = algo.init(init_key)
    if actor_params is not None:
        training_state = training_state.replace(actor_params=actor_params)
    if batch_stats is not None and hasattr(training_state, "actor_batch_stats"):
        training_state = training_state.replace(actor_batch_stats=batch_stats)
        algo._default_actor_bs = batch_stats

    # Splitbelt env returns dict obs (state + privileged_state). Actor sees only "state".
    # Detect whether ckpt used obs normalization (FastSAC defaults: False).
    use_obs_norm = bool(meta.get("fast_sac_config", {}).get("obs_normalization", False))

    def obs_normalize_fn(o):
        flat = o["state"] if isinstance(o, dict) else o
        if use_obs_norm and norm_state is not None:
            flat = norm_normalize(flat, norm_state)
        return flat

    from jax_rl.envs.wrappers import wrap_for_training

    def run_one(env):
        action_repeat = int(getattr(env._config, "action_repeat", 4))
        env_w = wrap_for_training(env, episode_length=args.episode_length, action_repeat=action_repeat)
        return evaluate(
            select_action_fn=algo.select_action,
            actor_params=actor_params,
            env=env_w,
            num_episodes=args.num_episodes,
            episode_length=args.episode_length,
            key=jax.random.PRNGKey(args.seed),
            obs_normalize_fn=obs_normalize_fn,
        )

    if args.pairs is not None:
        # Differential sweep
        print(f"\nDifferential sweep on {args.base_env}")
        print(f"  Train range: vL∈[0.3,1.5], ratio∈[0.5,2.0] (so max train ratio=2x, max diff~1.5)")
        print(f"  episodes/pair: {args.num_episodes}, episode_length: {args.episode_length}\n")
        print(f"  (vL, vR)        | ratio  diff  | {'mean':>8} {'std':>7} {'min':>8} {'max':>8}  | OOD?")
        print("-" * 84)
        for spec in args.pairs:
            vL, vR = (float(x) for x in spec.split(","))
            env = make_env("split_constant", {"vL": vL, "vR": vR}, args.base_env)
            m = run_one(env)
            ratio = vR / vL
            diff = vR - vL
            abs_ood = (vL < 0.3 or vL > 1.5) or (vR < 0.3 or vR > 1.5)
            ratio_ood = ratio < 0.5 or ratio > 2.0
            flags = []
            if abs_ood:   flags.append("ABS")
            if ratio_ood: flags.append("RATIO")
            flag = ",".join(flags) or ""
            print(f"  ({vL:.2f}, {vR:.2f})    | {ratio:>4.2f}x {diff:>+5.2f}  | "
                  f"{m['eval_mean']:>8.1f} {m['eval_std']:>7.1f} {m['eval_min']:>8.1f} {m['eval_max']:>8.1f}  | {flag}")
    else:
        # Tied speed sweep
        print(f"\nTied-speed sweep on {args.base_env} (PoseDR train range: v∈[0.3, 1.5])")
        print(f"  episodes/v: {args.num_episodes}, episode_length: {args.episode_length}\n")
        print(f"{'v':>6} | {'mean':>8} {'std':>7} {'min':>8} {'max':>8}  | OOD?")
        print("-" * 60)

        results = {}
        for v in args.speeds:
            env = make_ood_env(v, args.base_env)
            m = run_one(env)
            is_ood = v < 0.3 or v > 1.5
            flag = "OOD" if is_ood else ""
            print(f"{v:>6.2f} | {m['eval_mean']:>8.1f} {m['eval_std']:>7.1f} {m['eval_min']:>8.1f} {m['eval_max']:>8.1f}  | {flag}")
            results[v] = m

        in_dist = [m["eval_mean"] for v, m in results.items() if 0.3 <= v <= 1.5]
        if in_dist:
            peak = max(in_dist)
            print(f"\nIn-dist peak: {peak:.1f}")
            for v, m in sorted(results.items()):
                print(f"  v={v}: {m['eval_mean']:.1f} ({100 * m['eval_mean'] / peak:+.0f}% of peak)")


if __name__ == "__main__":
    main()
