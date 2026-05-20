"""Probe a trained G1 ckpt for foot-lift behavior.

Dumps per-step foot z, target rz from gait phase, and feet_phase reward
value. Goal: confirm whether the reward already saturates ~1.0 even when
the robot shuffles (foot_z stays near 0).

Usage:
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl uv run python \
    scripts/diagnostics/g1_foot_lift_probe.py \
    --checkpoint checkpoints/20260508_014027_fast_sac_g1warpjoystickholosoft_seed0/best \
    --steps 200 --out projects/adaptation/diagnostics/foot_probe_v22.csv
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import csv
import jax
import jax.numpy as jnp
import numpy as np
import optax

from mujoco_playground import registry as pg_registry
from mujoco_playground._src import gait

from jax_rl.training.checkpointing import load_actor_for_inference
import jax_rl.training.env_setup  # noqa: F401
from jax_rl.training.env_backends.mjx_backend import maybe_load_custom_env
from jax_rl.utils.normalization import normalize as norm_normalize
from jax_rl.utils.normalization import init as norm_init


def build_algo(meta, obs_dim, action_dim):
    algo = meta.get("algo", "fast_sac")
    sc = meta.get("fast_sac_config" if algo == "fast_sac" else "flash_sac_config", {})
    dummy_opt = optax.adam(1e-3)
    if algo == "fast_sac":
        from jax_rl.algos.fast_sac import FastSAC
        from jax_rl.configs.fast_sac_config import FastSACConfig
        cfg = FastSACConfig(
            hidden_dim=tuple(sc.get("hidden_dim", (512, 256, 128))),
            activation=sc.get("activation", "swish"),
            q_layer_norm=sc.get("q_layer_norm", True),
            target_entropy_scale=sc.get("target_entropy_scale", 0.0),
            num_atoms=sc.get("num_atoms", 101),
            v_min=sc.get("v_min", -20.0),
            v_max=sc.get("v_max", 20.0),
            q_aggregation=sc.get("q_aggregation", "avg"),
            critic_hidden_dim=tuple(sc["critic_hidden_dim"]) if sc.get("critic_hidden_dim") else None,
        )
        return FastSAC(cfg, obs_dim, action_dim, dummy_opt, dummy_opt, gamma=0.99)
    elif algo == "flash_sac":
        from jax_rl.algos.flash_sac import FlashSAC
        from jax_rl.configs.flash_sac_config import FlashSACConfig
        cfg = FlashSACConfig(
            num_blocks=sc.get("num_blocks", 2),
            actor_hidden_dim=sc.get("actor_hidden_dim", 128),
            critic_hidden_dim=sc.get("critic_hidden_dim", 256),
            expansion=sc.get("expansion", 4),
            num_atoms=sc.get("num_atoms", 101),
            v_min=sc.get("v_min", -5.0),
            v_max=sc.get("v_max", 5.0),
            sigma_target=sc.get("sigma_target", 0.15),
        )
        return FlashSAC(cfg, obs_dim, action_dim, dummy_opt, dummy_opt, gamma=0.99)
    raise ValueError(f"unsupported algo: {algo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--env", default="G1WarpJoystickHoloSoft")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", default="projects/adaptation/diagnostics/foot_probe.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cmd-x", type=float, default=None,
                    help="If set, lock cmd_x to this value (default: env-sampled)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ── load ckpt + env ────────────────────────────────────────────────────
    meta, actor_params, norm_state, _ = load_actor_for_inference(args.checkpoint)
    algo_name = meta.get("algo", "fast_sac")
    use_obs_norm = False
    for k in (f"{algo_name}_config", "fast_sac_config", "flash_sac_config"):
        if k in meta and meta[k].get("obs_normalization", False):
            use_obs_norm = True
            break
    print(f"[ckpt] algo={algo_name} obs_norm={use_obs_norm}")

    env = maybe_load_custom_env(args.env)
    if env is None:
        env = pg_registry.load(args.env)

    # Unwrap to base env (find underlying G1WarpJoystick for accessors).
    base = env
    while hasattr(base, "env"):
        base = base.env

    swing_height = base._config.reward_config.gait_swing_height
    sigma = base._config.reward_config.feet_phase_sigma
    print(f"[env] swing_height={swing_height} feet_phase_sigma={sigma}")

    env_step = jax.jit(env.step)
    key = jax.random.PRNGKey(args.seed)
    key, reset_key = jax.random.split(key)
    state = env.reset(reset_key)

    # Optional cmd lock — patch _cmd_a to zero out non-x components so the
    # Markov-chain `sample_command` cannot resample cmd_x off the lock.
    if args.cmd_x is not None:
        base._cmd_a = jnp.array([0.0, 0.0, 0.0])
        cmd = state.info["command"].at[0].set(args.cmd_x).at[1].set(0.0).at[2].set(0.0)
        state.info["command"] = cmd

    raw_obs = state.obs
    policy_obs = raw_obs["state"] if isinstance(raw_obs, dict) else raw_obs
    obs_dim = policy_obs.shape[-1]
    action_dim = env.action_size

    algo = build_algo(meta, obs_dim, action_dim)
    key, init_key = jax.random.split(key)
    ts = algo.init(init_key)
    ts = ts.replace(actor_params=actor_params)
    if norm_state is None:
        norm_state = norm_init(obs_dim)

    # ── rollout + per-step probe ───────────────────────────────────────────
    feet_site_id = base._feet_site_id
    rows = []
    for t in range(args.steps):
        obs = state.obs["state"] if isinstance(state.obs, dict) else state.obs
        if use_obs_norm:
            obs = norm_normalize(norm_state, obs)
        obs_b = obs[None, :] if obs.ndim == 1 else obs
        key, ak = jax.random.split(key)
        action = algo.select_action(ts.actor_params, obs_b, ak, deterministic=True)
        action = jnp.asarray(action).reshape(-1)

        phase = state.info["phase"]
        rz = gait.get_rz(phase, swing_height=swing_height)
        foot_pos = state.data.site_xpos[feet_site_id]
        foot_z = foot_pos[..., -1]
        err = float(jnp.sum(jnp.square(foot_z - rz)))
        feet_phase_rew = float(jnp.exp(-err / sigma))

        cmd = np.asarray(state.info["command"])
        rows.append({
            "t": t,
            "phase_L": float(phase[0]),
            "phase_R": float(phase[1]),
            "rz_L": float(rz[0]),
            "rz_R": float(rz[1]),
            "foot_z_L": float(foot_z[0]),
            "foot_z_R": float(foot_z[1]),
            "err": err,
            "feet_phase_rew": feet_phase_rew,
            "cmd_x": cmd[0],
            "cmd_y": cmd[1],
            "cmd_yaw": cmd[2],
            "reward_total": float(state.reward),
            "done": int(state.done),
        })

        state = env_step(state, action)
        if bool(state.done):
            print(f"[done] step {t}")
            break

    # ── write csv + summary ────────────────────────────────────────────────
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[out] {args.out} ({len(rows)} steps)")

    arr = lambda k: np.array([r[k] for r in rows])
    print("\n=== summary (post-warmup steps 30..end) ===")
    skip = min(30, len(rows) // 4)
    fpr = arr("feet_phase_rew")[skip:]
    fzL = arr("foot_z_L")[skip:]
    fzR = arr("foot_z_R")[skip:]
    rzL = arr("rz_L")[skip:]
    rzR = arr("rz_R")[skip:]
    print(f"feet_phase_rew  mean={fpr.mean():.3f}  median={np.median(fpr):.3f}  min={fpr.min():.3f}  p10={np.percentile(fpr,10):.3f}")
    print(f"foot_z_L  mean={fzL.mean():.4f}  max={fzL.max():.4f}  p90={np.percentile(fzL,90):.4f}")
    print(f"foot_z_R  mean={fzR.mean():.4f}  max={fzR.max():.4f}  p90={np.percentile(fzR,90):.4f}")
    print(f"rz_L      mean={rzL.mean():.4f}  max={rzL.max():.4f}  p90={np.percentile(rzL,90):.4f}")
    print(f"rz_R      mean={rzR.mean():.4f}  max={rzR.max():.4f}  p90={np.percentile(rzR,90):.4f}")
    print(f"target swing_height={swing_height}  sigma={sigma}")
    if fpr.mean() > 0.85 and max(fzL.max(), fzR.max()) < swing_height * 0.5:
        print("\n>>> CONCLUSION: reward saturates near 1.0 while foot stays well below "
              "swing_height. Hypothesis 1a (feet_phase saturation) confirmed.")
    elif max(fzL.max(), fzR.max()) >= swing_height * 0.7:
        print("\n>>> CONCLUSION: foot lift is reaching target; not a feet_phase problem. "
              "Look elsewhere (cmd range, push events).")


if __name__ == "__main__":
    main()
