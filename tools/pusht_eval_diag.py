"""Re-run eval episodes and log per-step env stats (coverage, pos_err, angle_err)."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_rl.algos.sac import SAC
from jax_rl.configs.sac_config import SACConfig
from train_pusht import _make_env, policy_to_gym_action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--obs-type", type=str, default="environment_state_agent_pos")
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--action-repeat", type=int, default=2)
    ap.add_argument("--reward-mode", type=str, default="contact_gated")
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=150)
    args = ap.parse_args()

    actor_params = np.load(args.ckpt, allow_pickle=True).item()

    env_probe = _make_env(args.reward_mode, args.obs_type, args.frame_stack, args.action_repeat)
    obs_dim = int(env_probe.observation_space.shape[0])
    action_dim = int(env_probe.action_space.shape[0])
    env_probe.close()

    algo = SAC(SACConfig(hidden_dim=(256, 256), critic_hidden_dim=(256, 256)),
               obs_dim, action_dim, optax.adam(1e-3), optax.adam(1e-3))

    @jax.jit
    def act_det(p, o):
        return algo.select_action(p, o, jax.random.PRNGKey(0), deterministic=True)

    @jax.jit
    def act_sto(p, o, k):
        return algo.select_action(p, o, k, deterministic=False)

    for mode, act_fn in [("det", lambda p, o, k: act_det(p, o)),
                         ("sto", lambda p, o, k: act_sto(p, o, k))]:
        print(f"\n=== {mode.upper()} ===")
        print(f"{'ep':>2} {'seed':>5} {'steps':>6} {'final_cov':>10} {'peak_cov':>10} {'final_pos_err':>14} {'final_angle_err':>16} {'final_return':>13}")
        for i in range(args.n_episodes):
            seed = 1000 + i
            env = _make_env(args.reward_mode, args.obs_type, args.frame_stack, args.action_repeat)
            obs, _ = env.reset(seed=seed)
            key = jax.random.PRNGKey(i + 7777)
            total_r = 0.0
            peak_cov = 0.0
            info = {}
            n_steps = 0
            for _ in range(args.max_steps):
                key, ak = jax.random.split(key)
                a_pm1 = np.asarray(act_fn(actor_params, jnp.asarray(obs[None]), ak))[0]
                obs, r, term, trunc, info = env.step(policy_to_gym_action(a_pm1).astype(np.float32))
                total_r += float(r)
                peak_cov = max(peak_cov, float(info.get("coverage", 0)))
                n_steps += 1
                if term or trunc:
                    break
            final_cov = float(info.get("coverage", 0))
            pos_err = float(np.linalg.norm(
                np.array(env.unwrapped.block.position) - env.unwrapped.goal_pose[:2]
            ))
            angle_err_raw = float(env.unwrapped.block.angle) - float(env.unwrapped.goal_pose[2])
            angle_err = abs(np.arctan2(np.sin(angle_err_raw), np.cos(angle_err_raw)))
            angle_err_deg = np.degrees(angle_err)
            print(f"{i:>2} {seed:>5} {n_steps:>6} {final_cov:>10.4f} {peak_cov:>10.4f} "
                  f"{pos_err:>14.2f} {angle_err_deg:>14.2f}° {total_r:>13.2f}")
            env.close()


if __name__ == "__main__":
    main()
