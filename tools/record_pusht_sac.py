"""Record trained FastSAC policy on gym-pusht. Loads actor_params saved by train_pusht.py."""
import argparse
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image

from jax_rl.algos.sac import SAC
from jax_rl.configs.sac_config import SACConfig
from jax_rl.envs.manipulation.pusht import PushTEnv


def policy_to_gym_action(a_pm1):
    return (a_pm1 + 1.0) * 0.5 * 512.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to actor_params_*.npy file")
    ap.add_argument("--reward-mode", type=str, default="dense")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    actor_params = np.load(ckpt_path, allow_pickle=True).item()

    env = PushTEnv(obs_type="state", reward_mode=args.reward_mode, render_mode="rgb_array")
    obs, info = env.reset(seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Vanilla SAC — no C51, matches training-time config.
    algo_cfg = SACConfig(hidden_dim=(256, 256), critic_hidden_dim=(256, 256))
    dummy_opt = optax.adam(1e-3)
    algo = SAC(algo_cfg, obs_dim, action_dim, dummy_opt, dummy_opt)

    @jax.jit
    def act(actor_params, obs):
        return algo.select_action(actor_params, obs, jax.random.PRNGKey(0), deterministic=True)

    frames = [env.render()]
    total_r = 0.0
    for _ in range(args.max_steps):
        a_pm1 = np.asarray(act(actor_params, jnp.asarray(obs[None])))[0]
        a_gym = policy_to_gym_action(a_pm1).astype(np.float32)
        obs, r, term, trunc, info = env.step(a_gym)
        total_r += float(r)
        frames.append(env.render())
        if term or trunc:
            break

    print(f"total_r={total_r:.2f}  final_coverage={info.get('coverage', 0):.3f}  "
          f"success={info.get('is_success', False)}  frames={len(frames)}")

    out_path = Path(args.out) if args.out else \
        Path.cwd() / ".temp" / f"pusht_sac_{args.reward_mode}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / "_rec_frames"
    tmp.mkdir(exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(tmp / f"{i:04d}.png")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "10",
        "-i", str(tmp / "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(out_path),
    ], check=True, capture_output=True)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    env.close()
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
