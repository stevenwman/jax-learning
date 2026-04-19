"""Record trained v9 SAC policy — 5 seeds, gym-pusht with full wrapper stack."""
import argparse
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image

from jax_rl.algos.sac import SAC
from jax_rl.configs.sac_config import SACConfig
from jax_rl.envs.manipulation.pusht import PushTEnv
from train_pusht import _make_env, policy_to_gym_action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--obs-type", type=str, default="environment_state_agent_pos")
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--action-repeat", type=int, default=2)
    ap.add_argument("--reward-mode", type=str, default="contact_gated")
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=150)  # 300 env / 2 repeat
    ap.add_argument("--out-prefix", type=str, default=".temp/pusht_v9_ep")
    args = ap.parse_args()

    actor_params = np.load(args.ckpt, allow_pickle=True).item()

    # Build one env to get obs_dim
    env_probe = _make_env(args.reward_mode, args.obs_type, args.frame_stack, args.action_repeat)
    obs_dim = int(env_probe.observation_space.shape[0])
    action_dim = int(env_probe.action_space.shape[0])
    env_probe.close()

    algo_cfg = SACConfig(hidden_dim=(256, 256), critic_hidden_dim=(256, 256))
    dummy_opt = optax.adam(1e-3)
    algo = SAC(algo_cfg, obs_dim, action_dim, dummy_opt, dummy_opt)

    @jax.jit
    def act(actor_params, obs, key):
        return algo.select_action(actor_params, obs, key, deterministic=True)

    out_dir = Path(args.out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(0)

    for i in range(args.n_episodes):
        env = _make_env(args.reward_mode, args.obs_type, args.frame_stack, args.action_repeat)
        # unwrapped render env (pygame)
        render_env = PushTEnv(obs_type="state", reward_mode=args.reward_mode, render_mode="rgb_array")
        obs, _ = env.reset(seed=1000 + i)
        render_env.reset(seed=1000 + i)

        frames = [render_env.render()]
        total_r = 0.0
        for _ in range(args.max_steps):
            key, ak = jax.random.split(key)
            a_pm1 = np.asarray(act(actor_params, jnp.asarray(obs[None]), ak))[0]
            a_gym = policy_to_gym_action(a_pm1).astype(np.float32)
            obs, r, term, trunc, info = env.step(a_gym)
            # Mirror action on render_env (same step rate because of repeat).
            for _ in range(args.action_repeat):
                _, _, rt, ru, ri = render_env.step(a_gym)
                frames.append(render_env.render())
                if rt or ru:
                    break
            total_r += float(r)
            if term or trunc:
                break

        cov = info.get("coverage", 0.0)
        success = info.get("is_success", False)
        print(f"ep {i}: total_r={total_r:.1f}  cov={cov:.3f}  success={success}  frames={len(frames)}")

        out_mp4 = Path(f"{args.out_prefix}{i}.mp4")
        tmp = out_dir / f"_v9_ep{i}"
        tmp.mkdir(exist_ok=True)
        for k, f in enumerate(frames):
            Image.fromarray(f).save(tmp / f"{k:04d}.png")
        subprocess.run([
            "ffmpeg", "-y", "-framerate", "20",  # 10Hz * 2 repeat
            "-i", str(tmp / "%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            str(out_mp4),
        ], check=True, capture_output=True)
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()
        env.close()
        render_env.close()


if __name__ == "__main__":
    main()
