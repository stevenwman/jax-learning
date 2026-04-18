"""Replay a LeRobot pusht expert demo in gym-pusht, record video.

Usage:
    uv run python -m jax_rl.envs.manipulation.pusht.demos.replay
    uv run python -m jax_rl.envs.manipulation.pusht.demos.replay --episode 5 --seed 42
    uv run python -m jax_rl.envs.manipulation.pusht.demos.replay --out /tmp/demo.mp4

Note: the env seed used here won't match the one used during data collection
(LeRobot doesn't store it), so replayed actions operate on a different initial
state than the demo was recorded from. This visualizes *what an expert did*,
not an exact reproduction. For pixel-exact demo playback, use the upstream
video at `videos/observation.image/chunk-000/file-000.mp4` on HF.
"""
import argparse
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import subprocess
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401 — registers env
import numpy as np
from PIL import Image

from jax_rl.envs.manipulation.pusht.demos import get_episode, n_episodes


def replay_episode(episode_idx: int, env_seed: int, out_path: Path) -> None:
    ep = get_episode(episode_idx)
    actions = ep["actions"]
    print(f"episode {episode_idx}: {len(actions)} frames")

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    obs, info = env.reset(seed=env_seed)

    frames = [env.render()]
    total_r = 0.0
    for a in actions:
        obs, r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
        total_r += float(r)
        frames.append(env.render())
        if term or trunc:
            break

    coverage = info.get("coverage", 0)
    success = info.get("is_success", False)
    print(f"replay: total_r={total_r:.2f}  final_coverage={coverage:.3f}  "
          f"success={success}  frames={len(frames)}")

    tmp = out_path.parent / f"_replay_frames_{episode_idx}"
    tmp.mkdir(parents=True, exist_ok=True)
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

    print(f"saved {out_path}")
    env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0,
                    help=f"Episode index (0 to {n_episodes()-1})")
    ap.add_argument("--seed", type=int, default=0, help="env reset seed")
    ap.add_argument("--out", type=str, default=None,
                    help="Output mp4 path (default: .temp/pusht_demo_ep{N}.mp4)")
    args = ap.parse_args()

    if args.episode < 0 or args.episode >= n_episodes():
        raise SystemExit(f"episode must be in [0, {n_episodes()-1}]")

    out_path = Path(args.out) if args.out else \
        Path.cwd() / ".temp" / f"pusht_demo_ep{args.episode}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    replay_episode(args.episode, args.seed, out_path)


if __name__ == "__main__":
    main()
