"""Replay a LeRobot pusht expert demo in gym-pusht, record video."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import subprocess
from pathlib import Path

import gymnasium as gym
import gym_pusht
import numpy as np
from datasets import load_dataset
from PIL import Image


def main(episode_idx: int = 0):
    print(f"loading lerobot/pusht...")
    ds = load_dataset("lerobot/pusht", split="train")

    # Filter to one episode
    ep_rows = [r for r in ds if r["episode_index"] == episode_idx]
    print(f"episode {episode_idx}: {len(ep_rows)} frames")

    actions = np.array([r["action"] for r in ep_rows], dtype=np.float32)
    print(f"actions: shape={actions.shape}  range=[{actions.min():.1f}, {actions.max():.1f}]")

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    obs, info = env.reset(seed=42)  # note: dataset seeding may differ, replay won't be exact

    frames = [env.render()]
    total_r = 0.0
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        frames.append(env.render())
        if term or trunc:
            break

    coverage = info.get("coverage", 0)
    success = info.get("is_success", False)
    print(f"replay: total_r={total_r:.2f}  final_coverage={coverage:.3f}  success={success}  frames={len(frames)}")

    # Render mp4
    out_dir = Path(__file__).parent.parent / ".temp"
    tmp = out_dir / "_demo_frames"
    tmp.mkdir(exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(tmp / f"{i:04d}.png")

    out_mp4 = out_dir / f"pusht_demo_ep{episode_idx}.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "10",
        "-i", str(tmp / "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(out_mp4),
    ], check=True, capture_output=True)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    print(f"saved {out_mp4}")


if __name__ == "__main__":
    main(episode_idx=0)
