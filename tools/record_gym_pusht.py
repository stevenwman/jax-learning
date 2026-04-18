"""Record gym-pusht rollout as video. Random policy baseline for visual check."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import subprocess
from pathlib import Path

import gymnasium as gym
import gym_pusht
import numpy as np
from PIL import Image


def main():
    out_dir = Path(__file__).parent.parent / ".temp"
    out_dir.mkdir(exist_ok=True)

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    frames = [env.render()]

    # Scripted heuristic: approach then push block toward target (center)
    ep_r = 0.0
    for t in range(300):
        # Simple heuristic: aim pusher at block, then push through block toward center.
        pusher = obs[:2]
        block = obs[2:4]
        to_center = np.array([256.0, 256.0]) - block
        to_center_unit = to_center / (np.linalg.norm(to_center) + 1e-6)
        # Approach point behind block (opposite from center)
        approach = block - 60.0 * to_center_unit
        to_approach = approach - pusher
        dist = np.linalg.norm(to_approach)
        if dist > 20.0:
            action = approach.astype(np.float32)
        else:
            # Push through block toward center
            action = (block + 120.0 * to_center_unit).astype(np.float32)
        action = np.clip(action, 0, 512)

        obs, r, term, trunc, info = env.step(action)
        ep_r += r
        frames.append(env.render())
        if term or trunc:
            break

    print(f"Episode reward: {ep_r:.2f}  coverage_last: {info.get('coverage', 0):.3f}  steps: {len(frames)-1}")

    # Save frames to temp dir
    tmp = out_dir / "_gym_pusht_frames"
    tmp.mkdir(exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(tmp / f"{i:04d}.png")

    out_mp4 = out_dir / "gym_pusht_heuristic.mp4"
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
    env.close()


if __name__ == "__main__":
    main()
