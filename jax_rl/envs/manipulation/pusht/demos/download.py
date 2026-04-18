"""Download lerobot/pusht demos and pack into a single .npz.

Source: https://huggingface.co/datasets/lerobot/pusht
License: Apache 2.0 (same as gym-pusht)

Produces: pusht_demos.npz
    obs_state     (N, 2)    agent xy (DP original has 2d; full 5d state not in this dataset)
    actions       (N, 2)    agent target xy
    rewards       (N,)      DP coverage-based reward (per step)
    dones         (N,)      episode terminal flag
    successes     (N,)      is_success flag
    episode_idx   (N,)      which episode each transition belongs to
    frame_idx     (N,)      frame within episode
    ep_bounds     (206, 2)  (start, end) row index per episode, inclusive start / exclusive end
"""
import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset


def main(out_path: str):
    print("loading lerobot/pusht (~25k frames, 206 eps)...")
    ds = load_dataset("lerobot/pusht", split="train")
    n = len(ds)

    obs_state = np.zeros((n, 2), dtype=np.float32)
    actions = np.zeros((n, 2), dtype=np.float32)
    rewards = np.zeros((n,), dtype=np.float32)
    dones = np.zeros((n,), dtype=bool)
    successes = np.zeros((n,), dtype=bool)
    episode_idx = np.zeros((n,), dtype=np.int32)
    frame_idx = np.zeros((n,), dtype=np.int32)

    for i, row in enumerate(ds):
        obs_state[i] = row["observation.state"]
        actions[i] = row["action"]
        rewards[i] = row["next.reward"]
        dones[i] = row["next.done"]
        successes[i] = row["next.success"]
        episode_idx[i] = row["episode_index"]
        frame_idx[i] = row["frame_index"]
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{n}")

    # Compute per-episode (start, end) bounds
    n_eps = int(episode_idx.max() + 1)
    ep_bounds = np.zeros((n_eps, 2), dtype=np.int32)
    starts = np.where(np.concatenate([[True], np.diff(episode_idx) > 0]))[0]
    ends = np.concatenate([starts[1:], [n]])
    ep_bounds[:, 0] = starts
    ep_bounds[:, 1] = ends

    out_path = Path(out_path)
    np.savez_compressed(
        out_path,
        obs_state=obs_state, actions=actions, rewards=rewards,
        dones=dones, successes=successes,
        episode_idx=episode_idx, frame_idx=frame_idx,
        ep_bounds=ep_bounds,
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"saved {out_path} ({size_mb:.2f} MB, {n} transitions, {n_eps} episodes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "pusht_demos.npz"))
    args = ap.parse_args()
    main(args.out)
