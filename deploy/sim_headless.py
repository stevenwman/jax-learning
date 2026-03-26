#!/usr/bin/env python3
"""Headless unitree_mujoco simulator for sim2sim testing over SSH.

Runs MuJoCo physics + DDS bridge (same as unitree_mujoco) but without the viewer.
Optionally records video to file.

Usage:
    # Just physics + DDS (no video):
    deploy/.venv/bin/python deploy/sim_headless.py

    # With video recording:
    deploy/.venv/bin/python deploy/sim_headless.py --record sim2sim_test.mp4 --duration 10

    # Then in another terminal, run your policy:
    deploy/.venv/bin/python deploy/deploy_go2.py --checkpoint checkpoints/.../best --sim --vx 0.5
"""
import argparse
import sys
import os
import time
import threading
import numpy as np

# Must set before mujoco import for headless rendering
os.environ.setdefault("MUJOCO_GL", "egl")

# Add unitree_mujoco to path for the bridge module
UNITREE_MUJOCO = os.environ.get(
    "UNITREE_MUJOCO",
    os.path.expanduser("~/.local/share/unitree/unitree_mujoco")
)
sys.path.insert(0, os.path.join(UNITREE_MUJOCO, "simulate_python"))

import mujoco

# Patch config before importing bridge (bridge reads config at import time)
import config
config.USE_JOYSTICK = 0  # no joystick in headless mode
config.PRINT_SCENE_INFORMATION = False
config.ENABLE_ELASTIC_BAND = False

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge


def run_headless(
    duration: float = 0.0,
    record_path: str | None = None,
    fps: int = 30,
):
    """Run simulator headless with optional video recording."""

    robot_scene = os.path.join(UNITREE_MUJOCO, "unitree_robots", "go2", "scene.xml")
    if not os.path.exists(robot_scene):
        print(f"ERROR: Scene not found at {robot_scene}")
        print(f"Set UNITREE_MUJOCO env var to your unitree_mujoco clone path")
        sys.exit(1)

    mj_model = mujoco.MjModel.from_xml_path(robot_scene)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = config.SIMULATE_DT

    # Setup video recording
    renderer = None
    frames = []
    record_interval = 1.0 / fps  # seconds between frames
    last_frame_time = 0.0

    if record_path:
        renderer = mujoco.Renderer(mj_model, width=640, height=480)
        print(f"Recording to {record_path} at {fps} fps")

    # Initialize DDS bridge
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    bridge = UnitreeSdk2Bridge(mj_model, mj_data)

    sim_time = 0.0
    step_count = 0
    run_forever = duration <= 0

    print(f"Headless simulator running (dt={config.SIMULATE_DT}s, Go2)")
    print(f"  DDS domain={config.DOMAIN_ID}, interface='{config.INTERFACE}'")
    if run_forever:
        print(f"  Duration: indefinite (Ctrl+C to stop)")
    else:
        print(f"  Duration: {duration}s")
    print(f"  Waiting for lowcmd from policy...")

    try:
        while run_forever or sim_time < duration:
            step_start = time.perf_counter()

            mujoco.mj_step(mj_model, mj_data)
            sim_time += config.SIMULATE_DT
            step_count += 1

            # Capture frame for video
            if renderer and (sim_time - last_frame_time) >= record_interval:
                renderer.update_scene(mj_data)
                frame = renderer.render()
                frames.append(frame.copy())
                last_frame_time = sim_time

            # Log every 2 seconds
            if step_count % int(2.0 / config.SIMULATE_DT) == 0:
                base_pos = mj_data.qpos[:3]
                base_height = base_pos[2]
                print(f"  t={sim_time:.1f}s | base_z={base_height:.3f}m | "
                      f"pos=[{base_pos[0]:.2f}, {base_pos[1]:.2f}]")

            # Maintain real-time
            elapsed = time.perf_counter() - step_start
            sleep_time = config.SIMULATE_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\nStopped at t={sim_time:.1f}s ({step_count} steps)")

    # Save video
    if renderer and frames:
        print(f"Saving {len(frames)} frames to {record_path}...")
        try:
            import imageio
            imageio.mimwrite(record_path, frames, fps=fps)
            print(f"  Saved: {record_path}")
        except ImportError:
            # Fallback: save as numpy
            npz_path = record_path.replace(".mp4", ".npz")
            np.savez_compressed(npz_path, frames=np.array(frames))
            print(f"  imageio not in deploy venv, saved frames to {npz_path}")
            print(f"  Convert with: uv run python -c \"import imageio, numpy as np; d=np.load('{npz_path}'); imageio.mimwrite('{record_path}', d['frames'], fps={fps})\"")

        if renderer:
            renderer.close()


def main():
    parser = argparse.ArgumentParser(description="Headless Go2 simulator for sim2sim")
    parser.add_argument("--duration", type=float, default=0, help="Sim duration in seconds (0=run forever)")
    parser.add_argument("--record", type=str, default=None, help="Record video to file (e.g. sim2sim.mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Video recording fps (default 30)")
    args = parser.parse_args()

    run_headless(
        duration=args.duration,
        record_path=args.record,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
