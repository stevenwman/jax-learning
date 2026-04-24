"""H2 smoke: warmup runs without crash with seed_steps=8 and num_envs=2.

seed_steps=8, num_envs=2 → 16 transitions, horizon=3, need H+1=4 per window,
batch_size=4 windows. 16 transitions gives a comfortable margin for rejection
sampling (4x oversample = 16 candidates, all from 16-4=12 valid starts).
"""
import os
import subprocess
import sys

import pytest


@pytest.mark.slow
def test_h2_smoke_warmup():
    env = {**os.environ}
    env.pop("CUDA_VISIBLE_DEVICES", None)

    # total_timesteps == seed_steps → main loop range(8, 8, 2) is empty, skipped.
    # Validates warmup + gradient burst in isolation without running main loop.
    script = """
import sys
sys.path.insert(0, '/home/stevenman/Desktop/Work/Research/jax-learning/.worktrees/tdmpc2-impl')
import dataclasses
from jax_rl.configs.env_presets import get_tdmpc2_preset
from train_tdmpc2 import train
cfg = get_tdmpc2_preset('CheetahRun')
cfg = dataclasses.replace(cfg, num_envs=2, seed_steps=8, batch_size=4)
train(cfg, 'CheetahRun', total_timesteps=8, seed=0)
print('H2_OK')
"""
    result = subprocess.run(
        ["uv", "run", "python", "-c", script],
        cwd="/home/stevenman/Desktop/Work/Research/jax-learning/.worktrees/tdmpc2-impl",
        env=env,
        capture_output=True,
        timeout=600,
    )
    stdout = result.stdout.decode()
    stderr = result.stderr.decode()
    assert "H2_OK" in stdout, (
        f"Warmup failed.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    )
    assert "warmup complete" in stdout, (
        f"'warmup complete' not in stdout.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    )
