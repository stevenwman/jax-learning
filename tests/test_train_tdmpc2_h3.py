"""H3 smoke: main loop runs for a few steps without crash."""
import os
import subprocess
import sys
import pytest


@pytest.mark.slow
def test_h3_smoke_main_loop():
    env = {**os.environ}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    script = """
import sys
sys.path.insert(0, '/home/stevenman/Desktop/Work/Research/jax-learning/.worktrees/tdmpc2-impl')
import dataclasses
from jax_rl.configs.env_presets import get_tdmpc2_preset
from train_tdmpc2 import train

cfg = get_tdmpc2_preset('CheetahRun')
# Tiny: 2 envs, 4 seed steps, batch 4. Main loop runs for a few outer iters.
cfg = dataclasses.replace(
    cfg, num_envs=2, seed_steps=4, batch_size=4, utd=1,
    eval_every=1000, total_steps=20,
)
# Run for total_timesteps = 4 (seed) + 8 (main, 4 iters * 2 envs) = 12
train(cfg, 'CheetahRun', total_timesteps=12, seed=0)
print('H3_OK')
"""
    result = subprocess.run(
        ["uv", "run", "python", "-c", script],
        cwd="/home/stevenman/Desktop/Work/Research/jax-learning/.worktrees/tdmpc2-impl",
        env=env,
        capture_output=True, timeout=900,
    )
    stdout = result.stdout.decode()
    stderr = result.stderr.decode()
    assert "H3_OK" in stdout, f"Main loop failed.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
