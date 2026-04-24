"""H4 smoke: eval runs and reports mppi/prior returns."""
import os
import subprocess
import pytest


@pytest.mark.slow
def test_h4_smoke_eval():
    env = {**os.environ}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    script = """
import sys
sys.path.insert(0, '/home/stevenman/Desktop/Work/Research/jax-learning/.worktrees/tdmpc2-impl')
import dataclasses
from jax_rl.configs.env_presets import get_tdmpc2_preset
from train_tdmpc2 import train

cfg = get_tdmpc2_preset('CheetahRun')
cfg = dataclasses.replace(
    cfg, num_envs=2, seed_steps=4, batch_size=4, utd=1,
    num_eval_envs=1, eval_every=4, total_steps=20,
    # Cut mppi cost for speed
    mppi_iterations=1, num_samples=8, num_elites=2, num_pi_trajs=1,
)
train(cfg, 'CheetahRun', total_timesteps=12, seed=0)
print('H4_OK')
"""
    result = subprocess.run(
        ["uv", "run", "python", "-c", script],
        cwd="/home/stevenman/Desktop/Work/Research/jax-learning/.worktrees/tdmpc2-impl",
        env=env,
        capture_output=True, timeout=1200,
    )
    stdout = result.stdout.decode()
    stderr = result.stderr.decode()
    assert "H4_OK" in stdout, f"H4 failed.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    # Must have at least one EVAL line with the expected fields
    assert "EVAL" in stdout, f"No EVAL output in stdout:\n{stdout}"
    assert "mppi=" in stdout and "prior=" in stdout and "gap=" in stdout
