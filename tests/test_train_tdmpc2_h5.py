"""H5 smoke: checkpoint files get created under --ckpt-dir."""
import os
import subprocess
import tempfile
import pytest


@pytest.mark.slow
def test_h5_ckpt_written():
    env = {**os.environ}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    with tempfile.TemporaryDirectory() as tmp:
        script = f"""
import sys
sys.path.insert(0, '/home/stevenman/Desktop/Work/Research/jax-learning/.worktrees/tdmpc2-impl')
import dataclasses
from jax_rl.configs.env_presets import get_tdmpc2_preset
from train_tdmpc2 import train

cfg = get_tdmpc2_preset('CheetahRun')
cfg = dataclasses.replace(
    cfg, num_envs=2, seed_steps=4, batch_size=4, utd=1,
    num_eval_envs=1, eval_every=4, total_steps=20,
    mppi_iterations=1, num_samples=8, num_elites=2, num_pi_trajs=1,
)
train(cfg, 'CheetahRun', total_timesteps=12, seed=0, ckpt_dir='{tmp}')
print('H5_OK')
"""
        result = subprocess.run(
            ["uv", "run", "python", "-c", script],
            cwd="/home/stevenman/Desktop/Work/Research/jax-learning/.worktrees/tdmpc2-impl",
            env=env,
            capture_output=True, timeout=1200,
        )
        stdout = result.stdout.decode()
        stderr = result.stderr.decode()
        assert "H5_OK" in stdout, f"H5 failed.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        # Check that checkpoint files were written
        assert os.path.exists(os.path.join(tmp, "actor_params.npz"))
        assert os.path.exists(os.path.join(tmp, "world_model_params.npz"))
        assert os.path.exists(os.path.join(tmp, "meta.json"))
        assert os.path.exists(os.path.join(tmp, "metrics.csv"))
        # Best dir should exist (first eval is always a new best)
        assert os.path.exists(os.path.join(tmp, "best"))
