"""H1 smoke: init-only path works without crash."""
import os
import subprocess
import sys

import pytest


@pytest.mark.slow
def test_h1_smoke_init_only():
    # CheetahRun requires MJX (CUDA). Clear CUDA_VISIBLE_DEVICES so the child
    # process gets the GPU even if the pytest runner suppressed it.
    env = {**os.environ}
    env.pop("CUDA_VISIBLE_DEVICES", None)

    result = subprocess.run(
        [sys.executable, "train_tdmpc2.py",
         "--env", "CheetahRun", "--total-timesteps", "0",
         "--seed", "0", "--num-envs", "2"],
        cwd="/home/stevenman/Desktop/Work/Research/jax-learning/.worktrees/tdmpc2-impl",
        capture_output=True, timeout=300,
        env=env,
    )
    stdout = result.stdout.decode()
    stderr = result.stderr.decode()
    assert result.returncode == 0, (
        f"Non-zero exit:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )
    assert "init-only smoke, exiting cleanly" in stdout
