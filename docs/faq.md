# FAQ & Troubleshooting

## General

### Do I need a physical robot?

No. The entire training pipeline runs in simulation. A physical Go2 is only needed if you want to deploy a trained policy via the `deploy/` scripts.

### What GPU do I need?

Any NVIDIA GPU with CUDA support. An RTX 3060 or better is recommended. VRAM requirements depend on the environment and number of parallel environments:

| Setup | Approximate VRAM |
|---|---|
| DM Control, 1024 envs | ~8 GB |
| Go2 Warp, 1024 envs | ~12 GB |

Reduce `--num-envs` if you hit memory limits.

### Does this work on Mac or Windows?

No. The framework requires CUDA, which means Linux with an NVIDIA GPU. There is no Metal or CPU-only fallback for training at reasonable speed.

### How long does training take?

Rough estimates on an RTX 4090:

| Environment | Time |
|---|---|
| CartpoleBalance (SAC) | ~2 minutes |
| Go2 locomotion (PPO, Warp) | ~30 minutes |
| HumanoidRun (SAC, 100M steps) | 2-4 hours |

JIT compilation adds 1-3 minutes of overhead at the start of every run.

---

## Installation

### `uv sync` fails

Check the following:

1. **Python version** — requires Python 3.13+. Run `python --version` to verify.
2. **CUDA toolkit** — must be installed and on your `PATH`. Run `nvcc --version` to check.
3. **Network** — `uv sync` downloads packages; verify internet connectivity.

### JAX reports `CpuDevice` instead of `CudaDevice`

JAX fell back to CPU, which means CUDA is not available to it. Common causes:

- CUDA drivers are not installed or are too old. Run `nvidia-smi` to check.
- The CUDA toolkit version does not match the installed `jaxlib` wheel.
- `LD_LIBRARY_PATH` does not include the CUDA libraries.

!!! warning
    Training on CPU is orders of magnitude slower and not a supported configuration.

### "No module named mujoco"

`uv sync` likely failed silently or was not run. Re-run `uv sync` and check for errors in the output.

---

## Training

### Training shows 0.0 return for the first iterations

This is normal. JAX JIT-compiles the training step on the first call, which takes 1-3 minutes depending on the environment and network size. Returns will start appearing after compilation finishes.

### NaN in training

MuJoCo physics can produce NaN values when simulations diverge (e.g., extreme forces, interpenetration). The framework includes guards that reset NaN'd environments automatically.

If NaN values are persistent:

- Reduce `--num-envs` — fewer parallel environments means less chance of divergent states
- Check your reward function for divisions by zero or unbounded terms
- Verify PD gains and physics solver settings are consistent

### Out of GPU memory

Reduce `--num-envs`. Memory scales linearly with the number of parallel environments.

!!! tip
    For DM Control tasks, 256-512 envs is often sufficient. Reserve 1024+ envs for Go2 Warp where throughput matters more.

### Eval score is stuck

Common causes, in order of likelihood:

1. **Learning rate too high or too low** — try the preset defaults first
2. **Entropy coefficient** — if using SAC, check that target entropy is appropriate for your env scale
3. **Reward function** — verify rewards are being computed correctly (log individual reward terms)
4. **Not enough training steps** — some environments need 50M+ steps

See the [Lessons Learned](reference/lessons-learned.md) page for environment-specific tuning advice.

### `MUJOCO_GL` errors

On headless servers (no display), MuJoCo's default OpenGL backend fails. Set the environment variable before running:

```bash
export MUJOCO_GL=egl
```

Or prefix your command:

```bash
MUJOCO_GL=egl uv run python train_offpolicy.py --algo sac ...
```

---

## Recording

### Black video output

The renderer cannot create an OpenGL context. Set `MUJOCO_GL=egl` before recording:

```bash
MUJOCO_GL=egl uv run python record_video.py ...
```

### Recorded behavior does not match training

The recording script must apply the same wrappers as training (frame stacking, action delay, observation normalization). If `record_video.py` skips a wrapper that was active during training, the policy receives different inputs and produces different behavior.

!!! warning
    Always use `record_video.py` from this repo rather than a generic MuJoCo viewer. The repo script applies the correct wrapper stack automatically.
