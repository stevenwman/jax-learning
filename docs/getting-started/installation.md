# Installation

## Prerequisites

- **Python 3.13+**
- **NVIDIA GPU with CUDA** — JAX compiles and runs all training on GPU. CPU-only is not supported for training. RTX 3060 (12GB) or higher recommended. 1024 parallel envs requires ~8GB VRAM (DM Control) or ~12GB (Go2 Warp).
- **Linux** — CUDA JAX does not support macOS or Windows.
- **[uv](https://docs.astral.sh/uv/)** — Fast Python package manager. Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh` if you don't have it.

!!! tip "No GPU? Try it in the cloud"
    If you don't have an NVIDIA GPU but want to try the Quickstart, [Google Colab](https://colab.research.google.com/) offers a free T4 GPU tier that can run the CartpoleBalance example. You'll need to `!pip install -e .` the repo inside the notebook instead of using `uv sync`. Go2 training is too memory-intensive for the free tier; it needs a paid Colab Pro GPU or a local RTX 3060+.

## Install

Clone the repository and install dependencies:

```bash
git clone https://github.com/stevenwman/jax-learning.git
cd jax-learning
uv sync
```

This installs the core dependencies: JAX with CUDA support, MuJoCo, and all RL training packages.

### Optional dependency groups

```bash
# Development (tests + Weights & Biases logging)
uv sync --group dev

# Documentation (mkdocs + plugins)
uv sync --group docs
```

!!! note "JAX CUDA support"
    `jax[cuda13]` is already in the project dependencies and installs automatically with `uv sync`. You do not need to install JAX separately.

## Verify your installation

### Check GPU access

```bash
uv run python -c "import jax; print(jax.devices())"
```

You should see output like:

```
[CudaDevice(id=0)]
```

!!! warning "If you see `CpuDevice` instead"
    JAX fell back to CPU. Check that your NVIDIA drivers and CUDA toolkit are installed correctly. Run `nvidia-smi` to verify your GPU is visible to the system.

### Check MuJoCo

```bash
uv run python -c "import mujoco; print(mujoco.__version__)"
```

This should print a version string (e.g., `3.x.x`). If it errors, your MuJoCo installation may be incomplete.

## Next steps

Once everything is installed, head to the [Quickstart](quickstart.md) to train your first policy.
