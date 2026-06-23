# mud_eval — setup from a fresh clone

Goal: a new machine can `git clone` jax-learning, pull the Newton physics
submodule, build one dedicated venv, and run the Newton triple-mud eval on a
JAX-trained Go2 policy. **No dependency on any private/student repo** — the
physics backend is the official `newton-physics/newton` submodule, and the
student's `mpm_go2_multi` MPM example is vendored in-repo under `mpm_go2_multi/`.

## What's in git vs what you fetch/build locally

| In git (tracked) | Fetched / built locally (gitignored) |
|---|---|
| `mud_*.py`, `record_*.py`, `probe_*.py` (drivers) | `vendor/newton/` — submodule, `git submodule update` |
| `mpm_go2_multi/` (relocated student MPM example) | `.venv/` — built per step 2 |
| `requirements.txt`, this `SETUP.md` | a checkpoint to eval (step 3) |
| `.gitmodules` (pins official newton @ `8c37ad26`) | |
| the Go2 model (`jax_rl/envs/locomotion/xmls/unitree_go2/`) | |

The Go2 `go2.xml` + meshes are already tracked in the repo under
`jax_rl/envs/locomotion/xmls/unitree_go2/`; the eval references them directly, so
there's nothing to copy.

## 1. Clone with the Newton submodule

```bash
git clone --recurse-submodules <jax-learning-url> jax-learning
# (already cloned without --recurse-submodules?)
git submodule update --init --recursive
```
This fetches the **official** `newton-physics/newton` into `projects/mud_eval/vendor/newton`,
pinned to commit `8c37ad26` (newton 0.1.3-dev; the exact build the example was
written against — the only deltas from the student's copy are 9 cosmetic
viewer/debug lines, zero physics impact). The submodule is **never modified**.

## 2. Build the dedicated venv (separate from the training env)

The training env uses `mujoco 3.6` (MJX/mujoco_playground); Newton 0.1.3 needs
`mujoco 3.7.0` + the **pre-per-world-batching** `mujoco-warp 0.0.2` — hence a
*separate* venv. The version combo is load-bearing; do not bump blindly.

```bash
cd projects/mud_eval
uv venv --python 3.13 .venv

# physics stack (pinned in requirements.txt: warp 1.12 / mujoco 3.7.0 / mujoco-warp 0.0.2 / trimesh / pycollada / numpy / pyyaml)
VIRTUAL_ENV=$(pwd)/.venv uv pip install -r requirements.txt

# JAX inference stack for the policy (jax runs the actor on CPU — fine).
# The trailing mujoco/mujoco-warp/warp pins are MANDATORY so the jax solve
# can't bump mujoco off 3.7.0.
VIRTUAL_ENV=$(pwd)/.venv uv pip install \
  "jax==0.9.0" "jaxlib==0.9.0" "jax-cuda13-plugin==0.9.0" "jax-cuda13-pjrt==0.9.0" \
  flax==0.12.2 optax==0.2.6 orbax-checkpoint==0.11.32 ml_collections==1.1.0 \
  distrax==0.1.7 chex==0.1.91 imageio imageio-ffmpeg pyglet \
  "mujoco==3.7.0" "mujoco-warp==0.0.2" "warp-lang==1.12.0"

# torch — match the box's GPU (cu128 = Blackwell/RTX-50; use cu124/cu121 for older)
VIRTUAL_ENV=$(pwd)/.venv uv pip install torch --torch-backend=cu128
```

## 3. Point at a checkpoint and run

**Bring your own checkpoint** — `checkpoints/` is gitignored, so a fresh clone has
none. Train a Go2 policy (`scripts/train_ppo_fast.py` / `train_fast_sac.py`) or copy
an existing checkpoint dir into `checkpoints/`. Any JAX-trained Go2 checkpoint works
(joint-PD = 12-d action; OSC/var-impedance = pass `--controller osc`).

Run from `projects/mud_eval/` — the scripts self-add the repo root to `sys.path`
(no `PYTHONPATH` needed) and force the JAX actor onto CPU (`JAX_PLATFORMS=cpu`;
Newton/warp owns the GPU):

```bash
cd projects/mud_eval
.venv/bin/python record_traverse_maxfwd.py \
    ../../checkpoints/<ckpt_dir> --frames 120 --controller joint-pd
```
`record_traverse_maxfwd.py` drives the Go2 forward through the thin/medium/thick
mud, two-way-coupled to the MPM, and writes an mp4. `probe_forces.py` measures
per-foot mud reaction forces.

## How the import wiring works (why it's self-sufficient)

- `import newton` / `from newton.solvers import SolverImplicitMPM` / `import
  newton.examples` → the **submodule** (`vendor/newton/` on `sys.path`).
- `import mpm_go2_multi.example_mpm_go2_multi` → the **in-repo** relocated student
  code (`mud_eval/` on `sys.path`). Its cross-imports were re-rooted from
  `newton.examples.mpm.mpm_go2_multi.*` → `mpm_go2_multi.*`.
- `from jax_rl.algos.fast_sac import FastSAC` → the jax-learning repo (PYTHONPATH).
  (We import `jax_rl.algos.*` directly, never `jax_rl.training.*`, which would pull
  `mujoco_playground` and bump mujoco off 3.7.0.)
