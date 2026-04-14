# Go2 Deployment Pipeline

Deploy trained JAX RL policies on the Unitree Go2 robot — in simulation (unitree_mujoco) or on real hardware (Go2 EDU).

## Prerequisites

- Python 3.12 (NOT 3.13 — CycloneDDS doesn't support it yet)
- cmake, gcc (for building CycloneDDS C library)
- `uv` package manager
- A trained Go2 checkpoint with `actor_params.npy` in `checkpoints/<run>/best/`

## Why a Separate Venv?

The deploy package uses its own Python 3.12 venv (`deploy/.venv`), separate from the training venv (`.venv`, Python 3.13). Reasons:
- CycloneDDS 0.10.x has a C extension incompatible with Python 3.13
- Deploy code is pure numpy — no JAX dependency at runtime
- Lab members deploying on robots don't need the full JAX/MJX training stack
- Training venv stays untouched — zero risk of breaking it

## Setup

### Quick setup (recommended)

```bash
bash deploy/setup_deploy_deps.sh
```

This script handles everything below automatically. Run from the project root.

### Manual setup

#### 1. Build CycloneDDS C library (one-time)

```bash
git clone --depth 1 -b releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds.git ~/.local/share/unitree/cyclonedds
cd ~/.local/share/unitree/cyclonedds
mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=Release
cmake --build . --target install -j$(nproc)
export CYCLONEDDS_HOME=~/.local/share/unitree/cyclonedds/install
```

#### 2. Create deploy venv and install deps

```bash
cd deploy/
uv venv --python 3.12
uv sync --extra robot   # installs numpy + cyclonedds
```

#### 3. Install unitree_sdk2_python

```bash
git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2_python.git ~/.local/share/unitree/unitree_sdk2_python
CYCLONEDDS_HOME=~/.local/share/unitree/cyclonedds/install uv pip install -e ~/.local/share/unitree/unitree_sdk2_python --no-deps
```

#### 4. Clone unitree_mujoco (for sim2sim)

```bash
git clone --depth 1 https://github.com/unitreerobotics/unitree_mujoco.git ~/.local/share/unitree/unitree_mujoco
```

#### Verify

```bash
cd /path/to/jax-learning
deploy/.venv/bin/python -c "from unitree_sdk2py.core.channel import ChannelFactoryInitialize; print('SDK OK')"
```

## Quick Start

**Important:** Deploy scripts use `deploy/.venv/bin/python`, NOT `uv run python` (which uses the training venv).

### Sim2sim (unitree_mujoco)

Terminal 1 — start the simulator:
```bash
cd ~/.local/share/unitree/unitree_mujoco/simulate_python
python3 unitree_mujoco.py
```

Terminal 2 — run your trained policy:
```bash
cd /path/to/jax-learning
deploy/.venv/bin/python deploy/deploy_go2.py \
    --checkpoint checkpoints/<your_run>/best \
    --sim \
    --vx 0.5
```

### Sim2real (Go2 EDU)

```bash
deploy/.venv/bin/python deploy/deploy_go2.py \
    --checkpoint checkpoints/<your_run>/best \
    --interface enp2s0 \
    --vx 0.3 \
    --stand-duration 3.0
```

**Safety:** Start with low velocity (`--vx 0.3`). Be ready to Ctrl+C. Have someone ready to catch the robot.

## End-to-End Example: Train → Sim2Sim

### 1. Train a policy (from project root, uses training venv)

**Recommended: FastSAC on Warp backend** (trains on unitree's exact MJCF — no sim2sim gap):
```bash
# FastSAC on Go2 Warp (unitree MJCF, full collision geometry)
uv run python train_fast_sac.py --env Go2WarpJoystickFlat \
    --num-envs 1024 --total-timesteps 20000000 --seed 42 --wandb

# With per-step domain randomization (recommended for robustness):
uv run python train_fast_sac.py --env Go2WarpJoystickFlat \
    --num-envs 1024 --total-timesteps 50000000 --seed 42 --wandb --reset-mode per_step
```

**Alternative: PPO** (faster wall-clock, but on-policy so less sample-efficient):
```bash
uv run python train_ppo_fast.py --env Go2WarpJoystickFlat --num-envs 1024 \
    --total-timesteps 50000000 --seed 42 --wandb --reset-mode per_step
```

```bash
# Check progress
grep "EVAL" /tmp/claude-*/tasks/*.output | tail -5
```

### 2. Record a video of the trained policy
```bash
MUJOCO_GL=egl uv run python record_video.py \
    --checkpoint checkpoints/<your_run>/best
```

### 3. Test in unitree_mujoco (sim2sim, uses deploy venv)
```bash
# Direct sim2sim (no DDS, PD at physics rate — recommended for testing)
deploy/.venv/bin/python deploy/sim2sim_direct.py \
    --checkpoint checkpoints/<your_run>/best \
    --vx 0.5 --duration 10 --record /tmp/sim2sim.mp4

# Or via DDS bridge (closer to real robot deployment)
# Terminal 1: start simulator
cd ~/.local/share/unitree/unitree_mujoco/simulate_python && python3 unitree_mujoco.py
# Terminal 2: run policy
deploy/.venv/bin/python deploy/deploy_go2.py \
    --checkpoint checkpoints/<your_run>/best --sim --vx 0.5
```

### Key concepts
- **Training env (Warp)** uses MuJoCo Warp backend with unitree's go2.xml — same MJCF as the deploy sim. Motor actuators + external PD at physics rate (Kp=20, Kd=0.5). Policy outputs position targets, env computes torque via PD. **This is the recommended training path for deployment.**
- **Training env (MJX)** uses MJX (JAX) backend with Menagerie's go2_mjx.xml — simplified collision geometry. Archived; has sim2sim gap with unitree model. Motor actuators + external PD (Kp=35, Kd=0.1).
- **Deploy env** uses standard MuJoCo (unitree_mujoco's Go2 model) with the same motor + PD setup. The deploy code loads the policy as pure numpy — no JAX needed.
- **Two venvs**: training (`.venv/`, Python 3.13, JAX) and deploy (`deploy/.venv/`, Python 3.12, CycloneDDS). They don't share dependencies.
- **PD gains must match**: Warp-trained policies use Kp=20/Kd=0.5, MJX-trained use Kp=35/Kd=0.1. The sim2sim script must use the same gains as training.

## Architecture

```
deploy/
├── go2_constants.py      # Joint remapping (SDK↔policy order), default pose, PD gains
├── policy_runner.py      # Loads JAX checkpoint, runs MLP inference with pure numpy
├── obs_builder.py        # Robot sensors → 48d observation vector
├── robot_interface.py    # CycloneDDS pub/sub (rt/lowstate, rt/lowcmd)
├── deploy_go2.py         # Main script: FSM (idle→stand→policy), 50Hz loop
└── sim_headless.py       # Headless simulator for SSH testing + video recording
```

The deploy package has **zero JAX dependency** at runtime. Policy weights are loaded as numpy arrays and inference is plain matrix multiplication.

## Joint Ordering

Our training env (Warp/unitree) uses: **FR, FL, RR, RL** (unitree actuator order)
Unitree SDK uses: **FR, FL, RR, RL**

The env handles remapping internally via `act_to_joint`. The deploy code handles any remaining remapping automatically.

## Observation Space

The policy expects 48d input. Dims 0:3 (local linear velocity) are zeroed during deployment since they're not directly available on hardware. If this degrades performance, options:
1. Retrain without linvel (recommended)
2. Use Unitree's built-in velocity estimate from `rt/sportmodestate`
3. Estimate from IMU integration (drift-prone)

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: cyclonedds` | CycloneDDS not built or CYCLONEDDS_HOME not set | Rebuild, export env var |
| `Cannot find libddsc` | cyclonedds C library not in LD_LIBRARY_PATH | `export LD_LIBRARY_PATH=$CYCLONEDDS_HOME/lib:$LD_LIBRARY_PATH` |
| No state received | Simulator not running, or wrong domain ID | Start unitree_mujoco first; use `--sim` flag |
| Robot falls immediately | Joint remapping wrong, or default pose mismatch | Check `go2_constants.py` values against your env |
| Robot vibrates | Kp too high for real hardware | Use `--kp 20` (Unitree official for Go2) |
| Policy does nothing | Obs normalization mismatch, or linvel dependency | Check if checkpoint used `--obs-norm`; try providing linvel |
