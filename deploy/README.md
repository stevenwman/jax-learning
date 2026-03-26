# Go2 Deployment Pipeline

Deploy trained JAX RL policies on the Unitree Go2 robot — in simulation (unitree_mujoco) or on real hardware (Go2 EDU).

## Prerequisites

- Python 3.10+ (managed by `uv` in this project)
- cmake, gcc (for building CycloneDDS)
- A trained Go2 checkpoint with `actor_params.npy` in `checkpoints/<run>/best/`

## Setup

### 1. Install CycloneDDS C library (one-time)

The Unitree SDK requires CycloneDDS 0.10.x as a C library. Must be built from source.

```bash
# Clone and build
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x /tmp/cyclonedds
cd /tmp/cyclonedds
mkdir build install
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install

# Set env var (add to ~/.bashrc for persistence)
export CYCLONEDDS_HOME=/tmp/cyclonedds/install
```

Verify: `ls $CYCLONEDDS_HOME/lib/libddsc*` should show shared libraries.

### 2. Install unitree_sdk2_python

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git /tmp/unitree_sdk2_python
cd /tmp/unitree_sdk2_python
uv pip install -e .
```

Verify:
```bash
uv run python -c "from unitree_sdk2py.core.channel import ChannelFactoryInitialize; print('SDK OK')"
```

### 3. Install unitree_mujoco (for sim2sim)

```bash
git clone https://github.com/unitreerobotics/unitree_mujoco.git /tmp/unitree_mujoco
```

No pip install needed — it's a standalone simulator you run directly.

Verify:
```bash
cd /tmp/unitree_mujoco/simulate_python
python3 unitree_mujoco.py  # Should open MuJoCo viewer with Go2
```

## Quick Start

### Sim2sim (unitree_mujoco)

Terminal 1 — start the simulator:
```bash
cd /tmp/unitree_mujoco/simulate_python
python3 unitree_mujoco.py
```

Terminal 2 — run your trained policy:
```bash
cd /path/to/jax-learning
uv run python deploy/deploy_go2.py \
    --checkpoint checkpoints/<your_run>/best \
    --sim \
    --vx 0.5
```

### Sim2real (Go2 EDU)

```bash
uv run python deploy/deploy_go2.py \
    --checkpoint checkpoints/<your_run>/best \
    --interface enp2s0 \
    --vx 0.3 \
    --stand-duration 3.0
```

**Safety:** Start with low velocity (`--vx 0.3`). Be ready to Ctrl+C. Have someone ready to catch the robot.

## Architecture

```
deploy/
├── go2_constants.py      # Joint remapping (SDK↔policy order), default pose, PD gains
├── policy_runner.py      # Loads JAX checkpoint, runs MLP inference with pure numpy
├── obs_builder.py        # Robot sensors → 48d observation vector
├── robot_interface.py    # CycloneDDS pub/sub (rt/lowstate, rt/lowcmd)
└── deploy_go2.py         # Main script: FSM (idle→stand→policy), 50Hz loop
```

The deploy package has **zero JAX dependency** at runtime. Policy weights are loaded as numpy arrays and inference is plain matrix multiplication.

## Joint Ordering

Our training env (MJX/Menagerie) uses: **FL, FR, RL, RR**
Unitree SDK uses: **FR, FL, RR, RL**

The deploy code handles this remapping automatically. You never need to think about it.

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
