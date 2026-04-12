# Sim-to-Real

*Advanced — requires a trained checkpoint and (for real robot deployment) a Unitree Go2 EDU.*

This tutorial covers the full pipeline from training a locomotion policy in simulation to running it on a real Unitree Go2 robot. Training on the exact deployment MJCF model eliminates most of the sim-to-real gap.

## The Pipeline

```
Train (GPU, Warp)  -->  Sim2Sim (CPU, MuJoCo)  -->  Real Robot (Go2 EDU)
   go2.xml                  go2.xml                    DDS @ 50Hz
   1024 envs                1 env                      numpy inference
   JAX + Warp               standard MuJoCo            no JAX needed
```

## Step 1: Why Warp Matters

Warp (via `Go2WarpJoystickFlat`) is the sole active Go2 training backend. It uses Unitree's exact MJCF (`go2.xml`) so the policy sees the same collision geometry, contact dynamics, and PD behavior it will encounter on the real robot, giving direct sim-to-real transfer with no extra sim2sim step.

The earlier MJX backend (`Go2JoystickFlat`, Menagerie `go2_mjx.xml`) has been archived. For historical reference, here is why it was insufficient:

| Feature | MJX (archived) | Warp (active) |
|---|---|---|
| **MJCF source** | Menagerie `go2_mjx.xml` | Unitree `go2.xml` |
| **Collision geometry** | Convex hulls (simplified) | Cylinders + boxes (exact) |
| **PD gains** | Kp=35, Kd=0.1 | Kp=20, Kd=0.5 |
| **Sim-to-real gap** | Required sim2sim transfer; often did not transfer cleanly | Direct -- same model as deploy |

MJX-trained policies required an additional sim2sim transfer step and frequently failed to transfer cleanly because the Menagerie model has different solver settings and simplified geometry. Warp eliminates this problem entirely.

## Step 2: Train with Domain Randomization

Domain randomization is critical for real-world robustness. It randomizes physical parameters during training so the policy learns to handle uncertainty:

- **Friction:** floor and foot friction coefficients
- **Mass:** per-link body mass perturbations
- **Damping / armature / friction loss:** joint-level mechanical properties
- **Motor strength:** per-actuator gain heterogeneity
- **PD gain scaling:** Kp x0.8-1.3, Kd x0.5-1.5

```bash
uv run python train_fast_sac.py \
    --env Go2WarpJoystickFlat \
    --num-envs 1024 \
    --total-timesteps 20000000 \
    --reset-mode per_step
```

!!! warning
    Without domain randomization, policies tend to exploit specific physics parameters and fail on real hardware. Always use `--reset-mode per_step` for policies intended for deployment — it enables `DomainRandWrapper`, which applies the env's declared DR specs (friction, mass, motor strength, etc.) per episode.

## Step 3: Locate the Checkpoint

After training, the best checkpoint is saved automatically:

```
checkpoints/<run-dir>/best/
├── meta.json           # training config, algo type, obs/action dims
├── actor_params.npy    # actor params + normalization stats (pickled dict)
├── metrics.csv         # training curve
└── orbax/              # full training state for resume
```

Only `meta.json` and `actor_params.npy` are needed for deployment. The `.npy` file is a pickled dict containing `actor_params`, `norm_mean`, `norm_mean_of_squares`, `norm_count`, and optionally `actor_batch_stats` (FlashSAC). The deploy code loads these as numpy arrays and runs inference with pure matrix multiplication -- no JAX required.

## Step 4: Sim2Sim Validation

Before going to hardware, validate the policy in a CPU-based MuJoCo simulation using the same Unitree MJCF:

```bash
MUJOCO_GL=egl uv run python deploy/sim2sim_direct.py \
    --checkpoint checkpoints/<run-dir> \
    --vx 0.5 \
    --record /tmp/sim2sim.mp4
```

This runs a single environment on CPU MuJoCo with the same `go2.xml` and PD gains as training. The `--record` flag saves a video so you can visually verify the gait before deploying.

!!! tip
    If the policy works in Warp training but fails in sim2sim, check that PD gains match. Warp-trained policies use Kp=20/Kd=0.5.

## Step 5: Deploy Environment Setup

The deploy code runs in a **separate Python 3.12 venv** because CycloneDDS (used for communication with the Go2) requires Python <3.13. The training venv (Python 3.13) is untouched.

### Quick setup (recommended)

```bash
bash deploy/setup_deploy_deps.sh
```

This script builds the CycloneDDS C library, creates the deploy venv, installs dependencies, and clones the Unitree SDK. Run it once from the project root.

### Manual setup

See `deploy/README.md` for step-by-step instructions covering:

1. Building the CycloneDDS C library
2. Creating the deploy venv with `uv venv --python 3.12`
3. Installing `unitree_sdk2_python`
4. Cloning `unitree_mujoco` for sim2sim testing

### Verify the setup

```bash
deploy/.venv/bin/python -c \
    "from unitree_sdk2py.core.channel import ChannelFactoryInitialize; print('SDK OK')"
```

## Step 6: Deploy to the Real Robot

```bash
deploy/.venv/bin/python deploy/deploy_go2.py \
    --checkpoint checkpoints/<run-dir>/best \
    --interface enp2s0 \
    --vx 0.3 \
    --stand-duration 3.0
```

The deploy script runs a 50Hz finite state machine (FSM):

1. **Idle** -- wait for robot connection
2. **Stand** -- move joints to default pose over `--stand-duration` seconds
3. **Policy** -- run the neural network at 50Hz, sending joint position targets via DDS

!!! warning "Safety"
    - Start with low velocity (`--vx 0.3`)
    - Be ready to Ctrl+C at any time
    - Have someone ready to catch the robot
    - Test in sim2sim first

## ONNX Export

ONNX export is available via `jax_rl/utils/export.py`. It builds a standalone ONNX graph of the actor MLP (512-256-128) with no JAX dependency at runtime, targeting Jetson deployment with `onnxruntime`. Numpy inference at 50Hz is also sufficient for the Go2's control loop -- the actor network evaluates in <1ms on a laptop CPU.

## Key Insights

1. **Train on the exact deploy model.** Using Unitree's `go2.xml` via Warp eliminates the sim-to-sim gap entirely.
2. **Domain randomization is non-negotiable.** Friction, mass, COM, and motor strength randomization force the policy to be robust to the physical uncertainty of the real world.

3. **PD gains must match.** The same Kp/Kd values used during training must be used in deployment. Warp-trained policies use Kp=20, Kd=0.5 (Unitree's official gains).

4. **Two venvs, zero conflicts.** Training (Python 3.13, JAX) and deploy (Python 3.12, CycloneDDS) are completely isolated. Neither can break the other.

## Next Steps

- [**Lessons Learned**](../reference/lessons-learned.md) -- hard-won insights from training and deployment
- [**Glossary**](../glossary.md) -- definitions for domain randomization, PD gains, sim-to-real, and other terms
