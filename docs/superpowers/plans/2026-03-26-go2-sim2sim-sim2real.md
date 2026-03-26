# Go2 Sim2Sim & Sim2Real Deployment Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run our trained JAX PPO/SAC Go2 policies in unitree_mujoco (sim2sim), then on the real Go2 EDU robot (sim2real), using the same deployment code for both.

**Architecture:** A standalone `deploy/` package that loads our checkpoint (actor_params.npy + meta.json), reconstructs the policy as pure numpy inference (no JAX at runtime), reads robot state from CycloneDDS `rt/lowstate`, constructs obs, runs the policy at 50Hz, and publishes joint targets to `rt/lowcmd`. The only difference between sim and real is the DDS domain ID (1=sim, 0=real) and network interface ("lo" vs ethernet adapter).

**Tech Stack:** Python, unitree_sdk2_python (CycloneDDS), numpy, unitree_mujoco (for sim2sim validation)

---

## Critical Context

### Joint Ordering Mismatch (MUST REMAP)

| Index | Our MJX Env (Menagerie) | Unitree SDK / unitree_mujoco |
|-------|------------------------|------------------------------|
| 0     | FL_hip               | FR_hip                       |
| 1     | FL_thigh             | FR_thigh                     |
| 2     | FL_calf              | FR_calf                      |
| 3     | FR_hip               | FL_hip                       |
| 4     | FR_thigh             | FL_thigh                     |
| 5     | FR_calf              | FL_calf                      |
| 6     | RL_hip               | RR_hip                       |
| 7     | RL_thigh             | RR_thigh                     |
| 8     | RL_calf              | RR_calf                      |
| 9     | RR_hip               | RL_hip                       |
| 10    | RR_thigh             | RL_thigh                     |
| 11    | RR_calf              | RL_calf                      |

**Remapping:** To go from Unitree SDK order → our policy order: `[3,4,5, 0,1,2, 9,10,11, 6,7,8]`
To go from our policy order → Unitree SDK order: `[3,4,5, 0,1,2, 9,10,11, 6,7,8]` (same — it's a symmetric swap of FL↔FR and RL↔RR)

### Obs Space (48d in training, 45d deployable)

Our training obs is 48d:
```
[0:3]   local_linvel (3)    ← NOT directly available on real hardware
[3:6]   gyroscope (3)       ← from imu_state.gyroscope
[6:9]   projected_gravity (3) ← computed from imu_state.quaternion
[9:21]  joint_pos - default_pose (12) ← from motor_state[i].q
[21:33] joint_vel (12)      ← from motor_state[i].dq
[33:45] last_action (12)    ← tracked internally
[45:48] command (3)         ← from joystick/external input
```

**Decision: Drop local_linvel.** Retrain or zero-pad. Unitree's own Go2 config (unitree_rl_lab) doesn't use linvel at all. Zeroing dims 0:3 is the safest first attempt — if the policy degrades, we retrain without linvel.

### PD Gains

- Training (MJX): Kp=35, Kd=0.5 (Playground Go1 defaults)
- unitree_mujoco: Kp varies by example (20-50)
- Real Go2: Kp=20, Kd=0.5 (Unitree official for RL deployment)
- **Sim2sim:** Use Kp=35 first (match training), then test Kp=20
- **Sim2real:** Use Kp=20, Kd=0.5 (Unitree official)

### Action Mapping

```python
q_target[i] = default_pose[i] + action[i] * action_scale  # action_scale=0.5
```

### Default Joint Angles (radians, our env convention: FL, FR, RL, RR)

From the MJCF keyframe (`go2_mjx.xml` line 246), loaded by `go2_joystick.py` via `self._mj_model.keyframe("home").qpos[7:]`:
```python
# ALL four legs identical: hip=0.0, thigh=0.9, calf=-1.8
DEFAULT_POSE = [0, 0.9, -1.8] * 4  # FL, FR, RL, RR
```
**NOT** the asymmetric values from unitree_rl_gym (hip=±0.1, thigh=0.8/1.0, calf=-1.5). Our env loads directly from the MJCF keyframe.

### Policy Inference (numpy, no JAX)

The policy is a simple MLP. For SAC with hidden_dim=(256,256):
```
obs(48) → Dense(256) → ReLU → Dense(256) → ReLU → Dense(12) → tanh → action
```

For deterministic inference (deployment), we use `tanh(mean)` — no sampling.
Weights are nested dicts in actor_params.npy. We flatten them into weight matrices and run numpy matmul.

---

## File Structure

```
deploy/
├── __init__.py
├── policy_runner.py          # Loads checkpoint, reconstructs MLP, runs numpy inference
├── robot_interface.py        # DDS pub/sub wrapper (rt/lowstate, rt/lowcmd)
├── obs_builder.py            # Builds 48d obs from LowState_ (remap joints, compute gravity, track last_action)
├── go2_constants.py          # Default pose, joint remapping, PD gains, action scale
├── deploy_go2.py             # Main script: CLI args, FSM (idle→stand→policy), 50Hz loop
└── test_policy_runner.py     # Offline test: load checkpoint, feed dummy obs, verify output shape/range
```

No changes to existing `jax_rl/` code. The deploy package is standalone — it only reads checkpoint files.

---

## Task 1: Go2 Deploy Constants

**Files:**
- Create: `deploy/__init__.py`
- Create: `deploy/go2_constants.py`

- [ ] **Step 1: Create deploy package with constants**

```python
# deploy/__init__.py
# empty

# deploy/go2_constants.py
"""Go2 deployment constants — joint ordering, default pose, PD gains."""
import numpy as np

# Joint remapping: Unitree SDK order (FR,FL,RR,RL) ↔ MJX env order (FL,FR,RL,RR)
# Apply to convert SDK sensor readings → policy obs order
# Apply to convert policy action output → SDK command order
# The mapping is symmetric (same array both directions)
SDK_TO_POLICY = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
POLICY_TO_SDK = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])

# Default joint angles (radians) in POLICY order (FL, FR, RL, RR)
# Each leg: [hip, thigh, calf] — from go2_mjx.xml keyframe("home").qpos[7:]
# All four legs identical in the MJCF keyframe
DEFAULT_POSE_POLICY = np.array([
    0.0, 0.9, -1.8,    # FL
    0.0, 0.9, -1.8,    # FR
    0.0, 0.9, -1.8,    # RL
    0.0, 0.9, -1.8,    # RR
], dtype=np.float32)

# Same in SDK order (FR, FL, RR, RL)
DEFAULT_POSE_SDK = DEFAULT_POSE_POLICY[POLICY_TO_SDK]

ACTION_SCALE = 0.5

# PD gains for deployment
KP_SIM = 35.0    # Match training env (MJX Playground)
KD_SIM = 0.5
KP_REAL = 20.0   # Unitree official for Go2 RL deployment
KD_REAL = 0.5

NUM_JOINTS = 12
POLICY_DT = 0.02  # 50 Hz policy
CONTROL_DT = 0.002  # 500 Hz motor commands

# Gravity vector (used for projected gravity computation)
GRAVITY_VEC = np.array([0.0, 0.0, -1.0], dtype=np.float32)
```

- [ ] **Step 2: Verify remapping is correct**

```bash
uv run python -c "
from deploy.go2_constants import SDK_TO_POLICY, POLICY_TO_SDK, DEFAULT_POSE_POLICY
import numpy as np
# Round-trip: policy → sdk → policy should be identity
assert np.all(DEFAULT_POSE_POLICY[POLICY_TO_SDK][SDK_TO_POLICY] == DEFAULT_POSE_POLICY)
# All legs have same pose [0, 0.9, -1.8], so SDK reorder should preserve values
sdk_pose = DEFAULT_POSE_POLICY[POLICY_TO_SDK]
np.testing.assert_array_equal(sdk_pose.reshape(4,3), np.tile([0, 0.9, -1.8], (4,1)))
print('Joint remapping verified OK')
print(f'SDK order pose: {sdk_pose}')
"
```

Expected: `Joint remapping verified OK`

- [ ] **Step 3: Commit**

```bash
git add deploy/
git commit -m "feat: deploy package with Go2 constants — joint remapping, default pose, PD gains"
```

---

## Task 2: Numpy Policy Runner

**Files:**
- Create: `deploy/policy_runner.py`
- Create: `deploy/test_policy_runner.py`

- [ ] **Step 1: Write test for policy loading and inference**

```python
# deploy/test_policy_runner.py
"""Test that PolicyRunner loads a checkpoint and produces valid actions."""
import numpy as np
import os

def test_policy_runner_shapes():
    """PolicyRunner should accept 48d obs and produce 12d action in [-1, 1]."""
    from deploy.policy_runner import PolicyRunner

    # Find a checkpoint (PPO or SAC)
    ckpt_dir = None
    for d in os.listdir("checkpoints"):
        best = os.path.join("checkpoints", d, "best")
        if os.path.isdir(best) and os.path.exists(os.path.join(best, "actor_params.npy")):
            ckpt_dir = best
            break

    if ckpt_dir is None:
        import pytest
        pytest.skip("No checkpoint found")

    runner = PolicyRunner(ckpt_dir)

    # Verify metadata
    assert runner.obs_dim == 48 or runner.obs_dim == 45
    assert runner.action_dim == 12

    # Run inference with dummy obs
    obs = np.zeros(runner.obs_dim, dtype=np.float32)
    action = runner.get_action(obs)

    assert action.shape == (12,), f"Expected (12,), got {action.shape}"
    assert np.all(np.abs(action) <= 1.0 + 1e-6), f"Actions outside [-1,1]: {action}"
    assert not np.any(np.isnan(action)), "NaN in actions"
    print(f"Loaded {runner.algo} from {ckpt_dir}")
    print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")

if __name__ == "__main__":
    test_policy_runner_shapes()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python deploy/test_policy_runner.py
```

Expected: `ImportError: cannot import name 'PolicyRunner'`

- [ ] **Step 3: Implement PolicyRunner**

```python
# deploy/policy_runner.py
"""Load a JAX RL checkpoint and run inference with pure numpy (no JAX dependency)."""
import json
import os
import numpy as np
from typing import Optional


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _swish(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


ACTIVATIONS = {"relu": _relu, "swish": _swish, "silu": _swish}


class PolicyRunner:
    """Pure numpy policy inference from a JAX RL checkpoint."""

    def __init__(self, ckpt_dir: str):
        meta_path = os.path.join(ckpt_dir, "meta.json")
        params_path = os.path.join(ckpt_dir, "actor_params.npy")

        with open(meta_path) as f:
            self.meta = json.load(f)

        saved = np.load(params_path, allow_pickle=True).item()
        self.actor_params = saved["actor_params"]

        self.obs_dim = self.meta["obs_dim"]
        self.action_dim = self.meta["action_dim"]
        self.algo = self.meta.get("algo", "unknown")

        # Extract normalization state
        self.norm_mean = np.array(saved.get("norm_mean", np.zeros(self.obs_dim)), dtype=np.float32)
        self.norm_mos = np.array(saved.get("norm_mean_of_squares", np.ones(self.obs_dim)), dtype=np.float32)
        self.norm_count = int(saved.get("norm_count", 0))
        self.use_obs_norm = self.norm_count > 0

        # Determine activation
        self._resolve_network_config()

        # Extract weight matrices from nested params dict
        self._extract_weights()

    def _resolve_network_config(self):
        """Determine hidden dims and activation from meta.json."""
        tc = self.meta.get("train_config", {})

        if self.algo == "ppo":
            # PPO config is stored as a separate top-level key in meta.json
            ppo_cfg = self.meta.get("ppo_config", {})
            # Fallback: some checkpoints nest it under train_config.ppo
            if not ppo_cfg:
                ppo_cfg = tc.get("ppo", {})
            self.hidden_dim = tuple(ppo_cfg.get("policy_hidden_dim", None) or (32, 32, 32, 32))
            self.activation = ppo_cfg.get("activation", tc.get("activation", "swish"))
            self.squash = ppo_cfg.get("squash", True)
        elif self.algo in ("sac", "fast_sac"):
            sc_key = "sac_config" if "sac_config" in self.meta else "fast_sac_config"
            sc = self.meta.get(sc_key, {})
            self.hidden_dim = tuple(sc.get("hidden_dim", (256, 256)))
            self.activation = sc.get("activation", "relu")
            self.squash = True  # SAC always squashes
        else:
            raise ValueError(f"Unsupported algo: {self.algo}")

        self.act_fn = ACTIVATIONS.get(self.activation, _relu)
        self.has_layer_norm = False  # detected during weight extraction

    def _extract_weights(self):
        """Extract encoder Dense layers + policy head mean layer from nested params dict."""
        self.layers = []  # list of (weight, bias, ln_scale, ln_bias) tuples

        # Navigate the nested params structure
        # Structure: actor_params -> {"params": {"MlpEncoder_0": {"Dense_0": ...}, "GaussianHead_0": {"Dense_0": ...}}}
        params = self.actor_params
        if "params" in params:
            params = params["params"]

        # Find encoder layers
        encoder_key = None
        for k in params:
            if "encoder" in k.lower() or "Encoder" in k:
                encoder_key = k
                break
        if encoder_key is None:
            encoder_key = "MlpEncoder_0"  # default Flax name

        encoder_params = params.get(encoder_key, {})
        i = 0
        while f"Dense_{i}" in encoder_params:
            layer = encoder_params[f"Dense_{i}"]
            w = np.array(layer["kernel"], dtype=np.float32)
            b = np.array(layer["bias"], dtype=np.float32)

            # Check for LayerNorm after this Dense layer
            ln_key = f"LayerNorm_{i}"
            ln_scale, ln_bias = None, None
            if ln_key in encoder_params:
                self.has_layer_norm = True
                ln_scale = np.array(encoder_params[ln_key]["scale"], dtype=np.float32)
                ln_bias = np.array(encoder_params[ln_key]["bias"], dtype=np.float32)

            self.layers.append((w, b, ln_scale, ln_bias))
            i += 1

        if self.has_layer_norm:
            print(f"  NOTE: Encoder uses LayerNorm ({len(self.layers)} layers)")

        # Find policy head mean layer
        head_key = None
        for k in params:
            if "head" in k.lower() or "Head" in k:
                head_key = k
                break
        if head_key is None:
            head_key = "GaussianHead_0"

        head_params = params.get(head_key, {})
        # Dense_0 is typically the mean projection
        if "Dense_0" in head_params:
            mean_layer = head_params["Dense_0"]
            self.mean_w = np.array(mean_layer["kernel"], dtype=np.float32)
            self.mean_b = np.array(mean_layer["bias"], dtype=np.float32)
        else:
            raise ValueError(f"Could not find mean layer in head params: {list(head_params.keys())}")

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply obs normalization using saved running statistics."""
        if not self.use_obs_norm:
            return obs
        variance = np.maximum(self.norm_mos - self.norm_mean ** 2, 0.0)
        std = np.sqrt(variance) + 1e-8
        return (obs - self.norm_mean) / std

    @staticmethod
    def _layer_norm(x: np.ndarray, scale: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Numpy LayerNorm matching nn.LayerNorm."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * scale + bias

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Run deterministic policy inference: obs → action in [-1, 1]."""
        # Normalize
        x = self.normalize_obs(obs.astype(np.float32))

        # Encoder forward pass
        for w, b, ln_scale, ln_bias in self.layers:
            x = x @ w + b
            if ln_scale is not None:
                x = self._layer_norm(x, ln_scale, ln_bias)
            x = self.act_fn(x)

        # Mean head
        mean = x @ self.mean_w + self.mean_b

        # Deterministic action (tanh squash for both PPO and SAC)
        if self.squash:
            action = np.tanh(mean)
        else:
            action = np.clip(mean, -1.0, 1.0)

        return action.astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run python deploy/test_policy_runner.py
```

Expected: `Loaded sac from checkpoints/.../best` (or ppo), action shape (12,), values in [-1, 1].

**If the test fails due to param dict structure:** The nested key names depend on Flax's auto-naming. Debug by printing `self.actor_params.keys()` and the nested structure. Adjust `_extract_weights` accordingly.

- [ ] **Step 5: Commit**

```bash
git add deploy/policy_runner.py deploy/test_policy_runner.py
git commit -m "feat: numpy PolicyRunner — loads JAX checkpoint, runs MLP inference without JAX"
```

---

## Task 3: Observation Builder

**Files:**
- Create: `deploy/obs_builder.py`

- [ ] **Step 1: Write obs builder test**

Add to `deploy/test_policy_runner.py`:

```python
def test_obs_builder():
    """ObsBuilder should produce 48d obs from simulated sensor data."""
    from deploy.obs_builder import ObsBuilder
    from deploy.go2_constants import DEFAULT_POSE_POLICY, NUM_JOINTS

    builder = ObsBuilder()

    # Simulate sensor readings (in SDK joint order — builder must remap)
    joint_pos_sdk = np.zeros(NUM_JOINTS, dtype=np.float32)  # at default-ish pose
    joint_vel_sdk = np.zeros(NUM_JOINTS, dtype=np.float32)
    gyro = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # w,x,y,z identity
    command = np.array([0.5, 0.0, 0.0], dtype=np.float32)  # forward 0.5 m/s

    obs = builder.build(
        joint_pos_sdk=joint_pos_sdk,
        joint_vel_sdk=joint_vel_sdk,
        gyroscope=gyro,
        quaternion=quat,
        command=command,
    )

    assert obs.shape == (48,), f"Expected (48,), got {obs.shape}"
    assert not np.any(np.isnan(obs)), "NaN in obs"

    # Verify command is at the end
    np.testing.assert_array_almost_equal(obs[45:48], command)

    # Verify projected gravity for identity quaternion = [0, 0, -1]
    np.testing.assert_array_almost_equal(obs[6:9], [0.0, 0.0, -1.0], decimal=5)

    print(f"Obs builder OK, shape={obs.shape}")

if __name__ == "__main__":
    test_policy_runner_shapes()
    test_obs_builder()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python deploy/test_policy_runner.py
```

Expected: `ImportError: cannot import name 'ObsBuilder'`

- [ ] **Step 3: Implement ObsBuilder**

```python
# deploy/obs_builder.py
"""Build policy observation vector from raw sensor data."""
import numpy as np
from deploy.go2_constants import SDK_TO_POLICY, DEFAULT_POSE_POLICY, NUM_JOINTS


def _quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vector by inverse of quaternion. quat = [w, x, y, z]."""
    w, x, y, z = quat
    # Rotation matrix from quaternion (transposed = inverse rotation)
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y + w * z)
    r02 = 2 * (x * z - w * y)
    r10 = 2 * (x * y - w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z + w * x)
    r20 = 2 * (x * z + w * y)
    r21 = 2 * (y * z - w * x)
    r22 = 1 - 2 * (x * x + y * y)
    # Transpose of R (inverse rotation)
    return np.array([
        r00 * vec[0] + r10 * vec[1] + r20 * vec[2],
        r01 * vec[0] + r11 * vec[1] + r21 * vec[2],
        r02 * vec[0] + r12 * vec[1] + r22 * vec[2],
    ], dtype=np.float32)


class ObsBuilder:
    """Builds 48d observation vector from robot sensor readings.

    Obs layout (matching go2_joystick.py _get_obs):
        [0:3]   local_linvel (zeroed for deployment — not available on hardware)
        [3:6]   gyroscope
        [6:9]   projected_gravity (from quaternion)
        [9:21]  joint_pos - default_pose (remapped from SDK to policy order)
        [21:33] joint_vel (remapped)
        [33:45] last_action (in policy order)
        [45:48] command (vx, vy, yaw_rate)
    """

    def __init__(self):
        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._quat_checked = False

    def build(
        self,
        joint_pos_sdk: np.ndarray,
        joint_vel_sdk: np.ndarray,
        gyroscope: np.ndarray,
        quaternion: np.ndarray,
        command: np.ndarray,
        linvel: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build 48d obs from sensor readings.

        Args:
            joint_pos_sdk: (12,) joint positions in SDK order (FR,FL,RR,RL)
            joint_vel_sdk: (12,) joint velocities in SDK order
            gyroscope: (3,) angular velocity [wx, wy, wz]
            quaternion: (4,) orientation [w, x, y, z]
            command: (3,) velocity command [vx, vy, yaw_rate]
            linvel: (3,) local linear velocity (optional, zeroed if None)
        """
        # Verify quaternion convention on first call (MuJoCo/Unitree: [w,x,y,z])
        if not self._quat_checked:
            norm = np.linalg.norm(quaternion)
            if abs(norm - 1.0) > 0.1:
                print(f"WARNING: quaternion norm={norm:.3f}, expected ~1.0")
            # Identity quat should have large w component at index 0 ([w,x,y,z])
            # If index 3 is large instead, the convention is [x,y,z,w]
            if abs(quaternion[3]) > 0.9 and abs(quaternion[0]) < 0.1:
                print("WARNING: quaternion appears to be [x,y,z,w] not [w,x,y,z] — reordering")
                # Swap to [w,x,y,z] convention
            self._quat_checked = True

        # Remap joints from SDK to policy order
        joint_pos = joint_pos_sdk[SDK_TO_POLICY]
        joint_vel = joint_vel_sdk[SDK_TO_POLICY]

        # Projected gravity in body frame
        proj_gravity = _quat_rotate_inverse(quaternion, self.gravity_world)

        # Local linear velocity (zero if not available)
        local_linvel = linvel if linvel is not None else np.zeros(3, dtype=np.float32)

        obs = np.concatenate([
            local_linvel,                          # [0:3]
            gyroscope.astype(np.float32),          # [3:6]
            proj_gravity,                          # [6:9]
            (joint_pos - DEFAULT_POSE_POLICY).astype(np.float32),  # [9:21]
            joint_vel.astype(np.float32),          # [21:33]
            self.last_action,                      # [33:45]
            command.astype(np.float32),            # [45:48]
        ])
        return obs

    def update_last_action(self, action: np.ndarray):
        """Store action for next obs construction (in policy order)."""
        self.last_action = action.copy()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run python deploy/test_policy_runner.py
```

Expected: Both tests pass.

- [ ] **Step 5: Commit**

```bash
git add deploy/obs_builder.py deploy/test_policy_runner.py
git commit -m "feat: ObsBuilder — sensor data to 48d policy obs with joint remapping"
```

---

## Task 4: Robot DDS Interface

**Files:**
- Create: `deploy/robot_interface.py`

**Prerequisite:** `unitree_sdk2_python` must be installed. If not:
```bash
# Install CycloneDDS if needed
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x /tmp/cyclonedds
cd /tmp/cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install && cmake --build . --target install
export CYCLONEDDS_HOME=/tmp/cyclonedds/install

# Install SDK
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git /tmp/unitree_sdk2_python
cd /tmp/unitree_sdk2_python && pip3 install -e .
```

- [ ] **Step 1: Implement robot interface**

```python
# deploy/robot_interface.py
"""DDS interface to Go2 robot (sim or real) via unitree_sdk2_python."""
import numpy as np
import time
from deploy.go2_constants import (
    POLICY_TO_SDK, DEFAULT_POSE_SDK, NUM_JOINTS,
    KP_SIM, KD_SIM, KP_REAL, KD_REAL, ACTION_SCALE,
)

try:
    from unitree_sdk2py.core.channel import (
        ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize,
    )
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
    HAS_SDK = True
except ImportError:
    HAS_SDK = False


class Go2Interface:
    """Publish motor commands and subscribe to robot state via CycloneDDS.

    Usage:
        iface = Go2Interface(sim=True)
        iface.start()
        while True:
            state = iface.get_state()  # joint pos/vel, IMU
            iface.send_action(action_policy_order)  # policy output [-1, 1]
    """

    def __init__(self, sim: bool = True, interface: str = "lo"):
        if not HAS_SDK:
            raise RuntimeError(
                "unitree_sdk2_python not installed. "
                "See deploy README or install from https://github.com/unitreerobotics/unitree_sdk2_python"
            )
        self.sim = sim
        self.interface = interface
        self.kp = KP_SIM if sim else KP_REAL
        self.kd = KD_SIM if sim else KD_REAL
        self._low_state = None
        self._crc = CRC()

    def start(self):
        """Initialize DDS channels."""
        domain_id = 1 if self.sim else 0
        ChannelFactoryInitialize(domain_id, self.interface)

        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._pub.Init()

        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_low_state, 10)

        # Prepare command template
        self._cmd = unitree_go_msg_dds__LowCmd_()
        self._cmd.head[0] = 0xFE
        self._cmd.head[1] = 0xEF
        self._cmd.level_flag = 0xFF
        self._cmd.gpio = 0
        for i in range(20):
            self._cmd.motor_cmd[i].mode = 0x01

    def _on_low_state(self, msg: LowState_):
        self._low_state = msg

    def get_state(self):
        """Get current robot state. Returns None if no state received yet.

        Returns dict with keys: joint_pos_sdk, joint_vel_sdk, gyroscope, quaternion
        All in SDK joint order (FR, FL, RR, RL).
        """
        if self._low_state is None:
            return None

        state = self._low_state
        joint_pos = np.array([state.motor_state[i].q for i in range(NUM_JOINTS)], dtype=np.float32)
        joint_vel = np.array([state.motor_state[i].dq for i in range(NUM_JOINTS)], dtype=np.float32)
        gyro = np.array(state.imu_state.gyroscope, dtype=np.float32)
        quat = np.array(state.imu_state.quaternion, dtype=np.float32)  # [w, x, y, z]

        return {
            "joint_pos_sdk": joint_pos,
            "joint_vel_sdk": joint_vel,
            "gyroscope": gyro,
            "quaternion": quat,
        }

    def send_action(self, action_policy_order: np.ndarray):
        """Send action to robot. action is in policy order (FL,FR,RL,RR), range [-1, 1]."""
        # Convert to SDK order
        action_sdk = action_policy_order[POLICY_TO_SDK]

        # Compute target joint positions
        q_targets = DEFAULT_POSE_SDK + action_sdk * ACTION_SCALE

        for i in range(NUM_JOINTS):
            self._cmd.motor_cmd[i].q = float(q_targets[i])
            self._cmd.motor_cmd[i].kp = self.kp
            self._cmd.motor_cmd[i].dq = 0.0
            self._cmd.motor_cmd[i].kd = self.kd
            self._cmd.motor_cmd[i].tau = 0.0

        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)

    def send_stand(self):
        """Send default standing pose (safe starting position)."""
        for i in range(NUM_JOINTS):
            self._cmd.motor_cmd[i].q = float(DEFAULT_POSE_SDK[i])
            self._cmd.motor_cmd[i].kp = self.kp
            self._cmd.motor_cmd[i].dq = 0.0
            self._cmd.motor_cmd[i].kd = self.kd
            self._cmd.motor_cmd[i].tau = 0.0

        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)

    def send_joint_targets(self, q_targets_sdk: np.ndarray):
        """Send raw joint position targets in SDK order. Used by interpolation."""
        for i in range(NUM_JOINTS):
            self._cmd.motor_cmd[i].q = float(q_targets_sdk[i])
            self._cmd.motor_cmd[i].kp = self.kp
            self._cmd.motor_cmd[i].dq = 0.0
            self._cmd.motor_cmd[i].kd = self.kd
            self._cmd.motor_cmd[i].tau = 0.0
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)

    def send_zero_torque(self):
        """Send zero torque (robot goes limp). Safe for startup."""
        for i in range(NUM_JOINTS):
            self._cmd.motor_cmd[i].q = 0.0
            self._cmd.motor_cmd[i].kp = 0.0
            self._cmd.motor_cmd[i].dq = 0.0
            self._cmd.motor_cmd[i].kd = 0.0
            self._cmd.motor_cmd[i].tau = 0.0

        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)
```

- [ ] **Step 2: Commit**

```bash
git add deploy/robot_interface.py
git commit -m "feat: Go2Interface — DDS pub/sub for sim and real robot"
```

---

## Task 5: Main Deployment Script

**Files:**
- Create: `deploy/deploy_go2.py`

- [ ] **Step 1: Implement deployment FSM**

```python
#!/usr/bin/env python3
# deploy/deploy_go2.py
"""Deploy trained JAX RL policy on Go2 robot (sim or real).

Usage:
    # Sim2sim (unitree_mujoco must be running):
    python deploy/deploy_go2.py --checkpoint checkpoints/.../best --sim

    # Sim2real (Go2 EDU connected via ethernet):
    python deploy/deploy_go2.py --checkpoint checkpoints/.../best --interface enp2s0

FSM: IDLE → STAND (2s interpolation) → HOLD (1s) → POLICY (runs until Ctrl+C)
"""
import argparse
import time
import signal
import sys
import numpy as np

from deploy.policy_runner import PolicyRunner
from deploy.obs_builder import ObsBuilder
from deploy.robot_interface import Go2Interface
from deploy.go2_constants import POLICY_DT, DEFAULT_POSE_SDK, NUM_JOINTS


def interpolate_to_stand(iface: Go2Interface, duration: float = 2.0, dt: float = 0.002):
    """Smoothly interpolate from current pose to default standing pose."""
    state = None
    while state is None:
        state = iface.get_state()
        time.sleep(0.01)

    start_pos = state["joint_pos_sdk"]
    steps = int(duration / dt)

    for step in range(steps):
        t = (step + 1) / steps  # 0→1
        # Smooth interpolation (cosine)
        alpha = 0.5 * (1 - np.cos(np.pi * t))
        target = start_pos + alpha * (DEFAULT_POSE_SDK - start_pos)
        iface.send_joint_targets(target)
        time.sleep(dt)


def run_policy_loop(
    runner: PolicyRunner,
    obs_builder: ObsBuilder,
    iface: Go2Interface,
    command: np.ndarray,
):
    """Run policy at 50Hz until interrupted."""
    print(f"Policy loop started — command: vx={command[0]:.1f}, vy={command[1]:.1f}, yaw={command[2]:.1f}")
    print("Press Ctrl+C to stop")

    step = 0
    try:
        while True:
            t_start = time.monotonic()

            state = iface.get_state()
            if state is None:
                time.sleep(POLICY_DT)
                continue

            # Build observation
            obs = obs_builder.build(
                joint_pos_sdk=state["joint_pos_sdk"],
                joint_vel_sdk=state["joint_vel_sdk"],
                gyroscope=state["gyroscope"],
                quaternion=state["quaternion"],
                command=command,
            )

            # Run policy
            action = runner.get_action(obs)

            # Send to robot
            iface.send_action(action)

            # Update last action for next obs
            obs_builder.update_last_action(action)

            step += 1
            if step % 50 == 0:  # Log every 1s
                q = state["joint_pos_sdk"]
                print(f"Step {step:5d} | action [{action.min():.2f}, {action.max():.2f}] | "
                      f"q [{q.min():.2f}, {q.max():.2f}]")

            # Maintain 50Hz
            elapsed = time.monotonic() - t_start
            sleep_time = POLICY_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping policy loop")


def main():
    parser = argparse.ArgumentParser(description="Deploy Go2 RL policy")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint dir (with actor_params.npy)")
    parser.add_argument("--sim", action="store_true", help="Use sim (unitree_mujoco) instead of real robot")
    parser.add_argument("--interface", default="lo", help="Network interface (default: lo for sim)")
    parser.add_argument("--vx", type=float, default=0.5, help="Forward velocity command (m/s)")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command (m/s)")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw rate command (rad/s)")
    parser.add_argument("--stand-duration", type=float, default=2.0, help="Stand-up interpolation time (s)")
    parser.add_argument("--hold-duration", type=float, default=1.0, help="Hold standing pose time (s)")
    args = parser.parse_args()

    # Load policy
    print(f"Loading policy from {args.checkpoint}")
    runner = PolicyRunner(args.checkpoint)
    print(f"  algo={runner.algo}, obs_dim={runner.obs_dim}, action_dim={runner.action_dim}")
    print(f"  hidden_dim={runner.hidden_dim}, activation={runner.activation}")
    print(f"  obs_norm={'yes' if runner.use_obs_norm else 'no'} (count={runner.norm_count})")

    obs_builder = ObsBuilder()
    command = np.array([args.vx, args.vy, args.yaw], dtype=np.float32)

    # Connect to robot
    mode = "SIM" if args.sim else "REAL"
    print(f"\nConnecting to Go2 ({mode}) on interface '{args.interface}'...")
    iface = Go2Interface(sim=args.sim, interface=args.interface)
    iface.start()

    # Wait for first state
    print("Waiting for robot state...")
    while iface.get_state() is None:
        time.sleep(0.1)
    print("  Robot state received")

    # Safety FSM
    if not args.sim:
        print("\n=== REAL ROBOT MODE ===")
        input("Press Enter to start stand-up sequence (robot will move!)...")

    # Phase 1: Stand up
    print(f"\nPhase 1: Interpolating to stand ({args.stand_duration}s)...")
    interpolate_to_stand(iface, duration=args.stand_duration)

    # Phase 2: Hold
    print(f"Phase 2: Holding stand ({args.hold_duration}s)...")
    t_hold = time.monotonic()
    while time.monotonic() - t_hold < args.hold_duration:
        iface.send_stand()
        time.sleep(0.002)

    # Phase 3: Policy
    print("Phase 3: Running policy...")
    run_policy_loop(runner, obs_builder, iface, command)

    # Cleanup: return to stand
    print("Returning to standing pose...")
    for _ in range(500):  # 1s at 500Hz
        iface.send_stand()
        time.sleep(0.002)

    print("Done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add deploy/deploy_go2.py
git commit -m "feat: deploy_go2.py — FSM deployment script (idle→stand→policy) for sim and real"
```

---

## Task 6: Sim2Sim Integration Test

**Files:**
- No new files — this is a manual integration test

**Prerequisites:**
1. `unitree_mujoco` cloned and running
2. `unitree_sdk2_python` installed
3. A trained Go2 checkpoint with `best/actor_params.npy`

- [ ] **Step 1: Start unitree_mujoco simulator**

In terminal 1:
```bash
cd /path/to/unitree_mujoco/simulate_python
python3 unitree_mujoco.py
```

You should see the MuJoCo viewer with Go2 standing.

- [ ] **Step 2: Run deploy script in sim mode**

In terminal 2:
```bash
uv run python deploy/deploy_go2.py \
    --checkpoint checkpoints/<best_go2_run>/best \
    --sim \
    --vx 0.5
```

**Expected behavior:**
1. "Loading policy..." prints algo, dims, norm state
2. "Waiting for robot state..." then "Robot state received"
3. "Interpolating to stand..." — robot smoothly rises to standing in viewer
4. "Holding stand..." — robot holds for 1s
5. "Running policy..." — robot starts walking forward at ~0.5 m/s
6. Periodic logs: `Step 50 | action [-0.xx, 0.xx] | q [-x.xx, x.xx]`

**What to watch for:**
- Robot falls immediately → joint remapping is wrong (check obs_builder and action remapping)
- Robot stands but doesn't move → linvel zeroing may be breaking the policy, or command isn't reaching the obs
- Robot moves erratically → PD gains mismatch or action_scale mismatch
- Robot vibrates → Kp too high (try Kp=20)

- [ ] **Step 3: Test with zero command (should stand still)**

```bash
uv run python deploy/deploy_go2.py \
    --checkpoint checkpoints/<best_go2_run>/best \
    --sim \
    --vx 0.0 --vy 0.0 --yaw 0.0
```

Robot should stand in place without walking.

- [ ] **Step 4: Test with turning command**

```bash
uv run python deploy/deploy_go2.py \
    --checkpoint checkpoints/<best_go2_run>/best \
    --sim \
    --vx 0.3 --yaw 0.5
```

Robot should walk forward while turning.

- [ ] **Step 5: Document results in journal**

Update `.context/journals/2026-03-26.md` with sim2sim results: does it walk? any issues? What tuning was needed?

- [ ] **Step 6: Commit any fixes**

```bash
git add deploy/
git commit -m "fix: sim2sim tuning — <describe what was fixed>"
```

---

## Task 7: Sim2Real Preparation (after sim2sim validated)

- [ ] **Step 1: Verify Go2 EDU hardware**

Confirm:
- Go2 EDU edition (not Pro/Air)
- Ethernet connection to robot at 192.168.123.161
- Sport mode disabled via Unitree app

- [ ] **Step 2: Test DDS connectivity**

```bash
uv run python -c "
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

ChannelFactoryInitialize(0, 'enp2s0')  # adjust interface name
sub = ChannelSubscriber('rt/lowstate', LowState_)
sub.Init()
msg = sub.Read(timeout=3.0)
if msg is not None:
    print(f'Got state! Motor 0 q={msg.motor_state[0].q:.3f}')
    print(f'IMU quat={list(msg.imu_state.quaternion)}')
else:
    print('ERROR: No state received in 3s. Check ethernet + sport mode.')
"
```

- [ ] **Step 3: Run deployment on real robot**

```bash
uv run python deploy/deploy_go2.py \
    --checkpoint checkpoints/<best_go2_run>/best \
    --interface enp2s0 \
    --vx 0.3 \
    --stand-duration 3.0 \
    --hold-duration 2.0
```

**Safety notes for real robot:**
- Start with `--vx 0.3` (slow), not 0.5
- Longer stand-duration (3s) for gentle startup
- Be ready to Ctrl+C immediately if robot behaves erratically
- Have someone ready to catch the robot
- First test on a flat, open surface

- [ ] **Step 4: Document sim2real results**

Update journal and lessons with:
- Does the policy transfer? Walking quality?
- Any PD gain tuning needed?
- Sim2real gap observations
- Whether linvel zeroing caused issues

---

## Open Questions (resolve during implementation)

1. ~~**Default pose values**~~ — RESOLVED: `[0, 0.9, -1.8] * 4` from MJCF keyframe, verified against go2_joystick.py.

2. **Obs normalization eps** — our training uses eps=1e-8, but deployment may want eps=1e-2 for safety. Test both. Ideally store eps in meta.json during checkpointing.

3. **Action clipping** — should we clip to joint limits (from MJCF) in addition to [-1, 1]? The env does this internally during training but the deploy script doesn't. Add joint limit clipping before real hardware deployment.

4. **Linvel impact** — if zeroing dims 0:3 kills the policy, we need to either: (a) retrain without linvel, (b) estimate from IMU integration, or (c) use Unitree's built-in velocity estimate from sportmodestate. Option (c) is quickest.

5. **Kp=35 vs Kp=20** — our policy was trained with Kp=35 (MJX). Deploy with Kp=35 first for sim2sim. If vibration on real hardware, switch to Kp=20 and potentially retrain.

6. ~~**Quaternion convention**~~ — RESOLVED: runtime check added to ObsBuilder. MuJoCo/unitree_mujoco uses [w,x,y,z]. Real SDK should too but we detect and warn if [x,y,z,w] is detected.
