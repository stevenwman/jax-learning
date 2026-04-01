# Go2 Bongo Board Handstand — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Warp-based RL env for training a Go2 quadruped to balance a handstand on a bongo board.

**Architecture:** Subclass `Go2WarpEnv` with a new scene XML (Go2 + bongo board), handstand-specific obs/reward/termination, and board state extraction via named joint/body lookups. The bongo board MJCF is already built and validated.

**Tech Stack:** MuJoCo Warp backend, JAX, ml_collections, Playground registry

**Spec:** `docs/superpowers/specs/2026-04-01-bongo-handstand-design.md`

**Existing validated artifact:** `jax_rl/envs/locomotion/xmls/bongo_board.xml` (board + roller, equality constraint tested)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `jax_rl/envs/locomotion/xmls/go2_bongo_scene.xml` | Create | Scene: Go2 + board + floor + sensors + keyframe |
| `jax_rl/envs/locomotion/go2_bongo_handstand.py` | Create | Env class: config, obs, reward, termination, step, reset |
| `jax_rl/envs/locomotion/go2_constants.py` | Modify | Add `BONGO_SCENE_XML` path |
| `jax_rl/training/env_setup.py` | Modify | Register `Go2BongoHandstand` |
| `tests/test_go2_bongo_env.py` | Create | Smoke tests for the new env |
| `tools/bongo_board_test.py` | Modify | Add handstand pose render test |

---

### Task 1: Scene XML — Go2 + Bongo Board

**Files:**
- Create: `jax_rl/envs/locomotion/xmls/go2_bongo_scene.xml`
- Modify: `jax_rl/envs/locomotion/go2_constants.py`

- [ ] **Step 1: Create scene XML**

Model after `go2_warp_scene_flat.xml`. Include Go2 first, then bongo board. Add floor, lighting, IMU sensors (copy from warp scene), board contact sensors, and a placeholder handstand keyframe.

```xml
<mujoco model="go2 bongo handstand scene">
  <!-- Go2 FIRST so robot qpos occupies indices 0-18 -->
  <include file="unitree_go2/go2.xml"/>
  <include file="bongo_board.xml"/>

  <statistic center="0 0 0.3" extent="1.0" meansize="0.04"/>

  <visual>
    <headlight diffuse=".8 .8 .8" ambient=".2 .2 .2" specular="1 1 1"/>
    <rgba force="1 0 0 1"/>
    <global azimuth="120" elevation="-20" offwidth="3840" offheight="2160"/>
    <map force="0.01"/>
    <scale forcewidth="0.3" contactwidth="0.5" contactheight="0.2"/>
    <quality shadowsize="8192"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1"
      width="800" height="800"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="1 1 1" rgb2="1 1 1" markrgb="0 0 0" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
      texrepeat="5 5" reflectance="0"/>
  </asset>

  <worldbody>
    <geom name="floor" size="0 0 0.01" type="plane" material="groundplane"
      contype="1" conaffinity="0" priority="1" friction="0.6" condim="3"/>
  </worldbody>

  <sensor>
    <!-- Go2 IMU sensors (same as go2_warp_scene_flat.xml) -->
    <gyro site="imu" name="gyro"/>
    <accelerometer site="imu" name="accelerometer"/>
    <framelinvel objtype="site" objname="imu" name="global_linvel"/>
    <frameangvel objtype="site" objname="imu" name="global_angvel"/>
    <velocimeter site="imu" name="local_linvel"/>
    <framezaxis objtype="site" objname="imu" name="upvector"/>
    <framexaxis objtype="site" objname="imu" name="forwardvector"/>

    <!-- Per-foot position sensors -->
    <framepos objtype="site" objname="FL_foot" name="FL_pos"
      reftype="site" refname="imu"/>
    <framepos objtype="site" objname="FR_foot" name="FR_pos"
      reftype="site" refname="imu"/>
    <framepos objtype="site" objname="RL_foot" name="RL_pos"
      reftype="site" refname="imu"/>
    <framepos objtype="site" objname="RR_foot" name="RR_pos"
      reftype="site" refname="imu"/>

    <!-- Per-foot linear velocity sensors -->
    <framelinvel objtype="site" objname="FL_foot" name="FL_global_linvel"/>
    <framelinvel objtype="site" objname="FR_foot" name="FR_global_linvel"/>
    <framelinvel objtype="site" objname="RL_foot" name="RL_global_linvel"/>
    <framelinvel objtype="site" objname="RR_foot" name="RR_global_linvel"/>

    <!-- Floor contact sensors (for rear legs if they touch ground) -->
    <contact name="FL_floor_found" geom1="FL" geom2="floor"
      reduce="mindist" num="1" data="found"/>
    <contact name="FR_floor_found" geom1="FR" geom2="floor"
      reduce="mindist" num="1" data="found"/>
    <contact name="RL_floor_found" geom1="RL" geom2="floor"
      reduce="mindist" num="1" data="found"/>
    <contact name="RR_floor_found" geom1="RR" geom2="floor"
      reduce="mindist" num="1" data="found"/>

    <!-- Front foot contact with BOARD (not floor) -->
    <contact name="FL_board_found" geom1="FL" geom2="board_top"
      reduce="mindist" num="1" data="found"/>
    <contact name="FR_board_found" geom1="FR" geom2="board_top"
      reduce="mindist" num="1" data="found"/>
  </sensor>

  <!-- Placeholder handstand keyframe — robot inverted on board.
       qpos layout: [robot_base(7), robot_joints(12), board_base(7), roller_slide(1), roller_spin(1)]
       Total: 28 qpos values.
       Joint angles and base position TBD via geometric calculation + visual iteration. -->
  <keyframe>
    <key name="handstand"
      qpos="0 0 0.5 0 0 1 0
            0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8
            0 0 0.1318 1 0 0 0
            0 0"
      ctrl="0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8"/>
  </keyframe>
</mujoco>
```

Note: The handstand keyframe above is a rough placeholder. The robot quat `0 0 1 0` is 180deg around Y. Joint angles are copied from home pose and will be refined in Task 3.

- [ ] **Step 2: Add constant to `go2_constants.py`**

Add after the existing `WARP_SCENE_FLAT_XML` line:

```python
BONGO_SCENE_XML = ROOT_PATH / "go2_bongo_scene.xml"
```

- [ ] **Step 3: Verify scene loads with CPU MuJoCo**

```bash
uv run python -c "
import mujoco
from jax_rl.envs.locomotion.go2_warp_base import get_warp_assets
from jax_rl.envs.locomotion import go2_constants as consts
assets = get_warp_assets()
m = mujoco.MjModel.from_xml_string(open(str(consts.BONGO_SCENE_XML)).read(), assets=assets)
print(f'nq={m.nq}, nv={m.nv}, nu={m.nu}')
print(f'Bodies: {[m.body(i).name for i in range(m.nbody)]}')
print(f'Joints: {[m.joint(i).name for i in range(m.njnt)]}')
print(f'board qposadr: {m.joint(\"board_joint\").qposadr}')
print(f'roller_slide qposadr: {m.joint(\"roller_slide\").qposadr}')
"
```

Expected: `nq=28`, `nv=26`, `nu=12`. Board joint at qposadr=19, roller_slide at 26.

- [ ] **Step 4: Commit**

```bash
git add jax_rl/envs/locomotion/xmls/go2_bongo_scene.xml jax_rl/envs/locomotion/go2_constants.py
git commit -m "feat: add Go2 bongo board scene XML with handstand keyframe"
```

---

### Task 2: Env Class — Skeleton with Config and Registration

**Files:**
- Create: `jax_rl/envs/locomotion/go2_bongo_handstand.py`
- Modify: `jax_rl/training/env_setup.py`
- Create: `tests/test_go2_bongo_env.py`

- [ ] **Step 1: Write loading tests**

```python
"""Tests for Go2 Bongo Board Handstand environment."""
import jax
import jax.numpy as jnp
import pytest

from jax_rl.envs.locomotion.go2_bongo_handstand import (
    BongoHandstand,
    default_config,
)


@pytest.fixture
def env():
    return BongoHandstand(task="bongo_handstand")


@pytest.fixture
def state(env):
    return env.reset(jax.random.PRNGKey(0))


class TestBongoLoads:
    def test_action_size(self, env):
        assert env.action_size == 12

    def test_board_body_exists(self, env):
        board_id = env.mj_model.body("board").id
        assert board_id > 0

    def test_roller_joints_exist(self, env):
        slide_id = env.mj_model.joint("roller_slide").id
        spin_id = env.mj_model.joint("roller_spin").id
        assert slide_id > 0
        assert spin_id > 0

    def test_equality_constraint_exists(self, env):
        assert env.mj_model.neq > 0

    def test_config_impl_is_warp(self):
        cfg = default_config()
        assert cfg.impl == "warp"

    def test_obs_dict_keys(self, state):
        assert isinstance(state.obs, dict)
        assert "state" in state.obs
        assert "privileged_state" in state.obs

    def test_obs_dims_with_board_state(self, state):
        assert state.obs["state"].shape == (46,)

    def test_obs_dims_without_board_state(self):
        cfg = default_config()
        cfg.observe_board_state = False
        env = BongoHandstand(task="bongo_handstand", config=cfg)
        state = env.reset(jax.random.PRNGKey(0))
        assert state.obs["state"].shape == (42,)

    def test_reset_shapes(self, state):
        assert state.reward.shape == ()
        assert state.done.shape == ()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run python -m pytest tests/test_go2_bongo_env.py -v 2>&1 | head -20
```

Expected: ImportError — module doesn't exist yet.

- [ ] **Step 3: Write env class skeleton**

Create `jax_rl/envs/locomotion/go2_bongo_handstand.py`:

```python
"""Go2 Bongo Board Handstand environment (Warp backend).

Balance task: Go2 inverted on a bongo board, front legs down, rear legs up.
Uses unitree_mujoco go2.xml + bongo_board.xml via MuJoCo Warp.

Returns dict obs: {"state": 42-46d policy obs, "privileged_state": ~96d critic obs}.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from jax_rl.envs.locomotion import go2_warp_base
from jax_rl.envs.locomotion import go2_constants as consts


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=500,
        Kp=20.0,
        Kd=0.5,
        action_repeat=1,
        action_scale=0.25,
        soft_joint_pos_limit_factor=0.95,
        observe_board_state=True,
        target_handstand_height=0.45,
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                inverted_orientation=10.0,
                board_level=8.0,
                com_above_support=5.0,
                height=-5.0,
                roller_centered=-2.0,
                torques=-0.0002,
                action_rate=-0.01,
                termination=-1.0,
            ),
        ),
        impl="warp",
        contact_mode="training",
        naconmax=4 * 8192,
        naccdmax=5000,
        njmax=150,
    )


class BongoHandstand(go2_warp_base.Go2WarpEnv):
    """Go2 handstand balance on a bongo board (Warp backend)."""

    def __init__(
        self,
        task: str = "bongo_handstand",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=consts.BONGO_SCENE_XML.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        # Robot pose from handstand keyframe.
        self._init_q = jp.array(self._mj_model.keyframe("handstand").qpos)
        self._default_pose = jp.array(
            self._mj_model.keyframe("handstand").qpos[7:19]
        )

        # Soft joint limits (joints 1-12 are robot, skip freejoint 0).
        # With two free bodies, jnt_range has more entries. Use robot joints only.
        robot_jnt_ids = []
        for i in range(self._mj_model.njnt):
            jnt_name = self._mj_model.joint(i).name
            if "hip_joint" in jnt_name or "thigh_joint" in jnt_name or "calf_joint" in jnt_name:
                robot_jnt_ids.append(i)
        robot_jnt_range = self._mj_model.jnt_range[robot_jnt_ids]
        self._lowers, self._uppers = robot_jnt_range.T
        c = self._config.soft_joint_pos_limit_factor
        self._soft_lowers = self._lowers * c
        self._soft_uppers = self._uppers * c

        # Torso body (robot).
        self._torso_body_id = self._mj_model.body(consts.WARP_ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

        # Board/roller indices — resolved by name, never hardcoded.
        self._board_body_id = self._mj_model.body("board").id
        self._roller_slide_qposadr = self._mj_model.joint("roller_slide").qposadr
        self._roller_slide_dofadr = self._mj_model.joint("roller_slide").dofadr
        self._board_jnt_qposadr = self._mj_model.joint("board_joint").qposadr
        self._board_jnt_dofadr = self._mj_model.joint("board_joint").dofadr

        # Front foot contact with board sensors.
        self._fl_board_sensor = self._mj_model.sensor("FL_board_found").id
        self._fr_board_sensor = self._mj_model.sensor("FR_board_found").id

        # Board contact mode override.
        # IMPORTANT: must modify _mj_model THEN re-create _mjx_model,
        # because base class already called mjx.put_model() in __init__.
        if getattr(self._config, 'contact_mode', 'training') == 'training':
            board_gid = self._mj_model.geom("board_top").id
            self._mj_model.geom_solimp[board_gid, :3] = np.array(
                [0.9, 0.95, 0.023]
            )
            self._mj_model.geom_condim[board_gid] = 3
            self._mj_model.geom_friction[board_gid] = np.array(
                [0.8, 0.005, 0.001]
            )

        # Re-create Warp model after all _mj_model modifications.
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

    # ── Core env methods (stubs for now) ───────────────────────────

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        # Small perturbations on robot joint angles.
        rng, key = jax.random.split(rng)
        joint_noise = jax.random.uniform(key, (12,), minval=-0.05, maxval=0.05)
        qpos = qpos.at[7:19].set(qpos[7:19] + joint_noise)

        # Small perturbation on robot base position (±2cm).
        rng, key = jax.random.split(rng)
        base_noise = jax.random.uniform(key, (3,), minval=-0.02, maxval=0.02)
        qpos = qpos.at[0:3].set(qpos[0:3] + base_noise)

        # Small perturbation on board tilt (±2deg ≈ ±0.035 rad).
        # Perturb board quat with small rotation around X and Y.
        rng, key = jax.random.split(rng)
        board_qposadr = self._board_jnt_qposadr
        tilt_noise = jax.random.uniform(key, (2,), minval=-0.035, maxval=0.035)
        # Apply as small-angle quaternion perturbation.
        dq = jp.array([1.0, tilt_noise[0], tilt_noise[1], 0.0])
        dq = dq / jp.linalg.norm(dq)
        from mujoco.mjx._src import math as mjx_math
        board_quat = qpos[board_qposadr + 3 : board_qposadr + 7]
        new_quat = mjx_math.quat_mul(board_quat, dq)
        qpos = qpos.at[board_qposadr + 3 : board_qposadr + 7].set(new_quat)

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=jp.zeros(self.mjx_model.nu),
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            naccdmax=self._config.naccdmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        info = {
            "rng": rng,
            "last_act": jp.zeros(self.mjx_model.nu),
            "step_count": jp.int32(0),
            "reward_components": {
                k: jp.zeros(()) for k in self._config.reward_config.scales.keys()
            },
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        motor_targets = self._default_pose + action * self._config.action_scale

        # External PD at physics rate (same pattern as WarpJoystick).
        kp = self._kp
        kd = self._kd
        model = self.mjx_model
        a2j = self._act_to_joint

        data = state.data

        def substep(data, _):
            current_q = data.qpos[7:19]
            current_dq = data.qvel[6:18]
            tau_joint = kp * (motor_targets - current_q) + kd * (0.0 - current_dq)
            tau_act = tau_joint[a2j]
            data = data.replace(ctrl=tau_act)
            return mjx.step(model, data), None

        data = jax.lax.scan(substep, data, (), self.n_substeps)[0]

        obs = self._get_obs(data, state.info)
        done = self._get_termination(data)

        rewards = self._get_reward(data, action, state.info, done)
        rewards = {
            k: v * self._config.reward_config.scales[k]
            for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        state.info["reward_components"] = rewards
        state.info["last_act"] = action
        state.info["step_count"] = state.info["step_count"] + 1
        state.info["rng"], _ = jax.random.split(state.info["rng"])

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        done = done.astype(reward.dtype)
        return state.replace(data=data, obs=obs, reward=reward, done=done)

    # ── Observation ────────────────────────────────────────────────

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> Dict[str, jax.Array]:
        gyro = self.get_gyro(data)
        gravity = self.get_gravity(data)
        joint_angles = data.qpos[7:19]
        joint_vel = data.qvel[6:18]

        # Apply noise.
        noise_level = self._config.noise_config.level
        info["rng"], *noise_rngs = jax.random.split(info["rng"], 5)

        noisy_gyro = (
            gyro
            + (2 * jax.random.uniform(noise_rngs[0], shape=gyro.shape) - 1)
            * noise_level * self._config.noise_config.scales.gyro
        )
        noisy_gravity = (
            gravity
            + (2 * jax.random.uniform(noise_rngs[1], shape=gravity.shape) - 1)
            * noise_level * self._config.noise_config.scales.gravity
        )
        noisy_joint_angles = (
            joint_angles
            + (2 * jax.random.uniform(noise_rngs[2], shape=joint_angles.shape) - 1)
            * noise_level * self._config.noise_config.scales.joint_pos
        )
        noisy_joint_vel = (
            joint_vel
            + (2 * jax.random.uniform(noise_rngs[3], shape=joint_vel.shape) - 1)
            * noise_level * self._config.noise_config.scales.joint_vel
        )

        # Core robot state (42d).
        state_parts = [
            noisy_gyro,                                 # 3
            noisy_gravity,                              # 3
            noisy_joint_angles - self._default_pose,    # 12
            noisy_joint_vel,                            # 12
            info["last_act"],                           # 12
        ]  # Total: 42

        # Optional board state (4d).
        if self._config.observe_board_state:
            board_tilt_xy = self._get_board_tilt(data)
            roller_pos = data.qpos[self._roller_slide_qposadr]
            roller_vel = data.qvel[self._roller_slide_dofadr]
            state_parts.extend([
                board_tilt_xy,                          # 2
                roller_pos.reshape(1),                  # 1
                roller_vel.reshape(1),                  # 1
            ])  # Total: 46

        state = jp.concatenate(state_parts)

        # Privileged state for critic.
        fl_board = data.sensordata[
            self._mj_model.sensor_adr[self._fl_board_sensor]
        ]
        fr_board = data.sensordata[
            self._mj_model.sensor_adr[self._fr_board_sensor]
        ]

        board_tilt_xy_clean = self._get_board_tilt(data)
        roller_pos_clean = data.qpos[self._roller_slide_qposadr]
        roller_vel_clean = data.qvel[self._roller_slide_dofadr]

        # Board angular velocity (from board body).
        board_angvel = self._get_board_angvel(data)

        # CoM relative to board center.
        com_xy = data.subtree_com[self._torso_body_id][:2]
        board_xy = data.xpos[self._board_body_id][:2]
        com_rel_board = com_xy - board_xy

        privileged_state = jp.concatenate([
            state,                                      # 42 or 46
            gyro,                                       # 3 (unnoised)
            gravity,                                    # 3 (unnoised)
            joint_angles - self._default_pose,          # 12 (unnoised)
            joint_vel,                                  # 12 (unnoised)
            data.actuator_force,                        # 12
            jp.array([fl_board > 0, fr_board > 0]).astype(jp.float32),  # 2
            board_tilt_xy_clean,                        # 2
            roller_pos_clean.reshape(1),                # 1
            roller_vel_clean.reshape(1),                # 1
            board_angvel[:2],                           # 2
            com_rel_board,                              # 2
        ])  # Total: 42+52=94 or 46+52=98

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

    # ── Board state helpers ────────────────────────────────────────

    def _get_board_tilt(self, data: mjx.Data) -> jax.Array:
        """Board tilt as x,y components of board's z-axis in world frame.

        Returns [0, 0] when level. Magnitude increases with tilt.
        Uses sin(angle) which is linear near zero.
        """
        board_xmat = data.xmat[self._board_body_id].reshape(3, 3)
        return board_xmat[:2, 2]

    def _get_board_angvel(self, data: mjx.Data) -> jax.Array:
        """Board angular velocity in world frame.

        Extracted from the board body's qvel (freejoint angular vel).
        """
        # Freejoint has 6 DOFs: 3 linear + 3 angular.
        return data.qvel[self._board_jnt_dofadr + 3 : self._board_jnt_dofadr + 6]

    # ── Termination ────────────────────────────────────────────────

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        gravity = self.get_gravity(data)
        not_inverted = gravity[2] < 0.0  # should be positive when inverted

        board_tilt = self._get_board_tilt(data)
        board_too_tilted = jp.sum(board_tilt ** 2) > 0.25  # ~30 deg

        base_z = data.subtree_com[self._torso_body_id][2]
        too_low = base_z < 0.15

        roller_pos = data.qpos[self._roller_slide_qposadr]
        roller_at_limit = jp.abs(roller_pos) > 0.22

        return not_inverted | board_too_tilted | too_low | roller_at_limit

    # ── Rewards ────────────────────────────────────────────────────

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        gravity = self.get_gravity(data)
        target_gravity = jp.array([0.0, 0.0, 1.0])

        board_tilt = self._get_board_tilt(data)
        roller_pos = data.qpos[self._roller_slide_qposadr]

        com_xy = data.subtree_com[self._torso_body_id][:2]
        board_xy = data.xpos[self._board_body_id][:2]

        return {
            "inverted_orientation": jp.exp(
                -jp.sum((gravity - target_gravity) ** 2)
            ),
            "board_level": jp.exp(-jp.sum(board_tilt ** 2) / 0.1),
            "com_above_support": jp.exp(
                -jp.sum((com_xy - board_xy) ** 2) / 0.05
            ),
            "height": (
                data.subtree_com[self._torso_body_id][2]
                - self._config.target_handstand_height
            ) ** 2,
            "roller_centered": roller_pos ** 2,
            "torques": (
                jp.sqrt(jp.sum(jp.square(data.actuator_force)))
                + jp.sum(jp.abs(data.actuator_force))
            ),
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "termination": done,
        }
```

- [ ] **Step 4: Register in env_setup.py**

Add to `_register_custom_envs()` in `jax_rl/training/env_setup.py`, after the WarpJoystick registration block:

```python
from jax_rl.envs.locomotion.go2_bongo_handstand import BongoHandstand
from jax_rl.envs.locomotion.go2_bongo_handstand import default_config as bongo_default_config
if "Go2BongoHandstand" not in pg_locomotion._envs:
    pg_locomotion.register_environment(
        "Go2BongoHandstand",
        functools.partial(BongoHandstand, task="bongo_handstand"),
        bongo_default_config,
    )
```

- [ ] **Step 5: Run tests**

```bash
uv run python -m pytest tests/test_go2_bongo_env.py -v
```

Expected: All `TestBongoLoads` tests pass.

- [ ] **Step 6: Commit**

```bash
git add jax_rl/envs/locomotion/go2_bongo_handstand.py jax_rl/training/env_setup.py tests/test_go2_bongo_env.py
git commit -m "feat: add Go2BongoHandstand env class with obs, reward, termination"
```

---

### Task 3: Handstand Keyframe — Geometric Calculation + Visual Iteration

**Files:**
- Modify: `jax_rl/envs/locomotion/xmls/go2_bongo_scene.xml` (keyframe)
- Modify: `tools/bongo_board_test.py` (add pose render)

This is the most iterative task. The goal is a physically plausible handstand starting pose.

- [ ] **Step 1: Add pose render function to test script**

Add to `tools/bongo_board_test.py`:

```python
def render_handstand_pose(xml_path, output_path="/tmp/bongo_test/handstand_pose.png"):
    """Render a single frame of the handstand keyframe for visual inspection."""
    from jax_rl.envs.locomotion.go2_warp_base import get_warp_assets
    assets = get_warp_assets()
    model = mujoco.MjModel.from_xml_string(
        open(str(xml_path)).read(), assets=assets
    )
    data = mujoco.MjData(model)

    # Load handstand keyframe.
    key_id = model.key("handstand").id
    data.qpos[:] = model.key_qpos[key_id]
    data.ctrl[:] = model.key_ctrl[key_id]
    mujoco.mj_forward(model, data)

    # Render from multiple angles.
    renderer = mujoco.Renderer(model, height=720, width=1280)

    for angle_name, azimuth, elevation in [
        ("front", 180, -20),
        ("side", 90, -10),
        ("top", 180, -80),
    ]:
        renderer.update_scene(data, camera=mujoco.MjvCamera(
            azimuth=azimuth, elevation=elevation,
            distance=1.5, lookat=[0, 0, 0.3],
        ))
        frame = renderer.render()
        path = Path(output_path).parent / f"handstand_{angle_name}.png"
        mediapy.write_image(str(path), frame)
        print(f"Saved: {path}")

    renderer.close()
```

- [ ] **Step 2: Compute initial geometric estimate for handstand qpos**

The Go2 inverted on the board:
- Base quat: `[0, 0, 1, 0]` (180deg around Y)
- When inverted, the body is flipped: what was "up" is now "down"
- Front legs (FL, FR) point down toward the board
- Rear legs (RL, RR) point up
- Need to compute base z so front feet touch the board top (~0.1318 + 0.015 = 0.1468m above ground)

Compute base height from leg geometry and update the keyframe. This will require trial and error.

- [ ] **Step 3: Render pose and iterate**

```bash
MUJOCO_GL=egl uv run python -c "
from tools.bongo_board_test import render_handstand_pose
from jax_rl.envs.locomotion import go2_constants as consts
render_handstand_pose(str(consts.BONGO_SCENE_XML))
"
```

Copy to `tmp_videos/` for review. Iterate on joint angles and base z until the pose looks like a stable handstand with front feet on the board.

- [ ] **Step 4: Update keyframe in scene XML**

Once the pose is validated visually, update the `<keyframe>` in `go2_bongo_scene.xml` and the `target_handstand_height` in the config.

- [ ] **Step 5: Commit**

```bash
git add jax_rl/envs/locomotion/xmls/go2_bongo_scene.xml tools/bongo_board_test.py
git commit -m "feat: tuned handstand keyframe pose for bongo board env"
```

---

### Task 4: Step + Reward Tests

**Files:**
- Modify: `tests/test_go2_bongo_env.py`

- [ ] **Step 1: Add step and integration tests**

Append to test file:

```python
class TestBongoSteps:
    def test_step_zero_action(self, env, state):
        action = jnp.zeros(12)
        next_state = env.step(state, action)
        assert isinstance(next_state.obs, dict)
        assert next_state.obs["state"].shape == (46,)
        assert not jnp.any(jnp.isnan(next_state.obs["state"]))
        assert not jnp.any(jnp.isnan(next_state.reward))

    def test_step_random_action(self, env, state):
        key = jax.random.PRNGKey(42)
        action = jax.random.uniform(key, (12,), minval=-1.0, maxval=1.0)
        next_state = env.step(state, action)
        assert next_state.obs["state"].shape == (46,)

    def test_reward_finite_after_steps(self, env, state):
        action = jnp.zeros(12)
        for _ in range(5):
            state = env.step(state, action)
        assert jnp.isfinite(state.reward)

    def test_privileged_state_shape(self, state):
        priv = state.obs["privileged_state"]
        # 46 (state w/ board) + 52 (privileged extras) = 98
        assert priv.shape == (98,)

    def test_privileged_state_without_board(self):
        cfg = default_config()
        cfg.observe_board_state = False
        env = BongoHandstand(task="bongo_handstand", config=cfg)
        state = env.reset(jax.random.PRNGKey(0))
        # 42 (state w/o board) + 52 (privileged extras) = 94
        assert state.obs["privileged_state"].shape == (94,)


class TestBongoBatched:
    def test_make_envs_integration(self):
        from jax_rl.training.env_setup import make_envs
        from jax_rl.configs.train_config import TrainConfig

        cfg = TrainConfig(
            env_name="Go2BongoHandstand",
            num_envs=4,
            total_timesteps=1000,
        )
        env, env_step, env_state, eval_env, obs_dim, action_dim, key = make_envs(
            cfg, seed=0
        )
        assert obs_dim == 46
        assert action_dim == 12
        assert isinstance(env_state.obs, dict)
        assert env_state.obs["state"].shape == (4, 46)

        action = jnp.zeros((4, 12))
        next_state = env_step(env_state, action)
        assert isinstance(next_state.obs, dict)
        assert not jnp.any(jnp.isnan(next_state.obs["state"]))
```

- [ ] **Step 2: Run all tests**

```bash
uv run python -m pytest tests/test_go2_bongo_env.py -v
```

Expected: All pass.

- [ ] **Step 3: Run existing tests for regression**

```bash
uv run python -m pytest tests/ -v --timeout=120 2>&1 | tail -20
```

Expected: No regressions in existing tests.

- [ ] **Step 4: Commit**

```bash
git add tests/test_go2_bongo_env.py
git commit -m "test: add step, reward, and integration tests for bongo handstand env"
```

---

### Task 5: Doc Sync + Cleanup

**Files:**
- Modify: `.context/TODO.md`
- Modify: `.context/AGENT_HANDOFF.md`

- [ ] **Step 1: Update TODO.md**

Add under Active or Short-term:

```markdown
## Short-term — Bongo Board Handstand
- [x] Bongo board MJCF — board + roller, equality constraint, physics validated
- [x] Scene XML — Go2 + bongo board + floor + sensors
- [x] `Go2BongoHandstand` env — obs, reward, termination, step, reset
- [x] Registration + tests
- [ ] Handstand keyframe tuning — visual iteration needed
- [ ] Training run — smoke test PPO/FastSAC on bongo handstand
- [ ] Phase 1B: full approach + mount + handstand (future)
```

- [ ] **Step 2: Update AGENT_HANDOFF.md**

Add to Part 4 codebase overview and Part 5 current state as appropriate. Mention the bongo board env, its registration name, and key facts (balance task, 42/46d obs, subclasses Go2WarpEnv).

- [ ] **Step 3: Remove test videos directory**

```bash
rm -rf tmp_videos/
```

- [ ] **Step 4: Commit**

```bash
git add .context/TODO.md .context/AGENT_HANDOFF.md
git commit -m "docs: add bongo handstand env to TODO and agent handoff"
```
