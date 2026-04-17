"""Shape-agnostic planar push env on MuJoCo Warp.

Pusher (circle) pushes a block to a target pose on a flat table.
Block shape is swappable: T, circle, L, star, etc.
Obs is shape-agnostic: pusher_xy, block_xy, block_angle, target_xy, target_angle, vels.

Shapes are defined as lists of MuJoCo geom dicts. Adding a new shape = adding one dict.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env


# ═════════════════════════════════════════════════════════════════════
# Shape registry — add new shapes here
# ═════════════════════════════════════════════════════════════════════

SHAPES: dict[str, list[dict]] = {
    "T": [
        {"name": "stem", "type": "box", "size": "0.025 0.075 0.015",
         "pos": "0 -0.025 0"},
        {"name": "top", "type": "box", "size": "0.075 0.025 0.015",
         "pos": "0 0.075 0"},
    ],
    "L": [
        {"name": "vert", "type": "box", "size": "0.025 0.075 0.015",
         "pos": "-0.025 0 0"},
        {"name": "horiz", "type": "box", "size": "0.075 0.025 0.015",
         "pos": "0.025 -0.05 0"},
    ],
    "circle": [
        {"name": "disc", "type": "cylinder", "size": "0.06 0.015",
         "pos": "0 0 0"},
    ],
    "plus": [
        {"name": "h_bar", "type": "box", "size": "0.075 0.02 0.015",
         "pos": "0 0 0"},
        {"name": "v_bar", "type": "box", "size": "0.02 0.075 0.015",
         "pos": "0 0 0"},
    ],
}

# ═════════════════════════════════════════════════════════════════════
# XML template
# ═════════════════════════════════════════════════════════════════════

_XML_TEMPLATE = """\
<mujoco model="push_{shape}">
  <option timestep="{sim_dt}" integrator="implicitfast" iterations="50" ls_iterations="10">
    <flag contact="enable"/>
  </option>

  <visual>
    <headlight diffuse="0.25 0.25 0.25" ambient="0.15 0.15 0.15" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global offwidth="480" offheight="480"/>
  </visual>

  <default>
    <default class="pusher">
      <geom type="cylinder" size="0.025 0.015" rgba="0.9 0.2 0.2 1"
            friction="0.3 0.005 0.001" condim="3" priority="2" mass="0.1"
            solref="0.004 1" solimp="0.98 0.995 0.0005 0.5 2"/>
    </default>
    <default class="block">
      <geom rgba="0.2 0.6 0.9 1"
            friction="0.4 0.005 0.001" condim="3" priority="1" mass="0.05"
            solref="0.004 1" solimp="0.98 0.995 0.0005 0.5 2"/>
    </default>
    <default class="target">
      <geom rgba="0.2 0.8 0.3 0.3" contype="0" conaffinity="0" mass="0.001"/>
    </default>
  </default>

  <asset>
    <texture type="2d" name="grid" builtin="checker" width="512" height="512"
             rgb1="0.45 0.47 0.5" rgb2="0.38 0.4 0.43"/>
    <material name="grid" texture="grid" texrepeat="10 10" reflectance="0" specular="0" shininess="0"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.2 0.2 0.2" specular="0 0 0" ambient="0 0 0"/>

    <!-- Table surface -->
    <geom name="floor" type="plane" size="0.3 0.3 0.01" material="grid"
          pos="0 0 0" friction="0.4 0.005 0.001" condim="3"/>

    <!-- Walls -->
    <geom name="wall_xp" type="box" size="0.005 0.3 0.03" pos="0.305 0 0.03"
          rgba="0.5 0.5 0.5 0.15" contype="1" conaffinity="1"/>
    <geom name="wall_xn" type="box" size="0.005 0.3 0.03" pos="-0.305 0 0.03"
          rgba="0.5 0.5 0.5 0.15" contype="1" conaffinity="1"/>
    <geom name="wall_yp" type="box" size="0.3 0.005 0.03" pos="0 0.305 0.03"
          rgba="0.5 0.5 0.5 0.15" contype="1" conaffinity="1"/>
    <geom name="wall_yn" type="box" size="0.3 0.005 0.03" pos="0 -0.305 0.03"
          rgba="0.5 0.5 0.5 0.15" contype="1" conaffinity="1"/>

    <!-- Pusher -->
    <body name="pusher" pos="-0.15 0 0.015">
      <joint name="pusher_x" type="slide" axis="1 0 0" range="-0.28 0.28" damping="0.8"/>
      <joint name="pusher_y" type="slide" axis="0 1 0" range="-0.28 0.28" damping="0.8"/>
      <geom name="pusher" class="pusher"/>
    </body>

    <!-- Block (shape-specific geoms injected) -->
    <body name="block" pos="0.05 0 0.015">
      <joint name="block_x" type="slide" axis="1 0 0" damping="0.3"/>
      <joint name="block_y" type="slide" axis="0 1 0" damping="0.3"/>
      <joint name="block_yaw" type="hinge" axis="0 0 1" damping="0.1"/>
{block_geoms}
    </body>

    <!-- Target ghost (same geoms, transparent) -->
    <body name="target" pos="0 0.1 0.015" mocap="true">
{target_geoms}
    </body>
  </worldbody>

  <actuator>
    <position name="pusher_x" joint="pusher_x" kp="80" ctrlrange="-0.28 0.28"
              forcerange="-20 20"/>
    <position name="pusher_y" joint="pusher_y" kp="80" ctrlrange="-0.28 0.28"
              forcerange="-20 20"/>
  </actuator>

  <keyframe>
    <key name="home" qpos="-0.15 0 0.05 0 0"/>
  </keyframe>
</mujoco>
"""


def _build_geom_xml(geoms: list[dict], cls: str, prefix: str = "") -> str:
    """Build geom XML lines from shape spec."""
    lines = []
    for g in geoms:
        attrs = [f'name="{prefix}{g["name"]}"', f'class="{cls}"']
        attrs.append(f'type="{g["type"]}"')
        attrs.append(f'size="{g["size"]}"')
        if "pos" in g:
            attrs.append(f'pos="{g["pos"]}"')
        lines.append(f'      <geom {" ".join(attrs)}/>')
    return "\n".join(lines)


def build_xml(shape: str, sim_dt: float = 0.002) -> str:
    """Build complete MJCF XML for a given shape."""
    if shape not in SHAPES:
        raise ValueError(f"Unknown shape '{shape}'. Available: {list(SHAPES.keys())}")
    geoms = SHAPES[shape]
    block_xml = _build_geom_xml(geoms, "block", prefix="block_")
    target_xml = _build_geom_xml(geoms, "target", prefix="target_")
    return _XML_TEMPLATE.format(
        shape=shape,
        sim_dt=sim_dt,
        block_geoms=block_xml,
        target_geoms=target_xml,
    )


# ═════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════

def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=200,
        shape="T",
        action_mode="position",  # "position" | "velocity" | "teleport"
        max_pusher_speed=0.25,   # m/s, used by teleport mode
        reward_type="dense",     # "dense" or "sparse"
        pos_threshold=0.02,      # success threshold (m)
        angle_threshold=0.15,    # success threshold (rad)
        randomize_target=True,
        randomize_block=True,
        table_half=0.22,         # spawn region half-size (within walls)
        impl="warp",
    )


# ═════════════════════════════════════════════════════════════════════
# Env
# ═════════════════════════════════════════════════════════════════════

class PushEnv(mjx_env.MjxEnv):
    """Shape-agnostic planar pushing task.

    Obs (16d, shape-agnostic):
        [0:2]   pusher_xy
        [2:4]   block_xy
        [4:6]   block_angle (sin, cos)
        [6:8]   target_xy
        [8:10]  target_angle (sin, cos)
        [10:12] pusher_vel_xy
        [12:14] block_vel_xy
        [14:16] last_action

    Action (2d): pusher target XY (position mode) or pusher dXY (velocity mode).
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)

        shape = self._config.shape
        xml_str = build_xml(shape, sim_dt=self._config.sim_dt)

        # Write to temp file (MjModel needs a path or string)
        self._mj_model = mujoco.MjModel.from_xml_string(xml_str)
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        # Joint indices: qpos = [pusher_x, pusher_y, block_x, block_y, block_yaw]
        self._pusher_qpos_idx = jp.array([0, 1])
        self._block_qpos_idx = jp.array([2, 3])
        self._block_yaw_idx = 4
        self._pusher_qvel_idx = jp.array([0, 1])
        self._block_qvel_idx = jp.array([2, 3])

        self._action_mode = self._config.action_mode

    # ── Properties ────────────────────────────────────────────────

    @property
    def xml_path(self) -> str:
        return f"push_{self._config.shape}_generated"

    @property
    def action_size(self) -> int:
        return 2

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    # ── Reset ─────────────────────────────────────────────────────

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k1, k2, k3, k4 = jax.random.split(rng, 5)

        half = self._config.table_half

        # Randomize block position + angle
        block_xy = jax.lax.cond(
            self._config.randomize_block,
            lambda: jax.random.uniform(k1, (2,), minval=-half * 0.6, maxval=half * 0.6),
            lambda: jp.array([0.05, 0.0]),
        )
        block_yaw = jax.lax.cond(
            self._config.randomize_block,
            lambda: jax.random.uniform(k2, (), minval=-jp.pi, maxval=jp.pi),
            lambda: jp.float32(0.0),
        )

        # Randomize target pose
        target_xy = jax.lax.cond(
            self._config.randomize_target,
            lambda: jax.random.uniform(k3, (2,), minval=-half * 0.6, maxval=half * 0.6),
            lambda: jp.array([0.0, 0.1]),
        )
        target_yaw = jax.lax.cond(
            self._config.randomize_target,
            lambda: jax.random.uniform(k4, (), minval=-jp.pi, maxval=jp.pi),
            lambda: jp.float32(0.0),
        )

        # Pusher starts at fixed offset from block
        pusher_xy = block_xy + jp.array([-0.12, 0.0])
        pusher_xy = jp.clip(pusher_xy, -0.28, 0.28)

        qpos = jp.array([pusher_xy[0], pusher_xy[1],
                          block_xy[0], block_xy[1], block_yaw])
        qvel = jp.zeros(self._mj_model.nv)

        data = mjx.make_data(self._mj_model, impl=self._config.impl)
        data = data.replace(qpos=qpos, qvel=qvel)

        # Set mocap body (target ghost) position
        mocap_pos = jp.array([[target_xy[0], target_xy[1], 0.015]])
        mocap_quat = _yaw_to_quat(target_yaw)
        data = data.replace(
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat[None],
        )
        data = mjx.forward(self._mjx_model, data)

        obs = self._get_obs(data, target_xy, target_yaw, jp.zeros(2))
        reward, done = jp.zeros(2)

        info = {
            "target_xy": target_xy,
            "target_yaw": target_yaw,
            "step_count": jp.int32(0),
            "success": jp.float32(0),
            "last_action": jp.zeros(2),
        }
        metrics = {
            "pos_error": jp.float32(0),
            "angle_error": jp.float32(0),
            "success": jp.float32(0),
        }

        return mjx_env.State(data, obs, reward, done, metrics, info)

    # ── Step ──────────────────────────────────────────────────────

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.clip(action, -1.0, 1.0)

        data = state.data

        if self._action_mode == "velocity":
            # Delta mode: action = delta position per step (1cm per unit).
            current_pos = data.qpos[self._pusher_qpos_idx]
            target = current_pos + action * 0.01
            target = jp.clip(target, -0.28, 0.28)
        elif self._action_mode == "teleport":
            # gym-pusht style: move toward commanded target, capped at
            # max_pusher_speed per control step.
            current_pos = data.qpos[self._pusher_qpos_idx]
            commanded = action * 0.28
            delta = commanded - current_pos
            max_step = self._config.max_pusher_speed * self.dt
            dist = jp.linalg.norm(delta) + 1e-8
            scale = jp.minimum(1.0, max_step / dist)
            target = current_pos + delta * scale
            target = jp.clip(target, -0.28, 0.28)
        else:
            # Position-PD mode: action = target XY (scaled to table).
            target = action * 0.28

        data = data.replace(ctrl=target)

        # Substep
        def substep(data, _):
            return mjx.step(self._mjx_model, data), None
        data = jax.lax.scan(substep, data, (), self.n_substeps)[0]

        target_xy = state.info["target_xy"]
        target_yaw = state.info["target_yaw"]

        # Compute reward
        block_xy = data.qpos[self._block_qpos_idx]
        block_yaw = data.qpos[self._block_yaw_idx]
        pos_error = jp.linalg.norm(block_xy - target_xy)
        angle_error = _angle_dist(block_yaw, target_yaw)

        if self._config.reward_type == "sparse":
            success = (pos_error < self._config.pos_threshold) & \
                      (angle_error < self._config.angle_threshold)
            reward = success.astype(jp.float32)
        else:
            # Dense: negative distance + angle penalty + success bonus
            reward = -pos_error - 0.3 * angle_error
            success = (pos_error < self._config.pos_threshold) & \
                      (angle_error < self._config.angle_threshold)
            reward = reward + 5.0 * success.astype(jp.float32)

        step_count = state.info["step_count"] + 1
        done = jp.float32(0)  # episode ends by truncation only

        obs = self._get_obs(data, target_xy, target_yaw, action)

        info = {
            **state.info,
            "step_count": step_count,
            "success": success.astype(jp.float32),
            "last_action": action,
        }
        metrics = {
            "pos_error": pos_error,
            "angle_error": angle_error,
            "success": success.astype(jp.float32),
        }

        return state.replace(data=data, obs=obs, reward=reward,
                              done=done, info=info, metrics=metrics)

    # ── Obs ───────────────────────────────────────────────────────

    def _get_obs(self, data, target_xy, target_yaw, last_action) -> jax.Array:
        pusher_xy = data.qpos[self._pusher_qpos_idx]
        block_xy = data.qpos[self._block_qpos_idx]
        block_yaw = data.qpos[self._block_yaw_idx]

        pusher_vel = data.qvel[self._pusher_qvel_idx]
        block_vel = data.qvel[self._block_qvel_idx]

        return jp.concatenate([
            pusher_xy,                               # [0:2]
            block_xy,                                # [2:4]
            jp.array([jp.sin(block_yaw), jp.cos(block_yaw)]),  # [4:6]
            target_xy,                               # [6:8]
            jp.array([jp.sin(target_yaw), jp.cos(target_yaw)]),  # [8:10]
            pusher_vel,                              # [10:12]
            block_vel,                               # [12:14]
            last_action,                             # [14:16]
        ])


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def _angle_dist(a: jax.Array, b: jax.Array) -> jax.Array:
    """Shortest angular distance between two angles."""
    diff = a - b
    return jp.abs(jp.arctan2(jp.sin(diff), jp.cos(diff)))


def _yaw_to_quat(yaw: jax.Array) -> jax.Array:
    """Convert yaw angle to quaternion [w, x, y, z]."""
    return jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
