"""Curriculum joystick env for Unitree Go2 (Warp backend).

Extends WarpJoystick with a procedurally generated terrain grid.  The terrain
MJCF is injected into a scene template at init time; the composed XML is
written to a PID-scoped temp file under xmls/ to avoid multiprocess races.

Terrain layout (GO2_DEFAULT_CFG): 10 rows × 4 cols, tile size 9.6×9.6 m.
Row 0 = flat/easy, row 9 = hardest.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco_playground._src import mjx_env

from jax_rl.envs.locomotion import go2_warp_base
from jax_rl.envs.locomotion.go2_warp_joystick import (
    WarpJoystick,
    default_config as _warp_default_config,
)
from jax_rl.envs.terrains.config import GO2_DEFAULT_CFG
from jax_rl.envs.terrains.generator import TerrainGenerator

_TEMPLATE_PATH = Path(__file__).parent / "xmls" / "go2_warp_curriculum_scene_template.xml"


def default_config() -> config_dict.ConfigDict:
    cfg = _warp_default_config()
    cfg.torque_speed_model = False
    cfg.terrain_seed = 0
    # Terrain grid has ~1500 geoms (vs ~100 for flat). Warp emits "nefc overflow
    # - please increase njmax" at init; safe to ignore — sim functions at
    # defaults (njmax=100, naconmax=32768). Bumping higher causes VRAM OOM.
    return cfg


class WarpJoystickCurriculum(WarpJoystick):
    """Joystick velocity tracking on a procedurally generated terrain grid.

    At init, the terrain MJCF is injected into a scene template and the
    composed XML is written to a PID-scoped file.  The grandparent
    ``Go2WarpEnv.__init__`` is called directly to bypass WarpJoystick's
    hardcoded flat-scene path.
    """

    def __init__(
        self,
        task: str = "flat_terrain",
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        if config is None:
            config = default_config()

        # Generate terrain MJCF fragment + spawn origins.
        seed = int(getattr(config, "terrain_seed", 0))
        gen = TerrainGenerator(GO2_DEFAULT_CFG)
        terrain_xml, origins = gen.generate(seed=seed)

        # Compose: inject terrain fragment into scene template.
        scene = _TEMPLATE_PATH.read_text()
        scene = scene.replace("<!-- TERRAIN_INJECT_POINT -->", terrain_xml)

        # Write to PID-scoped file to avoid multiprocess collisions.
        scene_path = _TEMPLATE_PATH.parent / f"_generated_curriculum_scene_{os.getpid()}.xml"
        scene_path.write_text(scene)

        # Call grandparent directly — WarpJoystick.__init__ hardcodes the flat
        # scene path, so we skip it and go straight to Go2WarpEnv.__init__.
        go2_warp_base.Go2WarpEnv.__init__(
            self,
            xml_path=str(scene_path),
            config=config,
            config_overrides=config_overrides,
        )
        # _post_init sets _init_q, joint limits, obs spec, reward spec, etc.
        self._post_init()

        # Terrain grid metadata for curriculum logic (Tasks 2.3+).
        self._terrain_origins = jp.array(origins)   # shape (10, 4, 3)
        self._num_rows = GO2_DEFAULT_CFG.num_rows    # 10
        self._num_cols = GO2_DEFAULT_CFG.num_cols    # 4
        self._tile_size = GO2_DEFAULT_CFG.tile_size  # (9.6, 9.6)

    # ── Task 2.3: spawn + goal sampling ─────────────────────────────────────

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, type_rng, level_rng, spawn_rng, yaw_rng = jax.random.split(rng, 5)

        # Random terrain_type for standalone reset (wrapper will override for curriculum)
        terrain_type = jax.random.randint(type_rng, (), 0, self._num_cols)
        terrain_level = jp.int32(0)

        # Sample spawn + goal in tile-local frame
        spawn_local, goal_local, spawn_yaw = self._sample_spawn_goal(
            terrain_type, self._tile_size, spawn_rng, yaw_rng
        )

        # Translate tile-local to world frame using origin for (level=0, type)
        tile_origin = self._terrain_origins[terrain_level, terrain_type]
        spawn_world_xy = spawn_local[:2] + tile_origin[:2]
        spawn_world_z = tile_origin[2] + spawn_local[2]
        goal_world_xy = goal_local[:2] + tile_origin[:2]

        # Base reset (initialises qpos from _init_q, sets up info dict)
        state = super().reset(rng)

        # Overwrite spawn pose
        new_qpos = state.data.qpos.at[0].set(spawn_world_xy[0])
        new_qpos = new_qpos.at[1].set(spawn_world_xy[1])
        new_qpos = new_qpos.at[2].set(spawn_world_z)
        # Yaw quaternion (wxyz order): pure rotation about z-axis
        qw = jp.cos(spawn_yaw / 2.0)
        qz = jp.sin(spawn_yaw / 2.0)
        new_qpos = new_qpos.at[3].set(qw)
        new_qpos = new_qpos.at[4].set(jp.float32(0.0))
        new_qpos = new_qpos.at[5].set(jp.float32(0.0))
        new_qpos = new_qpos.at[6].set(qz)
        state = state.replace(data=state.data.replace(qpos=new_qpos))

        # Curriculum bookkeeping
        target_speed = 0.5 + terrain_level.astype(jp.float32) / (self._num_rows - 1) * 1.0
        initial_distance = jp.linalg.norm(spawn_world_xy - goal_world_xy)

        state.info["terrain_level"] = terrain_level
        state.info["terrain_type"] = terrain_type
        state.info["goal_xy"] = goal_world_xy
        state.info["initial_distance"] = initial_distance
        state.info["episode_reached_goal"] = jp.bool_(False)
        state.info["episode_min_distance"] = initial_distance
        state.info["episode_fallen"] = jp.bool_(False)
        state.info["target_speed"] = target_speed

        return state

    def _sample_spawn_goal(self, terrain_type, tile_size, spawn_rng, yaw_rng):
        """Dispatch on terrain_type via lax.switch. Returns (spawn_local, goal_local, yaw)."""
        branches = [
            lambda r: self._edge_to_edge(r, yaw_rng, tile_size),   # col 0: Rough
            lambda r: self._rim_to_center(r, yaw_rng, tile_size),  # col 1: PyramidStairs
            lambda r: self._rim_to_center(r, yaw_rng, tile_size),  # col 2: InvertedPyramid
            lambda r: self._edge_to_edge(r, yaw_rng, tile_size),   # col 3: TiltedGrid
        ]
        return jax.lax.switch(terrain_type, branches, spawn_rng)

    def _edge_to_edge(self, rng, yaw_rng, tile_size):
        """Spawn on one edge, goal on the opposite edge. Random axis + direction."""
        axis_rng, side_rng, offset_rng = jax.random.split(rng, 3)
        hx = tile_size[0] / 2.0  # static Python float — ok, not traced
        hy = tile_size[1] / 2.0
        axis = jax.random.randint(axis_rng, (), 0, 2)   # 0=x-traverse, 1=y-traverse
        side = jax.random.randint(side_rng, (), 0, 2) * 2 - 1   # -1 or +1
        offset = jax.random.uniform(offset_rng, (), minval=-0.4, maxval=0.4)
        spawn_axis_val = side.astype(jp.float32) * hx * 0.9
        goal_axis_val = -side.astype(jp.float32) * hx * 0.9
        off = hy * offset
        spawn_x = jp.where(axis == 0, spawn_axis_val, off)
        spawn_y = jp.where(axis == 0, off, spawn_axis_val)
        goal_x = jp.where(axis == 0, goal_axis_val, off)
        goal_y = jp.where(axis == 0, off, goal_axis_val)
        spawn = jp.array([spawn_x, spawn_y, jp.float32(0.3)])
        goal = jp.array([goal_x, goal_y, jp.float32(0.0)])
        yaw = jax.random.uniform(yaw_rng, (), minval=-jp.pi, maxval=jp.pi)
        return spawn, goal, yaw

    def _rim_to_center(self, rng, yaw_rng, tile_size):
        """Spawn on rim (random angle), goal at tile center."""
        angle = jax.random.uniform(rng, (), minval=0.0, maxval=2.0 * jp.pi)
        r = min(tile_size[0], tile_size[1]) / 2.0 * 0.9  # static Python — ok
        spawn_x = r * jp.cos(angle)
        spawn_y = r * jp.sin(angle)
        spawn = jp.array([spawn_x, spawn_y, jp.float32(0.3)])
        goal = jp.array([jp.float32(0.0), jp.float32(0.0), jp.float32(0.0)])
        yaw = jax.random.uniform(yaw_rng, (), minval=-jp.pi, maxval=jp.pi)
        return spawn, goal, yaw

    # ── Task 2.4: goal-directed step ────────────────────────────────────────

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # Build goal-directed command in body frame
        goal_cmd = self._goal_directed_command(state)

        # Inject before super so reward terms (tracking_lin_vel, etc.) use it
        state.info["command"] = goal_cmd

        # Physics step — NOTE: super().step() overwrites info["command"] at the
        # end when steps_until_next_cmd <= 0.  We re-apply our command after.
        state = super().step(state, action)

        # Re-apply goal-directed command so next step starts from it, not random
        state.info["command"] = self._goal_directed_command(state)

        # Update episode tracking flags
        dist = jp.linalg.norm(state.data.qpos[:2] - state.info["goal_xy"])
        state.info["episode_min_distance"] = jp.minimum(
            state.info["episode_min_distance"], dist
        )
        state.info["episode_reached_goal"] = state.info["episode_reached_goal"] | (
            dist < jp.float32(0.5)
        )
        state.info["episode_fallen"] = state.done.astype(jp.bool_)

        return state

    def _goal_directed_command(self, state: mjx_env.State) -> jax.Array:
        """Compute [vx, vy=0, yaw_rate] in body frame pointing toward goal_xy."""
        robot_xy = state.data.qpos[:2]
        qw = state.data.qpos[3]
        qx = state.data.qpos[4]
        qy = state.data.qpos[5]
        qz = state.data.qpos[6]
        robot_yaw = jp.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        dx = state.info["goal_xy"][0] - robot_xy[0]
        dy = state.info["goal_xy"][1] - robot_xy[1]
        heading_world = jp.arctan2(dy, dx)
        yaw_error = heading_world - robot_yaw
        # Wrap to [-pi, pi]
        yaw_error = jp.mod(yaw_error + jp.pi, 2.0 * jp.pi) - jp.pi
        cmd_vx = state.info["target_speed"]
        cmd_vy = jp.float32(0.0)
        cmd_yaw_rate = jp.clip(2.0 * yaw_error, -1.5, 1.5)
        return jp.array([cmd_vx, cmd_vy, cmd_yaw_rate])
