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
from mujoco import mjx
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
    # Stronger action-rate penalty to discourage spiky actions on terrain.
    # Video analysis (2026-04-20) showed action saturation + velocity overshoot
    # preceding flips on pyramid_up L1 / pyramid_down L3. 5× stronger penalty
    # (was -0.01 from WarpJoystick default) encourages smoother gait.
    cfg.reward_config.scales.action_rate = -0.05
    # Weaker orientation penalty on terrain. Reward analysis (2026-04-21)
    # showed orient cost scales 190× from flat to tilted L2 (-0.01 → -1.71),
    # because upvector_z penalizes body-z misalignment with world-z — robot
    # upright on a tilted tile looks "wrong" to this cost. Reduce 5× so
    # terrain-induced tilt isn't catastrophically punished. Flipped detection
    # still handled via termination (upvector_z < 0).
    cfg.reward_config.scales.orientation = -1.0
    # Stronger base_height pressure. v11 policy transfer probe showed robot
    # crouching to ~0.20m (vs target 0.27m) — low stance saves balance on
    # terrain, tracking reward stays high, and the old weight -5 gave only
    # 0.024/step cost at crouch depth. Policy rationally ignored. Bump to
    # -20 to make stance height matter. Cost is overridden to feet-relative
    # (see _cost_base_height override below) to work on elevated terrain.
    cfg.reward_config.scales.base_height = -20.0
    # Drop world-frame feet_clearance + feet_height costs. These reward
    # foot elevation relative to WORLD z=0 (target 0.1m). On elevated
    # terrain (pyramid rings at z=0.3+), foot_z is already high while
    # grounded — penalty fires during normal stair walking. legged_gym
    # doesn't use these; feet_air_time alone handles gait shaping.
    cfg.reward_config.scales.feet_clearance = 0.0
    cfg.reward_config.scales.feet_height = 0.0
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

        # Base-contact sensor for terrain-agnostic fall detection.
        base_contact_sid = self._mj_model.sensor("base_contact").id
        self._base_contact_sensor_adr = int(self._mj_model.sensor_adr[base_contact_sid])

    # ── Task 2.3: spawn + goal sampling ─────────────────────────────────────

    # Per-episode probability of forcing zero linvel command for Class A
    # (rough/tilted). Gives robot full stand-still episodes as DR — otherwise
    # Bernoulli sampler rarely emits sustained low-speed. Wrapper resamples
    # this flag on each done; standalone reset samples fresh per-call.
    _ZERO_LINVEL_PROB = 0.15
    # Per-episode zero yaw_rate command. Higher prob when linvel already zero
    # (true stand-still episodes) vs when linvel is active (drive straight DR).
    _ZERO_YAW_PROB_IF_ZERO_LINVEL = 0.5
    _ZERO_YAW_PROB_OTHERWISE = 0.15

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, type_rng, level_rng, spawn_rng, yaw_rng, zero_rng, zero_yaw_rng = jax.random.split(rng, 7)

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
        target_speed = 0.5 + terrain_level.astype(jp.float32) / (self._num_rows - 1) * 0.5
        initial_distance = jp.linalg.norm(spawn_world_xy - goal_world_xy)

        is_goal = jp.asarray(self._IS_GOAL_DIRECTED, dtype=jp.bool_)[terrain_type]
        state.info["terrain_level"] = terrain_level
        state.info["terrain_type"] = terrain_type
        state.info["is_goal_directed"] = is_goal
        state.info["goal_xy"] = goal_world_xy
        state.info["spawn_xy"] = spawn_world_xy
        state.info["initial_distance"] = initial_distance
        state.info["episode_reached_goal"] = jp.bool_(False)
        state.info["episode_min_distance"] = initial_distance
        state.info["episode_fallen"] = jp.bool_(False)
        state.info["target_speed"] = target_speed
        force_zero_linvel = (
            jax.random.uniform(zero_rng, ()) < self._ZERO_LINVEL_PROB
        )
        yaw_prob = jp.where(
            force_zero_linvel,
            jp.float32(self._ZERO_YAW_PROB_IF_ZERO_LINVEL),
            jp.float32(self._ZERO_YAW_PROB_OTHERWISE),
        )
        force_zero_yaw = jax.random.uniform(zero_yaw_rng, ()) < yaw_prob
        state.info["force_zero_linvel"] = force_zero_linvel
        state.info["force_zero_yaw"] = force_zero_yaw

        return state

    # All 4 types are goal-directed: spawn on rim, walk to tile center.
    # Holonomic body-frame cmd; random yaw (parent's Bernoulli) + goal direction
    # together produce omnidirectional linvel DR in body frame — as yaw rotates,
    # the goal-directed cmd_vx/vy rotate through all body-frame directions.
    _IS_GOAL_DIRECTED = (True, True, True, True)

    def _sample_spawn_goal(self, terrain_type, tile_size, spawn_rng, yaw_rng):
        """Rim-to-center spawn for all terrain types. Returns (spawn_local, goal_local, yaw).

        Pyramids (types 1, 2) use face-toward-goal yaw — robot is not designed
        for sideways stair climbing, spawn random yaw makes success too rare.
        Rough (0) and tilted (3) keep random yaw for omnidirectional linvel DR
        (goal-directed holonomic cmd rotates through body frame as yaw spins).
        """
        spawn, goal, random_yaw = self._rim_to_center(spawn_rng, yaw_rng, tile_size)
        # For pyramids: override yaw to point at goal (world-frame atan2 on
        # local coords, since spawn/goal are tile-local and tile has no rot).
        goal_yaw = jp.arctan2(goal[1] - spawn[1], goal[0] - spawn[0])
        is_pyramid = (terrain_type == 1) | (terrain_type == 2)
        yaw = jp.where(is_pyramid, goal_yaw, random_yaw)
        return spawn, goal, yaw

    def _rim_to_center(self, rng, yaw_rng, tile_size):
        """Spawn on outer TILE EDGE (not rim-circle), goal at tile center.

        Pyramid/bowl rings are rectangular strips — a rim-circle at r≈4.3m lands
        ON TOP of ring-1 strips at diagonals (0.22m elevated at L5+), causing
        spawn clipping. Sampling on the axis-aligned outer edge guarantees
        spawn lands on ring 0 at z=0 regardless of angle.
        """
        axis_rng, side_rng, offset_rng = jax.random.split(rng, 3)
        hx = tile_size[0] / 2.0
        hy = tile_size[1] / 2.0
        axis = jax.random.randint(axis_rng, (), 0, 2)          # 0 = +/-x edge, 1 = +/-y edge
        side = jax.random.randint(side_rng, (), 0, 2) * 2 - 1  # -1 or +1
        # Offset along the chosen edge, kept well inside ring 0 (|other| < 0.7 * hy
        # to avoid overlap corners where ring 0 strip meets ring 1).
        offset = jax.random.uniform(offset_rng, (), minval=-0.7, maxval=0.7)
        edge_val = side.astype(jp.float32) * hx * 0.95
        off_val = hy * offset
        spawn_x = jp.where(axis == 0, edge_val, off_val)
        spawn_y = jp.where(axis == 0, off_val, edge_val)
        spawn = jp.array([spawn_x, spawn_y, jp.float32(0.3)])
        goal = jp.array([jp.float32(0.0), jp.float32(0.0), jp.float32(0.0)])
        yaw = jax.random.uniform(yaw_rng, (), minval=-jp.pi, maxval=jp.pi)
        return spawn, goal, yaw

    # ── Task 2.4: goal-directed step ────────────────────────────────────────

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # Unified (all types goal-directed): override vx, vy with holonomic
        # body-frame cmd toward tile center. yaw_rate stays from parent's
        # Bernoulli sampler (random rotation = body-frame linvel DR). Both
        # linvel and yaw have per-episode zero-cmd DR flags.
        #
        # After reach (episode_reached_goal=True, stays True for rest of
        # episode via OR accumulation), switch linvel cmd to zero. Robot
        # stands still at goal with random yaw_rate (parent's Bernoulli) —
        # naturally teaches "stop at target" and exercises yaw while
        # stationary. Episode continues to truncation, not cut on reach.
        force_zero = state.info["force_zero_linvel"]
        force_zero_yaw = state.info["force_zero_yaw"]

        def _override(cmd):
            # Read reached from current state.info (may have flipped in super.step)
            reached = state.info["episode_reached_goal"]
            zero_linvel = force_zero | reached
            body_vx, body_vy = self._goal_linvel_body(state)
            new_vx = jp.where(zero_linvel, jp.float32(0.0), body_vx)
            new_vy = jp.where(zero_linvel, jp.float32(0.0), body_vy)
            new_yaw = jp.where(force_zero_yaw, jp.float32(0.0), cmd[2])
            return cmd.at[0].set(new_vx).at[1].set(new_vy).at[2].set(new_yaw)

        state.info["command"] = _override(state.info["command"])

        state = super().step(state, action)

        # Re-apply post-step (parent's sampler may have clobbered command).
        state.info["command"] = _override(state.info["command"])

        # Episode tracking: reach (within 0.5m of goal) + min distance.
        robot_xy = state.data.qpos[:2]
        dist_to_goal = jp.linalg.norm(robot_xy - state.info["goal_xy"])

        state.info["episode_min_distance"] = jp.minimum(
            state.info["episode_min_distance"], dist_to_goal
        )
        state.info["episode_reached_goal"] = state.info["episode_reached_goal"] | (
            dist_to_goal < jp.float32(0.5)
        )
        state.info["episode_fallen"] = state.done.astype(jp.bool_)

        return state

    # ── Terrain-aware reward overrides ─────────────────────────────────────
    # Parent's _cost_orientation already uses sum(upvector[:2]²) which equals
    # sin²(tilt_from_world_up) — same as legged_gym's projected_gravity form.
    # Only the WEIGHT was misconfigured for terrain (dampened in default_config).
    # Parent's _cost_base_height is world-z subtree_com; that DOES break on
    # elevated/depressed terrain. Override below.

    def _cost_base_height(self, data: mjx.Data) -> jax.Array:
        """Override: feet-relative base height (terrain-invariant).

        Parent uses world-z subtree_com; false-penalizes robot in pyramid pit
        or on elevated stair. We measure torso COM height ABOVE median foot
        z — the stance height we actually care about. Target: 0.27m (same
        as parent's target, since robot kinematics haven't changed).
        """
        torso_z = data.subtree_com[self._torso_body_id][2]
        feet_z = data.site_xpos[self._feet_site_id][..., 2]
        ground_z = jp.median(feet_z)
        stance_height = torso_z - ground_z
        return jp.square(stance_height - 0.27)

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        """Terrain-agnostic termination.

        Replaces parent's `base_z < 0.18` check (which breaks on uneven
        terrain — robot standing in a bowl has low world-z but isn't fallen).
        Uses torso-ground contact sensor instead: any contact between
        base_link and any other geom = fall. Orientation check unchanged —
        full flip still terminates. Lying on one side (upvector ~0, legs
        propped) does NOT terminate — policy may still recover.
        """
        flipped = self.get_upvector(data)[-1] < 0.0
        base_contact = data.sensordata[self._base_contact_sensor_adr] > 0.0
        return flipped | base_contact

    def _goal_linvel_body(self, state: mjx_env.State) -> tuple:
        """Body-frame (vx, vy) derived from world-frame goal direction.

        Robot can walk toward goal by any gait — forward, sideways, diagonal —
        without having to turn first. yaw_rate is left untouched so the parent's
        Bernoulli sampler continues to drive DR-style random turning.
        """
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
        dist = jp.sqrt(dx * dx + dy * dy) + 1e-6
        # Unit vector toward goal in world frame, scaled to target_speed
        speed = state.info["target_speed"]
        world_vx = dx / dist * speed
        world_vy = dy / dist * speed
        # Rotate into body frame via yaw
        c = jp.cos(robot_yaw)
        s = jp.sin(robot_yaw)
        body_vx = c * world_vx + s * world_vy
        body_vy = -s * world_vx + c * world_vy
        return body_vx, body_vy
