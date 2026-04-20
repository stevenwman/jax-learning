"""Terrain Curriculum DR wrapper — extends DomainRandWrapper with per-env
curriculum advancement for terrain-based locomotion training.

After DomainRandWrapper's where_done merges reset/step states, this wrapper
overrides terrain-related fields for done envs:
  - terrain_type: preserved (fixed per env for all of training)
  - terrain_level: advanced/demoted based on episode outcome
  - goal_xy, spawn position: resampled for the new level
  - episode tracking flags: reset for the new episode
"""

from typing import Any, Literal, Optional

import jax
from jax import numpy as jp
from mujoco_playground._src import mjx_env

from jax_rl.envs.wrappers.domain_rand import DomainRandWrapper


class TerrainCurriculumDRWrapper(DomainRandWrapper):
    """DomainRandWrapper + per-env terrain curriculum advancement.

    Each env is assigned a fixed terrain_type (column) at init.  After each
    episode boundary (detected via DomainRandWrapper's done signal), the
    wrapper computes level promotion/demotion and resamples spawn+goal for
    the new (level, type) tile.

    Args:
        env: Unwrapped environment (must be a WarpJoystickCurriculum or similar
             with _terrain_origins, _num_rows, _num_cols, _sample_spawn_goal).
        episode_length: Max steps per episode.
        mode: Reset mode (only "per_step" supported).
        num_envs: Number of parallel environments.
    """

    _TC_KEY = '_tc'

    def __init__(
        self,
        env: Any,
        episode_length: int = 1000,
        mode: Literal["per_step"] = "per_step",
        num_envs: int = 1,
    ):
        super().__init__(env, episode_length=episode_length, mode=mode)
        self._num_envs = num_envs

        # Traverse wrapper chain to find the curriculum base env.
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        self._terrain_origins = base_env._terrain_origins  # (num_rows, num_cols, 3)
        self._num_rows = base_env._num_rows
        self._num_cols = base_env._num_cols
        self._base_env = base_env

    # ── Reset ───────────────────────────────────────────────────────

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Split off a TC-specific rng before super consumes it
        rng_split = jax.vmap(jax.random.split)(rng)
        tc_init_rng = rng_split[:, 0]
        super_rng = rng_split[:, 1]

        state = super().reset(super_rng)

        # Fixed terrain_type per env: env_id % num_cols
        env_ids = jp.arange(self._num_envs)
        fixed_types = env_ids % self._num_cols
        initial_levels = jp.zeros(self._num_envs, dtype=jp.int32)

        # Per-env RNG for spawn sampling
        tc_rng_split = jax.vmap(lambda k: jax.random.split(k, 3))(tc_init_rng)
        spawn_rngs = tc_rng_split[:, 0]
        yaw_rngs = tc_rng_split[:, 1]
        next_tc_rng = tc_rng_split[:, 2]

        # Vmap spawn+goal sampling
        spawn_xys, spawn_zs, goal_xys, yaws = jax.vmap(
            self._sample_for_env
        )(spawn_rngs, yaw_rngs, fixed_types, initial_levels)

        # Override qpos with correct spawn positions
        qpos = state.data.qpos
        qpos = qpos.at[:, 0].set(spawn_xys[:, 0])
        qpos = qpos.at[:, 1].set(spawn_xys[:, 1])
        qpos = qpos.at[:, 2].set(spawn_zs)
        qpos = qpos.at[:, 3].set(jp.cos(yaws / 2.0))
        qpos = qpos.at[:, 4].set(0.0)
        qpos = qpos.at[:, 5].set(0.0)
        qpos = qpos.at[:, 6].set(jp.sin(yaws / 2.0))
        state = state.replace(data=state.data.replace(qpos=qpos))

        # Override info fields
        initial_dist = jp.linalg.norm(spawn_xys - goal_xys, axis=-1)
        target_speed = 0.5 + initial_levels.astype(jp.float32) / (self._num_rows - 1) * 1.0

        _IS_GOAL = jp.asarray([False, True, True, False])
        state.info["terrain_type"] = fixed_types
        state.info["terrain_level"] = initial_levels
        state.info["is_goal_directed"] = _IS_GOAL[fixed_types]
        state.info["goal_xy"] = goal_xys
        state.info["spawn_xy"] = spawn_xys
        state.info["initial_distance"] = initial_dist
        state.info["episode_reached_goal"] = jp.zeros(self._num_envs, dtype=jp.bool_)
        state.info["episode_min_distance"] = initial_dist
        state.info["episode_max_dist_from_spawn"] = jp.zeros(self._num_envs, dtype=jp.float32)
        state.info["episode_fallen"] = jp.zeros(self._num_envs, dtype=jp.bool_)
        state.info["episode_tracking_error_sum"] = jp.zeros(self._num_envs, dtype=jp.float32)
        state.info["episode_step_count"] = jp.zeros(self._num_envs, dtype=jp.int32)
        state.info["target_speed"] = target_speed

        # Note: force_zero_linvel is already sampled per-env by the inner env's
        # reset (vmapped by DomainRandWrapper.reset). No wrapper-level override
        # needed — where_done merging on step will propagate fresh samples from
        # reset_state into done envs automatically.

        # Advancement flags (initialized to False; populated per step)
        state.info["episode_promoted"] = jp.zeros(self._num_envs, dtype=jp.bool_)
        state.info["episode_demoted"] = jp.zeros(self._num_envs, dtype=jp.bool_)

        # Curriculum wrapper state (separate key prefix from _dr)
        state.info[f'{self._TC_KEY}_rng'] = next_tc_rng

        return state

    # ── Step ────────────────────────────────────────────────────────

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # Save prev curriculum state BEFORE super().step() (which does where_done)
        prev_level = state.info["terrain_level"]
        prev_type = state.info["terrain_type"]
        prev_reached = state.info["episode_reached_goal"]
        prev_fallen = state.info["episode_fallen"]
        prev_min_dist = state.info["episode_min_distance"]
        prev_initial_dist = state.info["initial_distance"]
        prev_max_dist_from_spawn = state.info["episode_max_dist_from_spawn"]
        prev_tracking_err_sum = state.info["episode_tracking_error_sum"]
        prev_step_count = state.info["episode_step_count"]

        # Save and strip TC wrapper state so tree_map in super doesn't see it
        # (reset_state from inner env won't have these keys → tree mismatch)
        tc_rng = state.info.pop(f'{self._TC_KEY}_rng')
        state.info.pop("episode_promoted", None)
        state.info.pop("episode_demoted", None)

        # DomainRandWrapper.step: strips _dr state, resets+steps all envs,
        # computes done, where_done merges, reattaches _dr state
        state = super().step(state, action)

        # Get done signal from DomainRandWrapper
        done = state.info[f'{self._KEY}_episode_done']

        # Fall detection: prev_fallen (state.info["episode_fallen"]) is wiped
        # to False by where_done before we can read it next step. Infer fall
        # from the preserved truncation flag instead:
        #   done=1, truncation=1 → ended via timeout (no fall)
        #   done=1, truncation=0 → ended via env termination (= fall)
        truncation = state.info["truncation"]
        fall_at_done = (done > 0) & (truncation < 0.5)

        # Dual-class advancement:
        # Class B (goal-directed, pyramid/inv): reached → promote; fall OR
        #   no-progress-to-goal → demote
        # Class A (locomotion robustness, rough/tilted): survived + moved ≥ 2m
        #   → promote; fell OR didn't move → demote
        _IS_GOAL = jp.asarray([False, True, True, False])
        is_goal = _IS_GOAL[prev_type]

        # Class B logic
        reached = prev_reached
        no_progress_B = (~reached) & (~fall_at_done) & (prev_min_dist > 0.5 * prev_initial_dist)
        promote_B = reached & (~fall_at_done)
        demote_B = fall_at_done | no_progress_B

        # Class A logic: locomotion robustness = tracked cmd velocity well on
        # uneven/tilted ground. Uses body-frame linvel error averaged over
        # the episode. 0.3 m/s ≈ 25% of typical cmd magnitude = "decent track".
        mean_tracking_err = prev_tracking_err_sum / jp.maximum(
            prev_step_count.astype(jp.float32), jp.float32(1.0)
        )
        tracked_well = mean_tracking_err < jp.float32(0.3)
        promote_A = (~fall_at_done) & tracked_well
        demote_A = fall_at_done | (~tracked_well)

        promote = jp.where(is_goal, promote_B, promote_A) & (done > 0)
        demote = jp.where(is_goal, demote_B, demote_A) & (done > 0)

        delta = promote.astype(jp.int32) - demote.astype(jp.int32)
        new_level = jp.clip(prev_level + delta, 0, self._num_rows - 1)

        # RNG for spawn sampling
        tc_rng_split = jax.vmap(lambda k: jax.random.split(k, 3))(tc_rng)
        spawn_rngs = tc_rng_split[:, 0]
        yaw_rngs = tc_rng_split[:, 1]
        next_tc_rng = tc_rng_split[:, 2]

        # Resample spawn+goal for ALL envs (JAX both-paths), select for done only
        new_spawn_xys, new_spawn_zs, new_goal_xys, new_yaws = jax.vmap(
            self._sample_for_env
        )(spawn_rngs, yaw_rngs, prev_type, new_level)

        # Override qpos for done envs only (teleport to new spawn)
        done_bool = done > 0
        qpos = state.data.qpos
        qpos = qpos.at[:, 0].set(jp.where(done_bool, new_spawn_xys[:, 0], qpos[:, 0]))
        qpos = qpos.at[:, 1].set(jp.where(done_bool, new_spawn_xys[:, 1], qpos[:, 1]))
        qpos = qpos.at[:, 2].set(jp.where(done_bool, new_spawn_zs, qpos[:, 2]))
        qpos = qpos.at[:, 3].set(jp.where(done_bool, jp.cos(new_yaws / 2.0), qpos[:, 3]))
        qpos = qpos.at[:, 4].set(jp.where(done_bool, 0.0, qpos[:, 4]))
        qpos = qpos.at[:, 5].set(jp.where(done_bool, 0.0, qpos[:, 5]))
        qpos = qpos.at[:, 6].set(jp.where(done_bool, jp.sin(new_yaws / 2.0), qpos[:, 6]))
        state = state.replace(data=state.data.replace(qpos=qpos))

        # Override info fields
        new_initial_dist = jp.linalg.norm(new_spawn_xys - new_goal_xys, axis=-1)
        new_speed = 0.5 + new_level.astype(jp.float32) / (self._num_rows - 1) * 1.0

        state.info["terrain_level"] = jp.where(done_bool, new_level, prev_level)
        state.info["terrain_type"] = prev_type  # always preserved
        state.info["goal_xy"] = jp.where(
            done_bool[:, None], new_goal_xys, state.info["goal_xy"]
        )
        state.info["spawn_xy"] = jp.where(
            done_bool[:, None], new_spawn_xys, state.info["spawn_xy"]
        )
        state.info["initial_distance"] = jp.where(
            done_bool, new_initial_dist, state.info["initial_distance"]
        )
        state.info["episode_reached_goal"] = jp.where(
            done_bool, jp.bool_(False), state.info["episode_reached_goal"]
        )
        state.info["episode_min_distance"] = jp.where(
            done_bool, new_initial_dist, state.info["episode_min_distance"]
        )
        state.info["episode_max_dist_from_spawn"] = jp.where(
            done_bool, jp.float32(0.0), state.info["episode_max_dist_from_spawn"]
        )
        state.info["episode_fallen"] = jp.where(
            done_bool, jp.bool_(False), state.info["episode_fallen"]
        )
        state.info["episode_tracking_error_sum"] = jp.where(
            done_bool, jp.float32(0.0), state.info["episode_tracking_error_sum"]
        )
        state.info["episode_step_count"] = jp.where(
            done_bool, jp.int32(0), state.info["episode_step_count"]
        )
        state.info["target_speed"] = jp.where(
            done_bool, new_speed, state.info["target_speed"]
        )

        # Advancement flags for wandb logging
        state.info["episode_promoted"] = promote
        state.info["episode_demoted"] = demote

        # Re-attach TC wrapper state
        state.info[f'{self._TC_KEY}_rng'] = next_tc_rng

        return state

    # ── Helpers ──────────────────────────────────────────────────────

    def _sample_for_env(self, spawn_rng, yaw_rng, terrain_type, terrain_level):
        """Sample spawn+goal in world frame for a single env.

        Designed to be vmapped over the env batch dimension.

        Returns:
            spawn_xy: (2,) world-frame spawn xy
            spawn_z: scalar world-frame spawn z
            goal_xy: (2,) world-frame goal xy
            yaw: scalar spawn yaw
        """
        spawn_local, goal_local, yaw = self._base_env._sample_spawn_goal(
            terrain_type, self._base_env._tile_size, spawn_rng, yaw_rng
        )
        tile_origin = self._terrain_origins[terrain_level, terrain_type]
        spawn_world_xy = spawn_local[:2] + tile_origin[:2]
        spawn_world_z = tile_origin[2] + spawn_local[2]
        goal_world_xy = goal_local[:2] + tile_origin[:2]
        return spawn_world_xy, spawn_world_z, goal_world_xy, yaw
