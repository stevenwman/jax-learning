"""Flat-ground Go2 with delta-to-target obs + Lorentzian pos_track reward.

Prototype env (2026-05-12). Designed to share parameterization with the
splitbelt PosTrack env so cross-deploy is direct (same actor obs schema,
same reward formula, only diff = belts vs no belts).

API summary:
  - Obs (state, 45d):
      gyro (3), gravity (3), joint_pos_offset (12), joint_vel (12),
      last_act (12), delta_xy_yaw_body_frame (3)
  - NO `command` velocity input. Goal is encoded purely via delta-from-target.
  - Target trajectory: target_xy_world advances at nominal_vx forward (in
    target_yaw direction); target_yaw stays at init yaw (no yaw rate yet).
  - nominal_vx randomized per episode in `nominal_vx_range`.

Reward:
  - `pos_track_xy`: Lorentzian on |body-frame delta_xy|. Heavy-tail so
    gradient survives large lag.
  - `orient_track_yaw`: cos(d_yaw) — 1 when facing target heading.
  - + standard joystick regularizers (action_rate, orientation, base_height,
    feet_*, pose, etc.). No tracking_lin_vel, no stand_still.

Cross-deploy target: matching splitbelt env at eval time (target stationary,
belts moving). To make obs schema match, splitbelt env must also produce
the same `delta_xy_yaw_body_frame` term (TODO in splitbelt env).
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from jax_rl.envs.locomotion import go2_warp_joystick as base
from jax_rl.envs.obs_spec import ObsTerm, IncludeGroup
from jax_rl.envs.reward_spec import RewardTerm


def default_config() -> config_dict.ConfigDict:
    """Prototype config — minimal flat-ground PosTrack."""
    cfg = base.default_config()
    cfg.unlock()

    # Drop velocity-tracking + stand-still; replaced by pos_track terms below.
    cfg.reward_config.scales.tracking_lin_vel = 0.0
    cfg.reward_config.scales.tracking_ang_vel = 0.0  # yaw tracked via orient_track_yaw
    cfg.reward_config.scales.stand_still = 0.0

    # New pos-track terms (defaults; weights set on first preset).
    cfg.reward_config.scales.pos_track_xy = 10.0
    cfg.reward_config.scales.orient_track_yaw = 5.0
    cfg.reward_config.pos_track_lx = 0.5  # forward Lorentzian scale
    cfg.reward_config.pos_track_ly = 0.3  # lateral

    # Target trajectory params. nominal_vx sampled per episode. Range aligned
    # with splitbelt env's belt v_range so the speed-distribution training
    # axis matches for clean apples-to-apples comparison.
    cfg.nominal_vx_range = (0.3, 1.5)
    cfg.nominal_yaw_rate = 0.0  # constant heading for now

    # No DR knobs here — keep prototype clean. Add later if needed.
    return cfg


class WarpFlatPosTrack(base.WarpJoystickNoAccel):
    """Flat-ground PosTrack prototype. Extends NoAccel joystick env, overrides
    obs (no `command` field, adds delta_xy_yaw), reward (pos_track + orient_track
    replace vel tracking), and reset/step (manage target_xy world-frame state).
    """

    def _post_init(self) -> None:
        # Parent sets joint/sensor IDs, default_pose, etc. We'll overwrite its
        # _obs_groups and _reward_spec below.
        super()._post_init()

        noise = self._config.noise_config.scales
        # Strip parent's `command` ObsTerm — replaced by delta_xy_yaw.
        # Parent's state group (after NoAccel strips accelerometer):
        #   gyro, gravity, joint_pos_offset, joint_vel, last_act, command
        # We drop `command`, append `delta_xy_yaw_body_frame`.
        state_terms = [
            t for t in self._obs_groups["state"]
            if not (hasattr(t, "name") and t.name == "command")
        ]
        state_terms.append(
            ObsTerm("delta_xy_yaw_body_frame",
                    lambda data, info, **kw: self._compute_delta_body_frame(data, info),
                    noise_scale=0.0)
        )
        self._obs_groups["state"] = state_terms

        # Privileged adds clean unrotated world-frame delta for the critic.
        priv_terms = [t for t in self._obs_groups["privileged_state"]]
        priv_terms.append(
            ObsTerm("delta_xy_world",
                    lambda data, info, **kw: data.qpos[:2] - info["target_xy_world"],
                    noise_scale=0.0)
        )
        priv_terms.append(
            ObsTerm("target_velocity_xy",
                    lambda info, **kw: info["target_velocity_xy"],
                    noise_scale=0.0)
        )
        self._obs_groups["privileged_state"] = priv_terms

        # Reward spec: KEEP all parent terms (tracking_lin_vel, tracking_ang_vel,
        # stand_still remain in spec but their weights are 0 in default_config).
        # This ensures the reward-components dict keys match the scales config
        # exactly under JIT (mismatch causes a pytree-structure ValueError at
        # the first scan iteration). Append the two new pos-track terms.
        self._reward_spec.append(RewardTerm(
            "pos_track_xy",
            lambda data, info, **kw: self._reward_pos_track_xy(data, info),
        ))
        self._reward_spec.append(RewardTerm(
            "orient_track_yaw",
            lambda data, info, **kw: self._reward_orient_track_yaw(data, info),
        ))

    # ── Geometry helpers ──────────────────────────────────────────────────

    def _body_yaw(self, qpos: jax.Array) -> jax.Array:
        """Extract yaw (rotation about z) from quaternion `qpos[3:7]`."""
        w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
        # Yaw = atan2(2(wz + xy), 1 - 2(y² + z²))
        return jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _compute_delta_body_frame(self, data, info) -> jax.Array:
        """Return (delta_forward, delta_lateral, d_yaw) in body frame.

        delta_forward = forward error (how far behind / ahead robot is).
        delta_lateral = signed lateral error (left/right of target heading).
        d_yaw = body_yaw - target_yaw (signed).
        """
        dxy_world = data.qpos[:2] - info["target_xy_world"]
        body_yaw = self._body_yaw(data.qpos)
        c, s = jp.cos(body_yaw), jp.sin(body_yaw)
        # Rotate world delta into body frame: R(-yaw) @ dxy_world
        delta_x_body = c * dxy_world[0] + s * dxy_world[1]
        delta_y_body = -s * dxy_world[0] + c * dxy_world[1]
        d_yaw = body_yaw - info["target_yaw_world"]
        # Wrap d_yaw to [-pi, pi]
        d_yaw = jp.fmod(d_yaw + jp.pi, 2.0 * jp.pi) - jp.pi
        return jp.array([delta_x_body, delta_y_body, d_yaw])

    # ── Reward methods ────────────────────────────────────────────────────

    def _reward_pos_track_xy(self, data, info) -> jax.Array:
        """Lorentzian on body-frame delta_xy magnitude."""
        delta = self._compute_delta_body_frame(data, info)
        Lx = self._config.reward_config.pos_track_lx
        Ly = self._config.reward_config.pos_track_ly
        return 1.0 / (1.0 + (delta[0] / Lx) ** 2 + (delta[1] / Ly) ** 2)

    def _reward_orient_track_yaw(self, data, info) -> jax.Array:
        """cos(d_yaw): 1 = facing target heading, 0 = perpendicular, -1 = backward.
        Clamp to [0, 1] so the reward floor is 0, not -1."""
        d_yaw = self._body_yaw(data.qpos) - info["target_yaw_world"]
        return jp.maximum(jp.cos(d_yaw), 0.0)

    # ── Reset / step (custom — don't call parent's reset/step) ────────────

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Build qpos/qvel from scratch. Fixed spawn: home keyframe, no world-frame
        # randomization (body-frame obs is rotation-invariant, randomizing yaw
        # or xy adds nothing to training signal). Keep small qvel jitter only
        # for recovery resilience.
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(
            jax.random.uniform(key, (6,), minval=-0.2, maxval=0.2)
        )

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos, qvel=qvel,
            ctrl=jp.zeros(self.mjx_model.nu),
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            naccdmax=self._config.naccdmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        # Sample nominal_vx for this episode.
        rng, k = jax.random.split(rng)
        vmin, vmax = self._config.nominal_vx_range
        nominal_vx = jax.random.uniform(k, (), minval=vmin, maxval=vmax)

        target_yaw_world = self._body_yaw(data.qpos)
        target_xy_world = data.qpos[:2]
        target_velocity_xy = jp.array([
            nominal_vx * jp.cos(target_yaw_world),
            nominal_vx * jp.sin(target_yaw_world),
        ])

        info = {
            "rng": rng,
            "command": jp.zeros(3),  # vestigial slot (parent base infra may read)
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "swing_peak": jp.zeros(4),
            "step_count": jp.int32(0),
            "nominal_vx": nominal_vx,
            "target_xy_world": target_xy_world,
            "target_yaw_world": target_yaw_world,
            "target_velocity_xy": target_velocity_xy,
            "reward_components": {
                k: jp.zeros(()) for k in self._config.reward_config.scales.keys()
            },
        }

        metrics = {f"reward/{k}": jp.zeros(())
                   for k in self._config.reward_config.scales.keys()}

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        motor_targets = self._default_pose + action * self._config.action_scale

        # Optional velocity-kick (port from parent for robustness curriculum).
        step_count = state.info["step_count"]
        push_interval = 350
        rng, push_key = jax.random.split(state.info["rng"])
        push_vel = jax.random.uniform(push_key, (2,), minval=-0.75, maxval=0.75)
        do_push = (step_count > 0) & (step_count % push_interval == 0)
        data = state.data
        new_qvel = data.qvel.at[0:2].set(
            jp.where(do_push, data.qvel[0:2] + push_vel, data.qvel[0:2])
        )
        data = data.replace(qvel=new_qvel)

        kp = self._kp
        kd = self._kd
        model = self.mjx_model
        a2j = self._act_to_joint

        def substep(data, _):
            current_q = data.qpos[7:]
            current_dq = data.qvel[6:]
            tau_joint = kp * (motor_targets - current_q) + kd * (0.0 - current_dq)
            tau_joint = self._apply_torque_speed_limit(tau_joint, current_dq)
            tau_act = tau_joint[a2j]
            return mjx.step(model, data.replace(ctrl=tau_act)), None

        data = jax.lax.scan(substep, data, (), self.n_substeps)[0]

        # Foot contact
        contact = jp.array([
            data.sensordata[self._mj_model.sensor_adr[sid]] > 0
            for sid in self._feet_floor_found_sensor
        ])
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_fz = data.site_xpos[self._feet_site_id, -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        # Advance target BEFORE computing reward + obs.
        new_target_xy = state.info["target_xy_world"] + state.info["target_velocity_xy"] * self.dt
        state.info["target_xy_world"] = new_target_xy

        done = self._get_termination(data)
        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact
        )
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
        state.info["reward_components"] = rewards

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["step_count"] = step_count + 1
        state.info["rng"] = rng
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact

        obs = self._get_obs(data, state.info)
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        done = done.astype(reward.dtype)
        return state.replace(data=data, obs=obs, reward=reward, done=done)
