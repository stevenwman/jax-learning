"""Joystick velocity tracking task for Unitree Go2.

Adapted from Playground's Go1 Joystick env. Matches Go1 config exactly
(Kp=35, action_scale=0.5, reward weights, obs structure) for training
parity. Will switch to Unitree official values (Kp=20, action_scale=0.25)
for sim-to-real.

Returns dict obs: {"state": 48d policy obs, "privileged_state": 122d critic obs}.
Self-contained env: training script never touches obs construction,
reward computation, frame stacking, or action scaling.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from jax_rl.envs.locomotion import go2_base
from jax_rl.envs.locomotion import go2_constants as consts
from jax_rl.envs.obs_spec import ObsTerm, IncludeGroup, compute_obs
from jax_rl.envs.reward_spec import RewardTerm, compute_rewards


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,
        # PD gains — Kp=35 from Playground Go1. action_scale=0.5 to match Go1
        # (0.3 was too restrictive — only 15% thigh range, 34% calf range).
        Kp=35.0,
        Kd=0.1,  # Go2 hardware value (unitree_mujoco). Was 0.5 from Go1 Playground.
        action_repeat=1,
        action_scale=0.5,
        soft_joint_pos_limit_factor=0.95,
        # Observation noise — match Playground Go1 Joystick exactly.
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable.
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
                linvel=0.1,
            ),
        ),
        # Reward terms — Go1 PG defaults. Double damping fix is the real change.
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Tracking.
                tracking_lin_vel=10.0,
                tracking_ang_vel=5.0,
                # Base stability.
                lin_vel_z=-0.5,
                ang_vel_xy=-0.05,
                orientation=-5.0,
                # Regularization.
                torques=-0.0002,
                action_rate=-0.01,
                energy=-0.001,
                # Joint limits.
                dof_pos_limits=-1.0,
                # Feet.
                feet_air_time=0.1,
                feet_slip=-0.1,
                feet_clearance=-2.0,
                feet_height=-0.2,
                # Other.
                termination=-1.0,
                stand_still=-1.0,
                pose=0.5,
            ),
            tracking_sigma=0.25,
            max_foot_height=0.1,
        ),
        command_config=config_dict.create(
            # Max command amplitudes: [vx, vy, yaw_rate].
            a=[1.5, 0.8, 1.2],
            # Probability of non-zero command per axis.
            b=[0.9, 0.25, 0.5],
        ),
        impl="jax",
        naconmax=4 * 8192,
        njmax=40,
    )


class Joystick(go2_base.Go2Env):
    """Track a joystick velocity command with Go2."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=consts.SCENE_FLAT_XML.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

        # Soft joint limits (first joint is freejoint).
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

        self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]  # UNUSED — kept for potential mass-normalized rewards

        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in consts.FEET_SITES]
        )
        self._floor_geom_id = self._mj_model.geom("floor").id  # UNUSED — contact detection uses sensors now
        self._feet_geom_id = np.array(  # UNUSED — contact detection uses sensors now
            [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
        )

        # Foot linear velocity sensor addresses.
        foot_linvel_sensor_adr = []
        for site in consts.FEET_SITES:
            name = site.replace("_foot", "") + "_global_linvel"
            sensor_id = self._mj_model.sensor(name).id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        self._cmd_a = jp.array(self._config.command_config.a)
        self._cmd_b = jp.array(self._config.command_config.b)

        noise = self._config.noise_config.scales
        self._obs_groups = {
            "state": [
                ObsTerm("linvel", lambda data, **kw: self.get_local_linvel(data),
                        noise_scale=noise.linvel),
                ObsTerm("gyro", lambda data, **kw: self.get_gyro(data),
                        noise_scale=noise.gyro),
                ObsTerm("gravity", lambda data, **kw: self.get_gravity(data),
                        noise_scale=noise.gravity),
                ObsTerm("joint_pos_offset", lambda data, **kw: data.qpos[7:] - self._default_pose,
                        noise_scale=noise.joint_pos),
                ObsTerm("joint_vel", lambda data, **kw: data.qvel[6:],
                        noise_scale=noise.joint_vel),
                ObsTerm("last_act", lambda info, **kw: info["last_act"]),
                ObsTerm("command", lambda info, **kw: info["command"]),
            ],
            "privileged_state": [
                IncludeGroup("state"),
                ObsTerm("gyro_clean", lambda data, **kw: self.get_gyro(data)),
                ObsTerm("accelerometer", lambda data, **kw: self.get_accelerometer(data)),
                ObsTerm("gravity_clean", lambda data, **kw: self.get_gravity(data)),
                ObsTerm("linvel_clean", lambda data, **kw: self.get_local_linvel(data)),
                ObsTerm("angvel", lambda data, **kw: self.get_global_angvel(data)),
                ObsTerm("joint_pos_clean", lambda data, **kw: data.qpos[7:] - self._default_pose),
                ObsTerm("joint_vel_clean", lambda data, **kw: data.qvel[6:]),
                ObsTerm("actuator_force", lambda data, **kw: data.actuator_force),
                ObsTerm("last_contact", lambda info, **kw: info["last_contact"].astype(jp.float32)),
                ObsTerm("feet_vel", lambda data, **kw: data.sensordata[self._foot_linvel_sensor_adr].ravel()),
                ObsTerm("feet_air_time", lambda info, **kw: info["feet_air_time"]),
                ObsTerm("xfrc_applied", lambda data, **kw: data.xfrc_applied[self._torso_body_id, :3]),
            ],
        }

        self._reward_spec = [
            RewardTerm("tracking_lin_vel", lambda data, info, **kw:
                self._reward_tracking_lin_vel(info["command"], self.get_local_linvel(data))),
            RewardTerm("tracking_ang_vel", lambda data, info, **kw:
                self._reward_tracking_ang_vel(info["command"], self.get_gyro(data))),
            RewardTerm("lin_vel_z", lambda data, **kw:
                self._cost_lin_vel_z(self.get_global_linvel(data))),
            RewardTerm("ang_vel_xy", lambda data, **kw:
                self._cost_ang_vel_xy(self.get_global_angvel(data))),
            RewardTerm("orientation", lambda data, **kw:
                self._cost_orientation(self.get_upvector(data))),
            RewardTerm("torques", lambda data, **kw:
                self._cost_torques(data.actuator_force)),
            RewardTerm("action_rate", lambda action, info, **kw:
                self._cost_action_rate(action, info["last_act"], info["last_last_act"])),
            RewardTerm("energy", lambda data, **kw:
                self._cost_energy(data.qvel[6:], data.actuator_force)),
            RewardTerm("dof_pos_limits", lambda data, **kw:
                self._cost_joint_pos_limits(data.qpos[7:])),
            RewardTerm("feet_air_time", lambda info, first_contact, **kw:
                self._reward_feet_air_time(info["feet_air_time"], first_contact, info["command"])),
            RewardTerm("feet_slip", lambda data, contact, info, **kw:
                self._cost_feet_slip(data, contact, info)),
            RewardTerm("feet_clearance", lambda data, **kw:
                self._cost_feet_clearance(data)),
            RewardTerm("feet_height", lambda info, first_contact, **kw:
                self._cost_feet_height(info["swing_peak"], first_contact, info)),
            RewardTerm("termination", lambda done, **kw:
                self._cost_termination(done)),
            RewardTerm("stand_still", lambda data, info, **kw:
                self._cost_stand_still(info["command"], data.qpos[7:])),
            RewardTerm("pose", lambda data, **kw:
                self._reward_pose(data.qpos[7:])),
        ]

    # ── Core env methods ────────────────────────────────────────────────

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        # Randomize base position and yaw.
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)

        # Small initial velocity perturbation.
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(
            jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
        )

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=qpos[7:],
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        # Sample initial command.
        rng, key1, key2 = jax.random.split(rng, 3)
        time_until_next_cmd = jax.random.exponential(key1) * 5.0
        steps_until_next_cmd = jp.round(time_until_next_cmd / self.dt).astype(
            jp.int32
        )
        cmd = jax.random.uniform(
            key2, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
        )

        info = {
            "rng": rng,
            "command": cmd,
            "steps_until_next_cmd": steps_until_next_cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "swing_peak": jp.zeros(4),
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

        # Random velocity kick every ~250 steps (~5s at 50Hz).
        # Pushes the base with random xy velocity to train recovery.
        # Stronger and more frequent than before for MJX→CPU robustness.
        step_count = state.info["step_count"]
        push_interval = 350  # steps between pushes (~7s)
        rng, push_key = jax.random.split(state.info["rng"])
        push_vel = jax.random.uniform(push_key, (2,), minval=-0.75, maxval=0.75)
        do_push = (step_count > 0) & (step_count % push_interval == 0)
        data = state.data
        new_qvel = data.qvel.at[0:2].set(
            jp.where(do_push, data.qvel[0:2] + push_vel, data.qvel[0:2])
        )
        data = data.replace(qvel=new_qvel)

        # External PD at physics rate: recompute torque each substep from fresh
        # joint state. Matches unitree_mujoco / real robot motor model.
        kp, kd = self._kp, self._kd
        model = self.mjx_model

        def substep(data, _):
            current_q = data.qpos[7:]   # joint positions (skip freejoint)
            current_dq = data.qvel[6:]  # joint velocities (skip freejoint)
            tau = kp * (motor_targets - current_q) + kd * (0.0 - current_dq)
            data = data.replace(ctrl=tau)
            return mjx.step(model, data), None

        data = jax.lax.scan(substep, data, (), self.n_substeps)[0]

        # Foot contact detection.
        contact = jp.array([
            data.sensordata[self._mj_model.sensor_adr[sid]] > 0
            for sid in self._feet_floor_found_sensor
        ])
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        obs = self._get_obs(data, state.info)
        done = self._get_termination(data)

        # Compute weighted reward.
        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact
        )
        rewards = {
            k: v * self._config.reward_config.scales[k]
            for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # Store per-term weighted rewards for diagnostics.
        state.info["reward_components"] = rewards

        # Update info.
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["step_count"] = step_count + 1
        state.info["steps_until_next_cmd"] -= 1
        state.info["rng"], key1, key2 = jax.random.split(rng, 3)
        state.info["command"] = jp.where(
            state.info["steps_until_next_cmd"] <= 0,
            self.sample_command(key1, state.info["command"]),
            state.info["command"],
        )
        state.info["steps_until_next_cmd"] = jp.where(
            done | (state.info["steps_until_next_cmd"] <= 0),
            jp.round(jax.random.exponential(key2) * 5.0 / self.dt).astype(
                jp.int32
            ),
            state.info["steps_until_next_cmd"],
        )
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        done = done.astype(reward.dtype)
        return state.replace(data=data, obs=obs, reward=reward, done=done)

    # ── Observation ─────────────────────────────────────────────────────

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> Dict[str, jax.Array]:
        obs, info["rng"] = compute_obs(
            self._obs_groups,
            noise_level=self._config.noise_config.level,
            rng=info["rng"],
            data=data, info=info,
        )
        return obs

    # ── Termination ─────────────────────────────────────────────────────

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        flipped = self.get_upvector(data)[-1] < 0.0
        # Height termination: prevent crouching local optimum.
        # Go2 stands at ~0.30m; terminate below 0.18m (alexeiplatzer).
        base_z = data.subtree_com[self._torso_body_id][2]
        too_low = base_z < 0.18
        return flipped | too_low

    # ── Rewards ─────────────────────────────────────────────────────────

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics  # UNUSED — kept for potential curriculum-dependent rewards
        return compute_rewards(
            self._reward_spec,
            data=data, action=action, info=info,
            done=done, first_contact=first_contact, contact=contact,
        )

    # ── Tracking rewards ────────────────────────────────────────────────

    def _reward_tracking_lin_vel(
        self, commands: jax.Array, local_vel: jax.Array
    ) -> jax.Array:
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        return jp.exp(
            -lin_vel_error / self._config.reward_config.tracking_sigma
        )

    def _reward_tracking_ang_vel(
        self, commands: jax.Array, ang_vel: jax.Array
    ) -> jax.Array:
        ang_vel_error = jp.square(commands[2] - ang_vel[2])
        return jp.exp(
            -ang_vel_error / self._config.reward_config.tracking_sigma
        )

    # ── Base stability costs ────────────────────────────────────────────

    def _cost_lin_vel_z(self, global_linvel: jax.Array) -> jax.Array:
        return jp.square(global_linvel[2])

    def _cost_ang_vel_xy(self, global_angvel: jax.Array) -> jax.Array:
        return jp.sum(jp.square(global_angvel[:2]))

    def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
        return jp.sum(jp.square(torso_zaxis[:2]))

    # ── Regularization costs ────────────────────────────────────────────

    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _cost_energy(
        self, qvel: jax.Array, qfrc_actuator: jax.Array
    ) -> jax.Array:
        return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

    def _cost_action_rate(
        self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
    ) -> jax.Array:
        del last_last_act  # UNUSED — passed but never read; kept for potential jerk penalty
        return jp.sum(jp.square(act - last_act))

    # ── Joint costs ─────────────────────────────────────────────────────

    def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
        out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
        out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
        return jp.sum(out_of_limits)

    # ── Feet rewards ────────────────────────────────────────────────────

    def _reward_feet_air_time(
        self, air_time: jax.Array, first_contact: jax.Array,
        commands: jax.Array
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(commands)
        rew_air_time = jp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= cmd_norm > 0.01
        return rew_air_time

    def _cost_feet_slip(
        self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(info["command"])
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
        return jp.sum(vel_xy_norm_sq * contact) * (cmd_norm > 0.01)

    def _cost_feet_clearance(self, data: mjx.Data) -> jax.Array:
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_z = foot_pos[..., -1]
        delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
        return jp.sum(delta * vel_norm)

    def _cost_feet_height(
        self, swing_peak: jax.Array, first_contact: jax.Array,
        info: dict[str, Any]
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(info["command"])
        error = swing_peak / self._config.reward_config.max_foot_height - 1.0
        return jp.sum(jp.square(error) * first_contact) * (cmd_norm > 0.01)

    # ── Other rewards ───────────────────────────────────────────────────

    def _cost_termination(self, done: jax.Array) -> jax.Array:
        return done

    def _cost_stand_still(
        self, commands: jax.Array, qpos: jax.Array
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(commands)
        return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.01)

    def _reward_pose(self, qpos: jax.Array) -> jax.Array:
        # Per-joint weight: hip=1, thigh=1, calf=0.1 (same as Go1).
        weight = jp.array([1.0, 1.0, 0.1] * 4)
        return jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * weight))

    # ── Command sampling ────────────────────────────────────────────────

    def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
        rng, y_rng, w_rng, z_rng = jax.random.split(rng, 4)
        y_k = jax.random.uniform(
            y_rng, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
        )
        z_k = jax.random.bernoulli(z_rng, self._cmd_b, shape=(3,))
        w_k = jax.random.bernoulli(w_rng, 0.5, shape=(3,))
        x_kp1 = x_k - w_k * (x_k - y_k * z_k)
        return x_kp1
