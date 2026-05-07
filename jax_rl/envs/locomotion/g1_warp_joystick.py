"""Unitree G1 humanoid joystick env (Warp backend).

Track a 3-d (vx, vy, yaw_rate) command on flat ground. 29 actuators
(legs+waist+arms). Asymmetric AC: actor sees proprio + cmd; critic
also sees ground-truth linvel + angvel + actuator force + foot info.

Mirrors `Go2WarpJoystickFlatNoAccel` semantics (no accelerometer in
actor obs) so reward set + obs schema patterns transfer cleanly.
"""

from typing import Any, Dict, Optional, Union
from pathlib import Path

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from etils import epath
from mujoco_playground._src import mjx_env
from jax_rl.envs.locomotion import g1_constants as consts
from jax_rl.envs.locomotion import go2_sensors
from jax_rl.envs.obs_spec import ObsTerm, IncludeGroup, compute_obs
from jax_rl.envs.reward_spec import RewardTerm, compute_rewards


def get_warp_assets() -> Dict[str, bytes]:
    """Load XML and mesh assets from vendored G1 menagerie model."""
    assets = {}
    vendor_path = Path(__file__).parent / "xmls" / "unitree_g1"
    mjx_env.update_assets(assets, vendor_path, "*.xml")
    mjx_env.update_assets(assets, vendor_path / "assets")
    # Scene XML directory (for <include> resolution).
    mjx_env.update_assets(assets, Path(__file__).parent / "xmls", "*.xml")
    return assets


# Per-joint Kp/Kd matched to Unitree's RL deploy config (deploy/robots/g1_23dof/
# config/config.yaml from unitree_rl_lab). Stiffer hips/knees, softer
# ankles/wrists. Order matches consts.ACTUATOR_NAMES.
_KP_PER_ACTUATOR = (
    # Legs: hip_pitch/roll/yaw, knee, ankle_pitch/roll
    100., 100., 100., 150., 40., 40.,
    100., 100., 100., 150., 40., 40.,
    # Waist: yaw, roll, pitch
    200., 200., 200.,
    # Arms: shoulder p/r/y, elbow, wrist r/p/y
    40., 40., 40., 40., 40., 40., 40.,
    40., 40., 40., 40., 40., 40., 40.,
)
_KD_PER_ACTUATOR = (
    2., 2., 2., 4., 2., 2.,
    2., 2., 2., 4., 2., 2.,
    5., 5., 5.,
    10., 10., 10., 10., 10., 10., 10.,
    10., 10., 10., 10., 10., 10., 10.,
)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.5,
        soft_joint_pos_limit_factor=0.95,
        # Tilt termination: humanoid is more sensitive than quadruped.
        tilt_upvector_threshold=0.5,
        tilt_min_pelvis_z=0.55,
        target_pelvis_z=0.78,
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
                linvel=0.1,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_lin_vel=10.0,
                tracking_ang_vel=5.0,
                lin_vel_z=-0.5,
                ang_vel_xy=-0.05,
                orientation=-5.0,
                torques=-0.0001,
                action_rate=-0.01,
                energy=-0.0005,
                dof_pos_limits=-1.0,
                feet_air_time=0.5,
                feet_slip=-0.1,
                feet_clearance=-1.0,
                feet_height=-0.2,
                termination=-1.0,
                stand_still=-0.5,
                pose=0.5,
                base_height=-5.0,
            ),
            tracking_sigma=0.25,
            max_foot_height=0.15,  # taller than Go2's 0.10
        ),
        command_config=config_dict.create(
            # Smaller cmd ranges than Go2 — humanoid is slower/less stable.
            a=[1.0, 0.6, 0.8],
            b=[0.9, 0.25, 0.5],
        ),
        impl="warp",
        contact_mode="training",
        naconmax=4 * 8192,
        naccdmax=4000,
        njmax=100,
    )


class G1WarpJoystick(mjx_env.MjxEnv):
    """Track a joystick velocity command with G1 (Warp backend, 29 DOF)."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)
        del task  # placeholder — only flat_terrain supported for now

        self._model_assets = get_warp_assets()
        xml_path = consts.WARP_SCENE_FLAT_XML.as_posix()
        self._mj_model = mujoco.MjModel.from_xml_string(
            epath.Path(xml_path).read_text(), assets=self._model_assets
        )

        self._mj_model.opt.timestep = self._config.sim_dt
        self._mj_model.opt.ccd_iterations = 100

        # Per-actuator PD gains (humanoid varies by joint type).
        self._kp = jp.array(_KP_PER_ACTUATOR)
        self._kd = jp.array(_KD_PER_ACTUATOR)

        # actuator_forcerange comes from the XML class defaults.
        # MJX snapshot.
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = xml_path

        # Body / site IDs.
        self._pelvis_body_id = self._mj_model.body(consts.ROOT_BODY).id
        self._imu_site_id = self._mj_model.site(consts.IMU_SITE).id
        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in consts.FEET_SITES]
        )

        # Foot floor-found contact sensor IDs (defined in scene XML).
        self._feet_floor_found_sensor = [
            self._mj_model.sensor(f"{name}_floor_found").id
            for name in consts.FEET_SITES
        ]

        # Foot global linvel sensor address ranges.
        foot_linvel_sensor_adr = []
        for name in consts.FEET_LINVEL_SENSOR:
            sid = self._mj_model.sensor(name).id
            adr = self._mj_model.sensor_adr[sid]
            dim = self._mj_model.sensor_dim[sid]
            foot_linvel_sensor_adr.append(list(range(adr, adr + dim)))
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        self._post_init()

    # ── Properties / metadata ──────────────────────────────────────────

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    def get_control_metadata(self) -> dict:
        """Minimal control metadata for ckpt save. No deploy contract for G1
        yet (sim-only research), but downstream tooling expects a dict."""
        if not hasattr(self, "_default_pose"):
            raise RuntimeError("_default_pose not set; call _post_init first")
        policy_joint_names = [
            self._mj_model.jnt(i + 1).name for i in range(self._mj_model.nu)
        ]
        return {
            "Kp": _KP_PER_ACTUATOR,
            "Kd": _KD_PER_ACTUATOR,
            "action_scale": float(self._config.action_scale),
            "policy_dt": float(self._config.ctrl_dt),
            "physics_dt": float(self._config.sim_dt),
            "action_repeat": int(getattr(self._config, "action_repeat", 1)),
            "default_pose_policy": np.asarray(self._default_pose).tolist(),
            "policy_joint_names": policy_joint_names,
        }

    # ── Sensor accessors ──────────────────────────────────────────────

    def get_upvector(self, data):
        return go2_sensors.get_sensor_by_name(self.mj_model, data, consts.UPVECTOR_SENSOR)

    def get_gravity(self, data):
        return go2_sensors.get_gravity(data, self._imu_site_id)

    def get_global_linvel(self, data):
        return go2_sensors.get_sensor_by_name(self.mj_model, data, consts.GLOBAL_LINVEL_SENSOR)

    def get_global_angvel(self, data):
        return go2_sensors.get_sensor_by_name(self.mj_model, data, consts.GLOBAL_ANGVEL_SENSOR)

    def get_local_linvel(self, data):
        return go2_sensors.get_sensor_by_name(self.mj_model, data, consts.LOCAL_LINVEL_SENSOR)

    def get_gyro(self, data):
        return go2_sensors.get_sensor_by_name(self.mj_model, data, consts.GYRO_SENSOR)

    # ── Init helpers ──────────────────────────────────────────────────

    def _post_init(self) -> None:
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])
        if self._default_pose.shape != (consts.NUM_ACTUATORS,):
            raise RuntimeError(
                f"default_pose shape {self._default_pose.shape} != "
                f"({consts.NUM_ACTUATORS},). Keyframe 'home' qpos likely wrong size."
            )

        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

        self._cmd_a = jp.array(self._config.command_config.a)
        self._cmd_b = jp.array(self._config.command_config.b)

        noise = self._config.noise_config.scales
        self._obs_groups = {
            "state": [
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
                ObsTerm("gravity_clean", lambda data, **kw: self.get_gravity(data)),
                ObsTerm("linvel_clean", lambda data, **kw: self.get_local_linvel(data)),
                ObsTerm("angvel", lambda data, **kw: self.get_global_angvel(data)),
                ObsTerm("joint_pos_clean", lambda data, **kw: data.qpos[7:] - self._default_pose),
                ObsTerm("joint_vel_clean", lambda data, **kw: data.qvel[6:]),
                ObsTerm("actuator_force", lambda data, **kw: data.actuator_force),
                ObsTerm("last_contact", lambda info, **kw: info["last_contact"].astype(jp.float32)),
                ObsTerm("feet_vel", lambda data, **kw: data.sensordata[self._foot_linvel_sensor_adr].ravel()),
                ObsTerm("feet_air_time", lambda info, **kw: info["feet_air_time"]),
                ObsTerm("xfrc_applied", lambda data, **kw: data.xfrc_applied[self._pelvis_body_id, :3]),
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
            RewardTerm("base_height", lambda data, **kw:
                self._cost_base_height(data)),
        ]

    # ── Domain randomization ──────────────────────────────────────────

    def get_domain_randomization_spec(self):
        """Per-episode DR — friction, joint params, body mass, motor strength."""
        from jax_rl.envs.wrappers.domain_rand import DRSpec
        # Joint indices in qpos: 7..7+29=36. dof indices: 6..6+29=35.
        nu = consts.NUM_ACTUATORS
        return [
            DRSpec(name="friction", type="model", field="geom_friction",
                   column=0, min=0.5, max=1.5, per_element=False, operation="set",
                   description="Uniform friction across all geoms"),
            DRSpec(name="dof_damping", type="model", field="dof_damping",
                   indices=(6, 6 + nu), min=0.7, max=1.5, per_element=True,
                   description="Joint damping variation"),
            DRSpec(name="dof_armature", type="model", field="dof_armature",
                   indices=(6, 6 + nu), min=0.9, max=1.2, per_element=True,
                   description="Joint armature variation"),
            DRSpec(name="body_mass", type="model", field="body_mass",
                   min=0.85, max=1.15, per_element=True,
                   description="Per-link mass variation"),
            DRSpec(name="motor_strength", type="model", field="actuator_gainprm",
                   column=0, min=0.9, max=1.1, per_element=True,
                   description="Per-actuator motor heterogeneity"),
        ]

    # ── Core env methods ──────────────────────────────────────────────

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        # Randomize base xy + yaw.
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.2, maxval=0.2)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1.0]), yaw)
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)

        # Small initial velocity perturbation.
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(
            jax.random.uniform(key, (6,), minval=-0.3, maxval=0.3)
        )

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=self._default_pose,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            naccdmax=self._config.naccdmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        # Initial command.
        rng, key1, key2 = jax.random.split(rng, 3)
        time_until_next_cmd = jax.random.exponential(key1) * 5.0
        steps_until_next_cmd = jp.round(time_until_next_cmd / self.dt).astype(jp.int32)
        cmd = jax.random.uniform(
            key2, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
        )

        info = {
            "rng": rng,
            "command": cmd,
            "steps_until_next_cmd": steps_until_next_cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            "step_count": jp.int32(0),
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

        kp = self._kp
        kd = self._kd
        model = self.mjx_model

        def substep(data, _):
            # G1's actuators are in joint order (no remap). qpos[7:] is the 29
            # actuated joints; ctrl is also 29 in the same order.
            current_q = data.qpos[7:7 + consts.NUM_ACTUATORS]
            current_dq = data.qvel[6:6 + consts.NUM_ACTUATORS]
            tau = kp * (motor_targets - current_q) + kd * (0.0 - current_dq)
            data = data.replace(ctrl=tau)
            return mjx.step(model, data), None

        data = jax.lax.scan(substep, state.data, (), self.n_substeps)[0]

        # Foot contact (2 feet).
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
        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact
        )
        rewards = {
            k: v * self._config.reward_config.scales[k]
            for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
        state.info["reward_components"] = rewards

        # Update command resampling + housekeeping.
        rng, key1, key2 = jax.random.split(state.info["rng"], 3)
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["step_count"] = state.info["step_count"] + 1
        state.info["steps_until_next_cmd"] -= 1
        state.info["rng"] = rng
        state.info["command"] = jp.where(
            state.info["steps_until_next_cmd"] <= 0,
            self.sample_command(key1, state.info["command"]),
            state.info["command"],
        )
        state.info["steps_until_next_cmd"] = jp.where(
            done | (state.info["steps_until_next_cmd"] <= 0),
            jp.round(jax.random.exponential(key2) * 5.0 / self.dt).astype(jp.int32),
            state.info["steps_until_next_cmd"],
        )
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        done = done.astype(reward.dtype)
        return state.replace(data=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data, info):
        obs, info["rng"] = compute_obs(
            self._obs_groups,
            noise_level=self._config.noise_config.level,
            rng=info["rng"],
            data=data, info=info,
        )
        return obs

    def _get_termination(self, data):
        # Tilt: pelvis upvector z below threshold OR pelvis below min height.
        upvec = self.get_upvector(data)
        flipped = upvec[-1] < self._config.tilt_upvector_threshold
        pelvis_z = data.subtree_com[self._pelvis_body_id][2]
        too_low = pelvis_z < self._config.tilt_min_pelvis_z
        return flipped | too_low

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        del metrics
        return compute_rewards(
            self._reward_spec,
            data=data, action=action, info=info,
            done=done, first_contact=first_contact, contact=contact,
        )

    # ── Reward / cost helpers (ported from Go2 with biped tweaks) ──────

    def _reward_tracking_lin_vel(self, commands, local_vel):
        err = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        return jp.exp(-err / self._config.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, commands, ang_vel):
        err = jp.square(commands[2] - ang_vel[2])
        return jp.exp(-err / self._config.reward_config.tracking_sigma)

    def _cost_lin_vel_z(self, global_linvel):
        return jp.square(global_linvel[2])

    def _cost_ang_vel_xy(self, global_angvel):
        return jp.sum(jp.square(global_angvel[:2]))

    def _cost_orientation(self, torso_zaxis):
        return jp.sum(jp.square(torso_zaxis[:2]))

    def _cost_torques(self, torques):
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _cost_energy(self, qvel, qfrc_actuator):
        return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

    def _cost_action_rate(self, act, last_act, last_last_act):
        del last_last_act
        return jp.sum(jp.square(act - last_act))

    def _cost_joint_pos_limits(self, qpos):
        out = -jp.clip(qpos - self._soft_lowers, None, 0.0)
        out += jp.clip(qpos - self._soft_uppers, 0.0, None)
        return jp.sum(out)

    def _reward_feet_air_time(self, air_time, first_contact, commands):
        cmd_norm = jp.linalg.norm(commands)
        rew = jp.sum((air_time - 0.25) * first_contact)  # biped target air ~0.25s
        return rew * (cmd_norm > 0.01)

    def _cost_feet_slip(self, data, contact, info):
        cmd_norm = jp.linalg.norm(info["command"])
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        v2 = jp.sum(jp.square(vel_xy), axis=-1)
        return jp.sum(v2 * contact) * (cmd_norm > 0.01)

    def _cost_feet_clearance(self, data):
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_z = foot_pos[..., -1]
        delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
        return jp.sum(delta * vel_norm)

    def _cost_feet_height(self, swing_peak, first_contact, info):
        cmd_norm = jp.linalg.norm(info["command"])
        err = swing_peak / self._config.reward_config.max_foot_height - 1.0
        return jp.sum(jp.square(err) * first_contact) * (cmd_norm > 0.01)

    def _cost_termination(self, done):
        return done

    def _cost_stand_still(self, commands, qpos):
        cmd_norm = jp.linalg.norm(commands)
        return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.01)

    def _reward_pose(self, qpos):
        # Weight legs and waist more (must be functional); arms allowed to drift
        # more. Order matches consts.ACTUATOR_NAMES.
        weight = jp.array(
            # Legs (12)
            [1.0] * 12 +
            # Waist (3)
            [1.0] * 3 +
            # Arms (14): low weight so arms don't dominate the cost
            [0.1] * 14
        )
        return jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * weight))

    def _cost_base_height(self, data):
        z = data.subtree_com[self._pelvis_body_id][2]
        return jp.square(z - self._config.target_pelvis_z)

    # ── Command sampling (same Markov-chain pattern as Go2) ─────────────

    def sample_command(self, rng, x_k):
        rng, y_rng, w_rng, z_rng = jax.random.split(rng, 4)
        y = jax.random.uniform(y_rng, (3,), minval=-self._cmd_a, maxval=self._cmd_a)
        z = jax.random.bernoulli(z_rng, self._cmd_b, (3,))
        w = jax.random.bernoulli(w_rng, 0.5, (3,))
        return x_k - w * (x_k - y * z)
