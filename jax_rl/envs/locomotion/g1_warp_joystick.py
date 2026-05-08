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
from mujoco_playground._src import gait
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
    """G1 joystick config — recipe ported from mujoco_playground/g1/joystick.py.

    Key points:
    - Spawn from `knees_bent` keyframe (half-squat) — more stable than rigid
      `home` for SAC random-sigma exploration on a humanoid.
    - Reward weights: tracking 1.0/0.75 (small), termination -100 (huge "don't
      fall" gradient), pose -0.1 (cost form), joint_deviation_hip -0.25,
      joint_deviation_knee -0.1.
    - Most regularization (torques, action_rate, energy, feet_clearance, etc.)
      set to 0 — playground found these unnecessary or harmful early.
    - sim_dt=0.002 (10 substeps per ctrl_dt 0.02) for finer humanoid contact.
    """
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        # 0.25 from unitree_rl_lab. With knees_bent spawn + alive bonus,
        # policy doesn't need wide action range early.
        action_scale=0.25,
        soft_joint_pos_limit_factor=0.95,
        tilt_upvector_threshold=0.5,
        tilt_min_pelvis_z=0.55,
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
                # Match unitree_rl_lab/tasks/locomotion/robots/g1/29dof/
                # velocity_env_cfg.py weights as closely as our reward set
                # supports. Their full set has gait + feet_slide + undesired_
                # contacts which we don't implement yet (TODO if needed).
                # Tracking
                tracking_lin_vel=1.0,
                tracking_ang_vel=0.5,
                # Survival (per-step bonus, NO big termination cliff —
                # unitree relies on alive bonus alone, which avoids the
                # gradient-killing -100 we used in v3/v4).
                alive=0.15,
                termination=0.0,
                # Base motion costs
                lin_vel_z=-2.0,
                ang_vel_xy=-0.05,
                orientation=-5.0,
                base_height=-10.0,
                # Smoothness / regularization
                action_rate=-0.05,
                dof_pos_limits=-5.0,
                # Joint deviation (stronger on legs than arms)
                joint_deviation_hip=-1.0,
                joint_deviation_knee=-1.0,
                pose=-0.1,
                # Feet
                feet_slip=-0.25,
                feet_phase=1.0,
                # Defined here at 0 so reward_spec can include them; holosoma
                # preset overrides with real weights.
                close_feet_xy=0.0,
                feet_ori=0.0,
            ),
            tracking_sigma=0.25,
            max_foot_height=0.15,
            base_height_target=0.755,  # matches knees_bent spawn pelvis_z
            close_feet_threshold=0.15,
            # Gait CPG (port from mujoco_playground/g1/joystick): cubic-Bezier
            # foot-z trajectory at ~1.4Hz, left/right out-of-phase.
            gait_freq_range=(1.25, 1.5),
            gait_swing_height=0.08,
            gait_phase_offsets=(0.0, 3.141592653589793),  # [left_offset, right_offset]
            feet_phase_sigma=0.01,
        ),
        command_config=config_dict.create(
            # Narrow cmd ranges — early policy learns to STAND (cmd ≈ 0)
            # before being asked to walk. Mirrors unitree_rl_lab's curriculum
            # initial cmd range. Once standing trains, we'll expand to
            # [1.0, 0.5, 0.8] for actual locomotion.
            a=[0.1, 0.1, 0.1],
            b=[0.9, 0.25, 0.5],
        ),
        impl="warp",
        contact_mode="training",
        naconmax=4 * 8192,
        naccdmax=4000,
        njmax=100,
    )


# Per-joint pose weights from holosoma's g1_29dof_loco config (pose_weights):
# legs: [hip_pitch=0.01, hip_roll=1.0, hip_yaw=5.0, knee=0.01, ankle_pitch=5.0,
#        ankle_roll=5.0] × 2 (left, right)
# waist: [yaw, roll, pitch] = [50, 50, 50] (locked rigid)
# arms (7 per side): [shoulder_p, shoulder_r, shoulder_y, elbow, wrist_r,
#                     wrist_p, wrist_y] = [50] × 14 (locked rigid)
# Hip pitch + knee at 0.01 are basically free → policy can swing legs for stride
# without paying pose cost. Arms/waist at 50 → strong "keep arms still" signal.
_HOLOSOMA_POSE_WEIGHTS = (
    # Left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    0.01, 1.0, 5.0, 0.01, 5.0, 5.0,
    # Right leg
    0.01, 1.0, 5.0, 0.01, 5.0, 5.0,
    # Waist: yaw, roll, pitch
    50.0, 50.0, 50.0,
    # Left arm (7)
    50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
    # Right arm (7)
    50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
)


def default_config_holosoma() -> config_dict.ConfigDict:
    """Holosoma-matched reward set for G1 (per `holosoma/config_values/loco/g1/
    reward.py:g1_29dof_loco_fast_sac`).

    Differences vs `default_config`:
    - tracking_lin_vel 1.0 → 2.0
    - tracking_ang_vel 0.5 → 1.5
    - alive 0.15 → 10.0  (key for FastSAC variant per holosoma)
    - feet_phase 1.0 → 5.0 (sigma 0.01 → 0.008 — sharper foot-timing reward)
    - action_rate -0.05 → -2.0
    - orientation -5.0 → -10.0
    - pose -0.1 with uniform weights → -0.5 with per-joint holosoma weights
      (hip_pitch + knee free; waist + arms locked at 50)
    - close_feet_xy 0 → -10.0 (penalty if feet xy distance < 0.15m)
    - feet_ori 0 → -5.0 (penalize foot horizontal orientation deviation)
    - dof_pos_limits -5 → 0 (holosoma doesn't use explicit dof limit penalty;
      relies on pose for boundary)
    - joint_deviation_hip/knee → 0 (subsumed by per-joint pose weights)
    - lin_vel_z -2 → 0 (subsumed by orientation + base_height)
    - base_height -10 → 0 (subsumed by orientation; holosoma G1 doesn't
      penalize height directly in fast_sac variant)
    - feet_slip -0.25 → 0 (replaced by feet_phase + close_feet_xy combo)
    """
    cfg = default_config()
    cfg.unlock()
    cfg.reward_config.scales.tracking_lin_vel = 2.0
    cfg.reward_config.scales.tracking_ang_vel = 1.5
    cfg.reward_config.scales.alive = 10.0
    cfg.reward_config.scales.feet_phase = 5.0
    cfg.reward_config.scales.action_rate = -2.0
    cfg.reward_config.scales.orientation = -10.0
    cfg.reward_config.scales.pose = -0.5
    cfg.reward_config.scales.close_feet_xy = -10.0
    cfg.reward_config.scales.feet_ori = -5.0
    cfg.reward_config.scales.dof_pos_limits = 0.0
    cfg.reward_config.scales.joint_deviation_hip = 0.0
    cfg.reward_config.scales.joint_deviation_knee = 0.0
    cfg.reward_config.scales.lin_vel_z = 0.0
    cfg.reward_config.scales.base_height = 0.0
    cfg.reward_config.scales.feet_slip = 0.0
    cfg.reward_config.feet_phase_sigma = 0.008  # sharper than 0.01 default
    cfg.reward_config.gait_swing_height = 0.09  # holosoma uses 0.09 (we had 0.08)
    return cfg


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
        # Spawn from knees_bent (half-squat) — playground recipe. Stable
        # initial pose under random exploration; switching from "home" was
        # what made the difference between 0.1 eval and any actual learning.
        self._init_q = jp.array(self._mj_model.keyframe("knees_bent").qpos)
        self._default_pose = jp.array(self._mj_model.keyframe("knees_bent").qpos[7:])

        # Hip / knee joint indices in qpos[7:]. Used by joint_deviation costs.
        # (Hip pitch is excluded — its deviation is OK for stride.)
        hip_indices = []
        for side in ["left", "right"]:
            for jname in ["hip_roll", "hip_yaw"]:
                hip_indices.append(
                    self._mj_model.joint(f"{side}_{jname}_joint").qposadr[0] - 7
                )
        self._hip_indices = jp.array(hip_indices)

        knee_indices = []
        for side in ["left", "right"]:
            knee_indices.append(
                self._mj_model.joint(f"{side}_knee_joint").qposadr[0] - 7
            )
        self._knee_indices = jp.array(knee_indices)

        # Per-joint pose weights (used by holosoma preset; default config uses
        # uniform 1.0 — equivalent to dropping this and using sum(square(qpos -
        # default))). 29 entries matching ACTUATOR_NAMES order.
        self._pose_weights = jp.array(_HOLOSOMA_POSE_WEIGHTS, dtype=jp.float32)

        # Foot body IDs for close_feet + feet_ori costs.
        self._left_foot_body_id = self._mj_model.body("left_ankle_roll_link").id
        self._right_foot_body_id = self._mj_model.body("right_ankle_roll_link").id

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
            # Tracking
            RewardTerm("tracking_lin_vel", lambda data, info, **kw:
                self._reward_tracking_lin_vel(info["command"], self.get_local_linvel(data))),
            RewardTerm("tracking_ang_vel", lambda data, info, **kw:
                self._reward_tracking_ang_vel(info["command"], self.get_gyro(data))),
            # Survival
            RewardTerm("alive", lambda done, **kw: 1.0 - done.astype(jp.float32)),
            RewardTerm("termination", lambda done, **kw: self._cost_termination(done)),
            # Base motion
            RewardTerm("lin_vel_z", lambda data, **kw:
                self._cost_lin_vel_z(self.get_global_linvel(data))),
            RewardTerm("ang_vel_xy", lambda data, **kw:
                self._cost_ang_vel_xy(self.get_global_angvel(data))),
            RewardTerm("orientation", lambda data, **kw:
                self._cost_orientation(self.get_upvector(data))),
            RewardTerm("base_height", lambda data, **kw:
                self._cost_base_height(data)),
            # Smoothness
            RewardTerm("action_rate", lambda action, info, **kw:
                self._cost_action_rate(action, info["last_act"], info["last_last_act"])),
            RewardTerm("dof_pos_limits", lambda data, **kw:
                self._cost_joint_pos_limits(data.qpos[7:])),
            # Joint deviation
            RewardTerm("joint_deviation_hip", lambda data, info, **kw:
                self._cost_joint_deviation_hip(data.qpos[7:], info["command"])),
            RewardTerm("joint_deviation_knee", lambda data, **kw:
                self._cost_joint_deviation_knee(data.qpos[7:])),
            RewardTerm("pose", lambda data, **kw:
                self._cost_pose(data.qpos[7:])),
            # Feet
            RewardTerm("feet_slip", lambda data, contact, info, **kw:
                self._cost_feet_slip(data, contact, info)),
            RewardTerm("feet_phase", lambda data, info, **kw:
                self._reward_feet_phase(data, info["phase"])),
            RewardTerm("close_feet_xy", lambda data, **kw:
                self._cost_close_feet_xy(data)),
            RewardTerm("feet_ori", lambda data, **kw:
                self._cost_feet_ori(data)),
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

        # Gait phase: ticks at gait_freq Hz; left/right offset by π.
        gcfg = self._config.reward_config
        rng, gk = jax.random.split(rng)
        gait_freq = jax.random.uniform(
            gk, (), minval=gcfg.gait_freq_range[0], maxval=gcfg.gait_freq_range[1]
        )
        phase_dt = 2.0 * jp.pi * self.dt * gait_freq
        phase = jp.array(gcfg.gait_phase_offsets, dtype=jp.float32)

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
            "phase": phase,
            "phase_dt": phase_dt,
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
        # NO lower clip — termination=-100 (with -1 done weight = -100 reward
        # at fall-step) MUST propagate as a negative gradient. The Go2 pattern
        # of clip(0, 10000) silently kills the termination penalty; without
        # this fix policy sees fall == "0 reward" same as standing-still.
        reward = sum(rewards.values()) * self.dt
        state.info["reward_components"] = rewards

        # Advance gait phase. Wrap to [-pi, pi].
        phase_tp1 = state.info["phase"] + state.info["phase_dt"]
        state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2.0 * jp.pi) - jp.pi

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

    def _cost_pose(self, qpos):
        """Per-joint weighted squared deviation from default pose. The weights
        come from `_HOLOSOMA_POSE_WEIGHTS` (loaded into self._pose_weights in
        _post_init). holosoma config: hip_pitch + knee → 0.01 (free), waist
        + arms → 50 (locked rigid)."""
        return jp.sum(jp.square(qpos - self._default_pose) * self._pose_weights)

    def _cost_close_feet_xy(self, data):
        """Penalty (binary) when feet xy distance < threshold. Computed in
        base-local frame (rotated by pelvis yaw) so the lateral spacing is
        preserved regardless of base heading. Port from holosoma's
        penalty_close_feet_xy."""
        threshold = self._config.reward_config.close_feet_threshold
        left_xy = data.xpos[self._left_foot_body_id, :2]
        right_xy = data.xpos[self._right_foot_body_id, :2]
        # Base yaw from pelvis upvector + forwardvector — use forward x/y
        # components of the framexaxis sensor's value.
        fwd = self.get_forwardvector(data)
        # Rotate (left - right) into base-local frame and take |y component|.
        # base_yaw = atan2(fwd_y, fwd_x); cos/sin form below avoids the atan2.
        cos_y = fwd[0] / (jp.sqrt(fwd[0]**2 + fwd[1]**2) + 1e-6)
        sin_y = fwd[1] / (jp.sqrt(fwd[0]**2 + fwd[1]**2) + 1e-6)
        dx = left_xy[0] - right_xy[0]
        dy = left_xy[1] - right_xy[1]
        # Rotate (dx, dy) by -base_yaw to get base-local; we only need the
        # base-y component (lateral spacing).
        local_lateral = -sin_y * dx + cos_y * dy
        feet_distance = jp.abs(local_lateral)
        return (feet_distance < threshold).astype(jp.float32)

    def _cost_feet_ori(self, data):
        """Penalize foot orientation deviation from flat. For each foot,
        rotate gravity into the foot's local frame and take the magnitude of
        the horizontal component (sqrt of sum of squared x/y). Sums both feet.
        Port from holosoma's penalty_feet_ori."""
        # data.xmat is (nbody, 9) row-major rotation matrix per body.
        # gravity in world = (0, 0, -1).
        # gravity in body frame = R^T @ gravity = third column of R (since
        # gravity world = -ẑ_world; R^T @ (-ẑ_world) = -third row of R^T =
        # -third column of R = -(R[*, 2])). For our purposes only the |xy|
        # magnitudes matter — sign doesn't affect the cost.
        left_R = data.xmat[self._left_foot_body_id].reshape(3, 3)
        right_R = data.xmat[self._right_foot_body_id].reshape(3, 3)
        # gravity_in_foot_frame x/y components = first two entries of third
        # column of R^T = first two entries of third row of R.
        left_g_xy = left_R[2, :2]
        right_g_xy = right_R[2, :2]
        return (
            jp.sqrt(jp.sum(jp.square(left_g_xy)) + 1e-8)
            + jp.sqrt(jp.sum(jp.square(right_g_xy)) + 1e-8)
        )

    def get_forwardvector(self, data):
        return go2_sensors.get_sensor_by_name(self.mj_model, data, "forwardvector")

    def _cost_joint_deviation_hip(self, qpos, cmd):
        """Per-side hip roll/yaw deviation. Allows roll deviation when lateral
        cmd is high (lets the robot lean into a sideways step)."""
        error = qpos[self._hip_indices] - self._default_pose[self._hip_indices]
        weight = jp.where(
            cmd[1] > 0.1,
            jp.array([0.0, 1.0, 0.0, 1.0]),
            jp.array([1.0, 1.0, 1.0, 1.0]),
        )
        return jp.sum(jp.abs(error) * weight)

    def _cost_joint_deviation_knee(self, qpos):
        error = qpos[self._knee_indices] - self._default_pose[self._knee_indices]
        return jp.sum(jp.abs(error))

    def _cost_base_height(self, data):
        """Penalize squared deviation from target pelvis height. Strong
        signal to stay tall (unitree weights this -10)."""
        z = data.subtree_com[self._pelvis_body_id][2]
        return jp.square(z - self._config.reward_config.base_height_target)

    def _reward_feet_phase(self, data, phase):
        """Cubic-Bezier swing-phase foot height tracking.

        gait.get_rz(phi) returns target z for swing-foot at phase phi:
        rises 0 → swing_height during 0..pi, returns to 0 during pi..2pi.
        Per foot, we read its current world z and reward exp(-||z - z*||²).
        Left/right phases offset by pi → alternating gait emerges naturally.
        Ported from mujoco_playground/_src/locomotion/g1/joystick.py.
        """
        rz = gait.get_rz(phase, swing_height=self._config.reward_config.gait_swing_height)
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_z = foot_pos[..., -1]
        error = jp.sum(jp.square(foot_z - rz))
        return jp.exp(-error / self._config.reward_config.feet_phase_sigma)

    # ── Command sampling (same Markov-chain pattern as Go2) ─────────────

    def sample_command(self, rng, x_k):
        rng, y_rng, w_rng, z_rng = jax.random.split(rng, 4)
        y = jax.random.uniform(y_rng, (3,), minval=-self._cmd_a, maxval=self._cmd_a)
        z = jax.random.bernoulli(z_rng, self._cmd_b, (3,))
        w = jax.random.bernoulli(w_rng, 0.5, (3,))
        return x_k - w * (x_k - y * z)
