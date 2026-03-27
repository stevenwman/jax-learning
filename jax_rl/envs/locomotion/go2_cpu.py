"""CPU MuJoCo version of Go2 joystick env.

Sister env to go2_joystick.py — same MJCF, same overrides, same obs/action
interface. Runs on CPU mj_step instead of MJX. No JAX dependency at runtime.

Usage:
    env = Go2CpuEnv()
    obs = env.reset()
    for _ in range(1000):
        action = policy(obs)  # numpy array (12,)
        obs, reward, done, info = env.step(action, command)
"""

import mujoco
import numpy as np

from jax_rl.envs.locomotion.go2_base import get_assets
from jax_rl.envs.locomotion import go2_constants as consts


class Go2CpuEnv:
    """CPU MuJoCo Go2 env with identical physics to the MJX training env."""

    def __init__(self, kp: float = 35.0, kd: float = 0.1):
        from etils import epath

        assets = get_assets()
        xml = epath.Path(consts.SCENE_FLAT_XML.as_posix()).read_text()
        self.model = mujoco.MjModel.from_xml_string(xml, assets=assets)
        self.data = mujoco.MjData(self.model)

        # === EXACT COPY of go2_base.py overrides ===
        # If you change go2_base.py, update this too.
        self.model.opt.timestep = 0.004
        self.model.opt.ccd_iterations = 20

        self.model.actuator_gainprm[:, 0] = 1.0
        self.model.actuator_biasprm[:, :] = 0.0
        self.model.dof_damping[6:] = 0.1
        self.model.dof_frictionloss[6:] = 0.2

        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if 'calf' in name.lower():
                self.model.actuator_ctrlrange[i] = [-45.43, 45.43]
                self.model.actuator_forcerange[i] = [-45.43, 45.43]
            else:
                self.model.actuator_ctrlrange[i] = [-23.7, 23.7]
                self.model.actuator_forcerange[i] = [-23.7, 23.7]

        for jname in ["RL_thigh_joint", "RR_thigh_joint"]:
            jid = self.model.joint(jname).id
            self.model.jnt_range[jid] = [-0.5236, 4.5379]

        for foot_name in ["FL", "FR", "RL", "RR"]:
            gid = self.model.geom(foot_name).id
            self.model.geom_solimp[gid, :3] = [0.9, 0.95, 0.023]
            self.model.geom_condim[gid] = 3
            self.model.geom_friction[gid] = [0.6, 0.005, 0.0001]

        self.model.vis.global_.offwidth = 3840
        self.model.vis.global_.offheight = 2160

        # PD gains (same as go2_joystick default_config)
        self.kp = kp
        self.kd = kd

        # Env constants
        self.n_substeps = 5  # ctrl_dt(0.02) / sim_dt(0.004)
        self.default_pose = np.array(self.model.key_qpos[0][7:], dtype=np.float32)
        self.action_scale = 0.5
        self.imu_site_id = self.model.site('imu').id

        # Sensors
        gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'gyro')
        self.gyro_adr = self.model.sensor_adr[gyro_id]
        linvel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'local_linvel')
        self.linvel_adr = self.model.sensor_adr[linvel_id]

        # State
        self.last_action = np.zeros(12, dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Reset to home keyframe. Returns 48d obs."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.last_action = np.zeros(12, dtype=np.float32)
        return self._get_obs(np.zeros(3, dtype=np.float32))

    def step(self, action: np.ndarray, command: np.ndarray):
        """Step env with external PD (same as go2_joystick.step).

        Args:
            action: (12,) policy output in [-1, 1], policy order (FL,FR,RL,RR)
            command: (3,) velocity command [vx, vy, yaw_rate]

        Returns:
            obs: (48,) observation
            base_z: float, base height (for termination check)
        """
        motor_targets = self.default_pose + action * self.action_scale

        # PD at physics rate (same as go2_joystick substep loop)
        for _ in range(self.n_substeps):
            q = self.data.qpos[7:]
            dq = self.data.qvel[6:]
            tau = self.kp * (motor_targets - q) + self.kd * (0.0 - dq)
            self.data.ctrl[:] = np.clip(
                tau,
                self.model.actuator_ctrlrange[:, 0],
                self.model.actuator_ctrlrange[:, 1],
            )
            mujoco.mj_step(self.model, self.data)

        self.last_action = action.copy()
        obs = self._get_obs(command)
        base_z = self.data.qpos[2]
        return obs, base_z

    def _get_obs(self, command: np.ndarray) -> np.ndarray:
        """Build 48d obs (same layout as go2_joystick._get_obs, no noise)."""
        gravity = (self.data.site_xmat[self.imu_site_id].reshape(3, 3).T
                   @ np.array([0, 0, -1])).astype(np.float32)
        gyro = np.array(
            self.data.sensordata[self.gyro_adr:self.gyro_adr + 3],
            dtype=np.float32,
        )

        linvel = np.array(
            self.data.sensordata[self.linvel_adr:self.linvel_adr + 3],
            dtype=np.float32,
        )
        return np.concatenate([
            linvel,                                                  # [0:3] local linvel
            gyro,                                                    # [3:6]
            gravity,                                                 # [6:9]
            (self.data.qpos[7:] - self.default_pose).astype(np.float32),  # [9:21]
            self.data.qvel[6:].astype(np.float32),                  # [21:33]
            self.last_action,                                        # [33:45]
            command.astype(np.float32),                              # [45:48]
        ])
