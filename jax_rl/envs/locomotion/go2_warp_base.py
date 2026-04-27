"""Base class for Go2 Warp environments.

Uses unitree_mujoco's go2.xml (vendored) with MuJoCo Warp backend.
Full collision geometry (cylinders + boxes) -- no MJX simplifications.
"""

from typing import Any, Dict, Optional, Union
from pathlib import Path

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from etils import epath
from mujoco_playground._src import mjx_env
from jax_rl.envs.locomotion import go2_constants as consts
from jax_rl.envs.locomotion import go2_sensors


def get_warp_assets() -> Dict[str, bytes]:
    """Load XML and mesh assets from vendored unitree Go2 model."""
    assets = {}
    vendor_path = Path(__file__).parent / "xmls" / "unitree_go2"
    mjx_env.update_assets(assets, vendor_path, "*.xml")
    mjx_env.update_assets(assets, vendor_path / "assets")
    # Scene XML directory (for <include> resolution).
    mjx_env.update_assets(assets, Path(__file__).parent / "xmls", "*.xml")
    return assets


class Go2WarpEnv(mjx_env.MjxEnv):
    """Base class for Go2 Warp environments using unitree's MJCF."""

    def __init__(
        self,
        xml_path: str,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        self._model_assets = get_warp_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(
            epath.Path(xml_path).read_text(), assets=self._model_assets
        )

        # --- Always-applied overrides ---
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mj_model.opt.ccd_iterations = 100  # Warp needs many for cylinder+box geometry

        # --- contact_mode overrides ---
        # "training": firm foot contacts for crisp push-off (matches Go1 PG).
        # "deploy": keep unitree XML native contacts (condim=6, soft solimp).
        if getattr(self._config, 'contact_mode', 'training') == 'training':
            for foot_name in consts.FEET_GEOMS:
                gid = self._mj_model.geom(foot_name).id
                self._mj_model.geom_solimp[gid, :3] = np.array([0.9, 0.95, 0.023])
                self._mj_model.geom_condim[gid] = 3
                self._mj_model.geom_friction[gid] = np.array([0.6, 0.005, 0.0001])

        # Set actuator force limits (unitree XML has forcerange=[0,0] = unlimited).
        # Must match ctrlrange so PD torques are clamped to motor limits.
        for i in range(self._mj_model.nu):
            self._mj_model.actuator_forcerange[i] = self._mj_model.actuator_ctrlrange[i]

        # PD gains for external PD in step().
        self._kp = config.Kp
        self._kd = config.Kd

        # Actuator-to-joint remapping. In unitree's XML, qpos[7:] is in body-tree
        # order (FL,FR,RL,RR) but ctrl is in actuator order (FR,FL,RR,RL).
        # Build act_to_joint: for each actuator index, which joint index it drives.
        # Then ctrl[a] = tau_joint[act_to_joint[a]].
        act_to_joint = np.zeros(self._mj_model.nu, dtype=int)
        for i in range(self._mj_model.nu):
            jnt_id = self._mj_model.actuator_trnid[i, 0]
            act_to_joint[i] = jnt_id - 1  # joint index in qpos[7:]
        self._act_to_joint = jp.array(act_to_joint)

        # Torque-speed limits (optional; enabled via config.torque_speed_model).
        # Stall torque per joint = MJCF actuator_ctrlrange (remapped to joint order).
        # Velocity limit per joint from Unitree URDF, repeating (hip, thigh, calf) per leg.
        j2a = np.argsort(act_to_joint)
        ctrlrange_max = self._mj_model.actuator_ctrlrange[:, 1]
        self._stall_torque = jp.array(ctrlrange_max[j2a])
        self._velocity_limit = jp.tile(
            jp.array(consts.MOTOR_VELOCITY_LIMIT_PER_JOINT_TYPE), 4
        )
        self._torque_speed_model = bool(getattr(config, "torque_speed_model", False))

        # Rendering.
        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = xml_path
        self._imu_site_id = self._mj_model.site("imu").id

        # Contact sensor IDs.
        self._feet_floor_found_sensor = [
            self._mj_model.sensor(f"{geom}_floor_found").id
            for geom in consts.FEET_GEOMS
        ]

        # Torso body ID (base_link in unitree XML).
        self._torso_body_id = self._mj_model.body(consts.WARP_ROOT_BODY).id

    # ── Deploy / sim2sim parity metadata ───────────────────────────────

    def get_control_metadata(self) -> dict:
        """Return deploy-critical control parameters as a JSON-serializable dict.

        Picked up by `jax_rl/training/checkpointing.py:save_checkpoint` and
        written under `meta["control"]`. Read by `deploy/sim2sim_direct.py` and
        deploy-side tooling so they don't have to import (and stay in sync
        with) `deploy/go2_constants.py` constants per env. Closes the codex-
        audit P0 finding where sim2sim_direct used archived MJX gains
        (`KP_SIM=35.0`, `KD_SIM=0.1`) instead of Warp training's 20.0/0.5.
        """
        return {
            "Kp": float(self._config.Kp),
            "Kd": float(self._config.Kd),
            "action_scale": float(self._config.action_scale),
            "policy_dt": float(self._config.ctrl_dt),
            "physics_dt": float(self._config.sim_dt),
            "action_repeat": int(getattr(self._config, "action_repeat", 1)),
            "contact_mode": str(getattr(self._config, "contact_mode", "training")),
            "torque_speed_model": bool(getattr(self._config, "torque_speed_model", False)),
            "impl": str(getattr(self._config, "impl", "warp")),
            "joint_order": "policy_FL_FR_RL_RR",
            "action_order": "policy_FL_FR_RL_RR",
        }

    # ── Sensor readings (delegate to shared helpers) ───────────────────

    def get_upvector(self, data: mjx.Data) -> jax.Array:
        return go2_sensors.get_sensor_by_name(self.mj_model, data, consts.UPVECTOR_SENSOR)

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        return go2_sensors.get_gravity(data, self._imu_site_id)

    def get_global_linvel(self, data: mjx.Data) -> jax.Array:
        return go2_sensors.get_sensor_by_name(self.mj_model, data, consts.GLOBAL_LINVEL_SENSOR)

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        return go2_sensors.get_sensor_by_name(self.mj_model, data, consts.GLOBAL_ANGVEL_SENSOR)

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        return go2_sensors.get_sensor_by_name(self.mj_model, data, consts.LOCAL_LINVEL_SENSOR)

    def get_accelerometer(self, data: mjx.Data) -> jax.Array:
        return go2_sensors.get_sensor_by_name(self.mj_model, data, consts.ACCELEROMETER_SENSOR)

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        return go2_sensors.get_sensor_by_name(self.mj_model, data, consts.GYRO_SENSOR)

    def get_feet_pos(self, data: mjx.Data) -> jax.Array:
        return jp.vstack([
            go2_sensors.get_sensor_by_name(self.mj_model, data, name)
            for name in consts.FEET_POS_SENSOR
        ])

    # ── Actuator model ──────────────────────────────────────────────────

    def _apply_torque_speed_limit(
        self, tau_joint: jax.Array, dq: jax.Array
    ) -> jax.Array:
        """Clip joint torques by a linear torque-speed curve.

        tau_limit = stall_torque * max(1 - |dq| / velocity_limit, 0)

        No-op when config.torque_speed_model is False. When True, at |dq|=0
        the limit equals the MJCF ctrlrange (unchanged); at |dq|=velocity_limit
        the allowance reaches zero. Both inputs are in joint order.
        """
        if not self._torque_speed_model:
            return tau_joint
        scale = jp.maximum(1.0 - jp.abs(dq) / self._velocity_limit, 0.0)
        tau_limit = self._stall_torque * scale
        return jp.clip(tau_joint, -tau_limit, tau_limit)

    # ── Properties ──────────────────────────────────────────────────────

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
