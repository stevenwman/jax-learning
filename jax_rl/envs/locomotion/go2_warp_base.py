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
        self._mj_model.opt.ccd_iterations = 20

        # --- contact_mode overrides ---
        # "training": firm foot contacts for crisp push-off (matches Go1 PG).
        # "deploy": keep unitree XML native contacts (condim=6, soft solimp).
        if getattr(self._config, 'contact_mode', 'training') == 'training':
            import numpy as _np
            for foot_name in consts.FEET_GEOMS:
                gid = self._mj_model.geom(foot_name).id
                self._mj_model.geom_solimp[gid, :3] = _np.array([0.9, 0.95, 0.023])
                self._mj_model.geom_condim[gid] = 3
                self._mj_model.geom_friction[gid] = _np.array([0.6, 0.005, 0.0001])

        # PD gains for external PD in step().
        self._kp = config.Kp
        self._kd = config.Kd

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
