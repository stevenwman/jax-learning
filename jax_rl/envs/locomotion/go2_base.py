"""Base class for Go2 environments.

Follows Playground's Go1Env pattern (base.py) but adapted for Go2:
  - Loads Go2 MJCF from Menagerie (via scene XML include)
  - Sets PD gains (Kp=20, Kd=0.5 per Unitree official)
  - Provides sensor reading helpers
"""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from jax_rl.envs.locomotion import go2_constants as consts


def get_assets() -> Dict[str, bytes]:
    """Load XML and mesh assets for Go2 model."""
    assets = {}
    # Our scene XML files.
    mjx_env.update_assets(assets, consts.ROOT_PATH, "*.xml")
    # Menagerie Go2 model + mesh assets.
    menagerie_path = mjx_env.MENAGERIE_PATH / "unitree_go2"
    mjx_env.update_assets(assets, menagerie_path, "*.xml")
    mjx_env.update_assets(assets, menagerie_path / "assets")
    return assets


class Go2Env(mjx_env.MjxEnv):
    """Base class for Go2 environments."""

    def __init__(
        self,
        xml_path: str,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        self._model_assets = get_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(
            epath.Path(xml_path).read_text(), assets=self._model_assets
        )
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mj_model.opt.ccd_iterations = 20  # Match Go1 (default 4 is too few)

        # Convert actuators from general (built-in PD) to motor (direct torque).
        # Every other Go2 RL pipeline uses motor + external PD. The general
        # actuator's biastype="affine" interacts with MuJoCo's integrator
        # differently than external PD, breaking sim2sim/sim2real transfer.
        import numpy as _np
        self._mj_model.actuator_gainprm[:, 0] = 1.0   # gain=1 (ctrl = torque)
        self._mj_model.actuator_biasprm[:, :] = 0.0    # no bias (pure torque)
        self._mj_model.dof_damping[6:] = 0.0            # no implicit damping (PD handles it)

        # Store PD gains for use in step()
        self._kp = config.Kp
        self._kd = config.Kd

        # Set actuator ctrl/force ranges to torque limits (now ctrl=torque, not position).
        # Hip/abduction: ±23.7 Nm, Knee: ±45.43 Nm (matching unitree_mujoco go2.xml)
        for i in range(self._mj_model.nu):
            name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if 'calf' in name.lower():
                self._mj_model.actuator_ctrlrange[i] = _np.array([-45.43, 45.43])
                self._mj_model.actuator_forcerange[i] = _np.array([-45.43, 45.43])
            else:
                self._mj_model.actuator_ctrlrange[i] = _np.array([-23.7, 23.7])
                self._mj_model.actuator_forcerange[i] = _np.array([-23.7, 23.7])

        # Fix rear thigh joint range: Menagerie uses front range for all legs,
        # but real Go2 rear hips have different range [-0.5236, 4.5379].
        for jname in ["RL_thigh_joint", "RR_thigh_joint"]:
            jid = self._mj_model.joint(jname).id
            self._mj_model.jnt_range[jid] = _np.array([-0.5236, 4.5379])

        # Override Menagerie's soft foot contacts with Go1-style firm contacts.
        # Menagerie: solimp=0.015 1 0.031, condim=6 (marshmallow-soft, full friction)
        # Go1 (PG): solimp=0.9 0.95 0.023, condim=3 (firm, basic friction)
        # Soft contacts prevent crisp push-off needed for walking gaits.
        import numpy as _np
        for foot_name in ["FL", "FR", "RL", "RR"]:
            gid = self._mj_model.geom(foot_name).id
            self._mj_model.geom_solimp[gid, :3] = _np.array([0.9, 0.95, 0.023])
            self._mj_model.geom_condim[gid] = 3
            self._mj_model.geom_friction[gid] = _np.array([0.6, 0.005, 0.0001])

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

    # ── Sensor readings ─────────────────────────────────────────────────

    def get_upvector(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, consts.UPVECTOR_SENSOR
        )

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        return data.site_xmat[self._imu_site_id].T @ jp.array([0, 0, -1])

    def get_global_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, consts.GLOBAL_LINVEL_SENSOR
        )

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, consts.GLOBAL_ANGVEL_SENSOR
        )

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, consts.LOCAL_LINVEL_SENSOR
        )

    def get_accelerometer(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, consts.ACCELEROMETER_SENSOR
        )

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, consts.GYRO_SENSOR
        )

    def get_feet_pos(self, data: mjx.Data) -> jax.Array:
        return jp.vstack([
            mjx_env.get_sensor_data(self.mj_model, data, name)
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
