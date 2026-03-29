"""Shared sensor helpers for Go2 environments (MJX and Warp).

Standalone functions — no class dependency. Both Go2Env (MJX) and
Go2WarpEnv (Warp) call these via thin instance method wrappers.
"""

import jax.numpy as jp

from mujoco_playground._src import mjx_env


def get_sensor_by_name(mj_model, data, sensor_name):
    """Read sensor data by name."""
    return mjx_env.get_sensor_data(mj_model, data, sensor_name)


def get_gravity(data, imu_site_id):
    """Projected gravity in body frame from IMU site rotation matrix."""
    return data.site_xmat[imu_site_id].T @ jp.array([0, 0, -1])
