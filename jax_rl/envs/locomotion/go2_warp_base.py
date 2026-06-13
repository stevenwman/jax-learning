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


def torque_speed_clip(
    tau: jax.Array,
    dq: jax.Array,
    saturation: jax.Array,
    velocity_limit: jax.Array,
    effort_limit: jax.Array,
) -> jax.Array:
    """Clip joint torque by a linear DC-motor torque-speed curve (4-quadrant).

    Port of mjlab's ``DcMotorActuator._clip_effort``
    (``src/mjlab/actuator/dc_actuator.py``). All args broadcast per-joint.

      vel_at_eff = velocity_limit * (1 + effort_limit / saturation)
      v   = clip(dq, ±vel_at_eff)
      top = saturation * (1 - v / velocity_limit)      # upper envelope
      bot = saturation * (-1 - v / velocity_limit)     # lower envelope
      allow ∈ [max(bot, -effort_limit), min(top, effort_limit)]

    Properties (per joint): at dq=0 the cap is ±effort_limit; driving torque
    decays to 0 at |dq|=velocity_limit (no-load speed) while braking torque
    stays available; at |dq|≥vel_at_eff the window collapses to full braking,
    actively decelerating an over-spun joint. ``saturation`` is the stall
    (peak) torque, ``effort_limit`` the continuous (flat-top) torque ≤ stall.
    """
    vel_at_eff = velocity_limit * (1.0 + effort_limit / saturation)
    v = jp.clip(dq, -vel_at_eff, vel_at_eff)
    top = saturation * (1.0 - v / velocity_limit)
    bot = saturation * (-1.0 - v / velocity_limit)
    max_eff = jp.minimum(top, effort_limit)
    min_eff = jp.maximum(bot, -effort_limit)
    return jp.clip(tau, min_eff, max_eff)


def physical_armature(mj_model) -> np.ndarray:
    """Return a copy of ``dof_armature`` with mjlab per-joint rotor inertia set.

    Calf (knee) dofs get :data:`consts.MOTOR_ARMATURE_KNEE` (gear-9 cam), all
    other actuated leg joints get :data:`consts.MOTOR_ARMATURE_HIP` (gear-6).
    The 6 freejoint (base) dofs are left untouched. Joints are identified by
    name (``*_calf_joint``) so the result is independent of dof ordering.
    """
    arm = np.array(mj_model.dof_armature, copy=True)
    for i in range(mj_model.nu):
        jnt_id = mj_model.actuator_trnid[i, 0]
        adr = mj_model.jnt_dofadr[jnt_id]
        name = mj_model.jnt(jnt_id).name
        arm[adr] = (
            consts.MOTOR_ARMATURE_KNEE if name.endswith("calf_joint")
            else consts.MOTOR_ARMATURE_HIP
        )
    return arm


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

        # Terrain component (Flat | RoughHF), built from the legacy rough_* config
        # keys. Loaded via MjSpec so the floor can be reshaped programmatically
        # (rough heightfield) without a second scene XML: `apply` mutates the spec
        # before compile (geom type + hfield asset); `customize_model` fills the
        # exact hfield_data after compile. Flat → pure round-trip (verified
        # bit-identical to from_xml_string). See go2_warp_components.
        from jax_rl.envs.locomotion.go2_warp_components import terrain_from_config
        self._terrain = terrain_from_config(config)
        spec = mujoco.MjSpec.from_string(
            epath.Path(xml_path).read_text(), self._model_assets
        )
        spec.meshdir = ""                       # asset keys are bare filenames
        for _k, _v in self._model_assets.items():
            spec.assets[_k] = _v                # from_string doesn't load VFS bytes
        self._terrain.apply(spec)
        self._mj_model = spec.compile()

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
        # Continuous (flat-top) torque = stall * frac. frac=1.0 → no thermal
        # derating (the curve's flat top sits at the MJCF ctrlrange/stall).
        effort_frac = float(getattr(config, "continuous_effort_frac", 1.0))
        self._effort_limit = self._stall_torque * effort_frac

        # Actuation component (TorqueOnly | MotorModel), built from the legacy
        # config flags during the staged composition migration. Owns the per-joint
        # armature — set here, BEFORE mjx.put_model, so it reaches Λ — and the
        # per-substep torque-speed clip. See go2_warp_components.
        from jax_rl.envs.locomotion.go2_warp_components import actuation_from_config
        self._actuation = actuation_from_config(config)
        self._actuation.customize_model(self._mj_model)

        # Rendering.
        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        # Terrain post-compile data (e.g. rough-heightfield elevation into
        # hfield_data). No-op for Flat. Runs before mjx.put_model so the data
        # reaches the backend.
        self._terrain.customize_model(self._mj_model)

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
        written under `meta["control"]`. Read by `deploy/sim2sim_direct.py`,
        `deploy/robot_interface.py`, and `deploy/obs_builder.py` so they don't
        have to import (and stay in sync with) `deploy/go2_constants.py`
        constants per env. Closes the codex-audit P0 finding where sim2sim_direct
        used archived MJX gains (`KP_SIM=35.0`, `KD_SIM=0.1`) instead of Warp
        training's 20.0/0.5; also closes the deploy-contract drift where
        `default_pose` and joint remap lived only in deploy-side constants.

        Phase D2 (2026-04-27): adds `default_pose_policy` (sourced from XML
        keyframe — single source of truth), `default_pose_sdk` (remapped),
        plus `policy_joint_names` and `sdk_joint_names`. SDK order
        (`FR,FL,RR,RL`) is Unitree-spec for Go2, hardcoded here.
        """
        # Default pose in policy order, sourced from `self._default_pose`
        # which the subclass sets in _post_init from the appropriate keyframe
        # (joystick → "home", bongo handstand → "handstand"). Same array the
        # env uses for `joint_pos_offset` obs and `motor_targets` step.
        if not hasattr(self, "_default_pose"):
            raise RuntimeError(
                f"{type(self).__name__}.get_control_metadata: "
                f"self._default_pose not set. Subclass must set it in "
                f"_post_init before save_checkpoint runs."
            )
        default_pose_policy = np.asarray(self._default_pose, dtype=np.float32)
        if default_pose_policy.shape != (self._mj_model.nu,):
            raise RuntimeError(
                f"{type(self).__name__}.get_control_metadata: "
                f"_default_pose shape {default_pose_policy.shape} != "
                f"({self._mj_model.nu},). Did the subclass set the wrong slice?"
            )

        # Joint names in policy order (FL,FR,RL,RR per leg, hip→thigh→calf).
        # qpos[7:] joints are body-tree-ordered. mj_model.jnt(0) is the freejoint.
        policy_joint_names = [
            self._mj_model.jnt(i + 1).name for i in range(self._mj_model.nu)
        ]

        # Unitree SDK joint order for Go2 — fixed by hardware spec.
        # FR(0..2), FL(3..5), RR(6..8), RL(9..11).
        # POLICY_TO_SDK[i] = which policy index goes into SDK slot i.
        # Identical to `deploy/go2_constants.py::POLICY_TO_SDK` (kept in sync
        # by item 2's equality assertion at deploy load).
        policy_to_sdk = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        sdk_to_policy = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        default_pose_sdk = default_pose_policy[np.array(policy_to_sdk)]
        sdk_joint_names = [policy_joint_names[i] for i in policy_to_sdk]

        # OSC / variable-impedance controller config — so the Newton eval harness
        # (projects/mud_eval/mud_osc.py) configures itself from the ckpt instead of
        # GUESSING params (the param-mismatch that caused the mass-controller
        # catapult + bare-vs-Λ errors). None for joint-PD envs.
        osc_meta = None
        _osc = getattr(self._config, "osc", None)
        if _osc is not None:
            osc_meta = {
                "use_op_space_inertia": bool(_osc.use_op_space_inertia),
                "target_mode": str(_osc.target_mode),
                "ridge": float(_osc.ridge),
                "kp": [float(x) for x in _osc.kp],
                "kd": [float(x) for x in _osc.kd],
                "stiffness_granularity": str(getattr(_osc, "stiffness_granularity", "")),
                "damping_action": bool(getattr(_osc, "damping_action", False)),
                "mass_action": bool(getattr(_osc, "mass_action", False)),
                "var_s_min": float(getattr(_osc, "var_s_min", 0.25)),
                "var_s_max": float(getattr(_osc, "var_s_max", 2.0)),
                "var_zeta_min": float(getattr(_osc, "var_zeta_min", 0.5)),
                "var_zeta_max": float(getattr(_osc, "var_zeta_max", 2.0)),
                "var_a_min": float(getattr(_osc, "var_a_min", 0.0)),
                "var_a_max": float(getattr(_osc, "var_a_max", 2.0)),
                "var_xdd_ema": float(getattr(_osc, "var_xdd_ema", 1.0)),
            }

        return {
            "Kp": float(self._config.Kp),
            "Kd": float(self._config.Kd),
            "osc": osc_meta,
            "action_scale": float(self._config.action_scale),
            "policy_dt": float(self._config.ctrl_dt),
            "physics_dt": float(self._config.sim_dt),
            "action_repeat": int(getattr(self._config, "action_repeat", 1)),
            "contact_mode": str(getattr(self._config, "contact_mode", "training")),
            "torque_speed_model": bool(getattr(self._config, "torque_speed_model", False)),
            "impl": str(getattr(self._config, "impl", "warp")),
            "joint_order": "policy_FL_FR_RL_RR",
            "action_order": "policy_FL_FR_RL_RR",
            "default_pose_policy": default_pose_policy.tolist(),
            "default_pose_sdk": default_pose_sdk.tolist(),
            "policy_joint_names": policy_joint_names,
            "sdk_joint_names": sdk_joint_names,
            "policy_to_sdk": policy_to_sdk,
            "sdk_to_policy": sdk_to_policy,
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
        """Apply the Actuation component's torque-speed clip (joint order).

        Delegates to ``self._actuation.clip_torque`` — identity for TorqueOnly,
        the mjlab DC-motor curve for MotorModel. Kept under the old name so the
        controllers' substep loops call through unchanged.
        """
        return self._actuation.clip_torque(
            tau_joint, dq,
            self._stall_torque, self._velocity_limit, self._effort_limit,
        )

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
