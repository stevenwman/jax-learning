"""Joystick velocity tracking task for Unitree Go2 (Warp backend).

Parallel to go2_joystick.py but uses unitree_mujoco's go2.xml via Warp.
Full collision geometry (cylinders + boxes) -- no MJX simplifications.

Returns dict obs: {"state": 48d policy obs, "privileged_state": 122d critic obs}.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from jax_rl.envs.locomotion import go2_warp_base
from jax_rl.envs.locomotion import go2_constants as consts
from jax_rl.envs.obs_spec import ObsTerm, IncludeGroup, compute_obs
from jax_rl.envs.reward_spec import RewardTerm, compute_rewards


def default_config() -> config_dict.ConfigDict:
    """Default joint-PD joystick config. Canonical builder: go2_config()."""
    from jax_rl.envs.locomotion.go2_warp_variants import go2_config
    return go2_config()


class WarpJoystick(go2_warp_base.Go2WarpEnv):
    """Track a joystick velocity command with Go2 (Warp backend)."""

    # Scene XML to load. Subclasses override (e.g. a rough-heightfield scene).
    _scene_xml = consts.WARP_SCENE_FLAT_XML

    # gait_participation timer cap (steps). 25 @ dt=0.02 → every foot must make
    # contact within 0.5s or the penalty saturates. See the reward term.
    GAIT_FULL_CAP = 25.0

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=self._scene_xml.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        # Low-level controller (JointPD | OSC | VarImpedance), selected from the
        # (post-override) config. Function-local import to avoid a base↔components
        # cycle. Built before _post_init so setup() can cache controller geometry.
        from jax_rl.envs.locomotion.go2_warp_components import controller_from_config
        self._controller = controller_from_config(self._config)
        self._post_init()

    @property
    def action_size(self) -> int:
        return self._controller.action_size(self)

    def _post_init(self) -> None:
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

        # Soft joint limits (first joint is freejoint).
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

        self._torso_body_id = self._mj_model.body(consts.WARP_ROOT_BODY).id

        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in consts.FEET_SITES]
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
                ObsTerm("gyro", lambda data, **kw: self.get_gyro(data),
                        noise_scale=noise.gyro),
                ObsTerm("accelerometer", lambda data, **kw: self.get_accelerometer(data),
                        noise_scale=noise.accelerometer),
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
                ObsTerm("accelerometer_clean", lambda data, **kw: self.get_accelerometer(data)),
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
            RewardTerm("base_height", lambda data, **kw:
                self._cost_base_height(data)),
            # ── RMA "natural constraints" (arXiv 2107.04034) ───────────────
            # Bioenergetic terms that shape a natural gait WITHOUT explicit
            # swing/air-time shaping. Default scale 0; the RMA reward profile
            # turns them on at the paper's coefficients.
            # ground_impact/torque_smoothness need foot-force + prev-step carry
            # state. Subclasses with their own reset/step (postrack &c) don't
            # supply those and don't use RMA reward → no-op to 0 there (their
            # scale is 0 anyway). Guards are trace-time (None / static key check).
            RewardTerm("ground_impact", lambda info, foot_force=None, **kw:
                self._cost_ground_impact(foot_force, info["last_foot_force"])
                if foot_force is not None else jp.zeros(())),
            RewardTerm("torque_smoothness", lambda data, info, **kw:
                self._cost_torque_smoothness(data.actuator_force, info["last_torque"])
                if "last_torque" in info else jp.zeros(())),
            RewardTerm("action_magnitude", lambda action, **kw:
                self._cost_action_magnitude(action)),
            RewardTerm("joint_speed", lambda data, **kw:
                self._cost_joint_speed(data.qvel[6:])),
            # Gait-participation: penalize how long since ALL FOUR feet last each
            # made contact (a "since-reset" timer, NOT simultaneous 4-contact —
            # a trot is never 4-down-at-once). A hung foot never completes the
            # set → the timer (info["steps_since_full"]) grows → penalty. Capped
            # + normalized to [0,1]. Forces every foot to participate within
            # GAIT_FULL_CAP steps. Default scale 0; guarded for subclasses whose
            # own reset/step don't maintain the timer.
            RewardTerm("gait_participation", lambda info, **kw:
                jp.minimum(info["steps_since_full"], self.GAIT_FULL_CAP) / self.GAIT_FULL_CAP
                if "steps_since_full" in info else jp.zeros(())),
        ]

        # Controller geometry/gains (no-op for JointPD; OSC/VarImpedance cache
        # leg DoFs, foot sites, nominal foot positions, gains — see components).
        self._controller.setup(self)

        # Environmental force field (NoField unless config has a `mud` block).
        # Applied inside the controller's substep scan; NoField = identity.
        from jax_rl.envs.locomotion.go2_warp_components import field_from_config
        self._force_field = field_from_config(self._config)
        self._force_field.setup(self)

    # ── Domain Randomization ─────────────────────────────────────────────

    def get_domain_randomization_spec(self):
        """Declare per-episode domain randomization for DomainRandWrapper."""
        from jax_rl.envs.wrappers.domain_rand import DRSpec
        return [
            # Model-level DR
            DRSpec(name="friction", type="model", field="geom_friction",
                   column=0, min=0.3, max=1.5, per_element=False, operation="set",
                   description="Uniform friction across all geoms"),
            DRSpec(name="dof_damping", type="model", field="dof_damping",
                   indices=(6, 18), min=0.7, max=2.0, per_element=True,
                   description="Joint damping variation"),
            DRSpec(name="dof_armature", type="model", field="dof_armature",
                   indices=(6, 18), min=0.9, max=1.3, per_element=True,
                   description="Joint armature variation"),
            DRSpec(name="dof_frictionloss", type="model", field="dof_frictionloss",
                   indices=(6, 18), min=0.7, max=1.5, per_element=True,
                   description="Joint friction loss variation"),
            DRSpec(name="body_mass", type="model", field="body_mass",
                   min=0.8, max=1.2, per_element=True,
                   description="Per-link mass variation"),
            DRSpec(name="motor_strength", type="model", field="actuator_gainprm",
                   column=0, min=0.9, max=1.1, per_element=True,
                   description="Per-actuator motor heterogeneity"),
            # Torso COM jitter (x, y offset of body inertia center)
            DRSpec(name="torso_com_jitter", type="model", field="body_ipos",
                   indices=(1, 2), min=-0.03, max=0.03,
                   per_element=True, operation="add",
                   description="Torso COM offset (x, y)"),
            # Per-link inertia tensor scale
            DRSpec(name="body_inertia", type="model", field="body_inertia",
                   min=0.85, max=1.15, per_element=True,
                   description="Per-link inertia tensor variation"),
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
            ctrl=jp.zeros(self.mjx_model.nu),
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            naccdmax=self._config.naccdmax,
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

        # Per-episode kick magnitude (domain-randomized disturbance strength):
        # sample the kick-velocity bound once; step() kicks within ±this.
        rng, push_key = jax.random.split(rng)
        push_vel_max = jax.random.uniform(
            push_key,
            minval=self._config.push_config.vel_min,
            maxval=self._config.push_config.vel_max,
        )

        info = {
            "rng": rng,
            "command": cmd,
            "push_vel_max": push_vel_max,
            "steps_until_next_cmd": steps_until_next_cmd,
            # Sized by action_size (not nu) so subclasses whose policy action is
            # wider than the actuator count (e.g. variable-impedance adds per-foot
            # stiffness dims) get a correctly-shaped last_act. Identical to nu for
            # every fixed-action env.
            "last_act": jp.zeros(self.action_size),
            "last_last_act": jp.zeros(self.action_size),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "swing_peak": jp.zeros(4),
            # RMA smoothness/impact terms need the previous step's torque + foot
            # contact force (default 0 -> first step's delta is just the value).
            "last_torque": jp.zeros(self._mj_model.nu),
            "last_foot_force": jp.zeros(4),
            # Previous control-step foot velocity (4,3) for the virtual-mass
            # controller's ẍ = finite-diff foot accel. 0 at reset.
            "last_foot_vel": jp.zeros((4, 3)),
            # gait_participation: per-foot "touched since last reset" + a timer
            # counting steps since all four last completed a contact set.
            "feet_touched": jp.zeros(4, dtype=bool),
            "steps_since_full": jp.zeros(()),
            "step_count": jp.int32(0),
            "reward_components": {
                k: jp.zeros(()) for k in self._config.reward_config.scales.keys()
            },
        }

        # Per-episode force-field params (mud depth/coeffs &c). NoField returns {}
        # WITHOUT consuming rng → reset stays bit-identical for non-mud envs.
        info.update(self._force_field.sample(self, rng))

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        obs = self._get_obs(data, info)

        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _apply_control(
        self, data: mjx.Data, action: jax.Array, info: dict | None = None
    ) -> mjx.Data:
        """Low-level controller at physics rate (decimation), delegated to the
        Controller component (JointPD / OSC / VarImpedance) chosen from config.
        The OSC mechanics live on the controller (see go2_warp_components).
        ``info`` carries per-episode force-field params (mud &c) into the
        substep scan; None for callers that don't thread it (force field is
        then NoField/identity)."""
        return self._controller.apply(self, data, action, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # Random velocity kick every push_interval steps. The magnitude bound is
        # per-episode (info["push_vel_max"], sampled at reset) so disturbance
        # strength is domain-randomizable via push_config.vel_{min,max}.
        step_count = state.info["step_count"]
        push_interval = self._config.push_config.interval
        rng, push_key = jax.random.split(state.info["rng"])
        pmax = state.info["push_vel_max"]
        push_vel = jax.random.uniform(push_key, (2,), minval=-pmax, maxval=pmax)
        do_push = (step_count > 0) & (step_count % push_interval == 0)
        data = state.data
        new_qvel = data.qvel.at[0:2].set(
            jp.where(do_push, data.qvel[0:2] + push_vel, data.qvel[0:2])
        )
        data = data.replace(qvel=new_qvel)

        # Low-level controller at physics rate (decimation). Overridable.
        # Pass info so per-episode force-field params (mud &c) reach the substep.
        data = self._apply_control(data, action, state.info)

        # Foot contact detection. The floor-contact touch sensor reads the
        # per-foot normal-force MAGNITUDE; >0 == in contact, raw value == the
        # force used by the RMA ground-impact term.
        foot_force = jp.array([
            data.sensordata[self._mj_model.sensor_adr[sid]]
            for sid in self._feet_floor_found_sensor
        ])
        contact = foot_force > 0
        # gait_participation timer: accumulate which feet have touched since the
        # last reset; when all four have, reset timer + bitmap, else timer += 1.
        touched = state.info["feet_touched"] | contact
        all_touched = jp.all(touched)
        state.info["steps_since_full"] = jp.where(
            all_touched, 0.0, state.info["steps_since_full"] + 1.0)
        state.info["feet_touched"] = jp.where(
            all_touched, jp.zeros(4, dtype=bool), touched)
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
            data, action, state.info, state.metrics, done, first_contact, contact,
            foot_force,
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
        # RMA smoothness/impact carry-state (this step's torque + foot force
        # become next step's "previous").
        state.info["last_torque"] = data.actuator_force
        state.info["last_foot_force"] = foot_force
        # This step's foot velocity → next step's "previous" for the virtual-mass
        # ẍ finite difference (world-frame foot linvel sensor, (4,3)).
        state.info["last_foot_vel"] = (
            data.sensordata[self._foot_linvel_sensor_adr].reshape(4, 3))
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
        foot_force: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        del metrics  # UNUSED — kept for potential curriculum-dependent rewards
        return compute_rewards(
            self._reward_spec,
            data=data, action=action, info=info,
            done=done, first_contact=first_contact, contact=contact,
            foot_force=foot_force,
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

    # ── RMA "natural constraints" (arXiv 2107.04034, Sec III-A) ─────────────
    def _cost_ground_impact(
        self, foot_force: jax.Array, last_foot_force: jax.Array
    ) -> jax.Array:
        """RMA Ground Impact: -||f_t - f_{t-1}||^2 over the 4 feet (penalizes
        abrupt contact-force changes = hard foot strikes). foot_force = per-foot
        normal-force magnitude from the floor-contact touch sensors."""
        return jp.sum(jp.square(foot_force - last_foot_force))

    def _cost_torque_smoothness(
        self, torques: jax.Array, last_torque: jax.Array
    ) -> jax.Array:
        """RMA Smoothness: -||tau_t - tau_{t-1}||^2 (torque jerk)."""
        return jp.sum(jp.square(torques - last_torque))

    def _cost_action_magnitude(self, action: jax.Array) -> jax.Array:
        """RMA Action Magnitude: -||a||^2. Only the first 12 dims (joint position
        targets) — excludes the variable-impedance stiffness/damping dims so the
        term doesn't fight impedance modulation."""
        return jp.sum(jp.square(action[:12]))

    def _cost_joint_speed(self, qvel_joints: jax.Array) -> jax.Array:
        """RMA Joint Speed: -||qdot||^2 over the 12 actuated joints."""
        return jp.sum(jp.square(qvel_joints))

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
        weight = jp.array([1.0, 1.0, 0.1] * 4)
        return jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * weight))

    def _cost_base_height(self, data: mjx.Data) -> jax.Array:
        """Penalize deviation from target standing height (0.27m)."""
        base_z = data.subtree_com[self._torso_body_id][2]
        return jp.square(base_z - 0.27)

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


class WarpJoystickNoAccel(WarpJoystick):
    """Ablation: drop accelerometer from actor obs.

    Critic still sees `accelerometer_clean` (privileged-only term), so this
    isolates the question of whether the actor needs accel signal at all,
    not whether accel info is useful at the critic.

    Effective dims: state 48d → 45d, privileged_state 122d → 119d
    (privileged shrinks because it does IncludeGroup("state") which now
    pulls the smaller 45d state).
    """
    def _post_init(self) -> None:
        super()._post_init()
        self._obs_groups["state"] = [
            t for t in self._obs_groups["state"]
            if not (hasattr(t, "name") and t.name == "accelerometer")
        ]
