"""Go2 SplitbeltTreadmill env (Warp backend). See spec at
.superpowers/specs/2026-05-02-splitbelt-treadmill-env-design.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env

from jax_rl.envs.locomotion import go2_warp_base
from jax_rl.envs.locomotion import go2_constants as consts
from jax_rl.envs.locomotion import splitbelt_geom as geom
from jax_rl.envs.locomotion import splitbelt_schedules as sched
from jax_rl.envs.obs_spec import ObsTerm, IncludeGroup, compute_obs


_VALID_OBS_MODES = ("blind", "informed", "error", "history")

# Shared proprio name list — kept in sync with build_obs_groups bindings below.
_PROPRIO_NAMES = ("joint_pos", "joint_vel", "last_act", "gravity", "gyro", "command")


def obs_term_names(obs_mode: str) -> Dict[str, list[str]]:
    """Pure name-layout (S§5 obs modes). Hermetic — no env, no fn binding.

    Returns dict of {"state": [name, ...], "privileged_state": [name, ...]}.
    Mirrors the ObsTerm structure that `build_obs_groups` produces; used by tests
    and as the source-of-truth name list.
    """
    if obs_mode not in _VALID_OBS_MODES:
        raise ValueError(
            f"obs_mode must be one of {_VALID_OBS_MODES}, got {obs_mode!r}"
        )
    proprio = list(_PROPRIO_NAMES)
    if obs_mode == "blind" or obs_mode == "history":
        # history applies frame-stacking via wrapper; same name layout as blind.
        state_names = list(proprio)
    elif obs_mode == "informed":
        state_names = list(proprio) + ["belt_vel"]
    elif obs_mode == "error":
        state_names = list(proprio) + ["cmd_track_error", "drift_xy"]
    else:
        raise AssertionError("unreachable")
    privileged_names = list(proprio) + [
        "belt_vel", "cmd_track_error", "drift_xy",
        "base_lin_vel", "base_ang_vel",
    ]
    return {"state": state_names, "privileged_state": privileged_names}


def default_config() -> config_dict.ConfigDict:
    """Default config: tied belts at 0.5 m/s, blind obs, cmd=0 (smoke baseline).

    Reward scales ported VERBATIM from go2_warp_joystick.default_config (S§7.1).
    Empty / `.get(name, default)` patterns are forbidden — they silently produce
    ~10× weaker tracking reward and lose the calibration target (S§10.5).
    """
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1250,  # 25 s
        Kp=20.0,
        Kd=0.5,
        torque_speed_model=False,
        action_repeat=1,
        action_scale=1.0,
        soft_joint_pos_limit_factor=0.95,
        impl="warp",
        contact_mode="training",
        # Obs noise (consumed by compute_obs at step time; matches joystick env shape).
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03, joint_vel=1.5, gyro=0.2, gravity=0.05,
                linvel=0.1, accelerometer=0.1,
            ),
        ),
        # Splitbelt-specific
        obs_mode="blind",
        history_len=4,
        belt_layout=config_dict.create(
            left_y_min=-0.325, left_y_max=-0.025,
            right_y_min=0.025, right_y_max=0.325,
        ),
        schedule_kind="tied",
        schedule_params=config_dict.create(v=0.5),
        cmd_zero=True,
        treadmill_drift_lateral_weight=2.0,
        treadmill_drift_forward_weight=0.5,
        # Reward scales — ported verbatim from go2_warp_joystick:47-69.
        # Splitbelt-specific term `treadmill_drift` added; `stand_still` dropped
        # because cmd is always zero (the term collapses with treadmill_drift).
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_lin_vel=10.0,
                tracking_ang_vel=5.0,
                lin_vel_z=-0.5,
                ang_vel_xy=-0.05,
                orientation=-5.0,
                torques=-0.0002,
                action_rate=-0.01,
                energy=-0.001,
                dof_pos_limits=-1.0,
                feet_air_time=0.1,
                feet_slip=-0.1,
                feet_clearance=-2.0,
                feet_height=-0.2,
                termination=-1.0,
                pose=0.5,
                base_height=-5.0,
                # New for splitbelt:
                treadmill_drift=1.0,
            ),
            tracking_sigma=0.25,
            max_foot_height=0.1,
        ),
        # MJX/Warp tuning
        naconmax=4 * 8192,
        naccdmax=4000,
        njmax=100,
    )


def build_obs_groups(env: Any) -> Dict[str, list]:
    """Build real ObsTerm dispatch with env-method-bound lambdas (used in _post_init).

    Names must match `obs_term_names(env._config.obs_mode)` exactly — that is the
    contract validated in tests. Pattern follows go2_warp_joystick.py:128-159.
    """
    cfg = env._config
    layout = obs_term_names(cfg.obs_mode)
    noise = cfg.noise_config.scales

    term_factory = {
        "joint_pos": (lambda data, **kw: data.qpos[7:7+12] - env._default_pose, noise.joint_pos),
        "joint_vel": (lambda data, **kw: data.qvel[6:6+12], noise.joint_vel),
        "last_act": (lambda info, **kw: info["last_act"], 0.0),
        "gravity": (lambda data, **kw: env.get_gravity(data), noise.gravity),
        "gyro": (lambda data, **kw: env.get_gyro(data), noise.gyro),
        "command": (lambda info, **kw: info["command"], 0.0),
        "belt_vel": (lambda info, **kw: info["splitbelt"]["belt_vel"], 0.0),
        "cmd_track_error": (lambda info, **kw: info["splitbelt"]["cmd_track_error"], 0.0),
        "drift_xy": (lambda info, **kw: info["splitbelt"]["drift_xy"], 0.0),
        "base_lin_vel": (lambda data, **kw: env.get_local_linvel(data), 0.0),
        "base_ang_vel": (lambda data, **kw: env.get_global_angvel(data), 0.0),
    }

    def _build(names):
        return [ObsTerm(name=n, fn=term_factory[n][0], noise_scale=term_factory[n][1])
                for n in names]

    state_terms = _build(layout["state"])
    state_set = set(layout["state"])
    priv_extras = [n for n in layout["privileged_state"] if n not in state_set]
    privileged_terms = [IncludeGroup("state")] + _build(priv_extras)

    return {"state": state_terms, "privileged_state": privileged_terms}


class Go2WarpSplitbeltEnv(go2_warp_base.Go2WarpEnv):
    """Go2 on a split-belt treadmill (S§5.3)."""

    def __init__(
        self,
        task: str = "splitbelt",
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list]]] = None,
    ) -> None:
        del task  # accepted for Playground registry compatibility
        cfg = config if config is not None else default_config()
        xml_path = (
            Path(__file__).parent / "xmls" / "go2_warp_splitbelt_scene.xml"
        ).as_posix()
        super().__init__(
            xml_path=xml_path, config=cfg, config_overrides=config_overrides
        )
        self._post_init()

    @property
    def action_size(self) -> int:
        # Override Go2WarpEnv.action_size (which returns mjx_model.nu = 14).
        # Belt actuators are env-internal; the policy only controls 12 leg actuators.
        return 12

    def _post_init(self) -> None:
        # NOTE: Go2WarpEnv.__init__ does NOT call _post_init itself; subclass invokes
        # explicitly from its __init__ (joystick precedent).
        self._action_dim = 12
        self._init_q = jp.array(self._mj_model.keyframe("splitbelt_spawn").qpos)
        self._default_pose = jp.array(
            self._mj_model.keyframe("splitbelt_spawn").qpos[7:7+12]
        )

        # Soft joint limits — first joint is freejoint; next 12 are leg joints.
        self._lowers, self._uppers = self.mj_model.jnt_range[1:1+12].T
        self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

        self._torso_body_id = self._mj_model.body(consts.WARP_ROOT_BODY).id
        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in consts.FEET_SITES]
        )

        cfg = self._config
        self._belt_layout = geom.BeltLayout(
            left_y_min=cfg.belt_layout.left_y_min,
            left_y_max=cfg.belt_layout.left_y_max,
            right_y_min=cfg.belt_layout.right_y_min,
            right_y_max=cfg.belt_layout.right_y_max,
        )

        # Belt actuator + joint IDs.
        self._left_belt_act_id = self._mj_model.actuator("left_belt_vel").id
        self._right_belt_act_id = self._mj_model.actuator("right_belt_vel").id
        self._left_belt_qposadr = self._mj_model.joint("left_belt_joint").qposadr[0]
        self._right_belt_qposadr = self._mj_model.joint("right_belt_joint").qposadr[0]
        self._left_belt_dofadr = self._mj_model.joint("left_belt_joint").dofadr[0]
        self._right_belt_dofadr = self._mj_model.joint("right_belt_joint").dofadr[0]

        # Belt actuator addresses in `data.ctrl` (same as actuator id in MJX).
        self._left_belt_ctrl_idx = self._left_belt_act_id
        self._right_belt_ctrl_idx = self._right_belt_act_id

        # Contact-pair sensor IDs for foot_belt_id + termination cause (S§9.1).
        feet_order = ("FL", "FR", "RL", "RR")

        def _adr(name):
            sid = self._mj_model.sensor(name).id
            return self._mj_model.sensor_adr[sid]

        self._foot_left_belt_adr = jp.array(
            [_adr(f"{f}_left_belt") for f in feet_order]
        )
        self._foot_right_belt_adr = jp.array(
            [_adr(f"{f}_right_belt") for f in feet_order]
        )
        self._foot_floor_adr = jp.array(
            [_adr(f"{f}_floor_found") for f in feet_order]
        )
        self._torso_left_belt_adr = _adr("torso_left_belt")
        self._torso_right_belt_adr = _adr("torso_right_belt")
        self._torso_floor_adr = _adr("torso_floor")

        # Foot global linvel sensor addresses — needed by joystick reward helpers.
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

        # Leg-only act_to_joint slice (base class _act_to_joint has length nu=14).
        leg_act_ids = jp.array([
            self._mj_model.actuator(name).id for name in consts.LEG_ACTUATOR_NAMES
        ])
        self._leg_act_ids = leg_act_ids
        self._leg_act_to_joint = self._act_to_joint[leg_act_ids]

        # Restore belt actuator forcerange + RE-PUT model into MJX. Empirically
        # verified that Go2WarpEnv.__init__:64-67 clobbers forcerange BEFORE
        # mjx.put_model. XML default isn't enough — must mutate _mj_model and
        # re-snapshot. See spec §6.3 / round-3 audit.
        self._mj_model.actuator_forcerange[self._left_belt_act_id] = np.array([-200.0, 200.0])
        self._mj_model.actuator_forcerange[self._right_belt_act_id] = np.array([-200.0, 200.0])
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        assert float(self._mjx_model.actuator_forcerange[self._left_belt_act_id, 1]) > 100.0, (
            "belt actuator forcerange clobbered; mjx.put_model re-call failed"
        )

        self._obs_groups = build_obs_groups(self)
        self._schedule_T = int(cfg.episode_length)

    # ── Domain Randomization ─────────────────────────────────────────────

    def get_domain_randomization_spec(self):
        """Splitbelt reuses joystick's DR specs verbatim. Without this override,
        DR is silently no-op (base class does not define this method).
        """
        from jax_rl.envs.wrappers.domain_rand import DRSpec
        return [
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
            DRSpec(name="torso_com_jitter", type="model", field="body_ipos",
                   indices=(1, 2), min=-0.03, max=0.03,
                   per_element=True, operation="add",
                   description="Torso COM offset (x, y)"),
            DRSpec(name="body_inertia", type="model", field="body_inertia",
                   min=0.85, max=1.15, per_element=True,
                   description="Per-link inertia tensor variation"),
        ]

    # ── Control metadata override (deploy contract) ──────────────────────

    def get_control_metadata(self) -> dict:
        """Override base get_control_metadata — base shape-checks
        `_default_pose.shape == (mjx_model.nu,)` but splitbelt has nu=14 with
        len-12 default_pose. Slice to leg-only for the deploy contract.
        """
        from deploy.go2_constants import POLICY_TO_SDK
        default_pose_policy = np.asarray(self._default_pose, dtype=np.float32)
        assert default_pose_policy.shape == (12,), default_pose_policy.shape
        default_pose_sdk = default_pose_policy[np.array(POLICY_TO_SDK)]
        return {
            "default_pose_policy": default_pose_policy.tolist(),
            "default_pose_sdk": default_pose_sdk.tolist(),
            "policy_to_sdk": list(POLICY_TO_SDK),
            "sdk_to_policy": list(np.argsort(POLICY_TO_SDK)),
            "action_scale": float(self._config.action_scale),
            "Kp": float(self._config.Kp),
            "Kd": float(self._config.Kd),
            "ctrl_dt": float(self._config.ctrl_dt),
            "sim_dt": float(self._config.sim_dt),
            "contact_mode": self._config.contact_mode,
        }

    # ── Core env methods ────────────────────────────────────────────────

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k_qpos, k_sched = jax.random.split(rng, 3)

        # qpos: spawn keyframe + small per-joint noise on legs only.
        qpos = self._init_q
        qpos_noise = jax.random.uniform(
            k_qpos, (12,), minval=-0.01, maxval=0.01
        )
        qpos = qpos.at[7:7+12].set(qpos[7:7+12] + qpos_noise)

        # Belt slabs back to origin (override whatever was in keyframe).
        qpos = qpos.at[self._left_belt_qposadr].set(0.0)
        qpos = qpos.at[self._right_belt_qposadr].set(0.0)

        # Belt schedule sampled per episode (S§5.4).
        schedule_table = sched.sample_schedule(
            k_sched,
            T=self._schedule_T,
            kind=self._config.schedule_kind,
            params=dict(self._config.schedule_params),
        )

        # qvel: zeros for legs + base; belts pre-seeded to schedule[0].
        qvel = jp.zeros(self.mjx_model.nv)
        qvel = qvel.at[self._left_belt_dofadr].set(schedule_table[0, 0])
        qvel = qvel.at[self._right_belt_dofadr].set(schedule_table[0, 1])

        # Build initial mjx data (joystick precedent).
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

        # cmd: always-zero by default (S§7.4).
        cmd = jp.zeros(3)

        info = {
            "rng": rng,
            "step_idx": jp.int32(0),
            "belt_schedule": schedule_table,
            "command": cmd,
            "last_act": jp.zeros(self._action_dim),
            "last_last_act": jp.zeros(self._action_dim),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "swing_peak": jp.zeros(4),
            "splitbelt": {
                "foot_in_contact": jp.zeros(4, dtype=jp.bool_),
                "foot_pos_world": jp.zeros((4, 3)),
                "foot_belt_id": jp.full((4,), -1, dtype=jp.int32),
                "base_pos_world": jp.zeros(3),
                "base_vel_world": jp.zeros(3),
                "base_yaw": jp.float32(0.0),
                "belt_vel": schedule_table[0],
                "cmd_track_error": jp.zeros(3),
                "drift_xy": jp.zeros(2),
                "term_cause": jp.int32(0),
                "step_idx": jp.int32(0),
            },
            "reward_components": {
                k: jp.zeros(()) for k in self._config.reward_config.scales.keys()
            },
        }

        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        motor_targets = self._default_pose + action * self._config.action_scale

        step_idx = state.info["step_idx"]
        # Clamp index defense-in-depth (spec §6.3 invariant).
        safe_idx = jp.minimum(step_idx, self._schedule_T - 1)
        belt_vel_target = state.info["belt_schedule"][safe_idx]  # (2,)

        kp = self._kp
        kd = self._kd
        model = self.mjx_model
        leg_a2j = self._leg_act_to_joint
        leg_act_ids = self._leg_act_ids

        def substep(data, _):
            current_q = data.qpos[7:7+12]
            current_dq = data.qvel[6:6+12]
            tau_joint = kp * (motor_targets - current_q) + kd * (0.0 - current_dq)
            tau_joint = self._apply_torque_speed_limit(tau_joint, current_dq)
            leg_ctrl = tau_joint[leg_a2j]
            full_ctrl = data.ctrl.at[leg_act_ids].set(leg_ctrl)
            full_ctrl = full_ctrl.at[self._left_belt_ctrl_idx].set(belt_vel_target[0])
            full_ctrl = full_ctrl.at[self._right_belt_ctrl_idx].set(belt_vel_target[1])
            data = data.replace(ctrl=full_ctrl)
            return mjx.step(model, data), None

        data = jax.lax.scan(substep, state.data, (), self.n_substeps)[0]

        # Gait primitives (S§9.1).
        sensordata = data.sensordata
        foot_in_left = sensordata[self._foot_left_belt_adr] > 0.0
        foot_in_right = sensordata[self._foot_right_belt_adr] > 0.0
        foot_in_floor = sensordata[self._foot_floor_adr] > 0.0  # off-belt landings

        contact = foot_in_left | foot_in_right | foot_in_floor
        foot_belt_id = jp.where(
            foot_in_left, jp.int32(0),
            jp.where(foot_in_right, jp.int32(1), jp.int32(-1)),
        )

        foot_pos_world = data.site_xpos[self._feet_site_id]

        base_pos_world = data.qpos[:3]
        body_lin_vel = self.get_local_linvel(data)
        body_ang_vel = self.get_gyro(data)

        cmd = state.info["command"]
        cmd_track_error = jp.concatenate([
            cmd[:2] - body_lin_vel[:2],
            cmd[2:3] - body_ang_vel[2:3],
        ])
        drift_xy = base_pos_world[:2]  # treadmill_center = (0, 0)

        # Termination (S§7.3).
        fall_torso = (
            (sensordata[self._torso_left_belt_adr] > 0.0)
            | (sensordata[self._torso_right_belt_adr] > 0.0)
            | (sensordata[self._torso_floor_adr] > 0.0)
        )
        is_off_belt = jp.any(foot_in_floor)
        gravity_body = self.get_gravity(data)
        is_tilt = (gravity_body[2] < 0.5) | (base_pos_world[2] < 0.18)
        done = fall_torso | is_off_belt | is_tilt
        term_cause = jp.where(
            fall_torso, jp.int32(1),
            jp.where(is_off_belt, jp.int32(2),
                     jp.where(is_tilt, jp.int32(3), jp.int32(0))),
        )

        # Air-time bookkeeping (joystick pattern, but using splitbelt-OR contact).
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_fz = foot_pos_world[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        # Reward (S§7.1) — full joystick term set + treadmill_drift.
        rewards = {
            "tracking_lin_vel": self._reward_tracking_lin_vel(cmd, body_lin_vel),
            "tracking_ang_vel": self._reward_tracking_ang_vel(cmd, body_ang_vel),
            "lin_vel_z":        self._cost_lin_vel_z(self.get_global_linvel(data)),
            "ang_vel_xy":       self._cost_ang_vel_xy(self.get_global_angvel(data)),
            "orientation":      self._cost_orientation(self.get_upvector(data)),
            "torques":          self._cost_torques(data.actuator_force[:12]),
            "action_rate":      self._cost_action_rate(action, state.info["last_act"], state.info["last_last_act"]),
            "energy":           self._cost_energy(data.qvel[6:6+12], data.actuator_force[:12]),
            "dof_pos_limits":   self._cost_joint_pos_limits(data.qpos[7:7+12]),
            "feet_air_time":    self._reward_feet_air_time(state.info["feet_air_time"], first_contact, cmd),
            "feet_slip":        self._cost_feet_slip(data, contact, state.info),
            "feet_clearance":   self._cost_feet_clearance(data),
            "feet_height":      self._cost_feet_height(state.info["swing_peak"], first_contact, state.info),
            "termination":      self._cost_termination(done),
            "pose":             self._reward_pose(data.qpos[7:7+12]),
            "base_height":      self._cost_base_height(data),
            "treadmill_drift":  self._reward_treadmill_drift(drift_xy),
        }
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
        state.info["reward_components"] = rewards

        # Update info.
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["step_idx"] = step_idx + 1
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact

        state.info["splitbelt"] = {
            "foot_in_contact": contact,
            "foot_pos_world": foot_pos_world,
            "foot_belt_id": foot_belt_id,
            "base_pos_world": base_pos_world,
            "base_vel_world": data.qvel[:3],
            "base_yaw": jp.float32(0.0),
            "belt_vel": belt_vel_target,
            "cmd_track_error": cmd_track_error,
            "drift_xy": drift_xy,
            "term_cause": term_cause,
            "step_idx": step_idx,
        }

        obs = self._get_obs(data, state.info)
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["splitbelt/term_cause"] = term_cause.astype(reward.dtype)

        done = done.astype(reward.dtype)
        return state.replace(data=data, obs=obs, reward=reward, done=done)

    # ── Observation ─────────────────────────────────────────────────────

    def _get_obs(self, data, info):
        obs, info["rng"] = compute_obs(
            self._obs_groups,
            noise_level=self._config.noise_config.level,
            rng=info["rng"],
            data=data, info=info,
        )
        return obs

    # ── Reward helpers (copied from go2_warp_joystick.py:435-540) ────────

    def _reward_tracking_lin_vel(self, commands, local_vel):
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, commands, ang_vel):
        ang_vel_error = jp.square(commands[2] - ang_vel[2])
        return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

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
        del last_last_act  # unused — kept for joystick parity
        return jp.sum(jp.square(act - last_act))

    def _cost_joint_pos_limits(self, qpos):
        out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
        out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
        return jp.sum(out_of_limits)

    def _reward_feet_air_time(self, air_time, first_contact, commands):
        cmd_norm = jp.linalg.norm(commands)
        rew_air_time = jp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= cmd_norm > 0.01
        return rew_air_time

    def _cost_feet_slip(self, data, contact, info):
        cmd_norm = jp.linalg.norm(info["command"])
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
        return jp.sum(vel_xy_norm_sq * contact) * (cmd_norm > 0.01)

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
        error = swing_peak / self._config.reward_config.max_foot_height - 1.0
        return jp.sum(jp.square(error) * first_contact) * (cmd_norm > 0.01)

    def _cost_termination(self, done):
        return done

    def _reward_pose(self, qpos):
        weight = jp.array([1.0, 1.0, 0.1] * 4)
        return jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * weight))

    def _cost_base_height(self, data):
        base_z = data.subtree_com[self._torso_body_id][2]
        return jp.square(base_z - 0.27)

    # ── Splitbelt-specific reward ───────────────────────────────────────

    def _reward_treadmill_drift(self, drift_xy):
        wL = self._config.treadmill_drift_lateral_weight
        wF = self._config.treadmill_drift_forward_weight
        return -(wL * jp.square(drift_xy[1]) + wF * jp.square(drift_xy[0]))
