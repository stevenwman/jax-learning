"""Unitree G1 humanoid on splitbelt treadmill (Warp backend).

Subclasses G1WarpJoystick + adds:
- 2 extra actuators (left_belt_vel, right_belt_vel) hidden from policy
- Belt schedule sampling at reset (reuses splitbelt_schedules)
- Belt ctrl injection in step
- Off-belt termination via splitbelt_geom (position-based, robot-agnostic)
- splitbelt info dict for analysis

Cmd is zeroed (cmd_zero=True default) — policy task is to STAND on the
treadmill while belts drag feet backward. Same protocol family as Go2
splitbelt (A1/A2/A3/A4 via schedule_kind config).
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
from jax_rl.envs.reward_spec import RewardTerm
from jax_rl.envs.locomotion import g1_constants as consts
from jax_rl.envs.locomotion import go2_sensors
from jax_rl.envs.locomotion import splitbelt_geom as geom
from jax_rl.envs.locomotion import splitbelt_schedules as sched
from jax_rl.envs.locomotion.g1_warp_joystick import (
    G1WarpJoystick,
    default_config_holosoma_soft,
    default_config_holosoma_clearance,
    get_warp_assets,
)


SPLITBELT_SCENE_XML = consts.ROOT_PATH / "g1_warp_splitbelt_scene.xml"


def default_config() -> config_dict.ConfigDict:
    """G1 splitbelt config — inherits HoloSoft reward set (proven on flat
    G1 v18 eval 292) and adds splitbelt-specific config."""
    cfg = default_config_holosoma_soft()
    cfg.unlock()
    # Splitbelt: cmd is zero (task = stand on belts)
    cfg.command_config.a = [0.0, 0.0, 0.0]
    cfg.command_config.b = [0.0, 0.0, 0.0]
    # Splitbelt schedule (A2 prep — random per-episode constant belt speeds).
    cfg.schedule_kind = "random_per_episode"
    cfg.schedule_params = config_dict.create(
        v_range=(0.3, 1.0),         # narrower than Go2 (humanoid less robust)
        ratio_range=(0.5, 2.0),
    )
    cfg.belt_layout = config_dict.create(
        left_y_min=-0.500, left_y_max=0.000,
        right_y_min=0.000, right_y_max=0.500,
    )
    # Off-belt termination via foot position. G1's swing-foot lateral drift
    # can cross belt y-boundary momentarily; v19-v21 had this disabled.
    cfg.off_belt_termination = False
    # treadmill_drift: penalize body xy drift from origin (indirect "resist
    # belt drag" signal since actor has no belt_vel obs). Ported from
    # go2_warp_splitbelt. Lateral drift weighted 2× forward (lateral kills
    # balance; forward drift is the natural "fall behind" mode).
    cfg.reward_config.scales.treadmill_drift = 5.0
    cfg.reward_config.treadmill_drift_lateral_weight = 2.0
    cfg.reward_config.treadmill_drift_forward_weight = 0.5
    # If True, append belt_vel (R^2) to the actor's `state` obs. Otherwise
    # the actor is blind to the schedule (must infer from proprioception).
    # Per seed_prompt #2: highest leverage to close the 25→200 gap.
    cfg.informed_actor_obs = False
    return cfg


def default_config_informed() -> config_dict.ConfigDict:
    """Splitbelt with belt_vel exposed to the actor (R^2 added to `state`).

    Ports Go2 splitbelt's `informed` mode. Diagnostic-baseline config
    (random_per_episode + drift penalty + HoloSoft rewards) plus the one
    additional obs term. Expected to close the eval ~25 → 200+ gap.
    """
    cfg = default_config()
    cfg.unlock()
    cfg.informed_actor_obs = True
    return cfg


def default_config_tied() -> config_dict.ConfigDict:
    """A0/baseline: tied belts at fixed speed v=0.5. Useful for first
    smoke / establishing baseline before splitbelt protocols."""
    cfg = default_config()
    cfg.unlock()
    cfg.schedule_kind = "tied"
    cfg.schedule_params = config_dict.create(v=0.5)
    return cfg


def default_config_clearance_tied() -> config_dict.ConfigDict:
    """Splitbelt tied(0.5) + HoloClearance reward set + informed obs.

    Combines the foot-lift breakthrough (`feet_clearance_swing` linear
    bonus) with the splitbelt scene at the easiest (tied) belt config.
    Tests whether a real walker survives belt drag much better than the
    shuffle baselines that fell in 30-67 steps.
    """
    cfg = default_config_holosoma_clearance()
    cfg.unlock()
    cfg.command_config.a = [0.0, 0.0, 0.0]
    cfg.command_config.b = [0.0, 0.0, 0.0]
    cfg.schedule_kind = "tied"
    cfg.schedule_params = config_dict.create(v=0.5)
    cfg.belt_layout = config_dict.create(
        left_y_min=-0.500, left_y_max=0.000,
        right_y_min=0.000, right_y_max=0.500,
    )
    cfg.off_belt_termination = False
    cfg.reward_config.scales.treadmill_drift = 5.0
    cfg.reward_config.treadmill_drift_lateral_weight = 2.0
    cfg.reward_config.treadmill_drift_forward_weight = 0.5
    cfg.informed_actor_obs = True
    return cfg


def default_config_informed_tied() -> config_dict.ConfigDict:
    """Tied(v=0.5) + belt_vel exposed to actor. Isolates the obs-add from
    the belt-distribution-difficulty axis. Diagnostic on
    `G1WarpSplitbeltInformed` (random_per_episode v∈[0.3,1.0], ratio
    [0.5,2.0]) showed eval ~20 with mean episode 67 — robot fell before
    learning to use belt info because vR could reach 2.0 m/s. This
    preset locks both belts at v=0.5 to test whether informed obs alone
    helps when the dynamics are tractable.
    """
    cfg = default_config_tied()
    cfg.unlock()
    cfg.informed_actor_obs = True
    return cfg


class G1WarpSplitbeltEnv(G1WarpJoystick):
    """G1 humanoid on splitbelt apparatus."""

    def __init__(
        self,
        task: str = "splitbelt",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        # Skip G1WarpJoystick.__init__'s scene XML load; do our own.
        mjx_env.MjxEnv.__init__(self, config, config_overrides)
        del task

        self._model_assets = get_warp_assets()
        xml_path = SPLITBELT_SCENE_XML.as_posix()
        self._mj_model = mujoco.MjModel.from_xml_string(
            epath.Path(xml_path).read_text(), assets=self._model_assets
        )
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mj_model.opt.ccd_iterations = 100

        # Per-actuator PD gains (29 entries; leg/waist/arm). Belts handled
        # separately via velocity actuators (see _belt_ctrl in step).
        from jax_rl.envs.locomotion.g1_warp_joystick import _KP_PER_ACTUATOR, _KD_PER_ACTUATOR
        self._kp = jp.array(_KP_PER_ACTUATOR)
        self._kd = jp.array(_KD_PER_ACTUATOR)

        # Belt actuator + joint indices.
        self._left_belt_act_id = self._mj_model.actuator("left_belt_vel").id
        self._right_belt_act_id = self._mj_model.actuator("right_belt_vel").id
        self._left_belt_qposadr = self._mj_model.joint("left_belt_joint").qposadr[0]
        self._right_belt_qposadr = self._mj_model.joint("right_belt_joint").qposadr[0]
        self._left_belt_dofadr = self._mj_model.joint("left_belt_joint").dofadr[0]
        self._right_belt_dofadr = self._mj_model.joint("right_belt_joint").dofadr[0]

        # Restore belt actuator forcerange (Go2WarpEnv-style clobber prevention).
        # MJX freezes the model on put_model so we mutate before snapshot.
        self._mj_model.actuator_forcerange[self._left_belt_act_id] = np.array([-200.0, 200.0])
        self._mj_model.actuator_forcerange[self._right_belt_act_id] = np.array([-200.0, 200.0])

        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = xml_path

        # Belt forcerange smoke assert.
        assert float(self._mjx_model.actuator_forcerange[self._left_belt_act_id, 1]) > 100.0, (
            "belt actuator forcerange clobbered; mjx.put_model re-call failed"
        )

        # Body / site IDs (same as G1WarpJoystick).
        self._pelvis_body_id = self._mj_model.body(consts.ROOT_BODY).id
        self._imu_site_id = self._mj_model.site(consts.IMU_SITE).id
        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in consts.FEET_SITES]
        )

        # Foot global linvel sensor address ranges.
        foot_linvel_sensor_adr = []
        for name in consts.FEET_LINVEL_SENSOR:
            sid = self._mj_model.sensor(name).id
            adr = self._mj_model.sensor_adr[sid]
            dim = self._mj_model.sensor_dim[sid]
            foot_linvel_sensor_adr.append(list(range(adr, adr + dim)))
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        # Note: g1_warp_splitbelt_scene's foot×belt contact-found sensors not
        # used for off-belt detection (we use position-based instead per
        # splitbelt lessons). We do still need a "feet_floor_found" stub for
        # the parent class's contact-detection logic.
        # Use foot×left_belt + foot×right_belt as "is in contact with anything".
        self._left_foot_left_belt_adr = self._mj_model.sensor("left_foot_left_belt").id
        self._left_foot_right_belt_adr = self._mj_model.sensor("left_foot_right_belt").id
        self._right_foot_left_belt_adr = self._mj_model.sensor("right_foot_left_belt").id
        self._right_foot_right_belt_adr = self._mj_model.sensor("right_foot_right_belt").id
        # Map sensor ids → adr in sensordata.
        self._left_foot_contact_adrs = jp.array([
            self._mj_model.sensor_adr[self._left_foot_left_belt_adr],
            self._mj_model.sensor_adr[self._left_foot_right_belt_adr],
        ])
        self._right_foot_contact_adrs = jp.array([
            self._mj_model.sensor_adr[self._right_foot_left_belt_adr],
            self._mj_model.sensor_adr[self._right_foot_right_belt_adr],
        ])

        # Belt y-layout (used for off-belt detection).
        self._belt_layout = geom.BeltLayout(
            left_y_min=self._config.belt_layout.left_y_min,
            left_y_max=self._config.belt_layout.left_y_max,
            right_y_min=self._config.belt_layout.right_y_min,
            right_y_max=self._config.belt_layout.right_y_max,
        )

        # Schedule horizon (one extra row in case action_repeat indexes past T).
        self._schedule_T = int(self._config.episode_length) + 1

        self._post_init()

    # action_size hides belt actuators from the policy.
    @property
    def action_size(self) -> int:
        return consts.NUM_ACTUATORS  # 29 (legs+waist+arms), not nu=31

    def _post_init(self) -> None:
        # Override parent: spawn from knees_bent_belt keyframe (qpos has +2
        # belt slide entries). Default pose is leg+arm joints only (29).
        self._init_q = jp.array(self._mj_model.keyframe("knees_bent_belt").qpos)
        # qpos layout: 7 (free) + 29 (legs+arms) + 2 (belts) = 38.
        # default_pose for PD = qpos[7:7+29].
        self._default_pose = jp.array(
            self._mj_model.keyframe("knees_bent_belt").qpos[7:7 + consts.NUM_ACTUATORS]
        )

        # Skip parent _post_init's hip/knee setup if we want to keep it from
        # the parent's HoloSoft pose costs — but parent needs _hip_indices /
        # _knee_indices / _pose_weights. Re-build them here.
        super()._post_init()

        # Append splitbelt-specific reward term (treadmill_drift) to the
        # parent's reward_spec. Inserted as last term so it doesn't shift
        # existing indices. Penalizes lateral + forward body drift.
        self._reward_spec.append(RewardTerm(
            "treadmill_drift",
            lambda data, **kw: self._reward_treadmill_drift(data),
        ))

        # Splitbelt obs: belt_vel as R^2 — [vL, vR] commanded for the current
        # step. Off by default (preserves obs schema of v19-v22 ckpts);
        # `default_config_informed` flips both flags on for the eval-25→200
        # gap-closing experiment.
        if self._config.informed_actor_obs:
            from jax_rl.envs.obs_spec import ObsTerm
            belt_vel_term = ObsTerm(
                "belt_vel",
                lambda info, **kw: info["splitbelt"]["belt_vel"],
                noise_scale=0.0,
            )
            self._obs_groups["state"].append(belt_vel_term)
            self._obs_groups["privileged_state"].append(belt_vel_term)

    # ── reset / step overrides ────────────────────────────────────────

    def reset(self, rng: jax.Array) -> mjx_env.State:
        from jax_rl.envs.obs_spec import ObsTerm, IncludeGroup, compute_obs

        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        # Randomize base xy + yaw (modest perturbation — splitbelt is harder).
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.1, maxval=0.1)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.2, maxval=0.2))

        # Sample belt schedule (T × 2).
        rng, key = jax.random.split(rng)
        schedule_table = sched.sample_schedule(
            rng=key, T=self._schedule_T,
            kind=self._config.schedule_kind,
            params=dict(self._config.schedule_params),
        )

        # Initial belt joint velocity = -schedule[0] (drag-backward convention).
        qvel = qvel.at[self._left_belt_dofadr].set(-schedule_table[0, 0])
        qvel = qvel.at[self._right_belt_dofadr].set(-schedule_table[0, 1])

        data = mjx_env.make_data(
            self.mj_model, qpos=qpos, qvel=qvel,
            ctrl=jp.concatenate([self._default_pose, jp.zeros(2)]),  # leg pose + 2 belt zeros
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            naccdmax=self._config.naccdmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        # cmd is always zero on splitbelt.
        cmd = jp.zeros(3)

        # Gait phase init (same as parent).
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
            "steps_until_next_cmd": jp.int32(self._config.episode_length),
            "last_act": jp.zeros(self.action_size),
            "last_last_act": jp.zeros(self.action_size),
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            "step_count": jp.int32(0),
            "phase": phase,
            "phase_dt": phase_dt,
            "splitbelt": {
                "schedule_table": schedule_table,
                "belt_vel": schedule_table[0],
                "step_idx": jp.int32(0),
            },
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
        # Build full ctrl: 29 leg/arm motor targets + 2 belt vel.
        motor_targets = self._default_pose + action * self._config.action_scale
        step_idx = state.info["splitbelt"]["step_idx"]
        safe_idx = jp.minimum(step_idx, self._schedule_T - 1)
        belt_vel_target = state.info["splitbelt"]["schedule_table"][safe_idx]
        # Negate (drag-backward biomech convention — see splitbelt lesson).
        belt_ctrl = -belt_vel_target

        kp = self._kp
        kd = self._kd
        model = self.mjx_model
        nu_legs = consts.NUM_ACTUATORS

        def substep(data, _):
            current_q = data.qpos[7:7 + nu_legs]
            current_dq = data.qvel[6:6 + nu_legs]
            tau_legs = kp * (motor_targets - current_q) + kd * (0.0 - current_dq)
            full_ctrl = jp.concatenate([tau_legs, belt_ctrl])
            data = data.replace(ctrl=full_ctrl)
            return mjx.step(model, data), None

        data = jax.lax.scan(substep, state.data, (), self.n_substeps)[0]

        # Foot contact: any foot×belt sensor > 0.
        l_contact = (data.sensordata[self._left_foot_contact_adrs] > 0).any()
        r_contact = (data.sensordata[self._right_foot_contact_adrs] > 0).any()
        contact = jp.array([l_contact, r_contact])
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
        reward = sum(rewards.values()) * self.dt
        state.info["reward_components"] = rewards

        # Advance gait phase.
        phase_tp1 = state.info["phase"] + state.info["phase_dt"]
        state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2.0 * jp.pi) - jp.pi

        # Splitbelt info update.
        state.info["splitbelt"]["step_idx"] = step_idx + 1
        state.info["splitbelt"]["belt_vel"] = belt_vel_target

        # Housekeeping.
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["step_count"] = state.info["step_count"] + 1
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        done = done.astype(reward.dtype)
        return state.replace(data=data, obs=obs, reward=reward, done=done)

    # ── Reward helpers ────────────────────────────────────────────────

    def _reward_treadmill_drift(self, data):
        """Penalize body xy drift from origin. Ported from
        go2_warp_splitbelt. Returns negative quadratic of (lateral_w *
        y_drift² + forward_w * x_drift²)."""
        base_xy = data.qpos[:2]
        wL = self._config.reward_config.treadmill_drift_lateral_weight
        wF = self._config.reward_config.treadmill_drift_forward_weight
        return -(wL * jp.square(base_xy[1]) + wF * jp.square(base_xy[0]))

    def _get_termination(self, data):
        # Standard tilt termination from parent.
        upvec = self.get_upvector(data)
        flipped = upvec[-1] < self._config.tilt_upvector_threshold
        pelvis_z = data.subtree_com[self._pelvis_body_id][2]
        too_low = pelvis_z < self._config.tilt_min_pelvis_z

        # Off-belt: any foot xy outside both belts AND grounded.
        if self._config.off_belt_termination:
            foot_pos = data.site_xpos[self._feet_site_id]
            foot_xy = foot_pos[..., :2]
            foot_belt_id = geom.foot_belt_id(foot_xy, self._belt_layout)
            foot_off = (foot_belt_id == jp.int32(-1))
            foot_grounded = foot_pos[..., 2] < 0.05  # near ground
            off_belt = jp.any(foot_off & foot_grounded)
        else:
            off_belt = jp.bool_(False)

        return flipped | too_low | off_belt
