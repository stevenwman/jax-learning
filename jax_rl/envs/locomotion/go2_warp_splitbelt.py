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

from jax_rl.envs.locomotion import go2_warp_base
from jax_rl.envs.locomotion import go2_constants as consts
from jax_rl.envs.locomotion import splitbelt_geom as geom
from jax_rl.envs.locomotion import splitbelt_schedules as sched
from jax_rl.envs.obs_spec import ObsTerm, IncludeGroup


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
    Uses IncludeGroup("state") in privileged so the actor's terms are inherited
    automatically (saves duplication, idiomatic per obs_spec.py:46-49).
    """
    cfg = env._config
    layout = obs_term_names(cfg.obs_mode)
    noise = cfg.noise_config.scales

    # `data.qpos[7:]` and `data.qvel[6:]` are the real way to read joint pos/vel
    # for Go2 (post-base, post-freejoint). Joystick precedent: lines 137-140.
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
        # Privileged-only terms read TRUE base velocities (no noise).
        "base_lin_vel": (lambda data, **kw: env.get_local_linvel(data), 0.0),
        "base_ang_vel": (lambda data, **kw: env.get_global_angvel(data), 0.0),
    }

    def _build(names):
        return [ObsTerm(name=n, fn=term_factory[n][0], noise_scale=term_factory[n][1])
                for n in names]

    # State group: real ObsTerms.
    state_terms = _build(layout["state"])

    # Privileged: IncludeGroup("state") + privileged extras (the names in `layout`
    # that are NOT in `state`). Order preserved.
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
        # `task` accepted for Playground registry compatibility; not used internally.
        del task
        cfg = config if config is not None else default_config()
        xml_path = (
            Path(__file__).parent / "xmls" / "go2_warp_splitbelt_scene.xml"
        ).as_posix()
        super().__init__(
            xml_path=xml_path, config=cfg, config_overrides=config_overrides
        )
        self._post_init()

    def _post_init(self) -> None:
        # NOTE: go2_warp_base.Go2WarpEnv.__init__ does NOT call _post_init itself.
        # Joystick env calls it explicitly from its own __init__ (line 96).
        # Heavy lifting (sensor lookup, default_pose, belt IDs) lives here.
        # See Task 3.3 for the full implementation.
        self._action_dim = 12
        self._default_pose = jp.array(self._mj_model.keyframe("splitbelt_spawn").qpos[7:7+12])
        self._init_q = jp.array(self._mj_model.keyframe("splitbelt_spawn").qpos)
        self._obs_groups = build_obs_groups(self)
        # Task 3.3 fills in: belt_layout, belt actuator/joint IDs, contact-pair sensor IDs,
        # torso fall sensors, schedule_T.

    @property
    def action_size(self) -> int:
        # Override Go2WarpEnv.action_size (which returns mjx_model.nu = 14).
        # Belt actuators are env-internal (driven by schedule_table); the policy
        # only controls the 12 leg actuators. Without this override, the algo
        # would sample 14-d actions and shape-mismatch on every step.
        return 12

    def reset(self, rng: jax.Array):
        raise NotImplementedError("Implemented in Task 3.4")

    def step(self, state, action: jax.Array):
        raise NotImplementedError("Implemented in Task 3.5")
