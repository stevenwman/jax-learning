"""FactoryPegInsert — port of IsaacLab Factory PegInsert to MuJoCo Warp.

Architecture:
  - Subclass mujoco_playground._src.mjx_env.MjxEnv (same base as Go2 Warp envs)
  - impl="warp" set in config; routes through the existing mjx_backend
  - Dict obs {"state": (25,), "privileged_state": (~72,)} — asymmetric AC auto-engages
  - Held peg attached to gripper via <equality><weld/> (grasp_mode="weld";
    "friction" mode reserved for Phase 7+)
  - Geometry: capsule peg + bore tile ring + box bore floor (Phase 0b findings,
    see .context/journals/2026-05-27-factory-phase0.md)
  - Controller: zero-torque placeholder in this phase; OSC lands in Phase 2

Reference:
  - .superpowers/specs/2026-05-27-factory-mjx-warp-port.md (design)
  - .superpowers/plans/2026-05-27-factory-peg-insert-mvp.md (Plan 1)
  - IsaacLab v2.3.2 factory_env.py + factory_tasks_cfg.py:PegInsert
  - .context/lessons/algo_port_protocol.md (repo contract)
"""
import re
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from ml_collections import config_dict
import numpy as np
from mujoco_playground._src import mjx_env

from jax_rl.envs.obs_spec import ObsTerm, IncludeGroup, compute_obs


ASSETS_DIR = Path(__file__).parent / "assets"
PEG_ASSETS = ASSETS_DIR / "peg_insert"
PANDA_DIR = ASSETS_DIR / "franka_panda"
PANDA_XML = PANDA_DIR / "panda.xml"
SCENE_XML = PEG_ASSETS / "scene.xml"
BORE_TILES_DIR = PEG_ASSETS / "extracted" / "bore_tiles"


def default_config() -> config_dict.ConfigDict:
    """Default PegInsert env config. CLI overrides via dataclasses.replace pattern."""
    return config_dict.create(
        sim_dt=0.002,                       # 500Hz physics
        ctrl_dt=0.0083333,                  # 120Hz inner control rate
        decimation=4,                        # outer policy at 30Hz
        episode_length=900,                  # 30s @ 30Hz
        action_dim=6,
        grasp_mode="weld",                   # "weld" | "friction" (Phase 7+)

        # Action chain (Phase 1 stub doesn't actuate, but config is wired).
        # pos_action_bounds: cube around bore-opening-top within which the
        # commanded fingertip target must lie. IsaacLab uses (0.05, 0.05, 0.05)
        # AND starts the fingertip already inside their cube (peg tip ~1cm
        # below bore). We instead start the peg tip 2cm above the bore so
        # the visualization makes sense ("peg above hole, descend to insert"),
        # which puts the initial fingertip outside the z=0.05 bound. Widen
        # z to 0.10 so init is inside.
        #
        # ema_factor=0.2 per IsaacLab. (Briefly bumped to 1.0 while we
        # mis-diagnosed the v5 collapse as an EMA issue; the real cause was
        # actuator_mode="position_pd" silently disabling the OSC.)
        ema_factor=0.2,
        pos_threshold=(0.02, 0.02, 0.02),
        rot_threshold=(0.097, 0.097, 0.097),
        # pos_action_bounds: tightened xy from 0.05 → 0.02 (v7 finding —
        # the 5cm half-cube let SAC drive the peg to a corner 7cm offset
        # from the bolt; with the xy_engaged gate at 2.5mm there was no
        # gradient back). 2cm bounds keep the policy within the bonus's
        # 1cm loosened gate (see engage_xy_threshold).
        pos_action_bounds=(0.02, 0.02, 0.10),

        # Controller / actuator mode. "motor" rewrites arm actuators 0..6
        # in-place to gain=fixed(1) + bias=none so ctrl[:7] = raw torque,
        # consumed by the Phase 2 OSC controller (compute_osc_torque). The
        # only reason to flip this to "position_pd" now is the Phase 1
        # placeholder controller — used in early visual gates and weld diag
        # scripts before OSC landed. Gripper actuator (idx 7) is never
        # touched by either mode.
        #
        # NOTE: this default was "position_pd" through Phase 3 v1..v5,
        # which silently disabled the OSC and made every SAC run train
        # against an env that ignored the policy's actions. All five
        # collapsed at the d=0.066 baseline (Return 45 / 191).
        actuator_mode="motor",

        # OSC gains (Phase 2). task_deriv_gains default = critical damping
        # (2·sqrt(kp)) per task axis: ≈[20, 20, 20, 10.95, 10.95, 10.95].
        task_prop_gains=(100.0, 100.0, 100.0, 30.0, 30.0, 30.0),
        task_deriv_gains=(20.0, 20.0, 20.0, 10.954, 10.954, 10.954),
        kp_null=10.0,
        kd_null=6.3246,
        torque_limit=100.0,

        # Reward coefficients (IsaacLab PegInsert defaults — factory_tasks_cfg.py).
        # Earlier values (100, 2)/(50, 2)/(100, 0) were a plan-spec copy bug
        # that made the baseline term too narrow (kp_baseline ≈ 0.001 at init
        # vs 0.164 with the real Isaac coefs), starving SAC of a learning
        # gradient and collapsing the policy at return 45/episode in v1.
        #   - baseline (5, 4):  wide bell — "general movement toward fixed asset"
        #   - coarse   (50, 2): mid band  — "alignment"
        #   - fine     (100, 0): narrow   — "last-inch insertion"
        keypoint_coef_baseline=(5.0, 4.0),
        keypoint_coef_coarse=(50.0, 2.0),
        keypoint_coef_fine=(100.0, 0.0),
        keypoint_scale=0.05,
        engage_threshold=0.9,
        success_threshold=0.04,
        asset_height=0.025,                  # hole height
        # Peg geometry — half-length of the capsule along the body z-axis.
        # Used to compute entry_z = hole_top_z + peg_half_length in the
        # reward (peg body z at which the tip crosses the bore opening).
        # Matches scene.xml `<geom name="peg" type="capsule" size="r 0.021007"/>`.
        peg_half_length=0.021,
        # Per-episode hole-pose randomization (mocap_pos / mocap_quat).
        # Sampled uniformly each reset; 0 disables. xy and z are absolute
        # offsets from the scene.xml nominal pose; yaw is radians around z.
        hole_pos_xy_noise=0.02,              # ±2 cm in xy
        hole_pos_z_noise=0.01,               # ±1 cm in z
        hole_yaw_noise=0.0,                  # ±rad around z; 0 = disabled
                                              # (peg is axially symmetric so
                                              # yaw is currently moot — leave
                                              # off until rotation actions
                                              # land in Phase 5+)
        # XY + tilt gates for engaged / success bonuses. Both bonuses now
        # require the peg to be PHYSICALLY ALIGNABLE with the bore:
        #   - xy within 5 mm (peg radius 4 mm + clearance 1 mm)
        #   - tilt within 2° of vertical
        # Z depth is the only thing differing between engaged and success.
        # v11's 3 cm xy threshold without tilt let the policy park the peg
        # 2.8 cm off-axis at z=0.035 (peg hovering BESIDE the bolt, not in
        # it) and still trigger engaged for 779/900 steps — a quieter
        # variant of v6's exploit.
        engage_xy_threshold=0.005,
        success_xy_threshold=0.0025,

        # MJX/Warp contact / constraint buffer budgets. These are GLOBAL
        # across all envs in the batch. Bumped from Phase 0b defaults
        # (naconmax=32768, njmax=100) after a 256-env smoke train spammed
        # "broadphase overflow" / "nefc overflow" once the peg-bore
        # contacts ramped up — each env can hit ~28 contacts + ~67
        # constraints in steady state.
        impl="warp",
        naconmax=16 * 8192,                 # was 4 * 8192
        naccdmax=8000,                      # was 4000
        njmax=512,                          # was 100
    )


def _build_scene_xml() -> str:
    """Splice panda.xml into scene.xml + inject bore tile assets/geoms.

    Returns the fully-resolved MJCF string for from_xml_string(). We do this
    manually because <include> doesn't compose well with our compiler/meshdir
    settings (panda needs meshdir="assets" relative to panda.xml; bore tiles
    need different path resolution).
    """
    scene_text = SCENE_XML.read_text()

    # 1. Inject bore tile assets + geoms
    pieces = sorted(BORE_TILES_DIR.glob("bore_*.obj"))
    if not pieces:
        raise FileNotFoundError(
            f"No bore tiles found in {BORE_TILES_DIR}. Run "
            "scripts/factory/extract_usds.py + spike_drop.py to generate them."
        )
    assets_block = "\n".join(
        f'    <mesh name="bore_{i:02d}" file="{p.name}"/>'
        for i, p in enumerate(pieces)
    )
    geoms_block = "\n".join(
        f'      <geom name="bore_{i:02d}" type="mesh" mesh="bore_{i:02d}" '
        f'condim="6" friction="1.0 0.01 0.0001" '
        f'solref="0.004 1" solimp="0.98 0.995 0.001"/>'
        for i, p in enumerate(pieces)
    )
    scene_text = scene_text.replace("<!-- BORE_TILE_ASSETS -->", assets_block).replace(
        "<!-- BORE_TILE_GEOMS injected here -->", geoms_block
    )

    # 2. Splice panda.xml inline (replace <include>). MuJoCo's <include> doesn't
    #    handle meshdir collisions cleanly, so we paste panda's content directly.
    panda_text = PANDA_XML.read_text()
    m = re.search(r"<mujoco[^>]*>(.*)</mujoco>", panda_text, re.DOTALL)
    if not m:
        raise RuntimeError(f"Could not parse panda.xml at {PANDA_XML}")
    panda_inner = m.group(1)
    # Strip panda's compiler + option (our scene's win)
    panda_inner = re.sub(r"<compiler[^/]*/>", "", panda_inner)
    panda_inner = re.sub(r"<option[^/]*/>", "", panda_inner)

    scene_text = re.sub(
        r'<include\s+file="\.\./franka_panda/panda\.xml"\s*/>',
        panda_inner,
        scene_text,
    )
    return scene_text


def _build_assets_dict() -> Dict[str, bytes]:
    """Bundle all mesh bytes (panda meshes + bore tiles) for from_xml_string."""
    assets: Dict[str, bytes] = {}
    panda_assets_dir = PANDA_DIR / "assets"
    for p in panda_assets_dir.iterdir():
        if p.is_file():
            assets[p.name] = p.read_bytes()
    for p in BORE_TILES_DIR.glob("bore_*.obj"):
        assets[p.name] = p.read_bytes()
    return assets


def _apply_actuator_mode(mj_model: mujoco.MjModel, mode: str) -> None:
    """Rewrite arm actuators (idx 0..6) in-place per `mode`.

    "position_pd": no-op (menagerie panda default: gain=affine, bias=affine,
                   ctrl = q_target, torque = gain*ctrl + bias·(1,q,qdot)).
    "motor":       gain=fixed(1), bias=none → ctrl[:7] interpreted as raw
                   torque. Gripper actuator (idx 7) is left untouched.
                   ctrlrange is widened to match forcerange so torque isn't
                   silently clipped by the original position-PD ctrlrange.

    Must run before mjx.put_model so Warp picks up the new gain/bias schema.
    """
    if mode == "position_pd":
        return
    if mode != "motor":
        raise ValueError(
            f"actuator_mode must be 'position_pd' or 'motor', got {mode!r}"
        )

    arm_act_ids = [mj_model.actuator(f"actuator{i}").id for i in range(1, 8)]
    for aid in arm_act_ids:
        mj_model.actuator_gaintype[aid] = int(mujoco.mjtGain.mjGAIN_FIXED)
        mj_model.actuator_gainprm[aid, :] = 0.0
        mj_model.actuator_gainprm[aid, 0] = 1.0
        mj_model.actuator_biastype[aid] = int(mujoco.mjtBias.mjBIAS_NONE)
        mj_model.actuator_biasprm[aid, :] = 0.0
        # Widen ctrlrange to match forcerange so torque commands aren't clipped.
        mj_model.actuator_ctrlrange[aid] = mj_model.actuator_forcerange[aid]


# Default arm joint pose from IsaacLab factory_env_cfg.py:59 (reset_joints).
# Used as the seed for the IK that resolves the actual init pose. 7 arm
# joints; finger joints separately closed.
DEFAULT_ARM_QPOS = np.array(
    [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01]
)
# Nullspace posture target from IsaacLab factory_env_cfg.py:65
# (default_dof_pos_tensor) — a kinematically well-conditioned pose used by
# the OSC's joint-space PD in the task nullspace.
NULLSPACE_ARM_QPOS = np.array(
    [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
)
DEFAULT_FINGER_QPOS = np.array([0.04, 0.04])   # menagerie panda has 2 finger joints


def _solve_init_arm_qpos(
    mj_model: mujoco.MjModel,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    site_name: str = "fingertip_centered",
    arm_jnt_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"),
    seed_qpos: np.ndarray = DEFAULT_ARM_QPOS,
    max_iters: int = 500,
    pos_tol: float = 1e-4,
    rot_tol: float = 1e-3,
    damp: float = 1e-2,
    step_scale: float = 0.5,
) -> np.ndarray:
    """One-shot CPU IK for the panda arm. Returns 7-d joint angles.

    Matches IsaacLab Factory's reset target (hand sits at fixed_tip + hand_init_pos
    with gripper-down orientation). Uses damped Levenberg-Marquardt on a 6-D
    site Jacobian (linear + angular). Falls back to the seed pose if IK fails
    to converge — caller is responsible for sanity-checking the result.
    """
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = 0.0
    arm_qposadr = np.asarray(
        [mj_model.jnt_qposadr[mj_model.joint(n).id] for n in arm_jnt_names], dtype=np.int32
    )
    arm_dofadr = np.asarray(
        [mj_model.jnt_dofadr[mj_model.joint(n).id] for n in arm_jnt_names], dtype=np.int32
    )
    mj_data.qpos[arm_qposadr] = seed_qpos
    site_id = mj_model.site(site_name).id

    jacp = np.zeros((3, mj_model.nv))
    jacr = np.zeros((3, mj_model.nv))
    cur_quat = np.zeros(4)
    err_quat = np.zeros(4)
    rot_err = np.zeros(3)
    target_quat = target_quat / np.linalg.norm(target_quat)

    for _ in range(max_iters):
        mujoco.mj_kinematics(mj_model, mj_data)
        mujoco.mj_comPos(mj_model, mj_data)
        cur_pos = np.asarray(mj_data.site_xpos[site_id])
        pos_err = target_pos - cur_pos

        # Orientation error as axis-angle (small-angle stable via mju_quat2Vel).
        mujoco.mju_mat2Quat(cur_quat, mj_data.site_xmat[site_id])
        mujoco.mju_negQuat(err_quat, cur_quat)
        mujoco.mju_mulQuat(err_quat, target_quat, err_quat)
        mujoco.mju_quat2Vel(rot_err, err_quat, 1.0)

        if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < rot_tol:
            break

        mujoco.mj_jacSite(mj_model, mj_data, jacp, jacr, site_id)
        J_arm = np.concatenate([jacp[:, arm_dofadr], jacr[:, arm_dofadr]], axis=0)  # (6, 7)
        err6 = np.concatenate([pos_err, rot_err])
        # Damped pseudo-inverse step: dq = J^T (J J^T + λ²I)^-1 err
        JJt = J_arm @ J_arm.T + (damp ** 2) * np.eye(6)
        dq = J_arm.T @ np.linalg.solve(JJt, err6)
        mj_data.qpos[arm_qposadr] += step_scale * dq

    return mj_data.qpos[arm_qposadr].copy()


class FactoryPegInsert(mjx_env.MjxEnv):
    """Factory PegInsert (8mm peg → 8.1mm hole, 114µm clearance).

    Phase 1: env scaffold + obs/reward/reset. Controller stub returns zero torques.
    Phase 2: OSC controller wired in.
    Phase 3: Train SAC + asymmetric AC to ≥0.5 reward/step at 5M timesteps.
    """

    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        if config is None:
            config = default_config()
        super().__init__(config, config_overrides)

        # 1. Build full MJCF (panda spliced + bore tiles injected) + asset bytes
        scene_text = _build_scene_xml()
        assets = _build_assets_dict()
        self._mj_model = mujoco.MjModel.from_xml_string(scene_text, assets=assets)
        self._mj_model.opt.timestep = self._config.sim_dt

        # 1a. Optionally rewrite arm actuators to motor (Phase 2 OSC needs raw
        #     torque). Must happen BEFORE mjx.put_model so Warp sees the swap.
        _apply_actuator_mode(self._mj_model, self._config.actuator_mode)

        # 1b. Bump CCD iteration cap. Default 35 is fine for the nominal
        #     reset pose, but reset-noise (±0.05 rad arm qpos) puts peg + bore
        #     at occasional edge contact angles where the convex-hull CCD
        #     algorithm needs >35 iters to converge. MJWarp prints a runtime
        #     warning per such occurrence; 100 silences it without measurable
        #     step-time impact (the iter cap is per-contact-pair, not per-step).
        self._mj_model.opt.ccd_iterations = 100

        # 2. Hand to MJX. Lazy: only create Warp model when first accessed via
        #    self.mjx_model property. Lets env instantiate on CPU for hermetic
        #    schema tests; GPU is only needed for actual physics stepping.
        self._mjx_model_cache = None

        # 3. Cache body / site / joint IDs
        self._hand_body_id = self._mj_model.body("hand").id
        self._peg_body_id = self._mj_model.body("peg").id
        self._hole_body_id = self._mj_model.body("hole_base").id
        # hole_base is a mocap body — its world pose lives in
        # data.mocap_pos[hole_mocap_id], NOT mj_model.body_pos. mocap_id is
        # stored on the body itself (body_mocapid array, -1 for non-mocap).
        self._hole_mocap_id = int(self._mj_model.body_mocapid[self._hole_body_id])
        assert self._hole_mocap_id >= 0, "hole_base must be mocap='true' in scene.xml"
        # Nominal hole pose (matches scene.xml `pos`) — used as the anchor
        # around which per-episode hole noise is sampled.
        self._hole_nominal_pos = jp.asarray(
            self._mj_model.body_pos[self._hole_body_id])
        self._hole_nominal_quat = jp.asarray(
            self._mj_model.body_quat[self._hole_body_id])
        self._fingertip_site_id = self._mj_model.site("fingertip_centered").id

        # Panda arm joints: joint1..joint7 in menagerie naming
        arm_jnt_ids = []
        for i in range(1, 8):
            try:
                arm_jnt_ids.append(self._mj_model.joint(f"joint{i}").id)
            except KeyError:
                # Some panda models use different naming, e.g. "panda_joint1"
                arm_jnt_ids.append(self._mj_model.joint(f"panda_joint{i}").id)
        self._arm_jnt_ids = np.asarray(arm_jnt_ids, dtype=np.int32)
        # qpos addresses (handle peg freejoint qpos offset properly)
        self._arm_qposadr = np.asarray(
            [self._mj_model.jnt_qposadr[i] for i in self._arm_jnt_ids], dtype=np.int32
        )
        self._arm_dofadr = np.asarray(
            [self._mj_model.jnt_dofadr[i] for i in self._arm_jnt_ids], dtype=np.int32
        )

        # Peg freejoint qpos offset (xyz + quat = 7 values)
        peg_jnt_id = self._mj_model.body_jntadr[self._peg_body_id]
        self._peg_qpos_addr = int(self._mj_model.jnt_qposadr[peg_jnt_id])

        # Arm actuator indices into ctrl (for OSC torque write-out). Panda
        # actuators are named actuator1..actuator7 (arm) + actuator8 (gripper).
        self._arm_act_ids = np.asarray(
            [self._mj_model.actuator(f"actuator{i}").id for i in range(1, 8)],
            dtype=np.int32,
        )

        # 4. Solve init pose IK once (CPU). IsaacLab Factory PegInsert places
        #    the fingertip 4.7cm above the bore opening with gripper-down
        #    orientation. We mirror that target so the initial fingertip lies
        #    INSIDE the action_chain's pos_action_bounds cube — otherwise the
        #    target clip wipes the policy's +x/+z action gradient.
        self._init_arm_qpos = self._solve_init_pose()

        # 5. Set up obs schema
        self._setup_obs_groups()

    def _solve_init_pose(self) -> np.ndarray:
        """Run one-shot CPU IK so the peg tip starts 5cm above the bore opening.

        IsaacLab's hand_init_pos.z=0.047 is measured against their
        `fingertip_midpoint` body (~5cm above the peg tip). Menagerie's
        `fingertip_centered` site sits 10.34cm below the hand body origin,
        so with our weld geometry the fingertip is 4.76cm ABOVE the peg
        tip:

            peg_tip = hand - 0.13 (weld) - 0.021 (half-length) = hand - 0.151
            fingertip = hand - 0.1034
            ⇒ fingertip - peg_tip = 0.0476

        We want peg_tip 5cm above bore opening, so:
            target_fingertip_z = bore_top_z + 0.05 + 0.0476 ≈ bore_top_z + 0.098

        Hand orientation = current fingertip_quat at DEFAULT_ARM_QPOS, which
        is already gripper-down in menagerie. Close enough to seed the IK.
        """
        # Read hole pose from MJCF defaults.
        hole_pos = np.asarray(self._mj_model.body_pos[self._hole_body_id])
        bore_top_z = hole_pos[2] + self._config.asset_height       # 0.075 (hole_8mm)
        peg_tip_clearance = 0.02                                    # peg tip 2cm above bore
        fingertip_to_peg_tip = 0.0476                               # geom-derived (see docstring)
        target_pos = np.array([
            hole_pos[0], hole_pos[1],
            bore_top_z + peg_tip_clearance + fingertip_to_peg_tip,
        ])

        # Orientation target: take fingertip orient at DEFAULT_ARM_QPOS (the
        # menagerie panda's natural gripper-down at this arm config).
        mj_data_ref = mujoco.MjData(self._mj_model)
        mj_data_ref.qpos[self._arm_qposadr] = DEFAULT_ARM_QPOS
        mujoco.mj_kinematics(self._mj_model, mj_data_ref)
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, mj_data_ref.site_xmat[self._fingertip_site_id])

        return _solve_init_arm_qpos(
            self._mj_model, target_pos, target_quat,
            site_name="fingertip_centered",
            seed_qpos=DEFAULT_ARM_QPOS,
        )

    # ─────────────────────────────────────────────────────────────────
    # Obs schema
    # ─────────────────────────────────────────────────────────────────

    def _setup_obs_groups(self) -> None:
        """Declare obs terms. Replicates factory_env._get_observations layout."""
        self._obs_groups = {
            "state": [
                ObsTerm("fingertip_pos_rel_fixed", self._fingertip_pos_rel_fixed),
                ObsTerm("fingertip_quat",          self._fingertip_quat),
                ObsTerm("ee_linvel",               self._ee_linvel),
                ObsTerm("ee_angvel",               self._ee_angvel),
                ObsTerm("actions",                 lambda info, **kw: info["actions"]),
                ObsTerm("prev_actions",            lambda info, **kw: info["prev_actions"]),
            ],
            "privileged_state": [
                IncludeGroup("state"),
                ObsTerm("fingertip_pos",       self._fingertip_pos),
                ObsTerm("joint_pos",           self._joint_pos),
                ObsTerm("joint_vel",           self._joint_vel),
                ObsTerm("held_pos",            self._held_pos),
                ObsTerm("held_pos_rel_fixed",  self._held_pos_rel_fixed),
                ObsTerm("held_quat",           self._held_quat),
                ObsTerm("fixed_pos",           lambda info, **kw: info["fixed_pos"]),
                ObsTerm("fixed_quat",          lambda info, **kw: info["fixed_quat"]),
                ObsTerm("task_prop_gains",
                        lambda info, **kw: jp.asarray(self._config.task_prop_gains)),
                ObsTerm("ema_factor",
                        lambda info, **kw: jp.asarray([self._config.ema_factor])),
                ObsTerm("pos_threshold",
                        lambda info, **kw: jp.asarray(self._config.pos_threshold)),
                ObsTerm("rot_threshold",
                        lambda info, **kw: jp.asarray(self._config.rot_threshold)),
            ],
        }

    # Obs term callables. Each takes **kw (data + info passed by compute_obs).

    def _fingertip_pos(self, data, **kw):
        return data.site_xpos[self._fingertip_site_id]

    def _fingertip_quat(self, data, **kw):
        m = data.site_xmat[self._fingertip_site_id].reshape(3, 3)
        return _mat_to_quat(m)

    def _fingertip_pos_rel_fixed(self, data, info, **kw):
        return data.site_xpos[self._fingertip_site_id] - info["fixed_pos"]

    def _ee_linvel(self, data, info, **kw):
        prev = info["prev_fingertip_pos"]
        return (data.site_xpos[self._fingertip_site_id] - prev) / self._config.ctrl_dt

    def _ee_angvel(self, data, info, **kw):
        q_prev = info["prev_fingertip_quat"]
        q_curr = self._fingertip_quat(data)
        return _quat_finite_diff_angvel(q_prev, q_curr, self._config.ctrl_dt)

    def _joint_pos(self, data, **kw):
        return data.qpos[self._arm_qposadr]

    def _joint_vel(self, data, **kw):
        return data.qvel[self._arm_dofadr]

    def _held_pos(self, data, **kw):
        return data.xpos[self._peg_body_id]

    def _held_quat(self, data, **kw):
        return data.xquat[self._peg_body_id]

    def _held_pos_rel_fixed(self, data, info, **kw):
        return data.xpos[self._peg_body_id] - info["fixed_pos"]

    # ─────────────────────────────────────────────────────────────────
    # Stubs — reset/step land in Task 1.6 / 1.7
    # ─────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────
    # Reset + DR
    # ─────────────────────────────────────────────────────────────────

    def get_domain_randomization_spec(self):
        """Per-episode DR. Hand/bolt/peg initial pose noise.

        Phase 1 declares the specs but does not apply them — `reset()` uses
        deterministic nominal values. `DomainRandWrapper(per_step)` reads
        these specs and writes sampled values to `state.info` under the
        spec names; env then consumes them in reset. Wire-up lands in
        Phase 1.6 polish or Phase 3 if smoke train demands it.
        """
        from jax_rl.envs.wrappers.domain_rand import DRSpec
        return [
            DRSpec(name="bolt_pos_xy", type="runtime", min=-0.05, max=0.05,
                   description="Bolt position xy noise (±5cm cube)"),
            DRSpec(name="bolt_pos_z", type="runtime", min=-0.05, max=0.05,
                   description="Bolt position z noise"),
            DRSpec(name="bolt_yaw", type="runtime", min=-0.524, max=0.524,
                   description="Bolt yaw noise (±30°)"),
            DRSpec(name="hand_init_pos_xy", type="runtime", min=-0.02, max=0.02,
                   description="Hand initial pos xy noise (±2cm)"),
            DRSpec(name="hand_init_pos_z", type="runtime", min=-0.01, max=0.01,
                   description="Hand initial pos z noise (±1cm)"),
            DRSpec(name="hand_init_yaw", type="runtime", min=-0.26, max=0.26,
                   description="Hand initial yaw noise (±15° around 1.83 rad)"),
        ]

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset to nominal pose. DR application deferred (see DR spec note).

        - Arm at DEFAULT_ARM_QPOS (Isaac reset_joints)
        - Fingers closed (0.04 each — menagerie default open)
        - Hole at scene MJCF position (0.6, 0, 0.05)
        - Peg at weld-resolved position (mjx.forward derives from arm + weld eq)
        """
        rng, key_obs, key_qnoise, key_hole = jax.random.split(rng, 4)

        # Build a fresh data with the IK-resolved arm pose. NOTE: Warp's
        # make_data takes the CPU `mj_model` (not the mjx Model), per
        # mjx.io._make_data_warp. self._init_arm_qpos is precomputed in
        # __init__ so the fingertip starts inside the action_chain bounds
        # cube around the hole.
        data = mjx_env.make_data(
            self._mj_model,
            qpos=jp.zeros(self._mj_model.nq),
            qvel=jp.zeros(self._mj_model.nv),
            ctrl=jp.zeros(self._mj_model.nu),
            impl=self._config.impl,
            naconmax=self._config.naconmax,
            naccdmax=self._config.naccdmax,
            njmax=self._config.njmax,
        )

        # Per-episode hole pose randomization via mocap_pos / mocap_quat.
        # hole_base is a mocap body (scene.xml `mocap="true"`) — its world
        # pose lives in data.mocap_pos/mocap_quat and is kinematic (no
        # dynamics, infinite-mass contacts). Sampling here makes the same
        # trained policy generalize across hole locations without rebuilding
        # the model. State obs uses fingertip_pos_rel_fixed which is
        # hole-relative, so the actor naturally adapts; critic gets absolute
        # fixed_pos in privileged_state.
        hole_xy_lo = -self._config.hole_pos_xy_noise
        hole_z_lo = -self._config.hole_pos_z_noise
        hole_pos_offset = jax.random.uniform(
            key_hole, shape=(3,),
            minval=jp.array([hole_xy_lo, hole_xy_lo, hole_z_lo]),
            maxval=jp.array([-hole_xy_lo, -hole_xy_lo, -hole_z_lo]),
        )
        hole_mocap_pos = self._hole_nominal_pos + hole_pos_offset
        # Yaw noise wires in here when rotation actions land in Phase 5+.
        # For now hole_yaw_noise=0 → hole_mocap_quat = nominal.
        hole_mocap_quat = self._hole_nominal_quat
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._hole_mocap_id].set(hole_mocap_pos),
            mocap_quat=data.mocap_quat.at[self._hole_mocap_id].set(hole_mocap_quat),
        )
        # Per-episode arm qpos noise: ±0.02 rad per joint perturbs hand pose
        # by ~0.5-1.2 cm in xy, ±2° in tilt. v15.4/v15.5 with ±0.05 broke
        # the lucky-discovery determinism (training Return up to 6150 in
        # v15.4 but deterministic eval stuck at 1728 hover) — the wide
        # noise made policy unable to internalize descent as the
        # deterministic mean. Tightened range gives subtler exploration
        # while still preventing perfect hover memorization.
        arm_qpos_noise = jax.random.uniform(
            key_qnoise, shape=(len(self._arm_jnt_ids),),
            minval=-0.02, maxval=0.02,
        )
        qpos = data.qpos
        for i, jnt_id in enumerate(self._arm_jnt_ids):
            qpos = qpos.at[self._arm_qposadr[i]].set(
                self._init_arm_qpos[i] + arm_qpos_noise[i]
            )
        data = data.replace(qpos=qpos)
        # Pass 1: forward kinematics on the arm only so we know where the hand
        # frame sits. We then place the peg at the weld-resolved relpose so the
        # equality constraint solver has zero residual to chase at step 0.
        data = mjx.forward(self.mjx_model, data)
        hand_pos = data.xpos[self._hand_body_id]
        hand_mat = data.xmat[self._hand_body_id].reshape(3, 3)
        # relpose="0 0 -0.130 1 0 0 0" in body1=peg's frame means hand sits at
        # peg_origin + R_peg·(0,0,-0.13). With identity relpose orientation,
        # R_peg = R_hand, so peg_origin = hand_pos + R_hand·(0, 0, +0.130).
        peg_offset_in_hand = jp.array([0.0, 0.0, 0.130])
        peg_pos_init = hand_pos + hand_mat @ peg_offset_in_hand
        hand_quat = data.xquat[self._hand_body_id]
        peg_init = jp.concatenate([peg_pos_init, hand_quat])  # xyz + (w,x,y,z)
        qpos = jax.lax.dynamic_update_slice(
            data.qpos, peg_init, (self._peg_qpos_addr,)
        )
        data = data.replace(qpos=qpos)
        # Pass 2: re-forward with peg now placed at the weld equilibrium.
        data = mjx.forward(self.mjx_model, data)

        # Two pose frames live in info:
        #   - fixed_pos: hole body origin. Used as the reward target (peg
        #     seated position) and as the obs-frame for fingertip_pos_rel_fixed.
        #   - clip_anchor: bore opening top = fixed_pos + (0, 0, asset_height).
        #     Used by action_chain.clip_to_bounds — IsaacLab anchors there
        #     (fixed_pos_obs_frame in factory_env.py), NOT the hole body
        #     origin, so that the cube containing reachable fingertip
        #     targets sits at the bore tip where the policy commands ops.
        # Read hole pose from data.mocap_pos (post-DR sample), NOT from
        # mj_model.body_pos which only holds the scene.xml nominal anchor.
        fixed_pos = data.mocap_pos[self._hole_mocap_id]
        fixed_quat = data.mocap_quat[self._hole_mocap_id]
        clip_anchor = fixed_pos.at[2].add(self._config.asset_height)

        initial_fingertip_quat = self._fingertip_quat(data)
        info = {
            "fixed_pos": fixed_pos,
            "fixed_quat": fixed_quat,
            "clip_anchor": clip_anchor,
            # OSC target_quat accumulated from policy rot actions (6-DOF
            # controller). Reset to init_target_quat on episode boundary.
            "target_quat": initial_fingertip_quat,
            "init_target_quat": initial_fingertip_quat,
            "actions": jp.zeros(6),
            "prev_actions": jp.zeros(6),
            "step_count": jp.array(0, dtype=jp.int32),
            "prev_fingertip_pos": data.site_xpos[self._fingertip_site_id],
            "prev_fingertip_quat": initial_fingertip_quat,
            "rng": key_obs,
            "truncation": jp.array(0.0),
        }
        obs, info["rng"] = compute_obs(
            self._obs_groups,
            noise_level=0.0,
            rng=info["rng"],
            data=data, info=info,
        )
        return mjx_env.State(
            data=data, obs=obs, reward=jp.array(0.0), done=jp.array(0.0),
            metrics={}, info=info,
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Single env step. Phase 1: zero-torque controller (Phase 2 wires OSC).

        Pipeline:
          1. Reset EMA on done=True per-env (state.info trap)
          2. Apply EMA + denorm + clip to derive ctrl_target (Phase 2 uses this)
          3. Inner substep loop (decimation × sim_dt) with zero ctrl
          4. Recompute obs, reward, done
        """
        from jax_rl.envs.manipulation.factory.controller.action_chain import (
            apply_ema, denormalize, clip_to_bounds, reset_on_done, rotvec_to_quat,
        )
        from jax_rl.envs.manipulation.factory.controller.osc import (
            compute_osc_torque,
        )
        from jax_rl.envs.manipulation.factory.reward import compute_reward

        info = state.info
        # 1. State.info auto-reset trap — explicitly clear EMA on done.
        done_bool = state.done.astype(jp.bool_)
        ema_prev = reset_on_done(info["actions"], jp.atleast_1d(done_bool))[0]
        # 2. EMA + denorm + clip → OSC target_pos / target_quat (6-DOF).
        ema = apply_ema(action, ema_prev, ema_factor=self._config.ema_factor)
        pos_delta, rot_delta = denormalize(
            ema,
            jp.asarray(self._config.pos_threshold),
            jp.asarray(self._config.rot_threshold),
        )
        target_pos = clip_to_bounds(
            state.data.site_xpos[self._fingertip_site_id] + pos_delta,
            info["clip_anchor"],
            jp.asarray(self._config.pos_action_bounds),
        )
        # Accumulating target_quat: dq = rotvec_to_quat(rot_delta), composed
        # with previous target_quat (reset to init on episode boundary).
        target_quat_prev = jp.where(
            done_bool, info["init_target_quat"], info["target_quat"]
        )
        dq = rotvec_to_quat(rot_delta)
        target_quat = _quat_mul(target_quat_prev, dq)
        target_quat = target_quat / (jp.linalg.norm(target_quat) + 1e-12)

        # 3. Inner substep loop. Controller depends on actuator_mode.
        nu = self._mj_model.nu
        if self._config.actuator_mode == "motor":
            # Phase 2 OSC: recompute torque per substep (J, M, errors all
            # depend on q, qdot which advance inside the loop).
            arm_dof_ids = jp.asarray(self._arm_dofadr)
            arm_qpos_ids = jp.asarray(self._arm_qposadr)
            arm_act_ids = jp.asarray(self._arm_act_ids)
            kp_task = jp.asarray(self._config.task_prop_gains)
            kd_task = jp.asarray(self._config.task_deriv_gains)
            # Nullspace posture target = IsaacLab's default_dof_pos_tensor
            # (NOT the reset pose). Pulls arm into a kinematically benign
            # config in the task nullspace.
            q_default = jp.asarray(NULLSPACE_ARM_QPOS)
            site_id = self._fingertip_site_id

            def inner_step(_, d):
                # Feedforward gravity + Coriolis comp via qfrc_bias (MuJoCo's
                # inverse-dynamics bias term). Without this, OSC has to fight
                # gravity through pose error, which converges too slowly to
                # hold pose over decimation horizons.
                tau_arm = compute_osc_torque(
                    self.mjx_model, d,
                    target_pos=target_pos,
                    target_quat=target_quat,
                    site_id=site_id,
                    arm_dof_ids=arm_dof_ids,
                    arm_qpos_ids=arm_qpos_ids,
                    kp_task=kp_task,
                    kd_task=kd_task,
                    q_default=q_default,
                    kp_null=self._config.kp_null,
                    kd_null=self._config.kd_null,
                    torque_limit=self._config.torque_limit,
                    feedforward=d.qfrc_bias[arm_dof_ids],
                )
                ctrl = jp.zeros(nu).at[arm_act_ids].set(tau_arm)
                return mjx.step(self.mjx_model, d.replace(ctrl=ctrl))
        else:
            # Phase 1 placeholder (position_pd): hold arm at the IK-resolved
            # init pose so the visualization stays sane. ctrl=0 would yank
            # arm to q=0 (straight up) because panda's general+affine bias
            # actuators interpret ctrl as q_target.
            hold_ctrl = jp.zeros(nu).at[:7].set(jp.asarray(self._init_arm_qpos))

            def inner_step(_, d):
                return mjx.step(self.mjx_model, d.replace(ctrl=hold_ctrl))

        data = jax.lax.fori_loop(0, self._config.decimation, inner_step, state.data)

        # 4. Update info (advance step_count, store prev_fingertip for finite-diff)
        prev_fingertip_pos = state.data.site_xpos[self._fingertip_site_id]
        prev_fingertip_quat = self._fingertip_quat(state.data)
        new_info = {
            **info,                     # carries fixed_pos/quat, init_target_quat, rng
            "target_quat":          target_quat,
            "actions":              ema,
            "prev_actions":         info["actions"],
            "prev_fingertip_pos":   prev_fingertip_pos,
            "prev_fingertip_quat":  prev_fingertip_quat,
            "step_count":           info["step_count"] + 1,
        }

        # 5. Reward
        held_pos = data.xpos[self._peg_body_id]
        held_quat = data.xquat[self._peg_body_id]
        # Target = peg seated at hole (fixed_pos + bore opening at z=hole.z+0.025)
        target_pos = info["fixed_pos"]
        target_quat = info["fixed_quat"]
        peg_z = data.xpos[self._peg_body_id, 2]
        hole_top_z = info["fixed_pos"][2] + self._config.asset_height
        # entry_z = peg body z at which the tip crosses the bore opening.
        # Must track hole_top_z per-episode (hole_pos varies via mocap DR),
        # not stay at the nominal 0.096 constant.
        entry_z = hole_top_z + self._config.peg_half_length
        reward = compute_reward(
            held_pos=held_pos, held_quat=held_quat,
            target_pos=target_pos, target_quat=target_quat,
            peg_z=peg_z, hole_top_z=hole_top_z,
            keypoint_coef_baseline=self._config.keypoint_coef_baseline,
            keypoint_coef_coarse=self._config.keypoint_coef_coarse,
            keypoint_coef_fine=self._config.keypoint_coef_fine,
            asset_height=self._config.asset_height,
            engage_threshold=self._config.engage_threshold,
            success_threshold=self._config.success_threshold,
            keypoint_scale=self._config.keypoint_scale,
            engage_xy_threshold=self._config.engage_xy_threshold,
            success_xy_threshold=self._config.success_xy_threshold,
            entry_z=entry_z,
        )

        # 6. Done. TimeLimit only (no early termination — peg can't fall, welded).
        truncated = new_info["step_count"] >= self._config.episode_length
        terminated = jp.array(False)
        done = (truncated | terminated).astype(jp.float32)
        new_info["truncation"] = truncated.astype(jp.float32)

        # 7. New obs. compute_obs threads rng through info["rng"].
        obs, new_info["rng"] = compute_obs(
            self._obs_groups,
            noise_level=0.0,
            rng=new_info["rng"],
            data=data, info=new_info,
        )

        return mjx_env.State(
            data=data, obs=obs, reward=reward, done=done,
            metrics={}, info=new_info,
        )

    # ─────────────────────────────────────────────────────────────────
    # Public properties (mjx_env.MjxEnv contract)
    # ─────────────────────────────────────────────────────────────────

    @property
    def action_size(self) -> int:
        return self._config.action_dim

    @property
    def observation_size(self):
        from jax_rl.envs.obs_spec import schema_from_obs_groups
        return schema_from_obs_groups(self._obs_groups)

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self):
        """Lazy-construct the Warp/MJX model; only requires CUDA when first accessed."""
        if self._mjx_model_cache is None:
            self._mjx_model_cache = mjx.put_model(self._mj_model, impl=self._config.impl)
        return self._mjx_model_cache

    @property
    def xml_path(self) -> str:
        return str(SCENE_XML)


# ─────────────────────────────────────────────────────────────────────
# Math helpers (JIT-friendly)
# ─────────────────────────────────────────────────────────────────────

def _mat_to_quat(m: jp.ndarray) -> jp.ndarray:
    """3x3 rotation matrix → quat (w, x, y, z). JAX-stable 4-branch formula."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    def trace_pos(_):
        s = jp.sqrt(trace + 1.0) * 2
        return jp.array([0.25 * s,
                         (m[2, 1] - m[1, 2]) / s,
                         (m[0, 2] - m[2, 0]) / s,
                         (m[1, 0] - m[0, 1]) / s])

    def case_xx(_):
        s = jp.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return jp.array([(m[2, 1] - m[1, 2]) / s,
                         0.25 * s,
                         (m[0, 1] + m[1, 0]) / s,
                         (m[0, 2] + m[2, 0]) / s])

    def case_yy(_):
        s = jp.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return jp.array([(m[0, 2] - m[2, 0]) / s,
                         (m[0, 1] + m[1, 0]) / s,
                         0.25 * s,
                         (m[1, 2] + m[2, 1]) / s])

    def case_zz(_):
        s = jp.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        return jp.array([(m[1, 0] - m[0, 1]) / s,
                         (m[0, 2] + m[2, 0]) / s,
                         (m[1, 2] + m[2, 1]) / s,
                         0.25 * s])

    xx, yy, zz = m[0, 0], m[1, 1], m[2, 2]

    def trace_neg(_):
        return jax.lax.cond(
            (xx > yy) & (xx > zz),
            case_xx,
            lambda _: jax.lax.cond(yy > zz, case_yy, case_zz, operand=None),
            operand=None,
        )

    return jax.lax.cond(trace > 0, trace_pos, trace_neg, operand=None)


def _quat_mul(a: jp.ndarray, b: jp.ndarray) -> jp.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return jp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_conj(q: jp.ndarray) -> jp.ndarray:
    return jp.array([q[0], -q[1], -q[2], -q[3]])


def _quat_finite_diff_angvel(q_prev: jp.ndarray, q_curr: jp.ndarray, dt: float) -> jp.ndarray:
    """Body-frame angular velocity from successive quaternions."""
    qpc = _quat_mul(q_curr, _quat_conj(q_prev))
    return 2.0 * qpc[1:] / dt
