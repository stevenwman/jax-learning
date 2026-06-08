"""FactoryGearMesh — port of IsaacLab Factory GearMesh to MuJoCo Warp.

Architecture (parallels FactoryPegInsert):
  - Subclass mujoco_playground._src.mjx_env.MjxEnv (Warp backend)
  - Dict obs {"state": (25,), "privileged_state": (~72,)} — asymmetric AC
  - Held MEDIUM gear attached to gripper via <equality><weld/>
  - Geometry: plate primitives (slab + 3 cylinder pegs) +
              CoACD-decomposed gears (small/medium/large) +
              flanking gears hinge-constrained to their pegs
  - Controller: motor mode + OSC (shared with PegInsert)

Reference:
  - jax_rl/envs/manipulation/factory/factory_peg_insert.py (sibling env)
  - IsaacLab v2.3.2 factory_env.py + factory_tasks_cfg.py:GearMesh
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
GEAR_ASSETS = ASSETS_DIR / "gear_mesh"
PANDA_DIR = ASSETS_DIR / "franka_panda"
PANDA_XML = PANDA_DIR / "panda.xml"
SCENE_XML = GEAR_ASSETS / "scene.xml"
COACD_DIR = GEAR_ASSETS / "extracted" / "coacd"

# IsaacLab peg offsets in plate-local frame (x-only, y=0).
PEG_X = {"small": +0.0508, "medium": +0.0203, "large": -0.0302}
GEAR_MASS = {"small": 0.012, "medium": 0.018, "large": 0.019}


def default_config() -> config_dict.ConfigDict:
    """Default GearMesh env config. Mirrors PegInsert's defaults; reward + obs
    coefs may diverge once Phase 1 training data lands."""
    return config_dict.create(
        sim_dt=0.002,
        ctrl_dt=0.0083333,
        decimation=4,
        episode_length=900,
        action_dim=6,
        grasp_mode="weld",
        ema_factor=0.2,
        pos_threshold=(0.02, 0.02, 0.02),
        rot_threshold=(0.097, 0.097, 0.097),
        pos_action_bounds=(0.02, 0.02, 0.10),
        actuator_mode="motor",
        task_prop_gains=(100.0, 100.0, 100.0, 30.0, 30.0, 30.0),
        task_deriv_gains=(20.0, 20.0, 20.0, 10.954, 10.954, 10.954),
        kp_null=10.0,
        kd_null=6.3246,
        torque_limit=100.0,
        # Reward (reuse PegInsert's phased reward — Phase B aligned-descent +
        # dead-end penalty works for gear-onto-peg too).
        keypoint_coef_baseline=(5.0, 4.0),
        keypoint_coef_coarse=(50.0, 2.0),
        keypoint_coef_fine=(100.0, 0.0),
        keypoint_scale=0.05,
        # PegInsert's reward uses (asset_height, engage/success_threshold)
        # geometrically: insertion depth = hole_top_z - peg_z; "engaged" when
        # depth > engage_threshold * asset_height. For gear-on-peg, the
        # physical descent range from spawn-just-above-peg to seated is
        # ~0.008 m (gear visual centroid drops by gear_thickness/2 -
        # gear_bottom_offset). Setting asset_height to this descent so the
        # thresholds index the gear's actual reachable range.
        engage_threshold=0.5,                # engaged ~halfway through descent
        success_threshold=0.85,              # success ~fully seated
        asset_height=0.008,                  # gear descent range from start to seated
        peg_half_length=0.0125,              # medium gear half-thickness (gear extrusion / 2)
        # Per-episode plate-pose randomization.
        hole_pos_xy_noise=0.02,
        hole_pos_z_noise=0.01,
        hole_yaw_noise=0.0,
        # XY + tilt gates for engaged / success.
        engage_xy_threshold=0.005,
        success_xy_threshold=0.0025,
        # MJX/Warp buffer budgets — bigger than PegInsert since we have 392
        # CoACD geoms in the scene. Smoke train surfaced "CCD overflow →
        # increase naccdmax to 9000+" warnings; 16384 gives 1.5× margin.
        impl="warp",
        naconmax=16 * 8192,
        naccdmax=16384,
        njmax=20480,                  # Unified bitmask + 60-piece medium peaks
                                      # at ~15k nefc; 20k = ~30% safety margin.
        # "sdf" = production: flanking gears as single mesh-SDF geoms,
        #          medium uses symmetry-exploiting 60-piece decomp
        #          (40 tooth instances of 1 mesh + 20 hub-arc instances of 1).
        # "coacd" = legacy: all 3 gears as full CoACD parts (392 total).
        flanking_collision="coacd",
        # flanking_collision="coacd",   # uncomment + comment line above to swap
    )


def _build_scene_xml(flanking_collision: str = "sdf") -> str:
    """Splice panda.xml into scene.xml + inject collision blocks for 3 gears.

    flanking_collision: "coacd" → all 3 gears as CoACD parts (production).
                        "sdf"   → small/large as single mesh-SDF geoms,
                                  medium stays CoACD. Adds contype/conaffinity
                                  bitmasks to exclude SDF×cylinder pairs which
                                  Warp's SDF dispatcher doesn't handle (only
                                  PLANE/SPHERE/BOX/ELLIPSOID/MESH/SDF).
                                  Bit layout: peg=1, sdf_gear=2, default=4.
                                  - cylinder pegs:    contype=1 conaffinity=4
                                  - sdf flanking:     contype=2 conaffinity=4
                                  - medium CoACD:     contype=4 conaffinity=7
                                  - slab + render:    contype=4 conaffinity=7
                                  Pairs: peg×medium ✓, peg×sdf ✗, sdf×medium ✓,
                                  sdf×slab ✓ (slab is BOX, SDF-supported).
    """
    scene_text = SCENE_XML.read_text()

    sdf_mode = flanking_collision == "sdf"
    # Unified bitmask scheme (applies in BOTH modes):
    #   bit layout: peg=1, flanker_gear=2, medium_gear=4
    #   plate slab : contype=1 conaffinity=4   (excludes flanker contact)
    #   pegs       : contype=1 conaffinity=4
    #   flankers   : contype=2 conaffinity=4   (collide only with medium)
    #   medium     : contype=4 conaffinity=7   (collide with all)
    # Result: flanker ↔ plate = 0 → no contact. flanker rides on hinge only.
    scene_text = scene_text.replace(
        '<geom name="slab" type="box"',
        '<geom name="slab" type="box" contype="1" conaffinity="4"',
    )
    for n in ("peg_large", "peg_medium", "peg_small"):
        scene_text = scene_text.replace(
            f'<geom name="{n}" type="cylinder"',
            f'<geom name="{n}" type="cylinder" contype="1" conaffinity="4"',
        )

    for gear in ("small", "medium", "large"):
        use_sdf_gear = sdf_mode and gear in ("small", "large")
        bm = ('contype="2" conaffinity="4"' if gear in ("small", "large")
              else 'contype="4" conaffinity="7"')
        if use_sdf_gear:
            assets_block = (
                f'    <mesh name="gear_{gear}_sdf" '
                f'file="gear_{gear}.obj"/>'
            )
            geoms_block = (
                f'      <geom type="sdf" mesh="gear_{gear}_sdf" '
                f'mass="{GEAR_MASS[gear]:.5f}" {bm} condim="6" '
                f'friction="1.0 0.01 0.0001" '
                f'solref="0.004 1" solimp="0.98 0.995 0.001"/>'
            )
        elif gear == "medium":
            # 60-piece symmetry-exploiting decomp ALWAYS used for medium gear,
            # regardless of flanking mode. 40 tooth instances (9° apart) + 20
            # hub-arc instances (18° apart). Meshes axis-aligned with gear axis
            # at mesh-local (0,0) and rim/hub boundary at z=0; pos shifts to
            # body-local frame of original gear_medium mesh.
            tooth_mass = GEAR_MASS["medium"] * (40 * 0.298) / (40 * 0.298 + 20 * 0.507) / 40
            hub_mass   = GEAR_MASS["medium"] * (20 * 0.507) / (40 * 0.298 + 20 * 0.507) / 20
            assets_block = (
                '    <mesh name="gear_medium_tooth" file="gear_medium_tooth.obj"/>\n'
                '    <mesh name="gear_medium_hub" file="gear_medium_hub.obj"/>'
            )
            tooth_lines = [
                f'      <geom type="mesh" mesh="gear_medium_tooth" '
                f'pos="0.02025 0 0.015" euler="0 0 {i * 9}" '
                f'mass="{tooth_mass:.6f}" {bm} condim="6" '
                f'friction="1.0 0.01 0.0001" '
                f'solref="0.004 1" solimp="0.98 0.995 0.001"/>'
                for i in range(40)
            ]
            hub_lines = [
                f'      <geom type="mesh" mesh="gear_medium_hub" '
                f'pos="0.02025 0 0.015" euler="0 0 {i * 18}" '
                f'mass="{hub_mass:.6f}" {bm} condim="6" '
                f'friction="1.0 0.01 0.0001" '
                f'solref="0.004 1" solimp="0.98 0.995 0.001"/>'
                for i in range(20)
            ]
            geoms_block = "\n".join(tooth_lines + hub_lines)
        else:
            parts = sorted((COACD_DIR / f"gear_{gear}").glob("part_*.obj"))
            if not parts:
                raise FileNotFoundError(
                    f"No CoACD parts found in {COACD_DIR}/gear_{gear}. "
                    "Run .tmp/coacd_decompose_gears.py to regenerate."
                )
            m_each = GEAR_MASS[gear] / len(parts)
            assets_block = "\n".join(
                f'    <mesh name="gear_{gear}_p{i:02d}" '
                f'file="gear_{gear}_part_{i:02d}.obj"/>'
                for i, p in enumerate(parts)
            )
            geoms_block = "\n".join(
                f'      <geom type="mesh" mesh="gear_{gear}_p{i:02d}" '
                f'mass="{m_each:.5f}" {bm} condim="6" friction="1.0 0.01 0.0001" '
                f'solref="0.004 1" solimp="0.98 0.995 0.001"/>'
                for i, p in enumerate(parts)
            )
        scene_text = scene_text.replace(
            f"<!-- GEAR_{gear.upper()}_ASSETS -->", assets_block
        ).replace(
            f"<!-- GEAR_{gear.upper()}_GEOMS -->", geoms_block
        )

    # 2. Splice panda.xml inline (mirror PegInsert).
    panda_text = PANDA_XML.read_text()
    m = re.search(r"<mujoco[^>]*>(.*)</mujoco>", panda_text, re.DOTALL)
    if not m:
        raise RuntimeError(f"Could not parse panda.xml at {PANDA_XML}")
    panda_inner = m.group(1)
    panda_inner = re.sub(r"<compiler[^/]*/>", "", panda_inner)
    panda_inner = re.sub(r"<option[^/]*/>", "", panda_inner)
    scene_text = re.sub(
        r'<include\s+file="\.\./franka_panda/panda\.xml"\s*/>',
        panda_inner,
        scene_text,
    )
    return scene_text


def _build_assets_dict(flanking_collision: str = "sdf") -> Dict[str, bytes]:
    """Bundle panda meshes + gear OBJs for from_xml_string.

    flanking_collision:
      "sdf"   production: small+large SDF (full meshes), medium = 60-piece
              decomp (2 small meshes: gear_medium_tooth.obj + _hub.obj).
      "coacd" legacy: all 3 gears as full CoACD parts (392 OBJs).
    """
    assets: Dict[str, bytes] = {}
    panda_assets_dir = PANDA_DIR / "assets"
    for p in panda_assets_dir.iterdir():
        if p.is_file():
            assets[p.name] = p.read_bytes()
    GEAR_EXT_DIR = COACD_DIR.parent  # extracted/ dir holds gear_X.obj
    sdf_mode = flanking_collision == "sdf"
    # Medium ALWAYS uses the 60-piece decomp (2 mesh assets).
    assets["gear_medium_tooth.obj"] = (GEAR_EXT_DIR / "gear_medium_tooth.obj").read_bytes()
    assets["gear_medium_hub.obj"]   = (GEAR_EXT_DIR / "gear_medium_hub.obj").read_bytes()
    for gear in ("small", "large"):
        if sdf_mode:
            assets[f"gear_{gear}.obj"] = (GEAR_EXT_DIR / f"gear_{gear}.obj").read_bytes()
        else:
            for i, p in enumerate(sorted((COACD_DIR / f"gear_{gear}").glob("part_*.obj"))):
                assets[f"gear_{gear}_part_{i:02d}.obj"] = p.read_bytes()
    return assets


def _apply_actuator_mode(mj_model: mujoco.MjModel, mode: str) -> None:
    """Rewrite arm actuators to motor mode (raw torque). Mirror PegInsert."""
    if mode == "position_pd":
        return
    if mode != "motor":
        raise ValueError(f"actuator_mode must be 'position_pd' or 'motor', got {mode!r}")
    arm_act_ids = [mj_model.actuator(f"actuator{i}").id for i in range(1, 8)]
    for aid in arm_act_ids:
        mj_model.actuator_gaintype[aid] = int(mujoco.mjtGain.mjGAIN_FIXED)
        mj_model.actuator_gainprm[aid, :] = 0.0
        mj_model.actuator_gainprm[aid, 0] = 1.0
        mj_model.actuator_biastype[aid] = int(mujoco.mjtBias.mjBIAS_NONE)
        mj_model.actuator_biasprm[aid, :] = 0.0
        mj_model.actuator_ctrlrange[aid] = mj_model.actuator_forcerange[aid]


DEFAULT_ARM_QPOS = np.array(
    [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01]
)
NULLSPACE_ARM_QPOS = np.array(
    [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
)
DEFAULT_FINGER_QPOS = np.array([0.04, 0.04])


def _solve_init_arm_qpos(mj_model, target_pos, target_quat, *,
                          site_name="fingertip_centered",
                          arm_jnt_names=("joint1","joint2","joint3","joint4","joint5","joint6","joint7"),
                          seed_qpos=DEFAULT_ARM_QPOS, max_iters=500,
                          pos_tol=1e-4, rot_tol=1e-3, damp=1e-2, step_scale=0.5):
    """Damped-LS IK on a 6-D site Jacobian. Mirror PegInsert."""
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

    jacp = np.zeros((3, mj_model.nv)); jacr = np.zeros((3, mj_model.nv))
    cur_quat = np.zeros(4); err_quat = np.zeros(4); rot_err = np.zeros(3)
    target_quat = target_quat / np.linalg.norm(target_quat)

    for _ in range(max_iters):
        mujoco.mj_kinematics(mj_model, mj_data)
        mujoco.mj_comPos(mj_model, mj_data)
        cur_pos = np.asarray(mj_data.site_xpos[site_id])
        pos_err = target_pos - cur_pos
        mujoco.mju_mat2Quat(cur_quat, mj_data.site_xmat[site_id])
        mujoco.mju_negQuat(err_quat, cur_quat)
        mujoco.mju_mulQuat(err_quat, target_quat, err_quat)
        mujoco.mju_quat2Vel(rot_err, err_quat, 1.0)
        if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < rot_tol:
            break
        mujoco.mj_jacSite(mj_model, mj_data, jacp, jacr, site_id)
        J_arm = np.concatenate([jacp[:, arm_dofadr], jacr[:, arm_dofadr]], axis=0)
        err6 = np.concatenate([pos_err, rot_err])
        JJt = J_arm @ J_arm.T + (damp ** 2) * np.eye(6)
        dq = J_arm.T @ np.linalg.solve(JJt, err6)
        mj_data.qpos[arm_qposadr] += step_scale * dq

    return mj_data.qpos[arm_qposadr].copy()


class FactoryGearMesh(mjx_env.MjxEnv):
    """Factory GearMesh — Phase 1 scaffold (env loading + reset + zero-action steps).

    Phase 1: env loads, reset places medium gear above its peg via panda weld,
             policy step works with motor mode (zero torque).
    Phase 2: OSC controller (shared with PegInsert).
    Phase 3: Train SAC / FlashSAC.
    """

    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        if config is None:
            config = default_config()
        super().__init__(config, config_overrides)

        scene_text = _build_scene_xml(self._config.flanking_collision)
        assets = _build_assets_dict(self._config.flanking_collision)
        self._mj_model = mujoco.MjModel.from_xml_string(scene_text, assets=assets)
        self._mj_model.opt.timestep = self._config.sim_dt

        _apply_actuator_mode(self._mj_model, self._config.actuator_mode)
        self._mj_model.opt.ccd_iterations = 100

        self._mjx_model_cache = None

        # Cache body / site / joint IDs.
        self._hand_body_id = self._mj_model.body("hand").id
        self._medium_body_id = self._mj_model.body("g_medium").id
        self._small_body_id = self._mj_model.body("g_small").id
        self._large_body_id = self._mj_model.body("g_large").id
        self._plate_body_id = self._mj_model.body("plate_base").id
        self._plate_mocap_id = int(self._mj_model.body_mocapid[self._plate_body_id])
        assert self._plate_mocap_id >= 0, "plate_base must be mocap='true' in scene.xml"

        self._plate_nominal_pos = jp.asarray(self._mj_model.body_pos[self._plate_body_id])
        self._plate_nominal_quat = jp.asarray(self._mj_model.body_quat[self._plate_body_id])
        self._fingertip_site_id = self._mj_model.site("fingertip_centered").id

        # Arm joints
        arm_jnt_ids = [self._mj_model.joint(f"joint{i}").id for i in range(1, 8)]
        self._arm_jnt_ids = np.asarray(arm_jnt_ids, dtype=np.int32)
        self._arm_qposadr = np.asarray(
            [self._mj_model.jnt_qposadr[i] for i in self._arm_jnt_ids], dtype=np.int32
        )
        self._arm_dofadr = np.asarray(
            [self._mj_model.jnt_dofadr[i] for i in self._arm_jnt_ids], dtype=np.int32
        )

        # Medium gear freejoint qpos start (xyz + quat).
        med_jnt_id = self._mj_model.body_jntadr[self._medium_body_id]
        self._medium_qpos_addr = int(self._mj_model.jnt_qposadr[med_jnt_id])

        # Flanking gears' hinge dofs (for obs).
        self._small_hinge_jnt_id = self._mj_model.joint("g_small_hinge").id
        self._large_hinge_jnt_id = self._mj_model.joint("g_large_hinge").id
        self._small_hinge_qadr = int(self._mj_model.jnt_qposadr[self._small_hinge_jnt_id])
        self._large_hinge_qadr = int(self._mj_model.jnt_qposadr[self._large_hinge_jnt_id])
        # qvel index (≠ qpos for hinges sharing a freejoint-rooted body chain).
        self._small_hinge_dofadr = int(self._mj_model.jnt_dofadr[self._small_hinge_jnt_id])
        self._large_hinge_dofadr = int(self._mj_model.jnt_dofadr[self._large_hinge_jnt_id])

        self._arm_act_ids = np.asarray(
            [self._mj_model.actuator(f"actuator{i}").id for i in range(1, 8)],
            dtype=np.int32,
        )

        self._init_arm_qpos = self._solve_init_pose()
        self._setup_obs_groups()

    def _solve_init_pose(self) -> np.ndarray:
        """IK: fingertip ~5cm above the MEDIUM peg's top opening, gripper-down.

        Plate is at (0.6, 0, 0.05). Medium peg in plate-local frame at
        (+0.0203, 0, 0.025) (peg top at plate.z + 0.025).
        World target = plate_pos + (peg_x, 0, peg_top_z + clearance + fingertip_offset).
        """
        plate_pos = np.asarray(self._mj_model.body_pos[self._plate_body_id])
        peg_top_z = plate_pos[2] + 0.025                             # peg top in world
        gear_tip_clearance = 0.04                                     # gear bottom 4cm above peg top
        # Fingertip-to-gear-bottom geom math (post-flip):
        #   fingertip = hand - 0.1034 (menagerie panda)
        #   gear body = hand - 0.130 (relpose translation in body1 frame)
        #   gear visual centroid = body + 0.0147 world (after 180x flip, OBJ
        #     centroid_z=0.0147 above body's world-z because body's z is now
        #     world +z direction after the flip cancels gripper-down)
        #     ⇒ gear visual centroid = hand - 0.130 + 0.0147 = hand - 0.1153
        #   gear bottom = visual - (centroid_z - obj_bottom_z) = visual - 0.0097
        #     = hand - 0.125
        #   ⇒ fingertip - gear_bottom = (hand - 0.1034) - (hand - 0.125) = +0.0216
        fingertip_to_gear_bottom = 0.0216
        target_pos = np.array([
            plate_pos[0] + PEG_X["medium"],
            plate_pos[1],
            peg_top_z + gear_tip_clearance + fingertip_to_gear_bottom,
        ])

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

    def _setup_obs_groups(self) -> None:
        """Obs replicates PegInsert layout. Held = medium gear; fixed = medium-peg position."""
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

    # Obs callables (mirror PegInsert)
    def _fingertip_pos(self, data, **kw): return data.site_xpos[self._fingertip_site_id]
    def _fingertip_quat(self, data, **kw):
        m = data.site_xmat[self._fingertip_site_id].reshape(3, 3)
        return _mat_to_quat(m)
    def _fingertip_pos_rel_fixed(self, data, info, **kw):
        return data.site_xpos[self._fingertip_site_id] - info["fixed_pos"]
    def _ee_linvel(self, data, info, **kw):
        return (data.site_xpos[self._fingertip_site_id] - info["prev_fingertip_pos"]) / self._config.ctrl_dt
    def _ee_angvel(self, data, info, **kw):
        return _quat_finite_diff_angvel(info["prev_fingertip_quat"], self._fingertip_quat(data), self._config.ctrl_dt)
    def _joint_pos(self, data, **kw): return data.qpos[self._arm_qposadr]
    def _joint_vel(self, data, **kw): return data.qvel[self._arm_dofadr]
    def _held_pos(self, data, **kw):
        # Body origin ≠ gear visual centroid (mesh auto-recenter offsets each
        # part). Add the OBJ-frame centroid offset (0.0203, 0, 0.0147 — from
        # the full medium gear mesh), rotated into world by body's current
        # orientation.
        body_pos = data.xpos[self._medium_body_id]
        body_mat = data.xmat[self._medium_body_id].reshape(3, 3)
        return body_pos + body_mat @ jp.array([0.0203, 0.0, 0.0147])
    def _held_quat(self, data, **kw): return data.xquat[self._medium_body_id]
    def _held_pos_rel_fixed(self, data, info, **kw):
        return data.xpos[self._medium_body_id] - info["fixed_pos"]

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset: arm at IK-resolved pose, medium gear at weld equilibrium,
        flanking gears at θ=0, plate at nominal (DR applied per-episode)."""
        rng, key_obs, key_qnoise, key_plate = jax.random.split(rng, 4)

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

        # Per-episode plate DR.
        xy_lo = -self._config.hole_pos_xy_noise
        z_lo  = -self._config.hole_pos_z_noise
        plate_offset = jax.random.uniform(
            key_plate, shape=(3,),
            minval=jp.array([xy_lo, xy_lo, z_lo]),
            maxval=jp.array([-xy_lo, -xy_lo, -z_lo]),
        )
        plate_mocap_pos = self._plate_nominal_pos + plate_offset
        plate_mocap_quat = self._plate_nominal_quat
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._plate_mocap_id].set(plate_mocap_pos),
            mocap_quat=data.mocap_quat.at[self._plate_mocap_id].set(plate_mocap_quat),
        )

        # Per-episode arm qpos noise.
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
        data = mjx.forward(self.mjx_model, data)

        # Place medium gear at weld equilibrium.
        # relpose="0.0203 0 0.130 0 1 0 0" — 180° flip around x so gear OBJ +z
        # ends up aligned with world +z (right-side-up despite hand gripper-down).
        # body1_in_body2 = T_body2_in_body1^-1 = (R(180x) @ -p, q^-1).
        # R_180x @ (-0.0203, 0, -0.130) = (-0.0203, 0, +0.130).
        hand_pos = data.xpos[self._hand_body_id]
        hand_mat = data.xmat[self._hand_body_id].reshape(3, 3)
        gear_offset_in_hand = jp.array([-0.0203, 0.0, 0.130])
        gear_pos_init = hand_pos + hand_mat @ gear_offset_in_hand
        # gear quat = hand_quat * flip_x_quat (chain hand orient with flip).
        hand_quat = data.xquat[self._hand_body_id]
        flip_x_quat = jp.array([0.0, 1.0, 0.0, 0.0])
        gear_quat_init = _quat_mul(hand_quat, flip_x_quat)
        gear_init = jp.concatenate([gear_pos_init, gear_quat_init])
        qpos = jax.lax.dynamic_update_slice(
            data.qpos, gear_init, (self._medium_qpos_addr,)
        )
        data = data.replace(qpos=qpos)
        data = mjx.forward(self.mjx_model, data)

        # Build info.
        #   fixed_pos = gear's TARGET visual centroid (= seated gear pos in world).
        #     Plate top at plate_z + 0.005. Gear OBJ bottom at z=0.005, centroid
        #     at z=0.0157, so centroid is 0.0107m above bottom. When seated:
        #     centroid world_z = plate_z + 0.005 + 0.0107 = plate_z + 0.0157.
        #   hole_top_z stored separately for entry_z classification in step().
        plate_pos = data.mocap_pos[self._plate_mocap_id]
        plate_quat = data.mocap_quat[self._plate_mocap_id]
        target_local = jp.array([PEG_X["medium"], 0.0, 0.0147])
        fixed_pos = plate_pos + target_local
        fixed_quat = plate_quat
        hole_top_z = plate_pos[2] + 0.025                     # peg top in world
        clip_anchor = fixed_pos

        initial_fingertip_quat = self._fingertip_quat(data)
        info = {
            "fixed_pos": fixed_pos,
            "fixed_quat": fixed_quat,
            "hole_top_z": hole_top_z,
            "clip_anchor": clip_anchor,
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
            self._obs_groups, noise_level=0.0, rng=info["rng"],
            data=data, info=info,
        )
        return mjx_env.State(
            data=data, obs=obs, reward=jp.array(0.0), done=jp.array(0.0),
            metrics={}, info=info,
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        from jax_rl.envs.manipulation.factory.controller.action_chain import (
            apply_ema, denormalize, clip_to_bounds, reset_on_done, rotvec_to_quat,
        )
        from jax_rl.envs.manipulation.factory.controller.osc import compute_osc_torque
        from jax_rl.envs.manipulation.factory.reward import compute_reward

        info = state.info
        done_bool = state.done.astype(jp.bool_)
        ema_prev = reset_on_done(info["actions"], jp.atleast_1d(done_bool))[0]
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

        nu = self._mj_model.nu
        if self._config.actuator_mode == "motor":
            arm_dof_ids = jp.asarray(self._arm_dofadr)
            arm_qpos_ids = jp.asarray(self._arm_qposadr)
            arm_act_ids = jp.asarray(self._arm_act_ids)
            kp_task = jp.asarray(self._config.task_prop_gains)
            kd_task = jp.asarray(self._config.task_deriv_gains)
            q_default = jp.asarray(NULLSPACE_ARM_QPOS)
            site_id = self._fingertip_site_id

            def inner_step(_, d):
                tau_arm = compute_osc_torque(
                    self.mjx_model, d,
                    target_pos=target_pos, target_quat=target_quat,
                    site_id=site_id,
                    arm_dof_ids=arm_dof_ids, arm_qpos_ids=arm_qpos_ids,
                    kp_task=kp_task, kd_task=kd_task,
                    q_default=q_default,
                    kp_null=self._config.kp_null, kd_null=self._config.kd_null,
                    torque_limit=self._config.torque_limit,
                    feedforward=d.qfrc_bias[arm_dof_ids],
                )
                ctrl = jp.zeros(nu).at[arm_act_ids].set(tau_arm)
                return mjx.step(self.mjx_model, d.replace(ctrl=ctrl))
        else:
            hold_ctrl = jp.zeros(nu).at[:7].set(jp.asarray(self._init_arm_qpos))
            def inner_step(_, d):
                return mjx.step(self.mjx_model, d.replace(ctrl=hold_ctrl))

        data = jax.lax.fori_loop(0, self._config.decimation, inner_step, state.data)

        prev_fingertip_pos = state.data.site_xpos[self._fingertip_site_id]
        prev_fingertip_quat = self._fingertip_quat(state.data)
        new_info = {
            **info,
            "target_quat":          target_quat,
            "actions":              ema,
            "prev_actions":         info["actions"],
            "prev_fingertip_pos":   prev_fingertip_pos,
            "prev_fingertip_quat":  prev_fingertip_quat,
            "step_count":           info["step_count"] + 1,
        }

        held_pos = self._held_pos(data)
        # Tilt reward expects held z-axis aligned with world -z (gripper-down,
        # cos_dot = +1 at perfect alignment). Our medium gear is right-side-up
        # (its z-axis points +world_z due to the 180° weld flip). Pre-multiply
        # held_quat by flip_x_quat (0,1,0,0) so the reward sees gripper-down
        # orientation when the gear is actually right-side-up.
        flip_x_quat = jp.array([0.0, 1.0, 0.0, 0.0])
        held_quat_raw = data.xquat[self._medium_body_id]
        held_quat = _quat_mul(held_quat_raw, flip_x_quat)
        target_pos_r = info["fixed_pos"]
        target_quat_r = info["fixed_quat"]
        peg_z = held_pos[2]
        # hole_top_z = top of medium peg in world (separate from fixed_pos
        # which is now the gear's seated centroid, not peg top).
        hole_top_z = info["hole_top_z"]
        entry_z = hole_top_z + self._config.peg_half_length
        reward = compute_reward(
            held_pos=held_pos, held_quat=held_quat,
            target_pos=target_pos_r, target_quat=target_quat_r,
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

        # Tooth-meshing reward, gated by NOT seated. Meshgate confirmed during
        # this session to be a net win: more rotation pre-seat, none post-seat.
        from jax_rl.envs.manipulation.factory.reward import is_success as _is_success
        omega_small = data.qvel[self._small_hinge_dofadr]
        omega_large = data.qvel[self._large_hinge_dofadr]
        spin = jp.abs(omega_small) + jp.abs(omega_large)
        seated = _is_success(
            held_pos[:2], peg_z, held_quat,
            info["fixed_pos"][:2], hole_top_z,
            self._config.asset_height, self._config.success_threshold,
            xy_threshold=self._config.success_xy_threshold,
        ).astype(jp.float32)
        r_mesh = (1.0 - seated) * 1.0 * jp.tanh(spin / 2.0)
        reward = reward + r_mesh

        truncated = new_info["step_count"] >= self._config.episode_length
        done = truncated.astype(jp.float32)
        new_info["truncation"] = truncated.astype(jp.float32)

        obs, new_info["rng"] = compute_obs(
            self._obs_groups, noise_level=0.0, rng=new_info["rng"],
            data=data, info=new_info,
        )

        return mjx_env.State(
            data=data, obs=obs, reward=reward, done=done,
            metrics={}, info=new_info,
        )

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
        if self._mjx_model_cache is None:
            self._mjx_model_cache = mjx.put_model(self._mj_model, impl=self._config.impl)
        return self._mjx_model_cache

    @property
    def xml_path(self) -> str:
        return str(SCENE_XML)


# ─────────────────────────────────────────────────────────────────────
# Math helpers — shared with PegInsert (just copy to keep envs decoupled)
# ─────────────────────────────────────────────────────────────────────

def _mat_to_quat(m: jp.ndarray) -> jp.ndarray:
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    def trace_pos(_):
        s = jp.sqrt(trace + 1.0) * 2
        return jp.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                         (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    def case_xx(_):
        s = jp.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return jp.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                         (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    def case_yy(_):
        s = jp.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return jp.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                         0.25 * s, (m[1, 2] + m[2, 1]) / s])
    def case_zz(_):
        s = jp.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        return jp.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                         (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    xx, yy, zz = m[0, 0], m[1, 1], m[2, 2]
    def trace_neg(_):
        return jax.lax.cond(
            (xx > yy) & (xx > zz), case_xx,
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
    qpc = _quat_mul(q_curr, _quat_conj(q_prev))
    return 2.0 * qpc[1:] / dt
