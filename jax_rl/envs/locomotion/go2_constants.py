"""Unitree Go2 quadruped constants.

Differences from Playground's Go1:
  - Root body: "base" (Go1: "trunk")
  - Foot sites: "FL_foot" etc. (Go1: "FL")
  - Foot geoms: "FL" etc. (same as Go1)
  - Joint order: FL, FR, RL, RR (Go1 Playground: FR, FL, RR, RL)
  - Default height: 0.27m (Go1: 0.278m)
"""

from pathlib import Path

from mujoco_playground._src import mjx_env

# Path to our XML files.
ROOT_PATH = Path(__file__).parent / "xmls"

# ARCHIVED — MJX env used Menagerie scene XML (go2_base.py, go2_joystick.py).
# Kept for reference. Active env uses WARP_SCENE_FLAT_XML.
SCENE_FLAT_XML = ROOT_PATH / "go2_scene_flat.xml"

# Foot geom names (used for contact sensors).
# Order: FL, FR, RL, RR (matches Menagerie MJCF).
FEET_GEOMS = ["FL", "FR", "RL", "RR"]

# Foot site names (Go2 uses "*_foot" suffix unlike Go1).
FEET_SITES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

FEET_POS_SENSOR = ["FL_pos", "FR_pos", "RL_pos", "RR_pos"]

# Leg actuator names in MJX actuator order (verified against unitree_go2/go2.xml:227-238).
# Used by envs that add extra non-leg actuators (splitbelt: 12 leg + 2 belt) to filter
# `_act_to_joint` to leg-only entries. Matches SDK leg ordering: FR, FL, RR, RL.
LEG_ACTUATOR_NAMES = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)

# ARCHIVED — Menagerie root body name. Warp env uses WARP_ROOT_BODY.
ROOT_BODY = "base"

# Sensor names (matching scene XML + Menagerie sensors).
UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

WARP_ROOT_BODY = "base_link"  # unitree go2.xml (vs "base" in Menagerie)

WARP_SCENE_FLAT_XML = ROOT_PATH / "go2_warp_scene_flat.xml"
BONGO_SCENE_XML = ROOT_PATH / "go2_bongo_scene.xml"

# Per-joint-type velocity limits (rad/s). Source: unitree_rl_gym/resources/robots/go2/urdf/go2.urdf.
# Joint sub-order within each leg: (hip, thigh, calf). Legs repeat this triplet.
# Stall torques are read from MJCF actuator_ctrlrange at env init (per-joint),
# so they are not duplicated here.
MOTOR_VELOCITY_LIMIT_PER_JOINT_TYPE = (30.1, 30.1, 20.07)
