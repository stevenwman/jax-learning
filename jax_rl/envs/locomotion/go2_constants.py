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

SCENE_FLAT_XML = ROOT_PATH / "go2_scene_flat.xml"

# Foot geom names (used for contact sensors).
# Order: FL, FR, RL, RR (matches Menagerie MJCF).
FEET_GEOMS = ["FL", "FR", "RL", "RR"]

# Foot site names (Go2 uses "*_foot" suffix unlike Go1).
FEET_SITES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

FEET_POS_SENSOR = ["FL_pos", "FR_pos", "RL_pos", "RR_pos"]

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
