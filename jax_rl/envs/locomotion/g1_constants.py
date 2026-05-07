"""Unitree G1 humanoid constants.

29-actuator menagerie variant: 12 legs + 3 waist + 14 arms (incl. wrists).
qpos layout: 7 freejoint + 29 joints = 36.
"""

from pathlib import Path

ROOT_PATH = Path(__file__).parent / "xmls"

# Foot site names (G1 menagerie convention).
FEET_SITES = ["left_foot", "right_foot"]
FEET_POS_SENSOR = ["left_foot_pos", "right_foot_pos"]
FEET_LINVEL_SENSOR = ["left_foot_global_linvel", "right_foot_global_linvel"]

# Foot collision geom box (one per foot — single ref point, not the 3 capsules).
# Used for floor-contact pair sensors.
FEET_COLLISION_GEOMS = ["left_foot_box_collision", "right_foot_box_collision"]

# Actuator names in MJX actuator order (verified against menagerie g1_mjx.xml).
# 12 legs (L hip-pitch/roll/yaw, knee, ankle-pitch/roll, then R) +
# 3 waist (yaw, roll, pitch) +
# 14 arms (L shoulder-pitch/roll/yaw, elbow, wrist-roll/pitch/yaw; same R).
ACTUATOR_NAMES = (
    # Legs (12)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist (3)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # Left arm (7)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm (7)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
NUM_ACTUATORS = len(ACTUATOR_NAMES)  # 29

# Body / site references.
ROOT_BODY = "pelvis"
IMU_SITE = "imu_in_pelvis"

# Sensor names (defined in scene XML — same naming as Go2 except foot sensors).
UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

# Per-actuator force limits (Nm) from menagerie ctrlrange — used for
# torque-speed model normalisation if we ever enable it.
# Order matches ACTUATOR_NAMES.
ACTUATOR_FORCE_LIMITS = (
    # Legs: hip 88/88/88, knee 139, ankle 50/50
    88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
    # Waist: yaw 88, roll 50, pitch 50
    88.0, 50.0, 50.0,
    # Arms: shoulder 25 each, elbow 25, wrist roll 25, pitch/yaw 5
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
)

# Path to scene XML.
WARP_SCENE_FLAT_XML = ROOT_PATH / "g1_warp_scene_flat.xml"
