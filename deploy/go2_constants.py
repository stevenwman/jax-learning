"""Go2 deployment constants — joint ordering, default pose, PD gains."""
import numpy as np

# Joint remapping: Unitree SDK order (FR,FL,RR,RL) <-> MJX env order (FL,FR,RL,RR)
# The mapping is symmetric (same array both directions)
SDK_TO_POLICY = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
POLICY_TO_SDK = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])

# Default joint angles (radians) in POLICY order (FL, FR, RL, RR)
# Each leg: [hip, thigh, calf] — from go2_mjx.xml keyframe("home").qpos[7:]
# All four legs identical in the MJCF keyframe
DEFAULT_POSE_POLICY = np.array([
    0.0, 0.9, -1.8,    # FL
    0.0, 0.9, -1.8,    # FR
    0.0, 0.9, -1.8,    # RL
    0.0, 0.9, -1.8,    # RR
], dtype=np.float32)

# Same in SDK order (FR, FL, RR, RL)
DEFAULT_POSE_SDK = DEFAULT_POSE_POLICY[POLICY_TO_SDK]

ACTION_SCALE = 0.5

# PD gains for deployment
# WARNING: These must match the training env. Two configs exist:
#   MJX env (Go2JoystickFlat):      Kp=35, Kd=0.1
#   Warp env (Go2WarpJoystickFlat): Kp=20, Kd=0.5
# Using the wrong gains with a checkpoint = policy fights itself.
# TODO: read gains from checkpoint meta.json instead of hardcoding.
KP_SIM = 35.0    # MJX training env
KD_SIM = 0.1     # MJX training env
KP_WARP = 20.0   # Warp training env (unitree RL gains)
KD_WARP = 0.5    # Warp training env
KP_REAL = 20.0   # Unitree official for Go2 RL deployment
KD_REAL = 0.5    # Go2 hardware value

NUM_JOINTS = 12
POLICY_DT = 0.02  # 50 Hz policy
CONTROL_DT = 0.002  # 500 Hz motor commands

# Gravity vector (used for projected gravity computation)
GRAVITY_VEC = np.array([0.0, 0.0, -1.0], dtype=np.float32)
