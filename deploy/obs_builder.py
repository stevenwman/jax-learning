"""Build policy observation vector from raw sensor data."""
import numpy as np
from deploy.go2_constants import SDK_TO_POLICY, DEFAULT_POSE_POLICY, NUM_JOINTS


def _quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vector by inverse of quaternion. quat = [w, x, y, z].

    Equivalent to R(q)^T @ vec, which transforms from world frame to body frame.
    """
    w, x, y, z = quat
    # Rotation matrix columns
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y + w * z)
    r02 = 2 * (x * z - w * y)
    r10 = 2 * (x * y - w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z + w * x)
    r20 = 2 * (x * z + w * y)
    r21 = 2 * (y * z - w * x)
    r22 = 1 - 2 * (x * x + y * y)
    # R^T @ vec (transpose = inverse for rotation matrices)
    return np.array([
        r00 * vec[0] + r10 * vec[1] + r20 * vec[2],
        r01 * vec[0] + r11 * vec[1] + r21 * vec[2],
        r02 * vec[0] + r12 * vec[1] + r22 * vec[2],
    ], dtype=np.float32)


class ObsBuilder:
    """Builds observation vector from robot sensor readings.

    With n_frame_stack > 1, returns stacked obs (n_frame_stack * 48d). Newest frame at front, oldest at back.

    Obs layout (matching go2_warp_joystick.py _get_obs):
        [0:3]   local_linvel (zeroed for deployment — not available on hardware)
        [3:6]   gyroscope
        [6:9]   projected_gravity (from quaternion)
        [9:21]  joint_pos - default_pose (remapped from SDK to policy order)
        [21:33] joint_vel (remapped)
        [33:45] last_action (in policy order)
        [45:48] command (vx, vy, yaw_rate)
    """

    def __init__(self, n_frame_stack: int = 1):
        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._quat_checked = False
        self._n_frames = n_frame_stack
        self._raw_dim = 48
        self._frame_stack = np.zeros(self._n_frames * self._raw_dim, dtype=np.float32)
        self._initialized = False

    def build(
        self,
        joint_pos_sdk: np.ndarray,
        joint_vel_sdk: np.ndarray,
        gyroscope: np.ndarray,
        quaternion: np.ndarray,
        command: np.ndarray,
        linvel: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build 48d obs from sensor readings.

        Args:
            joint_pos_sdk: (12,) joint positions in SDK order (FR,FL,RR,RL)
            joint_vel_sdk: (12,) joint velocities in SDK order
            gyroscope: (3,) angular velocity [wx, wy, wz]
            quaternion: (4,) orientation [w, x, y, z]
            command: (3,) velocity command [vx, vy, yaw_rate]
            linvel: (3,) local linear velocity (optional, zeroed if None)
        """
        # Runtime quaternion convention check (first call only)
        if not self._quat_checked:
            norm = np.linalg.norm(quaternion)
            if abs(norm - 1.0) > 0.1:
                print(f"WARNING: quaternion norm={norm:.3f}, expected ~1.0")
            if abs(quaternion[3]) > 0.9 and abs(quaternion[0]) < 0.1:
                print("WARNING: quaternion appears to be [x,y,z,w] not [w,x,y,z]")
            self._quat_checked = True

        # Remap joints from SDK to policy order
        joint_pos = joint_pos_sdk[SDK_TO_POLICY]
        joint_vel = joint_vel_sdk[SDK_TO_POLICY]

        # Projected gravity in body frame
        proj_gravity = _quat_rotate_inverse(quaternion, self.gravity_world)

        # Local linear velocity (zero if not available)
        local_linvel = linvel if linvel is not None else np.zeros(3, dtype=np.float32)

        obs = np.concatenate([
            local_linvel,                                              # [0:3]
            gyroscope.astype(np.float32),                              # [3:6]
            proj_gravity,                                              # [6:9]
            (joint_pos - DEFAULT_POSE_POLICY).astype(np.float32),      # [9:21]
            joint_vel.astype(np.float32),                              # [21:33]
            self.last_action,                                          # [33:45]
            command.astype(np.float32),                                # [45:48]
        ])
        # Update frame stack: push new obs to front, shift old frames right.
        if self._n_frames > 1:
            if not self._initialized:
                # First call: fill all frames with initial obs.
                self._frame_stack = np.tile(obs, self._n_frames)
                self._initialized = True
            else:
                self._frame_stack[self._raw_dim:] = self._frame_stack[:-self._raw_dim]
                self._frame_stack[:self._raw_dim] = obs
            return self._frame_stack.copy()
        return obs

    def update_last_action(self, action: np.ndarray):
        """Store action for next obs construction (in policy order)."""
        self.last_action = action.copy()
