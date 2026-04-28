"""Build policy observation vector from raw sensor data.

Schema-driven: at training time we save `meta.json["obs_schema"]["state"]` =
ordered list of term names (e.g. `["gyro", "accelerometer", "gravity",
"joint_pos_offset", "joint_vel", "last_act", "command"]`). At deploy time,
`ObsBuilder.from_checkpoint(ckpt_dir)` reads that list and composes the obs
in the saved order, looking up each term in `SENSOR_FETCHERS`.

This makes deploy obs auto-sync with sim. To add a new obs term:
1. Add it to the env's `_obs_groups["state"]` (sim side).
2. Register a fetcher in `SENSOR_FETCHERS` (deploy side, this file).

To remove a term: just drop it from the env. New checkpoints serialise the
shorter schema; deploy auto-adapts at load time.

Old checkpoints without `obs_schema` fall back to the current default
schema (matches sim env as of 2026-04-24): gyro, accelerometer, gravity,
joint_pos_offset, joint_vel, last_act, command.
"""
import json
import os
import warnings
import numpy as np
from deploy.go2_constants import SDK_TO_POLICY, DEFAULT_POSE_POLICY, NUM_JOINTS


def _quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vector by inverse of quaternion. quat = [w, x, y, z].

    Equivalent to R(q)^T @ vec, which transforms from world frame to body frame.
    """
    w, x, y, z = quat
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y + w * z)
    r02 = 2 * (x * z - w * y)
    r10 = 2 * (x * y - w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z + w * x)
    r20 = 2 * (x * z + w * y)
    r21 = 2 * (y * z - w * x)
    r22 = 1 - 2 * (x * x + y * y)
    return np.array([
        r00 * vec[0] + r10 * vec[1] + r20 * vec[2],
        r01 * vec[0] + r11 * vec[1] + r21 * vec[2],
        r02 * vec[0] + r12 * vec[1] + r22 * vec[2],
    ], dtype=np.float32)


_GRAVITY_WORLD = np.array([0.0, 0.0, -1.0], dtype=np.float32)


# Term-name → sensor-fetcher. Each fn takes a `signals` dict from build()
# and returns a 1D np.float32 array. Add new terms here when sim env adds them.
def _signals(joint_pos_sdk, joint_vel_sdk, gyroscope, accelerometer, quaternion,
             command, last_action, default_pose_policy=DEFAULT_POSE_POLICY,
             sdk_to_policy=SDK_TO_POLICY):
    """Compute every available signal once; fetchers select from this dict.

    `default_pose_policy` and `sdk_to_policy` default to deploy/go2_constants.py
    constants for legacy callers; `ObsBuilder.from_checkpoint(strict=True)`
    overrides them with values from `meta["control"]` (single source of truth).
    """
    joint_pos = joint_pos_sdk[sdk_to_policy]
    joint_vel = joint_vel_sdk[sdk_to_policy]
    return {
        "gyro": gyroscope.astype(np.float32),
        "accelerometer": accelerometer.astype(np.float32),
        "gravity": _quat_rotate_inverse(quaternion, _GRAVITY_WORLD),
        "joint_pos_offset": (joint_pos - default_pose_policy).astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "last_act": last_action,
        "command": command.astype(np.float32),
    }


# Default schema (fallback for old ckpts without meta["obs_schema"]).
# Matches WarpJoystick._obs_groups["state"] as of 2026-04-24.
DEFAULT_STATE_SCHEMA = [
    "gyro", "accelerometer", "gravity",
    "joint_pos_offset", "joint_vel", "last_act", "command",
]

# Per-term widths. Used to compute raw_dim and validate at construction.
_TERM_DIMS = {
    "gyro": 3, "accelerometer": 3, "gravity": 3,
    "joint_pos_offset": 12, "joint_vel": 12,
    "last_act": 12, "command": 3,
}


class ObsBuilder:
    """Builds observation vector from robot sensor readings.

    Schema-driven: the order and presence of terms in the output is
    controlled by `state_schema` (a list of term names). Construct via
    `from_checkpoint(ckpt_dir)` to inherit the schema saved at training time.

    With n_frame_stack > 1, returns stacked obs (n_frame_stack * raw_dim).
    Newest frame at front, oldest at back.
    """

    def __init__(
        self,
        n_frame_stack: int = 1,
        state_schema: list[str] | None = None,
        default_pose_policy: np.ndarray | None = None,
        sdk_to_policy: np.ndarray | None = None,
    ):
        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._quat_checked = False
        self._n_frames = n_frame_stack
        self.state_schema = state_schema if state_schema is not None else list(DEFAULT_STATE_SCHEMA)
        self.default_pose_policy = (
            np.asarray(default_pose_policy, dtype=np.float32)
            if default_pose_policy is not None else DEFAULT_POSE_POLICY
        )
        self.sdk_to_policy = (
            np.asarray(sdk_to_policy, dtype=np.int64)
            if sdk_to_policy is not None else SDK_TO_POLICY
        )

        unknown = [t for t in self.state_schema if t not in _TERM_DIMS]
        if unknown:
            raise ValueError(
                f"ObsBuilder: unknown term(s) in state_schema: {unknown}. "
                f"Add a fetcher to deploy/obs_builder.py:_signals AND a width "
                f"to _TERM_DIMS, then retry."
            )

        self._raw_dim = sum(_TERM_DIMS[t] for t in self.state_schema)
        self._frame_stack = np.zeros(self._n_frames * self._raw_dim, dtype=np.float32)
        self._initialized = False

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: str,
        n_frame_stack: int = 1,
        strict: bool = False,
    ) -> "ObsBuilder":
        """Construct an ObsBuilder using the schema + default_pose saved in meta.json.

        With `strict=True`, raises if `obs_schema` or `meta["control"]` is
        missing. Use `strict=True` for real-robot arm (`deploy_go2.py` without
        `--sim`); `strict=False` for sim2sim/dry-run/legacy ckpts.

        Without strict, falls back to DEFAULT_STATE_SCHEMA + go2_constants
        DEFAULT_POSE_POLICY for ckpts predating self-describing meta
        (pre-2026-04-24).
        """
        meta_path = os.path.join(ckpt_dir, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        schema_dict = meta.get("obs_schema")
        control = meta.get("control")

        if schema_dict is None:
            if strict:
                raise RuntimeError(
                    f"ObsBuilder.from_checkpoint(strict=True): "
                    f"{ckpt_dir}/meta.json missing 'obs_schema'. Refuse to "
                    f"arm a real robot with a legacy ckpt — re-train or pass "
                    f"strict=False for dry-run only."
                )
            warnings.warn(
                f"{ckpt_dir}/meta.json has no obs_schema (pre-2026-04-24). "
                f"Falling back to DEFAULT_STATE_SCHEMA — verify it matches "
                f"what the policy was trained with.",
                stacklevel=2,
            )
            return cls(n_frame_stack=n_frame_stack)

        state_schema = schema_dict.get("state")
        if state_schema is None:
            raise KeyError("obs_schema in meta.json is missing 'state' key")

        # default_pose: prefer meta["control"]["default_pose_policy"] (single
        # source of truth), fall back to constants for older ckpts that have
        # obs_schema but no control block.
        default_pose_policy = None
        sdk_to_policy = None
        if control is not None:
            if "default_pose_policy" in control:
                default_pose_policy = np.asarray(
                    control["default_pose_policy"], dtype=np.float32
                )
            if "sdk_to_policy" in control:
                sdk_to_policy = np.asarray(control["sdk_to_policy"], dtype=np.int64)
        elif strict:
            raise RuntimeError(
                f"ObsBuilder.from_checkpoint(strict=True): "
                f"{ckpt_dir}/meta.json missing 'control' block."
            )

        return cls(
            n_frame_stack=n_frame_stack,
            state_schema=state_schema,
            default_pose_policy=default_pose_policy,
            sdk_to_policy=sdk_to_policy,
        )

    @property
    def raw_dim(self) -> int:
        return self._raw_dim

    def build(
        self,
        joint_pos_sdk: np.ndarray,
        joint_vel_sdk: np.ndarray,
        gyroscope: np.ndarray,
        accelerometer: np.ndarray,
        quaternion: np.ndarray,
        command: np.ndarray,
    ) -> np.ndarray:
        """Build obs vector by composing schema terms in saved order.

        Args:
            joint_pos_sdk: (12,) joint positions in SDK order (FR,FL,RR,RL)
            joint_vel_sdk: (12,) joint velocities in SDK order
            gyroscope: (3,) angular velocity [wx, wy, wz] in body frame (rad/s)
            accelerometer: (3,) specific force in body frame (m/s²)
            quaternion: (4,) orientation [w, x, y, z]
            command: (3,) velocity command [vx, vy, yaw_rate]
        """
        if not self._quat_checked:
            norm = np.linalg.norm(quaternion)
            if abs(norm - 1.0) > 0.1:
                print(f"WARNING: quaternion norm={norm:.3f}, expected ~1.0")
            if abs(quaternion[3]) > 0.9 and abs(quaternion[0]) < 0.1:
                print("WARNING: quaternion appears to be [x,y,z,w] not [w,x,y,z]")
            self._quat_checked = True

        signals = _signals(
            joint_pos_sdk, joint_vel_sdk, gyroscope, accelerometer,
            quaternion, command, self.last_action,
            default_pose_policy=self.default_pose_policy,
            sdk_to_policy=self.sdk_to_policy,
        )
        obs = np.concatenate([signals[name] for name in self.state_schema])

        if self._n_frames > 1:
            if not self._initialized:
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
