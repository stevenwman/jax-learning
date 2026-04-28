"""DDS interface to Go2 robot (sim or real) via unitree_sdk2_python."""
import warnings
import numpy as np
from deploy.go2_constants import (
    POLICY_TO_SDK, DEFAULT_POSE_SDK, NUM_JOINTS,
    KP_SIM, KD_SIM, KP_REAL, KD_REAL, ACTION_SCALE,
)

from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


class Go2Interface:
    """Publish motor commands and subscribe to robot state via CycloneDDS.

    Usage:
        iface = Go2Interface(sim=True)                     # legacy, uses constants
        iface = Go2Interface(sim=True, control_meta=meta)  # checkpoint-driven
        iface.start()
        while True:
            state = iface.get_state()
            iface.send_action(action_policy_order)

    When `control_meta` is provided (from `meta["control"]` written by
    `Go2WarpEnv.get_control_metadata()`), Kp/Kd/action_scale/default_pose/
    policy_to_sdk are read from it. Without it, falls back to
    `deploy/go2_constants.py` constants with a loud warning. The single-source-
    of-truth meta path closes the codex-audit P0 finding where real deploy
    silently drifted from training values.
    """

    def __init__(
        self,
        sim: bool = True,
        interface: str = "lo",
        control_meta: dict | None = None,
    ):
        self.sim = sim
        self.interface = interface
        self._low_state = None
        self._crc = CRC()

        if control_meta is not None:
            self.kp = float(control_meta["Kp"])
            self.kd = float(control_meta["Kd"])
            self.action_scale = float(control_meta["action_scale"])
            self.default_pose_sdk = np.asarray(
                control_meta["default_pose_sdk"], dtype=np.float32
            )
            self.policy_to_sdk = np.asarray(control_meta["policy_to_sdk"], dtype=np.int64)
            # Belt-and-suspenders: warn if meta drifts from go2_constants.py.
            # If meta and constants ever diverge, the meta wins (it came from the
            # actual training env), but the operator should know the constants
            # file is stale. Soft assertion — print, don't raise.
            self._sanity_check_against_constants()
        else:
            warnings.warn(
                "Go2Interface: no control_meta — falling back to "
                "deploy/go2_constants.py. OK for sim2sim/legacy ckpts, "
                "NOT recommended for real arm.",
                stacklevel=2,
            )
            self.kp = KP_SIM if sim else KP_REAL
            self.kd = KD_SIM if sim else KD_REAL
            self.action_scale = ACTION_SCALE
            self.default_pose_sdk = DEFAULT_POSE_SDK.astype(np.float32)
            self.policy_to_sdk = POLICY_TO_SDK

    def _sanity_check_against_constants(self) -> None:
        """Warn if meta values diverge from `deploy/go2_constants.py`."""
        mismatches = []
        if not np.allclose(self.default_pose_sdk, DEFAULT_POSE_SDK, atol=1e-5):
            mismatches.append(
                f"default_pose_sdk: meta={self.default_pose_sdk.tolist()} "
                f"vs constants={DEFAULT_POSE_SDK.tolist()}"
            )
        if not np.array_equal(self.policy_to_sdk, POLICY_TO_SDK):
            mismatches.append(
                f"policy_to_sdk: meta={self.policy_to_sdk.tolist()} "
                f"vs constants={POLICY_TO_SDK.tolist()}"
            )
        if not np.isclose(self.action_scale, ACTION_SCALE):
            mismatches.append(
                f"action_scale: meta={self.action_scale} vs constants={ACTION_SCALE}"
            )
        if mismatches:
            warnings.warn(
                "Go2Interface: meta diverges from deploy/go2_constants.py "
                "(meta wins). Consider updating constants:\n  "
                + "\n  ".join(mismatches),
                stacklevel=3,
            )

    def start(self):
        """Initialize DDS channels."""
        domain_id = 1 if self.sim else 0
        ChannelFactoryInitialize(domain_id, self.interface)

        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._pub.Init()

        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_low_state, 10)

        # Prepare command template
        self._cmd = unitree_go_msg_dds__LowCmd_()
        self._cmd.head[0] = 0xFE
        self._cmd.head[1] = 0xEF
        self._cmd.level_flag = 0xFF
        self._cmd.gpio = 0
        for i in range(20):
            self._cmd.motor_cmd[i].mode = 0x01

    def _on_low_state(self, msg: LowState_):
        self._low_state = msg

    def get_state(self):
        """Get current robot state. Returns None if no state received yet.

        Returns dict with: joint_pos_sdk, joint_vel_sdk, gyroscope, quaternion
        All joints in SDK order (FR, FL, RR, RL).
        """
        if self._low_state is None:
            return None

        state = self._low_state
        joint_pos = np.array([state.motor_state[i].q for i in range(NUM_JOINTS)], dtype=np.float32)
        joint_vel = np.array([state.motor_state[i].dq for i in range(NUM_JOINTS)], dtype=np.float32)
        gyro = np.array(state.imu_state.gyroscope, dtype=np.float32)
        quat = np.array(state.imu_state.quaternion, dtype=np.float32)

        accel = np.array(state.imu_state.accelerometer, dtype=np.float32)

        return {
            "joint_pos_sdk": joint_pos,
            "joint_vel_sdk": joint_vel,
            "gyroscope": gyro,
            "quaternion": quat,
            "accelerometer": accel,
        }

    def send_action(self, action_policy_order: np.ndarray):
        """Send action to robot. action in policy order (FL,FR,RL,RR), range [-1, 1]."""
        action_sdk = action_policy_order[self.policy_to_sdk]
        q_targets = self.default_pose_sdk + action_sdk * self.action_scale
        self.send_joint_targets(q_targets)

    def send_joint_targets(self, q_targets_sdk: np.ndarray):
        """Send raw joint position targets in SDK order."""
        for i in range(NUM_JOINTS):
            self._cmd.motor_cmd[i].q = float(q_targets_sdk[i])
            self._cmd.motor_cmd[i].kp = self.kp
            self._cmd.motor_cmd[i].dq = 0.0
            self._cmd.motor_cmd[i].kd = self.kd
            self._cmd.motor_cmd[i].tau = 0.0
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)

    def send_stand(self):
        """Send default standing pose (from meta when available, else constants)."""
        self.send_joint_targets(self.default_pose_sdk)

    def send_zero_torque(self):
        """Send zero torque (robot goes limp)."""
        for i in range(NUM_JOINTS):
            self._cmd.motor_cmd[i].q = 0.0
            self._cmd.motor_cmd[i].kp = 0.0
            self._cmd.motor_cmd[i].dq = 0.0
            self._cmd.motor_cmd[i].kd = 0.0
            self._cmd.motor_cmd[i].tau = 0.0
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)
