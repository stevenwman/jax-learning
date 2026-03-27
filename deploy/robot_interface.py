"""DDS interface to Go2 robot (sim or real) via unitree_sdk2_python."""
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
        iface = Go2Interface(sim=True)
        iface.start()
        while True:
            state = iface.get_state()
            iface.send_action(action_policy_order)
    """

    def __init__(self, sim: bool = True, interface: str = "lo"):
        self.sim = sim
        self.interface = interface
        self.kp = KP_SIM if sim else KP_REAL
        self.kd = KD_SIM if sim else KD_REAL
        self._low_state = None
        self._crc = CRC()

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
        action_sdk = action_policy_order[POLICY_TO_SDK]
        q_targets = DEFAULT_POSE_SDK + action_sdk * ACTION_SCALE
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
        """Send default standing pose."""
        self.send_joint_targets(DEFAULT_POSE_SDK)

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
