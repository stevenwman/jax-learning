"""Go2 RSL-RL policy wrapper.

Loads a Go2 locomotion checkpoint (.pt) and turns observations into joint
position targets for the Newton/MuJoCo control buffer.

Encapsulates the load + obs/action remap that was inlined into several
example_mpm_go2_*.py files.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import warp as wp

from mpm_go2_multi.example_robot_go2 import compute_obs, lab_to_mujoco, mujoco_to_lab


# Go2 Robot Configuration
INITIAL_Q = {
    # Front Left
    "FL_hip_joint": 0.1,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,

    # Front Right
    "FR_hip_joint": -0.1,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,

    # Rear Left
    "RL_hip_joint": 0.1,
    "RL_thigh_joint": 1.0,
    "RL_calf_joint": -1.5,

    # Rear Right
    "RR_hip_joint": -0.1,
    "RR_thigh_joint": 1.0,
    "RR_calf_joint": -1.5,
}

# PD control gains
PD_GAINS_KE = 60  # 40 #150
PD_GAINS_KD = 3.5  # 2.5 #5

# Action scaling
DEFAULT_ACTION_SCALE = 0.25

# Maps Newton's leg-grouped joint order [FL,FR,RL,RR] to the robot_lab/mud
# policy's [FR,FL,RR,RL] order (and back — it is an involution).
ROBOT_LAB_JOINT_SWAP = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


class Go2Policy:
    """Wraps a Go2 RSL-RL MLP policy and the lab<->mujoco joint remap.

    Call ``compute_joint_targets(state, command)`` each control tick; the
    returned wp.array has length ``6 + act_dim`` (first 6 entries are the
    free-joint padding expected by ``control.joint_target_pos``).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device,
        joint_pos_initial: torch.Tensor,
        action_scale: float = DEFAULT_ACTION_SCALE,
        obs_dim: int = 45,
        act_dim: int = 12,
        hidden_dims=(512, 256, 128),
        search_relative_to: Path | None = None,
    ):
        self.device = device
        self.action_scale = action_scale
        self.joint_pos_initial = joint_pos_initial
        self.act_dim = act_dim
        
        # PD control gains
        self.pd_ke = PD_GAINS_KE
        self.pd_kd = PD_GAINS_KD

        checkpoint = self._load_checkpoint(checkpoint_path, device, search_relative_to)
        # Infer the observation dim the checkpoint was trained with, instead of
        # trusting the obs_dim default: command-less policies use 45, policies
        # that include the 3-dim velocity command use 48.
        obs_dim = self._infer_obs_dim(checkpoint, default=obs_dim)
        self.net = self._build_mlp(obs_dim, act_dim, hidden_dims).to(device)
        self._load_weights(self.net, checkpoint)
        self.net.eval()

        # The deployment pipeline (obs layout/scales, joint ordering, per-joint
        # action scale) depends on which training convention the checkpoint came
        # from. The two are distinguished by the actor obs dim:
        #   48 -> legacy: includes base_lin_vel, type-grouped "lab" joint order,
        #         unit obs scales, single action scale.
        #   45 -> robot_lab/mud flat policy: drops base_lin_vel, leg-grouped
        #         [FR,FL,RR,RL] order, scaled ang_vel/joint_vel, per-joint scale.
        if obs_dim >= 48:
            self.include_base_lin_vel = True
            self.ang_vel_scale = 1.0
            self.joint_vel_scale = 1.0
            self.obs_joint_indices = torch.tensor(mujoco_to_lab, device=device)
            self.act_joint_indices = torch.tensor(lab_to_mujoco, device=device)
            self.action_scale_vec = action_scale  # scalar broadcast
        else:
            self.include_base_lin_vel = False
            self.ang_vel_scale = 0.25
            self.joint_vel_scale = 0.05
            # Newton joints are leg-grouped [FL,FR,RL,RR]; the policy expects
            # [FR,FL,RR,RL]. Swapping L<->R within each pair is its own inverse,
            # so the same index list serves both obs and action remaps.
            self.obs_joint_indices = torch.tensor(ROBOT_LAB_JOINT_SWAP, device=device)
            self.act_joint_indices = torch.tensor(ROBOT_LAB_JOINT_SWAP, device=device)
            # Per-joint action scale in Newton order: (hip, thigh, calf) x 4 legs.
            self.action_scale_vec = torch.tensor(
                [0.125, 0.25, 0.25] * 4, device=device, dtype=torch.float32
            ).unsqueeze(0)

        self.gravity_vec = torch.tensor(
            [0.0, 0.0, -1.0], device=device, dtype=torch.float32
        ).unsqueeze(0)

        self.last_action = torch.zeros(1, act_dim, device=device, dtype=torch.float32)
        self._padding = torch.zeros(6, device=device, dtype=torch.float32)

    @staticmethod
    def _load_checkpoint(path, device, search_relative_to):
        try:
            return torch.load(path, map_location=device)
        except FileNotFoundError:
            if search_relative_to is None:
                raise
            return torch.load(str(search_relative_to / path), map_location=device)

    @staticmethod
    def _infer_obs_dim(checkpoint, default):
        raw = checkpoint.get("actor_state_dict",
                             checkpoint.get("model_state_dict", checkpoint))
        for key, value in raw.items():
            if (key.endswith("0.weight")
                    and (key.startswith("mlp.") or key.startswith("actor."))
                    and hasattr(value, "shape")):
                return value.shape[1]
        return default

    @staticmethod
    def _build_mlp(in_dim, out_dim, hidden_dims):
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ELU(alpha=1.0)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _load_weights(net, checkpoint):
        raw = checkpoint.get("actor_state_dict",
                             checkpoint.get("model_state_dict", checkpoint))
        cleaned = {}
        for key, value in raw.items():
            if key.startswith("mlp."):
                cleaned[key[4:]] = value      # mlp.0.weight -> 0.weight
            elif key.startswith("actor."):
                cleaned[key[6:]] = value      # actor.0.weight -> 0.weight
        net.load_state_dict(cleaned)

    @torch.no_grad()
    def compute_joint_targets(self, state, command) -> wp.array:
        obs = compute_obs(
            self.last_action, state, self.joint_pos_initial, self.device,
            self.obs_joint_indices, self.gravity_vec, command,
            include_base_lin_vel=self.include_base_lin_vel,
            ang_vel_scale=self.ang_vel_scale, joint_vel_scale=self.joint_vel_scale,
        )
        self.last_action = self.net(obs)
        rearranged = torch.gather(self.last_action, 1,
                                  self.act_joint_indices.unsqueeze(0))
        target = self.joint_pos_initial + self.action_scale_vec * rearranged
        padded = torch.cat([self._padding, target.squeeze(0)])
        return wp.from_torch(padded, dtype=wp.float32, requires_grad=False)
