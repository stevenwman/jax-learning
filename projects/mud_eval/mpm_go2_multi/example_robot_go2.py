###########################################################################
# Example Robot Go2 Walk 
#
# Shows how to simulate Go2 using SolverMuJoCo and control it with a
# policy trained in PhysX.
#
# Command: python -m newton.examples robot_anymal_c_walk
#
###########################################################################

import torch
import warp as wp
from warp.torch import device_to_torch

wp.config.enable_backward = False

import newton
import newton.examples
import newton.utils
from newton import State
from newton.geometry import generate_terrain_grid

lab_to_mujoco = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
mujoco_to_lab = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]

@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate a vector by the inverse of a quaternion along the last dimension of q and v.    Args:
    q: The quaternion in (x, y, z, w). Shape is (..., 4).
    v: The vector in (x, y, z). Shape is (..., 3).    Returns:
    The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_w = q[..., 3]  # w component is at index 3 for XYZW format
    q_vec = q[..., :3]  # xyz components are at indices 0, 1, 2
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    # for two-dimensional tensors, bmm is faster than einsum
    if q_vec.dim() == 2:
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
    return a - b + c


def compute_obs(actions, state: State, joint_pos_initial, device, indices, gravity_vec, command,
                include_base_lin_vel=True, ang_vel_scale=1.0, joint_vel_scale=1.0):
    root_quat_w = torch.tensor(state.joint_q[3:7], device=device, dtype=torch.float32).unsqueeze(0)
    root_lin_vel_w = torch.tensor(state.joint_qd[:3], device=device, dtype=torch.float32).unsqueeze(0)
    root_ang_vel_w = torch.tensor(state.joint_qd[3:6], device=device, dtype=torch.float32).unsqueeze(0)

    joint_pos_current = torch.tensor(state.joint_q[7:], device=device, dtype=torch.float32).unsqueeze(0)
    joint_vel_current = torch.tensor(state.joint_qd[6:], device=device, dtype=torch.float32).unsqueeze(0)

    vel_b = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
    a_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
    grav = quat_rotate_inverse(root_quat_w, gravity_vec)
    
    joint_pos_rel = joint_pos_current - joint_pos_initial
    joint_vel_rel = joint_vel_current

    rearranged_joint_pos_rel = torch.index_select(joint_pos_rel, 1, indices)
    rearranged_joint_vel_rel = torch.index_select(joint_vel_rel, 1, indices)

    # robot_lab/mud Go2 policies (45-dim obs) drop base_lin_vel from the actor
    # obs (not observable on hardware) and scale ang_vel/joint_vel; the command
    # stays. Legacy 48-dim policies use unit scales and include base_lin_vel.
    obs_terms = []
    if include_base_lin_vel:
        obs_terms.append(vel_b)
    obs_terms += [a_vel_b * ang_vel_scale, grav, command,
                  rearranged_joint_pos_rel, rearranged_joint_vel_rel * joint_vel_scale, actions]
    obs = torch.cat(obs_terms, dim=1)

    return obs
