"""Run a jax-learning FastSAC Go2 policy inside the Newton sim — drop-in for the
example's torch `Go2Policy` (same `compute_joint_targets(state, command)`).

The POLICY is the real jax_rl network — we just `np.load` the checkpoint's
`actor_params.npy` artifact and build `jax_rl.algos.fast_sac.FastSAC`, then jit
its deterministic `select_action`. (We import from `jax_rl.algos` directly to
avoid `jax_rl.training.__init__`'s side-effect import of mujoco_playground, which
would bump mujoco off the 3.7.0 the Newton stack is pinned to.) jax runs the tiny
actor on CPU — fine at 50 Hz, and it keeps the GPU for the MPM sim.

The only real work here is the OBS ADAPTER: build jax-learning's 48-d WarpJoystick
`state` obs from Newton's `State`:
    [ gyro(3), accelerometer(3), gravity(3),
      joint_pos_offset(12), joint_vel(12), last_act(12), command(3) ]
Newton joint order (FL,FR,RL,RR, hip→thigh→calf) == the policy's
`policy_FL_FR_RL_RR`, so no joint remap. Action → joint targets:
    motor_targets = default_pose + action * action_scale   (joint-PD via the
    model's joint_target_ke/kd set from control.Kp/Kd).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
import warp as wp

from jax_rl.algos.fast_sac import FastSAC
from jax_rl.configs.fast_sac_config import FastSACConfig


def _build_fast_sac_policy(ckpt_dir: Path):
    """np.load the actor artifact + build the real FastSAC actor; return a jit'd
    deterministic select_action and (params, norm_state, meta)."""
    saved = np.load(ckpt_dir / "actor_params.npy", allow_pickle=True).item()
    params = saved["actor_params"]
    meta = json.load(open(ckpt_dir / "meta.json"))
    sc = meta["fast_sac_config"]
    dummy = optax.adam(1e-3)
    cfg = FastSACConfig(
        hidden_dim=tuple(sc["hidden_dim"]),
        activation=sc["activation"],
        q_layer_norm=sc.get("q_layer_norm", True),
        num_atoms=sc.get("num_atoms", 101),
        v_min=sc.get("v_min", -20.0),
        v_max=sc.get("v_max", 20.0),
        q_aggregation=sc.get("q_aggregation", "avg"),
        critic_hidden_dim=tuple(sc["critic_hidden_dim"]) if sc.get("critic_hidden_dim") else None,
    )
    sac = FastSAC(cfg, int(meta["obs_dim"]), int(meta["action_dim"]), dummy, dummy, gamma=0.99)
    key = jax.random.PRNGKey(0)

    @jax.jit
    def select(p, obs):
        return sac.select_action(p, obs, key, deterministic=True)

    norm = (np.asarray(saved["norm_mean"], np.float32),
            np.asarray(saved["norm_mean_of_squares"], np.float32),
            int(saved["norm_count"]))
    return select, params, norm, meta


def patched_config(ckpt_dir, base_config_path, out_path, mjcf_model: str | None = None,
                   spawn_xyz=(0.0, 1.5, 0.40), yaw_pi_mult: float = 0.0) -> str:
    """Write a config.yaml override so the Newton robot spawns at the POLICY's
    default pose and uses its training PD gains (Kp/Kd) — otherwise the policy
    sees a non-zero joint_pos_offset at spawn and a stiffer-than-trained PD.

    mjcf_model: if given, load this MJCF (abs path) instead of the example URDF
    (pathlib: here/<abs> == <abs>). The example's posing loop uses the buggy
    `joint_key.index(key)+6` idiom that overflows on go2.xml's 0-dof *_foot_joint,
    so we EMPTY initial_joint_q here and let mud_model's add_mjcf dispatch set the
    home pose directly (see mud_model.set_home_pose)."""
    import yaml
    cfg = yaml.safe_load(open(base_config_path))
    meta = json.load(open(Path(ckpt_dir) / "meta.json"))
    ctrl = meta["control"]
    names = ctrl["policy_joint_names"]
    pose = ctrl["default_pose_policy"]
    if mjcf_model is not None:
        cfg["robot"]["urdf_relative_path"] = str(mjcf_model)   # abs MJCF path
        cfg["policy"]["initial_joint_q"] = {}                   # skip the buggy loop
    else:
        cfg["policy"]["initial_joint_q"] = {n: float(v) for n, v in zip(names, pose)}
    cfg["policy"]["pd_gains_ke"] = float(ctrl["Kp"])
    cfg["policy"]["pd_gains_kd"] = float(ctrl["Kd"])
    cfg["policy"]["action_scale"] = float(ctrl["action_scale"])
    # Spawn UPRIGHT about +Z (example tilts via a non-z yaw axis). yaw_pi_mult
    # rotates about +Z: 0.5 => +90deg so body +X (the policy's "forward") points
    # to world +Y — the mud's long axis (thick y0-1 -> medium y1-2 -> thin y2-3).
    cfg["robot"]["initial_yaw_axis"] = [0.0, 0.0, 1.0]
    cfg["robot"]["initial_yaw_angle_pi_mult"] = float(yaw_pi_mult)
    # Spawn position. Traversal setup: y<0 on the flat ground plane
    # (builder.add_ground_plane), facing +Y, then walk forward into the mud.
    cfg["robot"]["initial_position"] = [float(v) for v in spawn_xyz]
    yaml.safe_dump(cfg, open(out_path, "w"))
    return str(out_path)


def _normalize(obs, norm, eps=1e-8):
    mean, msq, count = norm
    var = np.maximum(msq - mean * mean, 0.0)
    return (obs - mean) / np.sqrt(var + eps)


# ── obs adapter: Newton State -> 48-d jax obs ────────────────────────────────
def _quat_rotate_inverse(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """R(q)^T @ v (world->body). q=(x,y,z,w). Matches the vendored example's
    quat_rotate_inverse so the frame convention agrees with how Go2 is loaded."""
    qx, qy, qz, qw = q_xyzw
    qvec = np.array([qx, qy, qz], np.float32)
    a = v * (2.0 * qw * qw - 1.0)
    b = np.cross(qvec, v) * (2.0 * qw)
    c = qvec * (2.0 * float(qvec @ v))
    return a - b + c


class MudJaxPolicy:
    """Drop-in for the example's `Go2Policy`: same constructor kwargs + same
    `compute_joint_targets(state, command) -> wp.array(6 + act_dim)`."""

    def __init__(self, checkpoint_path, device, joint_pos_initial,
                 action_scale=0.5, search_relative_to: Path | None = None, **kw):
        ckpt = Path(checkpoint_path)
        if not ckpt.exists() and search_relative_to is not None:
            ckpt = Path(search_relative_to) / checkpoint_path
        self._select, self._params, self._norm, meta = _build_fast_sac_policy(ckpt)
        ctrl = meta["control"]
        self.action_scale = float(ctrl.get("action_scale", action_scale))
        self.default_pose = np.asarray(ctrl["default_pose_policy"], np.float32)  # FL,FR,RL,RR
        self.policy_dt = float(ctrl.get("policy_dt", 0.02))
        self.act_dim = int(meta["action_dim"])

        self.hold = False                      # if True: hold default pose (action=0) — for isolating spawn/contact from the policy
        self.last_act = np.zeros(self.act_dim, np.float32)
        self._prev_linvel_w = None
        self._g_world = np.array([0.0, 0.0, -9.81], np.float32)
        self._grav_dir = np.array([0.0, 0.0, -1.0], np.float32)
        self.device = device                  # torch device (unused for warp arrays)
        self._wp_device = wp.get_device()      # warp device for the control buffer (cuda:0)
        self.last_obs = None

    def _build_obs(self, state, command) -> np.ndarray:
        jq = np.asarray(state.joint_q.numpy(), np.float32)
        jqd = np.asarray(state.joint_qd.numpy(), np.float32)
        quat = jq[3:7]                          # (x,y,z,w)
        quat = quat / (np.linalg.norm(quat) + 1e-9)   # Newton free-joint quat isn't unit-norm
        linvel_w = jqd[0:3]
        angvel_w = jqd[3:6]
        joint_pos = jq[7:7 + self.act_dim]
        joint_vel = jqd[6:6 + self.act_dim]

        gyro = _quat_rotate_inverse(quat, angvel_w)
        gravity = _quat_rotate_inverse(quat, self._grav_dir)
        if self._prev_linvel_w is None:
            a_world = np.zeros(3, np.float32)
        else:
            a_world = (linvel_w - self._prev_linvel_w) / self.policy_dt
        self._prev_linvel_w = linvel_w.copy()
        accel = _quat_rotate_inverse(quat, a_world - self._g_world)   # specific force, body frame

        jpos_off = joint_pos - self.default_pose
        cmd = np.asarray(command, np.float32).reshape(-1)[:3]
        return np.concatenate([gyro, accel, gravity, jpos_off, joint_vel,
                               self.last_act, cmd]).astype(np.float32)

    def compute_joint_targets(self, state, command) -> wp.array:
        try:
            cmd = command.detach().cpu().numpy().reshape(-1)
        except AttributeError:
            cmd = np.asarray(command, np.float32).reshape(-1)
        obs = self._build_obs(state, cmd)
        self.last_obs = obs
        if self.hold:
            action = np.zeros(self.act_dim, np.float32)   # hold default pose
        else:
            action = np.asarray(self._select(self._params, jnp.asarray(_normalize(obs, self._norm))))
        self.last_act = action.astype(np.float32)
        targets = self.default_pose + action * self.action_scale   # joint order FL,FR,RL,RR
        padded = np.concatenate([np.zeros(6, np.float32), targets]).astype(np.float32)
        return wp.from_numpy(padded, dtype=wp.float32, device=self._wp_device)
