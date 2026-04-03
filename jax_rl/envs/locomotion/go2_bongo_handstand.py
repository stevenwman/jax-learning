"""Go2 Bongo Board Handstand environment (Warp backend).

Balance task: Go2 inverted on a bongo board, front legs down, rear legs up.
Uses unitree_mujoco go2.xml + bongo_board.xml via MuJoCo Warp.

Returns dict obs: {"state": 42-46d policy obs, "privileged_state": ~96d critic obs}.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
import numpy as np

from mujoco_playground._src import mjx_env
from jax_rl.envs.locomotion import go2_warp_base
from jax_rl.envs.locomotion import go2_constants as consts
from jax_rl.envs.obs_spec import ObsTerm, IncludeGroup, compute_obs
from jax_rl.envs.reward_spec import RewardTerm, compute_rewards


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=250,       # 5s — faster reward signal, falls are fast
        Kp=20.0,
        Kd=0.5,
        action_repeat=1,
        action_scale=1.0,         # full authority — let policy use full joint range
        soft_joint_pos_limit_factor=0.95,
        observe_board_state=True,
        target_handstand_height=0.55,
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Cost-based: survival is ceiling, everything else pulls down.
                # All costs are normalized to [0,1] before weighting.
                survival=10.0,                  # only positive term
                orientation_cost=-8.0,          # (gravity_error²)/4, range [0,1]
                board_tilt_cost=-6.0,           # (tilt²)/0.5, range [0,1]
                com_offset_cost=-5.0,           # (com_xy_error²)/0.1, range [0,1]
                height_cost=-3.0,               # (height_error²)/0.1, range [0,1]
                roller_cost=-2.0,               # (roller_pos²)/0.05, range [0,1]
                torque_cost=-0.5,               # normalized torques
                action_rate_cost=-0.5,          # (action_diff²)/1
                termination=-1.0,
            ),
        ),
        # Antagonistic pushes config.
        push_interval=99999,      # disabled — learn balance first, add pushes later
        push_robot_vel=0.5,       # ±m/s velocity kick on robot base
        push_board_vel=0.3,       # ±m/s velocity kick on board
        impl="warp",
        contact_mode="training",
        naconmax=4 * 8192,
        naccdmax=4000,   # same as joystick — board adds few extra contacts
        njmax=100,
    )


class BongoHandstand(go2_warp_base.Go2WarpEnv):
    """Go2 handstand balance on a bongo board (Warp backend)."""

    def __init__(
        self,
        task: str = "bongo_handstand",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=consts.BONGO_SCENE_XML.as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        # Robot pose from handstand keyframe.
        self._init_q = jp.array(self._mj_model.keyframe("handstand").qpos)
        self._default_pose = jp.array(
            self._mj_model.keyframe("handstand").qpos[7:19]
        )

        # Soft joint limits (joints 1-12 are robot, skip freejoint 0).
        # With two free bodies, jnt_range has more entries. Use robot joints only.
        robot_jnt_ids = []
        for i in range(self._mj_model.njnt):
            jnt_name = self._mj_model.joint(i).name
            if "hip_joint" in jnt_name or "thigh_joint" in jnt_name or "calf_joint" in jnt_name:
                robot_jnt_ids.append(i)
        robot_jnt_range = self._mj_model.jnt_range[robot_jnt_ids]
        self._lowers, self._uppers = robot_jnt_range.T
        c = self._config.soft_joint_pos_limit_factor
        self._soft_lowers = self._lowers * c
        self._soft_uppers = self._uppers * c

        # Torso body (robot).
        self._torso_body_id = self._mj_model.body(consts.WARP_ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

        # Board/roller indices — resolved by name, never hardcoded.
        # Convert to Python int so JAX sees static slice indices.
        self._board_body_id = self._mj_model.body("board").id
        self._roller_slide_qposadr = self._mj_model.joint("roller_slide").qposadr.item()
        self._roller_slide_dofadr = self._mj_model.joint("roller_slide").dofadr.item()
        self._board_jnt_qposadr = self._mj_model.joint("board_joint").qposadr.item()
        self._board_jnt_dofadr = self._mj_model.joint("board_joint").dofadr.item()

        # Front foot contact with board sensors.
        self._fl_board_sensor = self._mj_model.sensor("FL_board_found").id
        self._fr_board_sensor = self._mj_model.sensor("FR_board_found").id

        # Board-floor contact sensor (board edge touching ground = fail).
        self._board_floor_sensor_adr = self._mj_model.sensor_adr[
            self._mj_model.sensor("board_floor_found").id
        ]

        # Torso contact sensors (head/body touching floor or board = fail).
        # Body contact sensors: torso + front leg arms on floor/board = fail.
        # Only foot spheres (FL, FR) may touch the board.
        body_contact_names = [
            # Torso on floor/board
            "torso_box_floor", "torso_cyl_floor", "torso_nose_floor",
            "torso_box_board", "torso_cyl_board", "torso_nose_board",
            # Front leg arms/calves on board (no forearm balancing)
            "FL_hip_board", "FL_thigh_board",
            "FL_calf_upper_board", "FL_calf_lower_board",
            "FR_hip_board", "FR_thigh_board",
            "FR_calf_upper_board", "FR_calf_lower_board",
        ]
        self._body_contact_sensor_adr = [
            self._mj_model.sensor_adr[self._mj_model.sensor(n).id]
            for n in body_contact_names
        ]

        # Foot-floor contact sensors (ANY foot on floor = termination).
        self._feet_floor_sensors = [
            self._mj_model.sensor(f"{g}_floor_found").id
            for g in consts.FEET_GEOMS
        ]
        self._feet_floor_sensor_adr = [
            self._mj_model.sensor_adr[sid] for sid in self._feet_floor_sensors
        ]

        # Board contact mode override.
        # IMPORTANT: must modify _mj_model THEN re-create _mjx_model,
        # because base class already called mjx.put_model() in __init__.
        if getattr(self._config, 'contact_mode', 'training') == 'training':
            board_gid = self._mj_model.geom("board_top").id
            self._mj_model.geom_solimp[board_gid, :3] = np.array(
                [0.9, 0.95, 0.023]
            )
            self._mj_model.geom_condim[board_gid] = 3
            self._mj_model.geom_friction[board_gid] = np.array(
                [0.8, 0.005, 0.001]
            )

        # Re-create Warp model after all _mj_model modifications.
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        noise = self._config.noise_config.scales
        state_terms = [
            ObsTerm("gyro", lambda data, **kw: self.get_gyro(data),
                    noise_scale=noise.gyro),
            ObsTerm("gravity", lambda data, **kw: self.get_gravity(data),
                    noise_scale=noise.gravity),
            ObsTerm("joint_pos_offset", lambda data, **kw: data.qpos[7:19] - self._default_pose,
                    noise_scale=noise.joint_pos),
            ObsTerm("joint_vel", lambda data, **kw: data.qvel[6:18],
                    noise_scale=noise.joint_vel),
            ObsTerm("last_act", lambda info, **kw: info["last_act"]),
        ]
        if self._config.observe_board_state:
            state_terms.extend([
                ObsTerm("board_tilt", lambda data, **kw: self._get_board_tilt(data)),
                ObsTerm("roller_pos", lambda data, **kw: data.qpos[self._roller_slide_qposadr].reshape(1)),
                ObsTerm("roller_vel", lambda data, **kw: data.qvel[self._roller_slide_dofadr].reshape(1)),
            ])

        privileged_terms = [
            IncludeGroup("state"),
            ObsTerm("gyro_clean", lambda data, **kw: self.get_gyro(data)),
            ObsTerm("gravity_clean", lambda data, **kw: self.get_gravity(data)),
            ObsTerm("joint_pos_clean", lambda data, **kw: data.qpos[7:19] - self._default_pose),
            ObsTerm("joint_vel_clean", lambda data, **kw: data.qvel[6:18]),
            ObsTerm("actuator_force", lambda data, **kw: data.actuator_force),
            ObsTerm("foot_board_contact", lambda data, **kw: jp.array([
                data.sensordata[self._mj_model.sensor_adr[self._fl_board_sensor]] > 0,
                data.sensordata[self._mj_model.sensor_adr[self._fr_board_sensor]] > 0,
            ]).astype(jp.float32)),
            ObsTerm("board_tilt_clean", lambda data, **kw: self._get_board_tilt(data)),
            ObsTerm("roller_pos_clean", lambda data, **kw: data.qpos[self._roller_slide_qposadr].reshape(1)),
            ObsTerm("roller_vel_clean", lambda data, **kw: data.qvel[self._roller_slide_dofadr].reshape(1)),
            ObsTerm("board_angvel", lambda data, **kw: self._get_board_angvel(data)[:2]),
            ObsTerm("com_rel_board", lambda data, **kw:
                data.subtree_com[self._torso_body_id][:2] - data.xpos[self._board_body_id][:2]),
        ]

        self._obs_groups = {"state": state_terms, "privileged_state": privileged_terms}

        # Cost-based rewards: normalized quadratic costs in [0,1].
        # Survival is the ceiling; costs pull down. No saturation.
        target_gravity = jp.array([1.0, 0.0, 0.0])
        max_torque_norm = 45.43 * 12  # rough max: all joints at calf limit

        self._reward_spec = [
            RewardTerm("survival", lambda **kw:
                jp.float32(1.0)),
            RewardTerm("orientation_cost", lambda data, **kw:
                jp.clip(jp.sum((self.get_gravity(data) - target_gravity) ** 2) / 4.0, 0.0, 1.0)),
            RewardTerm("board_tilt_cost", lambda data, **kw:
                jp.clip(jp.sum(self._get_board_tilt(data) ** 2) / 0.5, 0.0, 1.0)),
            RewardTerm("com_offset_cost", lambda data, **kw:
                jp.clip(jp.sum((data.subtree_com[self._torso_body_id][:2]
                                - data.xpos[self._board_body_id][:2]) ** 2) / 0.1, 0.0, 1.0)),
            RewardTerm("height_cost", lambda data, **kw:
                jp.clip((data.subtree_com[self._torso_body_id][2]
                         - self._config.target_handstand_height) ** 2 / 0.1, 0.0, 1.0)),
            RewardTerm("roller_cost", lambda data, **kw:
                jp.clip(data.qpos[self._roller_slide_qposadr] ** 2 / 0.05, 0.0, 1.0)),
            RewardTerm("torque_cost", lambda data, **kw:
                jp.clip((jp.sqrt(jp.sum(jp.square(data.actuator_force)))
                         + jp.sum(jp.abs(data.actuator_force))) / max_torque_norm, 0.0, 1.0)),
            RewardTerm("action_rate_cost", lambda action, info, **kw:
                jp.clip(jp.sum(jp.square(action - info["last_act"])) / 12.0, 0.0, 1.0)),
            RewardTerm("termination", lambda done, **kw:
                done),
        ]

    # ── Core env methods ───────────────────────────────────────────

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        # Perturbations on robot joint angles (±0.1 rad).
        rng, key = jax.random.split(rng)
        joint_noise = jax.random.uniform(key, (12,), minval=-0.1, maxval=0.1)
        qpos = qpos.at[7:19].set(qpos[7:19] + joint_noise)

        # Perturbation on robot base xy position (±5cm) — offset from board center.
        rng, key = jax.random.split(rng)
        xy_noise = jax.random.uniform(key, (2,), minval=-0.05, maxval=0.05)
        qpos = qpos.at[0:2].set(qpos[0:2] + xy_noise)

        # Yaw perturbation on robot (±10deg ≈ ±0.175 rad).
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-0.175, maxval=0.175)
        yaw_quat = mjx_math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        robot_quat = qpos[3:7]
        qpos = qpos.at[3:7].set(mjx_math.quat_mul(robot_quat, yaw_quat))

        # Board tilt perturbation (±5deg ≈ ±0.087 rad).
        # Perturb board quat with small rotation around X and Y.
        rng, key = jax.random.split(rng)
        board_qposadr = self._board_jnt_qposadr
        tilt_noise = jax.random.uniform(key, (2,), minval=-0.087, maxval=0.087)
        # Apply as small-angle quaternion perturbation.
        dq = jp.array([1.0, tilt_noise[0], tilt_noise[1], 0.0])
        dq = dq / jp.linalg.norm(dq)
        board_quat = qpos[board_qposadr + 3 : board_qposadr + 7]
        new_quat = mjx_math.quat_mul(board_quat, dq)
        qpos = qpos.at[board_qposadr + 3 : board_qposadr + 7].set(new_quat)

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=jp.zeros(self.mjx_model.nu),
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            naccdmax=self._config.naccdmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        info = {
            "rng": rng,
            "last_act": jp.zeros(self.mjx_model.nu),
            "step_count": jp.int32(0),
            "reward_components": {
                k: jp.zeros(()) for k in self._config.reward_config.scales.keys()
            },
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        motor_targets = self._default_pose + action * self._config.action_scale

        # Antagonistic pushes: velocity kicks on robot and board.
        step_count = state.info["step_count"]
        push_interval = self._config.push_interval
        rng, robot_key, board_key = jax.random.split(state.info["rng"], 3)
        do_push = (step_count > 0) & (step_count % push_interval == 0)

        data = state.data

        # Push robot base (xy velocity kick).
        robot_push = jax.random.uniform(
            robot_key, (2,),
            minval=-self._config.push_robot_vel,
            maxval=self._config.push_robot_vel,
        )
        new_qvel = data.qvel.at[0:2].set(
            jp.where(do_push, data.qvel[0:2] + robot_push, data.qvel[0:2])
        )

        # Push board (xy velocity kick via board freejoint qvel).
        board_dofadr = self._board_jnt_dofadr
        board_push = jax.random.uniform(
            board_key, (2,),
            minval=-self._config.push_board_vel,
            maxval=self._config.push_board_vel,
        )
        new_qvel = new_qvel.at[board_dofadr:board_dofadr + 2].set(
            jp.where(do_push,
                     new_qvel[board_dofadr:board_dofadr + 2] + board_push,
                     new_qvel[board_dofadr:board_dofadr + 2])
        )
        data = data.replace(qvel=new_qvel)

        # External PD at physics rate.
        kp = self._kp
        kd = self._kd
        model = self.mjx_model
        a2j = self._act_to_joint

        def substep(data, _):
            current_q = data.qpos[7:19]
            current_dq = data.qvel[6:18]
            tau_joint = kp * (motor_targets - current_q) + kd * (0.0 - current_dq)
            tau_act = tau_joint[a2j]
            data = data.replace(ctrl=tau_act)
            return mjx.step(model, data), None

        data = jax.lax.scan(substep, data, (), self.n_substeps)[0]

        obs = self._get_obs(data, state.info)
        done = self._get_termination(data)

        rewards = self._get_reward(data, action, state.info, done)
        rewards = {
            k: v * self._config.reward_config.scales[k]
            for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, -10000.0, 10000.0)

        state.info["reward_components"] = rewards
        state.info["last_act"] = action
        state.info["step_count"] = state.info["step_count"] + 1
        state.info["rng"] = rng

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        done = done.astype(reward.dtype)
        return state.replace(data=data, obs=obs, reward=reward, done=done)

    # ── Observation ────────────────────────────────────────────────

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> Dict[str, jax.Array]:
        obs, info["rng"] = compute_obs(
            self._obs_groups,
            noise_level=self._config.noise_config.level,
            rng=info["rng"],
            data=data, info=info,
        )
        return obs

    # ── Board state helpers ────────────────────────────────────────

    def _get_board_tilt(self, data: mjx.Data) -> jax.Array:
        """Board tilt as x,y components of board's z-axis in world frame.

        Returns [0, 0] when level. Magnitude increases with tilt.
        Uses sin(angle) which is linear near zero.
        """
        board_xmat = data.xmat[self._board_body_id].reshape(3, 3)
        return board_xmat[:2, 2]

    def _get_board_angvel(self, data: mjx.Data) -> jax.Array:
        """Board angular velocity in world frame.

        Extracted from the board body's qvel (freejoint angular vel).
        """
        # Freejoint has 6 DOFs: 3 linear + 3 angular.
        return data.qvel[self._board_jnt_dofadr + 3 : self._board_jnt_dofadr + 6]

    # ── Termination ────────────────────────────────────────────────

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        gravity = self.get_gravity(data)
        # At ~97deg pitch, gravity_body ≈ [0.99, 0, 0.11].
        # gravity_body[0] > 0 means body X-axis points down (handstand).
        # Terminate if gravity_body[0] < 0.3 (tipped too far from handstand).
        not_handstand = gravity[0] < 0.3

        base_z = data.subtree_com[self._torso_body_id][2]
        too_low = base_z < 0.15

        # Contact-based termination (all via sensors — no position heuristics).
        # Any foot on floor.
        feet_on_floor = jp.any(jp.array([
            data.sensordata[adr] > 0
            for adr in self._feet_floor_sensor_adr
        ]))
        # Board edge on floor.
        board_on_floor = data.sensordata[self._board_floor_sensor_adr] > 0
        # Body contact: torso or front leg arms on floor/board.
        body_contact = jp.any(jp.array([
            data.sensordata[adr] > 0
            for adr in self._body_contact_sensor_adr
        ]))

        return (not_handstand | too_low
                | feet_on_floor | board_on_floor | body_contact)

    # ── Rewards ────────────────────────────────────────────────────

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        return compute_rewards(
            self._reward_spec,
            data=data, action=action, info=info, done=done,
        )
