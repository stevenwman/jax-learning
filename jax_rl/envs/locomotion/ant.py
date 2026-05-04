"""Ant environment — port of gymnasium AntEnv-v5 to MJX/Warp.

Behavioral port of `gymnasium.envs.mujoco.ant_v5.AntEnv`. Same observation,
reward, termination, and init distribution as Gym Ant-v5; runs through MJX
for GPU-parallel rollouts. Registered under env_name `AntMJX` (NOT `Ant` —
that string is taken by the CPU Gym lane in `gym_backend.py`).

Spec source: `gymnasium/envs/mujoco/ant_v5.py`. All scalar weights, the
asymmetric reset noise, and the (105d) observation layout are mirrored from
that file.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env


_XML_PATH = Path(__file__).parent / "xmls" / "ant.xml"


def default_config() -> config_dict.ConfigDict:
    """Default Ant-v5 config. All scalars locked from `ant_v5.py`."""
    return config_dict.create(
        ctrl_dt=0.05,  # frame_skip * sim_dt = 5 * 0.01
        sim_dt=0.01,
        episode_length=1000,
        action_repeat=1,  # frame_skip handled via n_substeps inside step()
        vision=False,
        impl="warp",
        # Warp contact-buffer sizing (mirrors mujoco_playground cheetah.py:42-43):
        naconmax=100_000,
        njmax=100,
        # Gym Ant-v5 defaults (locked from ant_v5.py:229-292):
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.5,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        healthy_z_min=0.2,
        healthy_z_max=1.0,
        contact_force_clip=1.0,  # symmetric clip [-1, 1]
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        include_cfrc_ext_in_observation=True,
        terminate_when_unhealthy=True,
        main_body_id=1,  # torso
    )


class Ant(mjx_env.MjxEnv):
    """Gym Ant-v5 port (registered to env_name 'AntMJX')."""

    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list]]] = None,
    ):
        if config is None:
            config = default_config()
        super().__init__(config, config_overrides)
        if self._config.vision:
            raise NotImplementedError(
                f"Vision not implemented for {self.__class__.__name__}."
            )
        self._xml_path = _XML_PATH.as_posix()
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    def _post_init(self) -> None:
        # init_qpos comes from mj_model.qpos0 (= torso body's pos="0 0 0.75"),
        # NOT from any <custom name="init_qpos"> tag (Brax-only metadata that
        # stock MuJoCo ignores; keeping it in the XML would mislead readers
        # since it claims z=0.55 while the real init is z=0.75). Matches Gym
        # ant_v5: MujocoEnv.__init__ does
        # `self.init_qpos = self.data.qpos.ravel().copy()` after mj_resetData.
        self._init_qpos = jp.asarray(self._mj_model.qpos0)
        # init_qvel: gym ant.xml has no <numeric init_qvel> and Gym defaults
        # to zeros via MujocoEnv.set_state convention.
        self._init_qvel = jp.zeros(self._mj_model.nv)

    # ------------------------------------------------------------------ reset
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng_q, rng_v = jax.random.split(rng, 3)

        # Asymmetric noise (Gym v5): uniform-bounded on qpos, gaussian on qvel,
        # both scaled by reset_noise_scale=0.1.
        noise_q = jax.random.uniform(
            rng_q,
            (self.mjx_model.nq,),
            minval=-self._config.reset_noise_scale,
            maxval=self._config.reset_noise_scale,
        )
        noise_v = self._config.reset_noise_scale * jax.random.normal(
            rng_v, (self.mjx_model.nv,)
        )
        qpos = self._init_qpos + noise_q
        qvel = self._init_qvel + noise_v

        # Forward naconmax/njmax to size Warp contact buffers (cheetah.py:91-92).
        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        # Seed all info keys that step() will populate so the wrapper's
        # jax.lax.scan over action_repeat sees a stable pytree structure
        # between reset (carry input) and step (carry output).
        info = {
            "rng": rng,
            "x_position": data.qpos[0],
            "y_position": data.qpos[1],
            "distance_from_origin": jp.linalg.norm(data.qpos[0:2]),
            "x_velocity": jp.zeros(()),
            "y_velocity": jp.zeros(()),
        }
        metrics = {
            "reward_forward": jp.zeros(()),
            "reward_ctrl": jp.zeros(()),
            "reward_contact": jp.zeros(()),
            "reward_survive": jp.zeros(()),
        }
        reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
        obs = self._get_obs(data)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    # ------------------------------------------------------------------- step
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # Capture pre-step CoM xy from torso body world-frame position.
        xy_before = state.data.xpos[self._config.main_body_id, :2]

        # Advance frame_skip=5 substeps:
        data = mjx_env.step(
            self.mjx_model, state.data, action, self.n_substeps
        )

        xy_after = data.xpos[self._config.main_body_id, :2]
        dt = self.dt  # ctrl_dt = 0.05
        x_velocity = (xy_after[0] - xy_before[0]) / dt
        y_velocity = (xy_after[1] - xy_before[1]) / dt

        # Single pass through health check; reuse for survive_r and done.
        unhealthy = self._is_unhealthy(data)
        reward, reward_info = self._get_reward(  # pylint: disable=redefined-outer-name
            data, action, x_velocity, unhealthy
        )
        obs = self._get_obs(data)

        if self._config.terminate_when_unhealthy:
            done = unhealthy.astype(jp.float32)
        else:
            done = jp.zeros((), dtype=jp.float32)

        # Match Gym Ant-v5 info dict (ant_v5.py:360-367).
        info = {
            **state.info,
            "x_position": data.qpos[0],
            "y_position": data.qpos[1],
            "distance_from_origin": jp.linalg.norm(data.qpos[0:2]),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
        }
        metrics = {**reward_info}

        return mjx_env.State(data, obs, reward, done, metrics, info)

    # --------------------------------------------------------------- helpers
    def _is_unhealthy(self, data) -> jax.Array:
        state_vec = jp.concatenate([data.qpos, data.qvel])
        finite = jp.all(jp.isfinite(state_vec))
        z = data.qpos[2]
        in_z = (z >= self._config.healthy_z_min) & (
            z <= self._config.healthy_z_max
        )
        healthy = finite & in_z
        return jp.logical_not(healthy)

    def _get_reward(self, data, action, x_velocity, unhealthy):
        forward_r = x_velocity * self._config.forward_reward_weight
        healthy = jp.logical_not(unhealthy)
        survive_r = jp.where(healthy, self._config.healthy_reward, 0.0)
        ctrl_c = self._config.ctrl_cost_weight * jp.sum(action ** 2)
        cfrc = jp.clip(
            data.cfrc_ext,
            -self._config.contact_force_clip,
            self._config.contact_force_clip,
        )
        contact_c = self._config.contact_cost_weight * jp.sum(cfrc ** 2)
        reward = forward_r + survive_r - ctrl_c - contact_c
        info = {
            "reward_forward": forward_r,
            "reward_survive": survive_r,
            "reward_ctrl": -ctrl_c,  # negated to match Gym v5 ant_v5.py:387
            "reward_contact": -contact_c,
        }
        return reward, info

    def _get_obs(self, data) -> jax.Array:
        qpos = data.qpos
        qvel = data.qvel
        if self._config.exclude_current_positions_from_observation:
            qpos = qpos[2:]  # skip xy
        parts = [qpos, qvel]
        if self._config.include_cfrc_ext_in_observation:
            # Skip worldbody (index 0); flatten (nbody-1, 6) → 78d.
            cfrc = data.cfrc_ext[1:].flatten()
            parts.append(cfrc)
        return jp.concatenate(parts)

    # -------------------------------------------------------------- properties
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
