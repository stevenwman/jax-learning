"""Shared runtime helpers for TD-MPC2: module construction, state init, eval rollout,
and checkpoint load. Used by both scripts/train_tdmpc2.py and scripts/eval_tdmpc2.py
to avoid duplicating ~200 LOC of init+eval plumbing.

Save side lives in train script (write-only path); load side lives here so eval
scripts can reconstruct a TDMPC2State from the npz dump.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from jax_rl.algos.tdmpc2 import (
    Encoder, Dynamics, Reward, QEnsemble, PolicyPrior,
    TDMPC2State,
    build_world_model_optimizer, build_policy_optimizer,
)
from jax_rl.configs.tdmpc2_config import TDMPC2Config
from jax_rl.configs.train_config import TrainConfig
from jax_rl.utils.qscale import qscale_init


def build_modules(cfg: TDMPC2Config):
    """Instantiate the five network modules from config."""
    encoder = Encoder(
        enc_dim=cfg.enc_dim, num_layers=cfg.num_enc_layers,
        latent_dim=cfg.latent_dim, simnorm_dim=cfg.simnorm_dim,
    )
    dynamics = Dynamics(
        mlp_dim=cfg.mlp_dim, latent_dim=cfg.latent_dim, simnorm_dim=cfg.simnorm_dim,
    )
    reward_net = Reward(mlp_dim=cfg.mlp_dim, num_bins=cfg.num_bins)
    q_ensemble = QEnsemble(
        mlp_dim=cfg.mlp_dim, num_bins=cfg.num_bins,
        num_q=cfg.num_q, dropout=cfg.dropout,
    )
    policy = PolicyPrior(
        mlp_dim=cfg.mlp_dim, action_dim=cfg.action_dim,
        log_std_min=cfg.log_std_min, log_std_max=cfg.log_std_max,
    )
    return encoder, dynamics, reward_net, q_ensemble, policy


def init_train_state(
    cfg: TDMPC2Config,
    obs_dim: int,
    seed: int,
) -> tuple[TDMPC2State, object, object]:
    """Build modules + init all params + init optimizers + build TDMPC2State.

    Returns (state, wm_optimizer, policy_optimizer).
    """
    encoder, dynamics, reward_net, q_ensemble, policy = build_modules(cfg)

    key = jax.random.PRNGKey(seed)
    ks = jax.random.split(key, 6)

    enc_params = encoder.init(ks[0], jnp.zeros((1, obs_dim)))
    dyn_params = dynamics.init(ks[1], jnp.zeros((1, cfg.latent_dim)),
                                jnp.zeros((1, cfg.action_dim)))
    rwd_params = reward_net.init(ks[2], jnp.zeros((1, cfg.latent_dim)),
                                  jnp.zeros((1, cfg.action_dim)))
    q_params = q_ensemble.init(
        {"params": ks[3]},
        jnp.zeros((1, cfg.latent_dim)), jnp.zeros((1, cfg.action_dim)),
        deterministic=True,
    )
    pol_params = policy.init(ks[4], jnp.zeros((1, cfg.latent_dim)), ks[5])

    target_enc = jax.tree_util.tree_map(lambda x: x, enc_params)
    target_dyn = jax.tree_util.tree_map(lambda x: x, dyn_params)
    target_rwd = jax.tree_util.tree_map(lambda x: x, rwd_params)
    target_q = jax.tree_util.tree_map(lambda x: x, q_params)

    wm_opt = build_world_model_optimizer(cfg)
    pol_opt = build_policy_optimizer(cfg)
    wm_params = {"encoder": enc_params, "dynamics": dyn_params,
                 "reward": rwd_params, "q_ensemble": q_params}
    wm_opt_state = wm_opt.init(wm_params)
    pol_opt_state = pol_opt.init(
        {"policy": pol_params, "q_ensemble": q_params},
    )

    state = TDMPC2State(
        encoder_params=enc_params,
        dynamics_params=dyn_params,
        reward_params=rwd_params,
        q_ensemble_params=q_params,
        policy_params=pol_params,
        encoder_target_params=target_enc,
        dynamics_target_params=target_dyn,
        reward_target_params=target_rwd,
        q_ensemble_target_params=target_q,
        world_model_opt_state=wm_opt_state,
        policy_opt_state=pol_opt_state,
        qscale=qscale_init(),
        prev_mean=jnp.zeros((cfg.num_envs, cfg.horizon, cfg.action_dim)),
        key=jax.random.PRNGKey(seed + 1000),
        step=jnp.array(0, dtype=jnp.int32),
    )
    return state, wm_opt, pol_opt


def _policy_prior_greedy(policy_params, z, policy):
    """Greedy prior action: tanh(Gaussian mean). Deterministic given the latent."""
    dummy_key = jax.random.PRNGKey(0)
    _, extras = policy.apply(policy_params, z, dummy_key)
    return jnp.tanh(extras["mean"])


def _pipe_obs(obs, dict_obs: bool):
    """Extract 'state' key if dict obs, else pass through."""
    if dict_obs:
        return obs["state"]
    return obs


def run_eval(
    state: TDMPC2State,
    env_bundle,
    plan_fn,
    modules: tuple,
    cfg: TDMPC2Config,
    key: jax.Array,
) -> dict:
    """Evaluate with both MPPI and policy-prior modes.

    Each mode gets its OWN prev_mean tensor (local to this call — NOT shared with
    state.prev_mean). Rolls cfg.num_eval_envs envs for one full episode; returns
    mean total reward across envs for each mode.
    """
    encoder, dynamics, reward_net, q_ensemble, policy = modules
    eval_env = env_bundle.eval_env
    num_eval = cfg.num_eval_envs
    dict_obs = env_bundle.dict_obs

    plan_params = {
        "encoder": state.encoder_params,
        "dynamics": state.dynamics_params,
        "reward": state.reward_params,
        "q_ensemble": state.q_ensemble_params,
        "policy": state.policy_params,
    }

    eval_step = env_bundle.env_step
    ep_len = cfg.episode_lengths[0] if cfg.episode_lengths else 1000

    def _rollout(mode: str, key: jax.Array) -> float:
        key, reset_key = jax.random.split(key)
        env_state = eval_env.reset(jax.random.split(reset_key, num_eval))
        eval_prev_mean = jnp.zeros((num_eval, cfg.horizon, cfg.action_dim))
        t0 = jnp.ones(num_eval, dtype=jnp.bool_)
        total_reward = jnp.zeros(num_eval)

        for _ in range(ep_len):
            obs = _pipe_obs(env_state.obs, dict_obs)
            z_0 = encoder.apply(plan_params["encoder"], obs)

            if mode == "mppi":
                key, plan_key = jax.random.split(key)
                plan_keys = jax.random.split(plan_key, num_eval)
                action, eval_prev_mean = plan_fn(
                    plan_params, z_0, eval_prev_mean, t0, cfg, plan_keys, True,
                )
            else:
                action = _policy_prior_greedy(plan_params["policy"], z_0, policy)

            env_state = eval_step(env_state, action)
            total_reward = total_reward + env_state.reward
            t0 = env_state.done.astype(jnp.bool_)

        return float(jnp.mean(total_reward))

    key, mppi_key, prior_key = jax.random.split(key, 3)
    mppi_return = _rollout("mppi", mppi_key)
    prior_return = _rollout("prior", prior_key)

    return {
        "mppi_return": mppi_return,
        "prior_return": prior_return,
        "mppi_prior_gap": mppi_return - prior_return,
    }


def build_train_config_from_tdmpc2(
    tdmpc2_cfg: TDMPC2Config,
    env_name: str,
    total_timesteps: int,
    seed: int,
) -> TrainConfig:
    """Adapter: minimal TrainConfig from TDMPC2Config + env info."""
    ep_len = tdmpc2_cfg.episode_lengths[0] if tdmpc2_cfg.episode_lengths else 1000
    return TrainConfig(
        env_name=env_name,
        total_timesteps=total_timesteps,
        num_envs=tdmpc2_cfg.num_envs,
        episode_length=ep_len,
        gamma=tdmpc2_cfg.discount,
        lr=tdmpc2_cfg.lr,
        reward_scaling=1.0,
        handle_truncation=True,
    )


def load_params_into_state(state: TDMPC2State, ckpt_dir: str) -> TDMPC2State:
    """Load actor_params.npz + world_model_params.npz from ckpt_dir into a freshly
    initialized state. Replaces online + target params (target := online).

    Save side lives in scripts/train_tdmpc2.py (_flatten_params_for_save / _save_checkpoint).
    """
    import os
    actor_path = os.path.join(ckpt_dir, "actor_params.npz")
    wm_path = os.path.join(ckpt_dir, "world_model_params.npz")

    pol_loaded = _load_into_pytree(state.policy_params, actor_path)
    wm_template = {
        "encoder": state.encoder_params,
        "dynamics": state.dynamics_params,
        "reward": state.reward_params,
        "q_ensemble": state.q_ensemble_params,
    }
    wm_loaded = _load_into_pytree(wm_template, wm_path)
    return state.replace(
        policy_params=pol_loaded,
        encoder_params=wm_loaded["encoder"],
        dynamics_params=wm_loaded["dynamics"],
        reward_params=wm_loaded["reward"],
        q_ensemble_params=wm_loaded["q_ensemble"],
        encoder_target_params=wm_loaded["encoder"],
        dynamics_target_params=wm_loaded["dynamics"],
        reward_target_params=wm_loaded["reward"],
        q_ensemble_target_params=wm_loaded["q_ensemble"],
    )


def _load_into_pytree(template, npz_path: str):
    """Inverse of train script's _flatten_params_for_save: load npz into a pytree
    shaped like `template`, matching by dotted path."""
    flat = np.load(npz_path)
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(template)
    new_leaves = []
    for path, _ in leaves_with_path:
        key = ".".join(
            str(p.key) if hasattr(p, "key") else str(p)
            for p in path
        )
        if key not in flat:
            raise KeyError(f"Missing param '{key}' in {npz_path}")
        new_leaves.append(jnp.asarray(flat[key]))
    return jax.tree_util.tree_unflatten(treedef, new_leaves)
