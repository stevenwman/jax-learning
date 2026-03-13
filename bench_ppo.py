"""Benchmark ppo.py vs ppo_jit.py vs ppo_scan.py — update step only."""

import time
import jax
import jax.numpy as jnp

from jax_rl.configs import PPOConfig, EncoderConfig, PolicyHeadConfig
from jax_rl.buffers import RolloutBuffer

num_envs = 64
num_steps = 64
obs_dim = 5
action_dim = 1
N = 5  # number of full update() calls to time


def make_config():
    """Fresh config each time (PPO __init__ mutates encoder_config)."""
    return PPOConfig(
        encoder=EncoderConfig(obs_dim=obs_dim, hidden_dim=(64, 64)),
        policy_head=PolicyHeadConfig(action_dim=action_dim, squash=False),
        num_envs=num_envs, num_steps=num_steps, minibatch_size=256,
    )


def make_batch():
    buf = RolloutBuffer(num_steps, num_envs, obs_dim, action_dim)
    for s in range(num_steps):
        buf.add(obs=jnp.ones((num_envs, obs_dim)), action=jnp.ones((num_envs, action_dim)),
                reward=jnp.ones(num_envs), done=jnp.zeros(num_envs),
                log_prob=jnp.zeros(num_envs), value=jnp.ones(num_envs))
    return buf.get(jnp.zeros(num_envs), gamma=0.99, gae_lambda=0.95)


def bench(label, ppo_cls, key, batch, obs):
    print("=" * 60)
    print(label)
    print("=" * 60)

    ppo = ppo_cls(make_config(), obs_dim, action_dim)
    key, ik = jax.random.split(key)
    state = ppo.init(ik)

    # Warmup update
    key, uk = jax.random.split(key)
    t0 = time.time()
    state, _ = ppo.update(state, batch, uk)
    jax.block_until_ready(state.actor_params)
    print(f"  update() warmup: {time.time() - t0:.2f}s")

    # Timed update
    t0 = time.time()
    for _ in range(N):
        key, uk = jax.random.split(key)
        state, metrics = ppo.update(state, batch, uk)
    jax.block_until_ready(state.actor_params)
    elapsed = time.time() - t0
    print(f"  update() × {N}: {elapsed:.2f}s → {elapsed / N:.3f}s each")

    # Warmup select_action
    key, ak = jax.random.split(key)
    t0 = time.time()
    a, lp, v = ppo.select_action(state, obs, ak)
    jax.block_until_ready(a)
    print(f"  select_action warmup: {time.time() - t0:.2f}s")

    # Timed select_action
    t0 = time.time()
    for _ in range(100):
        key, ak = jax.random.split(key)
        a, lp, v = ppo.select_action(state, obs, ak)
    jax.block_until_ready(a)
    elapsed = time.time() - t0
    print(f"  select_action × 100: {elapsed:.2f}s → {elapsed / 100:.4f}s each")
    print()

    return key


key = jax.random.PRNGKey(42)
batch = make_batch()
obs = jnp.ones((num_envs, obs_dim))

from jax_rl.algos.ppo import PPO as PPO_eager
key = bench("ppo.py — Python loops, no JIT on update/select", PPO_eager, key, batch, obs)

from jax_rl.algos.ppo_jit import PPO as PPO_jit
key = bench("ppo_jit.py — JIT closures, Python loops for epochs", PPO_jit, key, batch, obs)

from jax_rl.algos.ppo_scan import PPO as PPO_scan
key = bench("ppo_scan.py — JIT closures + scan (fully compiled update)", PPO_scan, key, batch, obs)
