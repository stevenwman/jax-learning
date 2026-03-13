"""Diagnose which operation in the training loop is slow."""

import time
import jax
import jax.numpy as jnp

from mujoco_playground import dm_control_suite
from mujoco_playground._src.wrapper import wrap_for_brax_training

from jax_rl.algos.ppo_jit import PPO
from jax_rl.configs import PPOConfig, EncoderConfig, PolicyHeadConfig
from jax_rl.utils.normalization import init as norm_init, update as norm_update, normalize as norm_normalize

num_envs = 64
num_steps = 64

# ── Setup ──
env = dm_control_suite.load("CartpoleBalance")
env = wrap_for_brax_training(env, episode_length=1000)

key = jax.random.PRNGKey(0)
env_state = env.reset(jax.random.split(key, num_envs))
obs_dim = env_state.obs.shape[-1]
action_dim = env.action_size

config = PPOConfig(
    encoder=EncoderConfig(obs_dim=obs_dim, hidden_dim=(64, 64)),
    policy_head=PolicyHeadConfig(action_dim=action_dim, squash=False),
    num_envs=num_envs, num_steps=num_steps, minibatch_size=256,
)
ppo = PPO(config, obs_dim, action_dim)
key, init_key = jax.random.split(key)
training_state = ppo.init(init_key)
norm_state = norm_init(obs_dim)

print(f"Setup done. obs_dim={obs_dim}, action_dim={action_dim}, num_envs={num_envs}")
print()

# ── Test 1: env.step alone ──
print("Test 1: env.step × 10 (warmup + timed)")
action = jax.random.uniform(key, (num_envs, action_dim), minval=-1, maxval=1)

# Warmup (JIT compile)
t0 = time.time()
env_state = env.step(env_state, action)
jax.block_until_ready(env_state.obs)
print(f"  env.step warmup (1 call): {time.time() - t0:.2f}s")

# Timed
t0 = time.time()
for _ in range(10):
    env_state = env.step(env_state, action)
jax.block_until_ready(env_state.obs)
print(f"  env.step timed (10 calls): {time.time() - t0:.2f}s → {(time.time() - t0) / 10:.3f}s each")
print()

# ── Test 2: select_action alone ──
print("Test 2: select_action × 10 (warmup + timed)")
obs = env_state.obs

# Warmup
t0 = time.time()
key, ak = jax.random.split(key)
a, lp, v = ppo.select_action(training_state, obs, ak)
jax.block_until_ready(a)
print(f"  select_action warmup (1 call): {time.time() - t0:.2f}s")

# Timed
t0 = time.time()
for _ in range(10):
    key, ak = jax.random.split(key)
    a, lp, v = ppo.select_action(training_state, obs, ak)
jax.block_until_ready(a)
print(f"  select_action timed (10 calls): {time.time() - t0:.2f}s → {(time.time() - t0) / 10:.3f}s each")
print()

# ── Test 3: norm_update + normalize ──
print("Test 3: norm_update+normalize × 10")
t0 = time.time()
for _ in range(10):
    norm_state = norm_update(norm_state, obs)
    normed = norm_normalize(norm_state, obs)
jax.block_until_ready(normed)
print(f"  norm × 10: {time.time() - t0:.2f}s → {(time.time() - t0) / 10:.3f}s each")
print()

# ── Test 4: full collect loop (no buffer) ──
print("Test 4: full collect loop (64 steps, no buffer.add)")
t0 = time.time()
for step in range(num_steps):
    obs = env_state.obs
    norm_state = norm_update(norm_state, obs)
    normed_obs = norm_normalize(norm_state, obs)
    key, ak = jax.random.split(key)
    action, log_prob, value = ppo.select_action(training_state, normed_obs, ak)
    clipped = jnp.clip(action, -1.0, 1.0)
    env_state = env.step(env_state, clipped)
jax.block_until_ready(env_state.obs)
print(f"  64-step collect: {time.time() - t0:.2f}s")
print()

# ── Test 5: minibatch_step ──
print("Test 5: _minibatch_step × 10 (warmup + timed)")
from jax_rl.buffers import RolloutBuffer
buf = RolloutBuffer(num_steps, num_envs, obs_dim, action_dim)
# fill with dummy data
for s in range(num_steps):
    buf.add(obs=jnp.zeros((num_envs, obs_dim)), action=jnp.zeros((num_envs, action_dim)),
            reward=jnp.zeros(num_envs), done=jnp.zeros(num_envs),
            log_prob=jnp.zeros(num_envs), value=jnp.zeros(num_envs))
batch = buf.get(jnp.zeros(num_envs), gamma=0.99, gae_lambda=0.95)
flat_obs = batch.obs.reshape(-1, obs_dim)
flat_act = batch.actions.reshape(-1, action_dim)
flat_lp = batch.log_probs.reshape(-1)
flat_ret = batch.returns.reshape(-1)
flat_adv = batch.advantages.reshape(-1)

mb = jnp.arange(256)
# Warmup
t0 = time.time()
state, metrics = ppo._minibatch_step(training_state, flat_obs[mb], flat_act[mb], flat_lp[mb], flat_ret[mb], flat_adv[mb])
jax.block_until_ready(state.actor_params)
print(f"  _minibatch_step warmup: {time.time() - t0:.2f}s")

# Timed
t0 = time.time()
for _ in range(10):
    state, metrics = ppo._minibatch_step(state, flat_obs[mb], flat_act[mb], flat_lp[mb], flat_ret[mb], flat_adv[mb])
jax.block_until_ready(state.actor_params)
print(f"  _minibatch_step × 10: {time.time() - t0:.2f}s → {(time.time() - t0) / 10:.3f}s each")
