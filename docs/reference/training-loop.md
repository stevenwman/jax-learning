# Annotated Training Loop (Off-Policy)

The API reference documents algorithms, buffers, and configs in isolation. This page shows how they fit together end-to-end. The real implementation lives in [`jax_rl/training/offpolicy_loop.py`](https://github.com/stevenwman/jax-learning/blob/main/jax_rl/training/offpolicy_loop.py); the snippet below is abbreviated for readability.

For PPO see `jax_rl/training/ppo_loop.py` — the on-policy structure is different (lax.scan rollout + epoch-based update).

```python
import jax, jax.numpy as jnp, optax
from jax_rl.algos.fast_sac import FastSAC
from jax_rl.training.env_setup import make_env_bundle
from jax_rl.training.obs_pipeline import ObsPipeline

# 1. Build env + algo --------------------------------------------------------
# env_bundle wraps the MuJoCo Playground env in our wrapper stack
# (VmapWrapper, EpisodeWrapper, AutoResetWrapper / DomainRandWrapper).
# env_step is JIT'd; env_state is a pytree of JAX arrays.
env_bundle = make_env_bundle(cfg, seed=0)
env_step, env_state = env_bundle.env_step, env_bundle.env_state

# Algorithms are constructed with their optimizer bound. Networks live as
# closures inside __init__ (JIT-friendly); see api/algos.md for the rationale.
algo = FastSAC(
    config=algo_cfg,
    obs_dim=env_bundle.obs_dim,
    action_dim=env_bundle.action_dim,
    optimizer=optax.adam(cfg.lr),
    alpha_optimizer=optax.adam(cfg.lr),
    critic_obs_dim=env_bundle.critic_obs_dim,  # asymmetric critic
)

# 2. Observation pipeline + replay buffer ------------------------------------
# ObsPipeline handles dict obs (state / privileged_state), running mean/std
# normalization, and frame stacking — keeping the algo agnostic to all that.
pipe = ObsPipeline(
    dict_obs=env_bundle.dict_obs,
    has_privileged=env_bundle.has_privileged,
    use_obs_norm=algo_cfg.obs_normalization,
    n_frame_stack=cfg.n_frame_stack,
)
buffer = pipe.make_buffer(
    env_bundle.obs_dim, env_bundle.action_dim, algo_cfg.buffer_size,
    critic_obs_dim=env_bundle.critic_obs_dim, num_envs=cfg.num_envs,
)
norm_state = pipe.init_norm_state(env_bundle.obs_dim)

# 3. Training state ----------------------------------------------------------
# TrainingState is a frozen pytree (actor/critic params, opt state, target
# params, alpha). algo.init returns a fresh one; algo.update returns a new one.
key = jax.random.PRNGKey(0)
key, init_key = jax.random.split(key)
training_state = algo.init(init_key)

# 4. Main loop ---------------------------------------------------------------
for outer_step in range(cfg.total_timesteps // cfg.num_envs):
    # 4a. Read the current obs from env_state (a pytree, not a fresh call).
    raw_obs = pipe.get_obs(env_state.obs)
    norm_state = pipe.update_stats(raw_obs, norm_state)
    obs = pipe.normalize_for_action(raw_obs, norm_state)

    # 4b. Random actions until min_buffer_size; then policy actions.
    # explore_fn closes over the JIT'd select_action — no Python overhead
    # past the first call.
    key, ak = jax.random.split(key)
    if len(buffer) < algo_cfg.min_buffer_size:
        action = jax.random.uniform(ak, (cfg.num_envs, env_bundle.action_dim),
                                    minval=-1.0, maxval=1.0)
    else:
        action = algo.select_action(training_state.actor_params, obs, ak)

    # 4c. env_step is JIT'd + vmapped across num_envs. One Python call
    # advances all parallel envs by one physics step on the GPU.
    env_state = env_step(env_state, action)

    # 4d. Buffer.add_batch handles the [num_envs, ...] batch in one shot.
    buffer.add_batch(
        obs=raw_obs, action=action,
        reward=env_state.reward * cfg.reward_scaling,
        next_obs=pipe.get_obs(env_state.obs),
        done=env_state.done,
        truncation=env_state.info.get("truncation", jnp.zeros_like(env_state.done)),
    )

    # 4e. UTD ratio: grad_updates_per_step gradient steps per env step.
    # algo.update is fully JIT'd — buffer.sample returns a JAX pytree that
    # stays on-device, so no host<->device transfers per step.
    if len(buffer) >= algo_cfg.min_buffer_size:
        for _ in range(algo_cfg.grad_updates_per_step):
            key, sk = jax.random.split(key)
            batch = buffer.sample(algo_cfg.batch_size, key=sk)
            batch = pipe.normalize_batch(batch, norm_state)
            training_state, metrics = algo.update(training_state, batch)
```

## Things to notice

- **No mutation.** `training_state`, `norm_state`, `env_state` are all rebound each iteration — JAX pytrees are functional. `buffer` is the one stateful exception (it owns a ring of preallocated device arrays).
- **One Python iteration = `num_envs` env steps.** With `num_envs=1024` you only loop ~20k times for a 20M-step run.
- **JIT boundaries are `env_step`, `algo.select_action`, `algo.update`.** Everything inside those compiles once and runs on-device.
- **Asymmetric critic is plumbed through `critic_obs` kwargs.** When the env returns dict obs with `"privileged_state"`, the buffer stores both views and the critic sees the privileged one. See [Asymmetric Critic](../tutorials/asymmetric-critic.md).
- **The real loop adds:** episode tracking, periodic eval + checkpointing, W&B logging, resume support, action-delay wrappers. Those are orthogonal to the core RL math.

## See also

- [Concepts](../getting-started/concepts.md) — high-level architecture
- [Architecture](architecture.md) — three-layer separation, wrapper pipeline, checkpoint format
- [Algorithms](../api/algos.md) — per-algo constructors and `update` signatures
