import jax
from jax._src.prng import value
import jax.numpy as jnp
from jaxlib.mlir.dialects.sparse_tensor import out

# def value_fn(params, state):
#     return jnp.dot(params, state)

# def td_loss(params, state, reward, next_state, gamma=0.99):
#     v_s = value_fn(params, state)
#     v_next = value_fn(params, next_state)
#     target = reward + gamma * v_next
#     return (v_s - target) ** 2

# params = jnp.array([0.1, 0.2, 0.3])
# state = jnp.array([1.0, 0.5, 0.0])
# next_state = jnp.array([0.8, 0.6, 0.1])
# reward = 1.0

# grad = jax.grad(td_loss, 0)(params, state, reward, next_state)
# print(grad)

# val, grad = jax.value_and_grad(td_loss, 0)(params, state, reward, next_state)
# print(val)
# print(grad)

# loss_fn = lambda p: td_loss(p, state, reward, next_state)
# val, grad = jax.value_and_grad(loss_fn)(params)
# print(val)
# print(grad)

# def compute_target_broken(reward, next_state, params, done, gamma=0.99):
#     v_next = value_fn(params, next_state)
#     if done:
#         return reward
#     else:
#         return reward + gamma * v_next

# def compute_target_fixed1(reward, next_state, params, done, gamma=0.99):
#     v_next = value_fn(params, next_state)
#     reward = jax.lax.cond(done, lambda r: r, lambda r: r + gamma * v_next, reward)
#     return reward

# def compute_target_fixed2(reward, next_state, params, done, gamma=0.99):
#     v_next = value_fn(params, next_state)
#     reward = jax.lax.select(done, reward, reward + gamma * v_next)
#     return reward

# def env_step(state, action):
#     """Dummy env: state evolves linearly, reward = -distance from origin"""
#     next_state = state * 0.9 + action * 0.1
#     reward = -jnp.sum(next_state ** 2)
#     done = jnp.sum(next_state ** 2) < 0.01
#     return next_state, reward, done

# def policy(params, state):
#     """Linear policy"""
#     return jnp.tanh(params @ state)


# def bad_rollout_loop(params, init_state, num_steps):
#     state = init_state
#     rewards = []
#     for _ in range(num_steps):
#         action = policy(params, state)
#         state, reward, done = env_step(state, action)
#         rewards.append(reward)
#     return jnp.array(rewards)

# def good_rollout_loop(params, init_state, num_steps):

#     def loop(carry, x):
#         state = carry
#         action = policy(params, state)
#         state, reward, done = env_step(state, action)
#         return state, reward

#     final_state, rewards = jax.lax.scan(loop, init_state, None, length=num_steps)
#     return rewards

# def update_stats(stats, observation):
#     """Online mean/variance update (Welford's algorithm)"""
#     mean, var, count = stats
#     count = count + 1
#     delta = observation - mean
#     mean = mean + delta / count
#     var = var + delta * (observation - mean)
#     normalized = (observation - mean) / (jnp.sqrt(var / count) + 1e-8)
#     return (mean, var, count), normalized  # return normalized obs? or raw?


# observations = jnp.array([
#     [1.0, 2.0],
#     [1.5, 2.5],
#     [0.5, 1.5],
#     [1.2, 2.2],
#     [0.8, 1.8],
# ])  # 5 timesteps, 2D observations

# init_stats = (jnp.zeros(2), jnp.zeros(2), 0.0)  # (mean, var, count)


# final, outputs = jax.lax.scan(update_stats, init_stats, observations)

# print(final)
# print(outputs)



# rewards = jnp.array([1.0, 0.0, 0.0, 1.0, 0.0])      # 5 timesteps
# values = jnp.array([0.5, 0.4, 0.3, 0.8, 0.2])       # V(s) estimates
# next_values = jnp.array([0.4, 0.3, 0.8, 0.2, 0.0])  # V(s') estimates
# dones = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0])        # episode ends at step 5

# gamma = 0.99
# lam = 0.95

# # delta_t = reward_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
# # A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}

# def update_advantage(A_tp1, x):
#     reward_t, V_tp1, V_t, done_t = x
#     delta_t = reward_t + gamma * V_tp1 * (1 - done_t) - V_t
#     A_t = delta_t + gamma * lam * (1 - done_t) * A_tp1
#     return A_t, A_t

# A_t, A_ts = jax.lax.scan(update_advantage, 0, (rewards, values, next_values, dones), reverse=True)
# print(A_t, A_ts)

# # def scan(f, init, xs, length=None):
# #   if xs is None:
# #     xs = [None] * length
# #   carry = init
# #   ys = []
# #   for x in xs:
# #     carry, y = f(carry, x)
# #     ys.append(y)
# #   return carry, np.stack(ys)

def single_env_step(state, action, params):
    next_state = state * 0.9 + action * 0.1
    reward = -jnp.dot(params, next_state ** 2)
    return next_state, reward

batched_step = jax.vmap(single_env_step, in_axes=(0,0,None))

states = jnp.array([
    [1.0, 0.0],  # env 0 state
    [0.5, 0.5],  # env 1 state
    [0.0, 1.0],  # env 2 state
])  # shape (3, 2) — 3 envs, 2D state

actions = jnp.array([
    [0.1, 0.0],  # env 0 action
    [0.0, 0.1],  # env 1 action
    [-0.1, 0.0], # env 2 action
])  # shape (3, 2)

params = jnp.array([1.0, 1.0])  # shape (2,) — shared across all envs

batched_step = jax.vmap(single_env_step, in_axes=(0, 0, None))
next_states, rewards = batched_step(states, actions, params)
print(next_states.shape, rewards.shape)