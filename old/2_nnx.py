from math import log
from flax import nnx
import optax
import jax
import jax.numpy as jnp

# class MLP(nnx.Module):
#     def __init__(self, din, dhidden, dout, rngs: nnx.Rngs):
#         self.linear1 = nnx.Linear(din, dhidden, rngs=rngs)
#         self.linear2 = nnx.Linear(dhidden, dout, rngs=rngs)

#     def __call__(self, x):
#         x = nnx.relu(self.linear1(x))
#         return self.linear2(x)

# model = MLP(4, 32, 4, rngs=nnx.Rngs(0))
# print(model.linear1.kernel.get_value().shape)


# optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

# @nnx.jit
# def train_step(model, optimizer, x, y):
#     def loss_fn(model):
#         y_pred = model(x)
#         return jnp.mean((y_pred - y) ** 2)

#     loss, grads = nnx.value_and_grad(loss_fn)(model)
#     optimizer.update(model, grads)
#     return loss

# fun = lambda x: x**2

# key = jax.random.PRNGKey(42)
# key, subkey = jax.random.split(key)
# x = jax.random.normal(subkey, (32, 4))

# y = jax.vmap(fun, (0))(x)

# for loop in range(100):
#     loss = train_step(model, optimizer, x, y)
#     print(loss)

class GaussianPolicy(nnx.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim, rngs: nnx.Rngs):
        self.net = nnx.Linear(obs_dim, hidden_dim, rngs=rngs)
        self.mean_head = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        self.log_std_head = nnx.Linear(hidden_dim, action_dim, rngs=rngs)

    def __call__(self, obs):
        h = nnx.relu(self.net(obs))
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        return mean, log_std

    def sample_action(self, obs, key):
        mean, log_std = self(obs)
        actions = jax.random.normal(key) * jnp.exp(log_std) + mean
        return actions, mean, log_std

policy = GaussianPolicy(3,15,1,nnx.Rngs(0))

key = jax.random.PRNGKey(42)
key,subkey = jax.random.split(key)

obs = jax.random.normal(subkey, (5,3))
key,subkey = jax.random.split(key)

for i in range(5):
    actions, mean, log_std = policy.sample_action(obs, subkey)
    print(actions)

for i in range(5):
    key,subkey = jax.random.split(key)
    actions, mean, log_std = policy.sample_action(obs, subkey)
    print(actions)