from flax import nnx
import jax

class GaussianHead(nnx.Module):

    def __init__(self, feature_dim: int, action_dim: int, rngs: nnx.Rngs) -> None:
        self.mu_net = nnx.Linear(feature_dim, action_dim, rngs=rngs)
        self.log_std_net = nnx.Linear(feature_dim, action_dim, rngs=rngs)

    def __call__(self, feature: jax.Array) -> tuple[jax.Array, jax.Array]: 
        return self.mu_net(feature), self.log_std_net(feature) 