import jax
import jax.numpy as jnp
import pytest


class TestTD3PolicyDelay:
    def test_td3_always_returns_actor_loss(self):
        """actor_loss should be non-zero in metrics even on critic-only steps."""
        from jax_rl.algos.td3 import TD3
        from jax_rl.configs.td3_config import TD3Config
        import optax

        cfg = TD3Config()
        algo = TD3(config=cfg, obs_dim=4, action_dim=2,
                   actor_optimizer=optax.adam(3e-4),
                   critic_optimizer=optax.adam(3e-4))
        key = jax.random.PRNGKey(0)
        ts = algo.init(key)

        # Create a dummy batch
        batch = {
            "obs": jnp.zeros((256, 4)),
            "action": jnp.zeros((256, 2)),
            "reward": jnp.zeros((256, 1)),
            "next_obs": jnp.zeros((256, 4)),
            "done": jnp.zeros((256, 1)),
            "truncation": jnp.zeros((256, 1)),
            "critic_obs": jnp.zeros((256, 4)),
            "critic_next_obs": jnp.zeros((256, 4)),
        }

        # Run enough updates to hit both actor and critic-only steps
        losses = []
        for _ in range(cfg.policy_delay + 2):
            ts, metrics = algo.update(ts, batch)
            losses.append(float(metrics["actor_loss"]))

        # After policy_delay steps, ALL should have non-zero actor_loss
        # (The first few might be 0 if no actor update happened yet, that's ok)
        # But once an actor update happens, subsequent critic-only steps should carry it forward
        has_nonzero = any(l != 0.0 for l in losses)
        assert has_nonzero, "No actor update happened at all"

        # After the first nonzero, none should be exactly 0.0
        first_nonzero_idx = next(i for i, l in enumerate(losses) if l != 0.0)
        for i in range(first_nonzero_idx, len(losses)):
            assert losses[i] != 0.0, f"Step {i} had actor_loss=0.0 after first real update"

    def test_fast_td3_always_returns_actor_loss(self):
        """Same test for FastTD3."""
        from jax_rl.algos.fast_td3 import FastTD3
        from jax_rl.configs.fast_td3_config import FastTD3Config
        import optax

        cfg = FastTD3Config()
        algo = FastTD3(config=cfg, obs_dim=4, action_dim=2,
                       actor_optimizer=optax.adam(3e-4),
                       critic_optimizer=optax.adam(3e-4))
        key = jax.random.PRNGKey(0)
        ts = algo.init(key)

        batch = {
            "obs": jnp.zeros((256, 4)),
            "action": jnp.zeros((256, 2)),
            "reward": jnp.zeros((256, 1)),
            "next_obs": jnp.zeros((256, 4)),
            "done": jnp.zeros((256, 1)),
            "truncation": jnp.zeros((256, 1)),
            "critic_obs": jnp.zeros((256, 4)),
            "critic_next_obs": jnp.zeros((256, 4)),
        }

        losses = []
        for _ in range(cfg.policy_delay + 2):
            ts, metrics = algo.update(ts, batch)
            losses.append(float(metrics["actor_loss"]))

        has_nonzero = any(l != 0.0 for l in losses)
        assert has_nonzero

        first_nonzero_idx = next(i for i, l in enumerate(losses) if l != 0.0)
        for i in range(first_nonzero_idx, len(losses)):
            assert losses[i] != 0.0, f"Step {i} had actor_loss=0.0 after first real update"
