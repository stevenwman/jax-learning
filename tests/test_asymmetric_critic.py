"""Tests for asymmetric off-policy critic.

Verifies that all 4 off-policy algos (SAC, TD3, FastSAC, FastTD3) correctly:
  1. Init Q-networks with critic_obs_dim != obs_dim
  2. Keep actor on obs_dim
  3. Run update with asymmetric batch (critic_obs wider than obs)
  4. Remain backward-compatible when critic_obs_dim is None
  5. Flow gradients through actor correctly
  6. Accept critic_obs in get_q_value

Also tests the replay buffer extra_obs_dims feature.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

OBS_DIM = 48
CRITIC_OBS_DIM = 122
ACTION_DIM = 12
BATCH = 32
HIDDEN = (32, 32)


def _make_asymmetric_batch():
    return {
        "obs": jnp.zeros((BATCH, OBS_DIM)),
        "critic_obs": jnp.zeros((BATCH, CRITIC_OBS_DIM)),
        "action": jnp.zeros((BATCH, ACTION_DIM)),
        "reward": jnp.zeros((BATCH, 1)),
        "next_obs": jnp.zeros((BATCH, OBS_DIM)),
        "critic_next_obs": jnp.zeros((BATCH, CRITIC_OBS_DIM)),
        "done": jnp.zeros((BATCH, 1)),
        "truncation": jnp.zeros((BATCH, 1)),
    }


def _make_symmetric_batch():
    return {
        "obs": jnp.ones((BATCH, OBS_DIM)),
        "critic_obs": jnp.ones((BATCH, OBS_DIM)),
        "action": jnp.zeros((BATCH, ACTION_DIM)),
        "reward": jnp.zeros((BATCH, 1)),
        "next_obs": jnp.ones((BATCH, OBS_DIM)),
        "critic_next_obs": jnp.ones((BATCH, OBS_DIM)),
        "done": jnp.zeros((BATCH, 1)),
        "truncation": jnp.zeros((BATCH, 1)),
    }


def _make_nonzero_batch():
    """Batch with nonzero values so gradients are nonzero."""
    return {
        "obs": jnp.ones((BATCH, OBS_DIM)) * 1.0,
        "critic_obs": jnp.ones((BATCH, CRITIC_OBS_DIM)) * 2.0,
        "action": jnp.zeros((BATCH, ACTION_DIM)),
        "reward": jnp.zeros((BATCH, 1)),
        "next_obs": jnp.ones((BATCH, OBS_DIM)) * 1.0,
        "critic_next_obs": jnp.ones((BATCH, CRITIC_OBS_DIM)) * 2.0,
        "done": jnp.zeros((BATCH, 1)),
        "truncation": jnp.zeros((BATCH, 1)),
    }


# ── SAC ──────────────────────────────────────────────────────────────────────

class TestSACAsymmetric:
    def _make(self, critic_obs_dim=CRITIC_OBS_DIM):
        from jax_rl.algos.sac import SAC
        from jax_rl.configs.sac_config import SACConfig
        cfg = SACConfig(hidden_dim=HIDDEN)
        return SAC(cfg, OBS_DIM, ACTION_DIM,
                   optax.adam(1e-3), optax.adam(1e-3),
                   critic_obs_dim=critic_obs_dim)

    def test_q_network_uses_critic_obs_dim(self):
        sac = self._make()
        state = sac.init(jax.random.PRNGKey(0))
        q_val = sac.q1.apply(state.q1_params,
                             jnp.zeros((1, CRITIC_OBS_DIM)),
                             jnp.zeros((1, ACTION_DIM)))
        assert q_val.shape == (1,)

    def test_actor_uses_obs_dim(self):
        sac = self._make()
        state = sac.init(jax.random.PRNGKey(0))
        action = sac.select_action(state.actor_params, jnp.zeros(OBS_DIM),
                                   jax.random.PRNGKey(1))
        assert action.shape == (ACTION_DIM,)

    def test_update_with_asymmetric_batch(self):
        sac = self._make()
        state = sac.init(jax.random.PRNGKey(0))
        new_state, metrics = sac.update(state, _make_asymmetric_batch())
        assert jnp.isfinite(metrics["q1_loss"])
        assert jnp.isfinite(metrics["actor_loss"])

    def test_backward_compatible(self):
        from jax_rl.algos.sac import SAC
        from jax_rl.configs.sac_config import SACConfig
        cfg = SACConfig(hidden_dim=HIDDEN)
        sac = SAC(cfg, OBS_DIM, ACTION_DIM, optax.adam(1e-3), optax.adam(1e-3))
        state = sac.init(jax.random.PRNGKey(0))
        new_state, metrics = sac.update(state, _make_symmetric_batch())
        assert jnp.isfinite(metrics["q1_loss"])

    def test_actor_gradient_flows(self):
        sac = self._make()
        state = sac.init(jax.random.PRNGKey(0))
        new_state, _ = sac.update(state, _make_nonzero_batch())
        actor_diff = jax.tree.map(lambda a, b: jnp.sum(jnp.abs(a - b)),
                                  state.actor_params, new_state.actor_params)
        total_diff = sum(jax.tree.leaves(actor_diff))
        assert total_diff > 0, "Actor params should change"

    def test_get_q_value_with_critic_obs(self):
        sac = self._make()
        state = sac.init(jax.random.PRNGKey(0))
        q = sac.get_q_value(state, jnp.zeros(OBS_DIM), jnp.zeros(ACTION_DIM),
                            critic_obs=jnp.zeros(CRITIC_OBS_DIM))
        assert jnp.isfinite(q)


# ── TD3 ──────────────────────────────────────────────────────────────────────

class TestTD3Asymmetric:
    def _make(self, critic_obs_dim=CRITIC_OBS_DIM):
        from jax_rl.algos.td3 import TD3
        from jax_rl.configs.td3_config import TD3Config
        cfg = TD3Config(hidden_dim=HIDDEN)
        return TD3(cfg, OBS_DIM, ACTION_DIM,
                   optax.adam(1e-3), optax.adam(1e-3),
                   critic_obs_dim=critic_obs_dim)

    def test_q_network_uses_critic_obs_dim(self):
        td3 = self._make()
        state = td3.init(jax.random.PRNGKey(0))
        q_val = td3.q1.apply(state.q1_params,
                             jnp.zeros((1, CRITIC_OBS_DIM)),
                             jnp.zeros((1, ACTION_DIM)))
        assert q_val.shape == (1,)

    def test_actor_uses_obs_dim(self):
        td3 = self._make()
        state = td3.init(jax.random.PRNGKey(0))
        action = td3.select_action(state.actor_params, jnp.zeros(OBS_DIM),
                                   jax.random.PRNGKey(1))
        assert action.shape == (ACTION_DIM,)

    def test_update_with_asymmetric_batch(self):
        td3 = self._make()
        state = td3.init(jax.random.PRNGKey(0))
        new_state, metrics = td3.update(state, _make_asymmetric_batch())
        assert jnp.isfinite(metrics["q1_loss"])

    def test_backward_compatible(self):
        from jax_rl.algos.td3 import TD3
        from jax_rl.configs.td3_config import TD3Config
        cfg = TD3Config(hidden_dim=HIDDEN)
        td3 = TD3(cfg, OBS_DIM, ACTION_DIM, optax.adam(1e-3), optax.adam(1e-3))
        state = td3.init(jax.random.PRNGKey(0))
        new_state, metrics = td3.update(state, _make_symmetric_batch())
        assert jnp.isfinite(metrics["q1_loss"])

    def test_actor_gradient_flows(self):
        td3 = self._make()
        state = td3.init(jax.random.PRNGKey(0))
        # TD3 has policy_delay=2 by default, need 2 updates for actor to change
        new_state, _ = td3.update(state, _make_nonzero_batch())
        new_state, _ = td3.update(new_state, _make_nonzero_batch())
        actor_diff = jax.tree.map(lambda a, b: jnp.sum(jnp.abs(a - b)),
                                  state.actor_params, new_state.actor_params)
        total_diff = sum(jax.tree.leaves(actor_diff))
        assert total_diff > 0, "Actor params should change after policy_delay steps"

    def test_get_q_value_with_critic_obs(self):
        td3 = self._make()
        state = td3.init(jax.random.PRNGKey(0))
        q = td3.get_q_value(state, jnp.zeros(OBS_DIM), jnp.zeros(ACTION_DIM),
                            critic_obs=jnp.zeros(CRITIC_OBS_DIM))
        assert jnp.isfinite(q)


# ── FastSAC ──────────────────────────────────────────────────────────────────

class TestFastSACAsymmetric:
    def _make(self, critic_obs_dim=CRITIC_OBS_DIM):
        from jax_rl.algos.fast_sac import FastSAC
        from jax_rl.configs.fast_sac_config import FastSACConfig
        cfg = FastSACConfig(hidden_dim=HIDDEN, critic_hidden_dim=HIDDEN,
                            num_atoms=11, v_min=-5.0, v_max=5.0)
        return FastSAC(cfg, OBS_DIM, ACTION_DIM,
                       optax.adam(1e-3), optax.adam(1e-3),
                       critic_obs_dim=critic_obs_dim)

    def test_q_network_uses_critic_obs_dim(self):
        algo = self._make()
        state = algo.init(jax.random.PRNGKey(0))
        logits = algo.q1.apply(state.q1_params,
                               jnp.zeros((1, CRITIC_OBS_DIM)),
                               jnp.zeros((1, ACTION_DIM)))
        assert logits.shape == (1, 11)  # num_atoms=11

    def test_actor_uses_obs_dim(self):
        algo = self._make()
        state = algo.init(jax.random.PRNGKey(0))
        action = algo.select_action(state.actor_params, jnp.zeros(OBS_DIM),
                                    jax.random.PRNGKey(1))
        assert action.shape == (ACTION_DIM,)

    def test_update_with_asymmetric_batch(self):
        algo = self._make()
        state = algo.init(jax.random.PRNGKey(0))
        new_state, metrics = algo.update(state, _make_asymmetric_batch())
        assert jnp.isfinite(metrics["q1_loss"])

    def test_backward_compatible(self):
        from jax_rl.algos.fast_sac import FastSAC
        from jax_rl.configs.fast_sac_config import FastSACConfig
        cfg = FastSACConfig(hidden_dim=HIDDEN, critic_hidden_dim=HIDDEN,
                            num_atoms=11, v_min=-5.0, v_max=5.0)
        algo = FastSAC(cfg, OBS_DIM, ACTION_DIM, optax.adam(1e-3), optax.adam(1e-3))
        state = algo.init(jax.random.PRNGKey(0))
        new_state, metrics = algo.update(state, _make_symmetric_batch())
        assert jnp.isfinite(metrics["q1_loss"])

    def test_actor_gradient_flows(self):
        algo = self._make()
        state = algo.init(jax.random.PRNGKey(0))
        # FastSAC has policy_delay=4 by default — but we use small config
        # Run enough updates to trigger actor update
        new_state = state
        for _ in range(4):
            new_state, _ = algo.update(new_state, _make_nonzero_batch())
        actor_diff = jax.tree.map(lambda a, b: jnp.sum(jnp.abs(a - b)),
                                  state.actor_params, new_state.actor_params)
        total_diff = sum(jax.tree.leaves(actor_diff))
        assert total_diff > 0, "Actor params should change after policy_delay steps"

    def test_get_q_value_with_critic_obs(self):
        algo = self._make()
        state = algo.init(jax.random.PRNGKey(0))
        q = algo.get_q_value(state, jnp.zeros(OBS_DIM), jnp.zeros(ACTION_DIM),
                             critic_obs=jnp.zeros(CRITIC_OBS_DIM))
        assert jnp.isfinite(q)


# ── FastTD3 ──────────────────────────────────────────────────────────────────

class TestFastTD3Asymmetric:
    def _make(self, critic_obs_dim=CRITIC_OBS_DIM):
        from jax_rl.algos.fast_td3 import FastTD3
        from jax_rl.configs.fast_td3_config import FastTD3Config
        cfg = FastTD3Config(hidden_dim=HIDDEN, critic_hidden_dim=HIDDEN,
                            num_atoms=11, v_min=-5.0, v_max=5.0)
        return FastTD3(cfg, OBS_DIM, ACTION_DIM,
                       optax.adam(1e-3), optax.adam(1e-3),
                       critic_obs_dim=critic_obs_dim)

    def test_q_network_uses_critic_obs_dim(self):
        algo = self._make()
        state = algo.init(jax.random.PRNGKey(0))
        logits = algo.q1.apply(state.q1_params,
                               jnp.zeros((1, CRITIC_OBS_DIM)),
                               jnp.zeros((1, ACTION_DIM)))
        assert logits.shape == (1, 11)

    def test_actor_uses_obs_dim(self):
        algo = self._make()
        state = algo.init(jax.random.PRNGKey(0))
        action = algo.select_action(state.actor_params, jnp.zeros(OBS_DIM),
                                    jax.random.PRNGKey(1))
        assert action.shape == (ACTION_DIM,)

    def test_update_with_asymmetric_batch(self):
        algo = self._make()
        state = algo.init(jax.random.PRNGKey(0))
        new_state, metrics = algo.update(state, _make_asymmetric_batch())
        assert jnp.isfinite(metrics["q1_loss"])

    def test_backward_compatible(self):
        from jax_rl.algos.fast_td3 import FastTD3
        from jax_rl.configs.fast_td3_config import FastTD3Config
        cfg = FastTD3Config(hidden_dim=HIDDEN, critic_hidden_dim=HIDDEN,
                            num_atoms=11, v_min=-5.0, v_max=5.0)
        algo = FastTD3(cfg, OBS_DIM, ACTION_DIM, optax.adam(1e-3), optax.adam(1e-3))
        state = algo.init(jax.random.PRNGKey(0))
        new_state, metrics = algo.update(state, _make_symmetric_batch())
        assert jnp.isfinite(metrics["q1_loss"])

    def test_actor_gradient_flows(self):
        algo = self._make()
        state = algo.init(jax.random.PRNGKey(0))
        new_state = state
        for _ in range(2):  # policy_delay=2
            new_state, _ = algo.update(new_state, _make_nonzero_batch())
        actor_diff = jax.tree.map(lambda a, b: jnp.sum(jnp.abs(a - b)),
                                  state.actor_params, new_state.actor_params)
        total_diff = sum(jax.tree.leaves(actor_diff))
        assert total_diff > 0, "Actor params should change after policy_delay steps"

    def test_get_q_value_with_critic_obs(self):
        algo = self._make()
        state = algo.init(jax.random.PRNGKey(0))
        q = algo.get_q_value(state, jnp.zeros(OBS_DIM), jnp.zeros(ACTION_DIM),
                             critic_obs=jnp.zeros(CRITIC_OBS_DIM))
        assert jnp.isfinite(q)


# ── Replay Buffer Extra Obs ──────────────────────────────────────────────────

class TestBufferExtraObs:
    def test_extra_obs_stored_and_sampled(self):
        from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
        buf = JaxReplayBuffer(48, 12, max_size=100,
                              extra_obs_dims={"critic_obs": 122})
        buf.add_batch(
            obs=jnp.ones((10, 48)),
            action=jnp.zeros((10, 12)),
            reward=jnp.zeros(10),
            next_obs=jnp.ones((10, 48)),
            done=jnp.zeros(10),
            critic_obs=jnp.ones((10, 122)) * 2.0,
            next_critic_obs=jnp.ones((10, 122)) * 3.0,
        )
        batch = buf.sample(8, key=jax.random.PRNGKey(0))
        assert "critic_obs" in batch
        assert "next_critic_obs" in batch
        assert batch["critic_obs"].shape == (8, 122)
        assert jnp.allclose(batch["critic_obs"], 2.0)
        assert jnp.allclose(batch["next_critic_obs"], 3.0)

    def test_no_extra_obs_unchanged(self):
        from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
        buf = JaxReplayBuffer(48, 12, max_size=100)
        buf.add_batch(
            obs=jnp.ones((10, 48)),
            action=jnp.zeros((10, 12)),
            reward=jnp.zeros(10),
            next_obs=jnp.ones((10, 48)),
            done=jnp.zeros(10),
        )
        batch = buf.sample(8, key=jax.random.PRNGKey(0))
        assert "critic_obs" not in batch
        assert batch["obs"].shape == (8, 48)

    def test_extra_obs_non_jit_path(self):
        """Non-JIT sample path (key=None) should also return extra obs."""
        from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
        buf = JaxReplayBuffer(48, 12, max_size=100,
                              extra_obs_dims={"critic_obs": 122})
        buf.add_batch(
            obs=jnp.ones((10, 48)),
            action=jnp.zeros((10, 12)),
            reward=jnp.zeros(10),
            next_obs=jnp.ones((10, 48)),
            done=jnp.zeros(10),
            critic_obs=jnp.ones((10, 122)) * 5.0,
            next_critic_obs=jnp.ones((10, 122)) * 6.0,
        )
        batch = buf.sample(8)  # key=None
        assert "critic_obs" in batch
        assert batch["critic_obs"].shape == (8, 122)

    def test_extra_obs_wrap_around(self):
        """Extra obs should survive buffer wrap-around."""
        from jax_rl.buffers.jax_replay_buffer import JaxReplayBuffer
        buf = JaxReplayBuffer(4, 2, max_size=10,
                              extra_obs_dims={"critic_obs": 8})
        # Fill past capacity
        for i in range(3):
            buf.add_batch(
                obs=jnp.ones((5, 4)) * float(i),
                action=jnp.zeros((5, 2)),
                reward=jnp.zeros(5),
                next_obs=jnp.ones((5, 4)) * float(i),
                done=jnp.zeros(5),
                critic_obs=jnp.ones((5, 8)) * float(i + 10),
                next_critic_obs=jnp.ones((5, 8)) * float(i + 20),
            )
        assert len(buf) == 10  # capped at max_size
        batch = buf.sample(5, key=jax.random.PRNGKey(42))
        assert batch["critic_obs"].shape == (5, 8)
        # Values should be from the last two batches (i=1 and i=2)
        assert jnp.all(batch["critic_obs"] >= 10.0)
