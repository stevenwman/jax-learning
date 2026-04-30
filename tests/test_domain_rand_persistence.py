"""Regression tests for per-episode DomainRand model-field persistence."""

import jax
import jax.numpy as jp
from mujoco_playground._src.mjx_env import State

from jax_rl.envs.wrappers.domain_rand import DRSpec, DomainRandWrapper


@jax.tree_util.register_pytree_node_class
class FakeModel:
    """Tiny pytree model with the tree_replace API DomainRandWrapper needs."""

    def __init__(self, geom_friction):
        self.geom_friction = geom_friction

    def tree_flatten(self):
        return (self.geom_friction,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(children[0])

    def tree_replace(self, replacements):
        geom_friction = replacements.get("geom_friction", self.geom_friction)
        return FakeModel(geom_friction)


class FakeEnv:
    """Vmap-compatible env stub exposing the model fields DomainRand reads."""

    action_size = 1

    def __init__(self, model):
        self._mjx_model = model
        self.unwrapped = self

    @property
    def mjx_model(self):
        return self._mjx_model

    def reset(self, rng):
        return State(
            data=jp.array(0.0),
            obs=jp.zeros((1,)),
            reward=jp.array(0.0),
            done=jp.array(0.0),
            metrics={},
            info={},
        )

    def step(self, state, action):
        return state.replace(
            obs=state.obs + jp.ones_like(state.obs),
            reward=jp.array(0.5),
            done=jp.array(0.0),
            info={},
        )


def _mock_wrapper_with_episode(specs, *, episode_length=1000):
    """Mock wrapper that supports reset() and step()."""
    w = DomainRandWrapper.__new__(DomainRandWrapper)
    w.env = FakeEnv(FakeModel(jp.ones((10, 3)) * 0.5))
    w.episode_length = episode_length
    w.mode = "per_step"
    w._model_specs = specs
    w._runtime_specs = []
    w.dr_specs = specs
    return w


def test_dr_fields_persist_across_steps_for_non_done_envs():
    """Non-done envs keep byte-identical DR fields across multiple steps."""
    specs = [
        DRSpec(
            name="friction",
            type="model",
            field="geom_friction",
            column=0,
            operation="uniform",
            min=0.0,
            max=1.0,
            per_element=False,
        )
    ]
    wrapper = _mock_wrapper_with_episode(specs)
    state = wrapper.reset(jax.random.split(jax.random.PRNGKey(0), 4))
    initial = state.info["_dr_dr_fields"]["geom_friction"]
    action = jp.zeros((4, 1))

    state = wrapper.step(state, action)
    after_step1 = state.info["_dr_dr_fields"]["geom_friction"]
    assert jp.array_equal(initial, after_step1)

    state = wrapper.step(state, action)
    after_step2 = state.info["_dr_dr_fields"]["geom_friction"]
    assert jp.array_equal(initial, after_step2)


def test_dr_fields_resample_only_for_done_envs():
    """Done envs receive fresh DR fields; active envs keep persisted fields."""
    specs = [
        DRSpec(
            name="friction",
            type="model",
            field="geom_friction",
            column=0,
            operation="uniform",
            min=0.0,
            max=1.0,
            per_element=False,
        )
    ]
    wrapper = _mock_wrapper_with_episode(specs, episode_length=2)
    state = wrapper.reset(jax.random.split(jax.random.PRNGKey(0), 2))
    initial = state.info["_dr_dr_fields"]["geom_friction"]

    # Start env 0 one step from truncation; env 1 remains active.
    state.info["_dr_steps"] = jp.array([1.0, 0.0])
    state = wrapper.step(state, jp.zeros((2, 1)))
    after = state.info["_dr_dr_fields"]["geom_friction"]

    assert not jp.array_equal(initial[0], after[0])
    assert jp.array_equal(initial[1], after[1])
