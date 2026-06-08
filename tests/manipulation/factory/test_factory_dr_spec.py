"""Hermetic test for FactoryPegInsert DR spec declaration."""
import pytest


@pytest.fixture(scope="module")
def env():
    from jax_rl.envs.manipulation.factory.factory_peg_insert import FactoryPegInsert
    return FactoryPegInsert()


def test_dr_specs_declared(env):
    """Env declares the 6 expected DR specs for per-episode randomization."""
    specs = env.get_domain_randomization_spec()
    names = {s.name for s in specs}
    expected = {
        "bolt_pos_xy", "bolt_pos_z", "bolt_yaw",
        "hand_init_pos_xy", "hand_init_pos_z", "hand_init_yaw",
    }
    assert expected == names, f"DR specs mismatch: missing={expected-names}, extra={names-expected}"


def test_dr_specs_are_runtime(env):
    """All Phase 1 DR specs are type='runtime' (env reads from state.info)."""
    specs = env.get_domain_randomization_spec()
    for s in specs:
        assert s.type == "runtime", f"{s.name} is type={s.type}, expected runtime"


def test_dr_spec_bounds_finite(env):
    """All DR specs have finite, sensibly-ranged bounds (no inf or NaN)."""
    import math
    specs = env.get_domain_randomization_spec()
    for s in specs:
        assert math.isfinite(s.min), f"{s.name}.min not finite"
        assert math.isfinite(s.max), f"{s.name}.max not finite"
        assert s.min < s.max, f"{s.name}: min={s.min} >= max={s.max}"
