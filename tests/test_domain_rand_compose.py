"""Tests for DomainRandWrapper model-DR field composition.

Codex audit (2026-04-27) flagged that `_sample_dr_fields` overwrites earlier
spec replacements when multiple specs target the same `mjx.Model` field
(e.g. `geom_friction` column 0 + column 1). Each spec's `at[...].set` was
applied to the ORIGINAL `field_data`, not the previous spec's output, so
later specs silently discarded earlier contributions.

Fix: read `current_field = replacements.get(spec.field, model.<field>)`
before each spec's op, so subsequent specs build on prior outputs.

These tests run hermetically on CPU — no env construction. They monkey-
construct `DomainRandWrapper` with a `SimpleNamespace` mock env exposing
just the `unwrapped.mjx_model.<field>` attribute the wrapper reads.
"""

import types

import jax
import jax.numpy as jp
import pytest

from jax_rl.envs.wrappers.domain_rand import DomainRandWrapper, DRSpec


def _mock_wrapper(model_field_name: str, model_field_array: jp.ndarray, specs: list[DRSpec]):
    """Bypass __init__'s env probing — assemble just the bits _sample_dr_fields reads."""
    w = DomainRandWrapper.__new__(DomainRandWrapper)
    w.env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            mjx_model=types.SimpleNamespace(**{model_field_name: model_field_array}),
        ),
    )
    w._model_specs = specs
    w._runtime_specs = []
    return w


def test_two_specs_on_distinct_columns_compose():
    """Both specs' contributions must survive after _sample_dr_fields."""
    specs = [
        DRSpec(name="c0", type="model", field="geom_friction", column=0,
               operation="set", min=0.7, max=0.7, per_element=False),
        DRSpec(name="c1", type="model", field="geom_friction", column=1,
               operation="set", min=0.3, max=0.3, per_element=False),
    ]
    w = _mock_wrapper("geom_friction", jp.ones((10, 3)) * 0.5, specs)

    fields = w._sample_dr_fields(jax.random.split(jax.random.PRNGKey(0), 4))
    gf = fields["geom_friction"]

    assert gf.shape == (4, 10, 3)
    assert jp.allclose(gf[..., 0], 0.7), "spec1 (col0=0.7) was overwritten"
    assert jp.allclose(gf[..., 1], 0.3), "spec2 (col1=0.3) not applied"
    assert jp.allclose(gf[..., 2], 0.5), "col2 was modified by accident"


def test_two_specs_on_same_column_compose_via_multiply():
    """Multiply ops on same field+column must chain (op2(op1(x)) not op2(x))."""
    specs = [
        DRSpec(name="m1", type="model", field="geom_friction", column=0,
               operation="multiply", min=2.0, max=2.0, per_element=False),
        DRSpec(name="m2", type="model", field="geom_friction", column=0,
               operation="multiply", min=3.0, max=3.0, per_element=False),
    ]
    w = _mock_wrapper("geom_friction", jp.ones((5, 1)) * 0.5, specs)

    fields = w._sample_dr_fields(jax.random.split(jax.random.PRNGKey(0), 2))
    # Pre-fix: m2 would compute from original 0.5 → 0.5 * 3 = 1.5 (m1 lost).
    # Post-fix: m2 builds on m1's output → 0.5 * 2 * 3 = 3.0.
    gf = fields["geom_friction"]
    assert jp.allclose(gf, 3.0), f"compose failed: expected 3.0, got {gf[0, 0, 0]}"


def test_single_spec_unchanged():
    """Single-spec case must be byte-identical to old behavior."""
    specs = [
        DRSpec(name="single", type="model", field="geom_friction", column=0,
               operation="set", min=0.42, max=0.42, per_element=False),
    ]
    w = _mock_wrapper("geom_friction", jp.ones((4, 3)) * 0.5, specs)

    fields = w._sample_dr_fields(jax.random.split(jax.random.PRNGKey(0), 2))
    gf = fields["geom_friction"]

    assert jp.allclose(gf[..., 0], 0.42)
    assert jp.allclose(gf[..., 1], 0.5)
    assert jp.allclose(gf[..., 2], 0.5)


def test_specs_on_different_fields_unaffected():
    """Specs on different fields are independent — composition only kicks in on
    repeated `spec.field`."""
    specs = [
        DRSpec(name="fric", type="model", field="geom_friction", column=0,
               operation="set", min=0.7, max=0.7, per_element=False),
        DRSpec(name="dmp", type="model", field="dof_damping",
               operation="set", min=0.1, max=0.1, per_element=False),
    ]
    arr_fric = jp.ones((10, 3)) * 0.5
    arr_dmp = jp.ones((12,)) * 0.05

    w = DomainRandWrapper.__new__(DomainRandWrapper)
    w.env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            mjx_model=types.SimpleNamespace(
                geom_friction=arr_fric, dof_damping=arr_dmp,
            ),
        ),
    )
    w._model_specs = specs
    w._runtime_specs = []

    fields = w._sample_dr_fields(jax.random.split(jax.random.PRNGKey(0), 2))
    assert "geom_friction" in fields and "dof_damping" in fields
    assert jp.allclose(fields["geom_friction"][..., 0], 0.7)
    assert jp.allclose(fields["dof_damping"], 0.1)
