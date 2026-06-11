"""tests/test_go2_warp_variants.py — variants-as-data registry tests."""
import numpy as np
import pytest


def _deep_eq(a, b, path=""):
    """ConfigDict-aware deep equality with float tolerance 0 (exact)."""
    da, db = a.to_dict(), b.to_dict()
    def rec(x, y, p):
        assert type(x) == type(y) or (isinstance(x, (int, float)) and isinstance(y, (int, float))), f"{p}: {x!r} vs {y!r}"
        if isinstance(x, dict):
            assert x.keys() == y.keys(), f"{p}: keys {sorted(x)} vs {sorted(y)}"
            for k in x: rec(x[k], y[k], f"{p}.{k}")
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y), f"{p}: len"
            for i, (xi, yi) in enumerate(zip(x, y)): rec(xi, yi, f"{p}[{i}]")
        else:
            assert x == y, f"{p}: {x!r} != {y!r}"
    rec(da, db, path)


def test_variant_configs_match_legacy_registration():
    """TRANSITIONAL: every variant's config == the legacy-registered config.

    Relies on mjx_backend's legacy closures still being registered. After
    Task 5 this test is REPLACED by the snapshot test."""
    from mujoco_playground import registry as pg_registry
    import jax_rl.training.env_backends.mjx_backend  # noqa: F401  (registers legacy)
    from jax_rl.envs.locomotion.go2_warp_variants import GO2_WARP_VARIANTS
    for name, v in GO2_WARP_VARIANTS.items():
        legacy = pg_registry.get_default_config(name)
        new = v.config()
        _deep_eq(legacy, new, path=name)


def test_registry_uses_variant_cls():
    from mujoco_playground import registry as pg_registry
    import jax_rl.training.env_backends.mjx_backend  # noqa: F401
    from jax_rl.envs.locomotion.go2_warp_variants import GO2_WARP_VARIANTS
    # construction-only; pick the 6 non-default-cls + 2 default-cls names to keep runtime sane
    check = {n: v for n, v in GO2_WARP_VARIANTS.items()
             if v.cls != "WarpJoystick"} | {
        "Go2WarpJoystickFlat": GO2_WARP_VARIANTS["Go2WarpJoystickFlat"],
        "Go2WarpOscFlatSoftPhysical": GO2_WARP_VARIANTS["Go2WarpOscFlatSoftPhysical"]}
    for name, v in check.items():
        env = pg_registry.load(name)
        assert type(env).__name__ == v.cls, f"{name}: {type(env).__name__} != {v.cls}"


def test_joint_pd_rejects_cartesian_knobs():
    from jax_rl.envs.locomotion.go2_warp_variants import go2_config
    with pytest.raises(ValueError, match="cartesian controller"):
        go2_config(controller="joint_pd", osc_kp=[1, 1, 1])


def test_fixed_gain_osc_rejects_var_impedance_knobs():
    from jax_rl.envs.locomotion.go2_warp_variants import go2_config
    with pytest.raises(ValueError, match="var_impedance"):
        go2_config(controller="osc", damping_action=True)


def test_variants_file_is_import_light():
    """The variants FILE must import only stdlib + ml_collections.

    Loads the file directly (importlib, by path) instead of via the package:
    `jax_rl/__init__.py` eagerly imports networks/algos (-> flax -> jax), so a
    package-path import would measure the parent init, not this file."""
    import sys, subprocess
    from jax_rl.envs.locomotion import go2_warp_variants
    path = go2_warp_variants.__file__
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, importlib.util; "
         f"spec = importlib.util.spec_from_file_location('go2_warp_variants', {path!r}); "
         "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); "
         "bad = [m for m in ('mujoco', 'jax', 'mujoco_playground') if m in sys.modules]; "
         "print(','.join(bad))"],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"heavy imports leaked: {out.stdout}"
