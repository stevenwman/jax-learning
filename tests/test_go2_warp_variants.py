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


def test_go2_presets_resolve_and_unknown_raises():
    from jax_rl.configs import env_presets as ep
    from jax_rl.envs.locomotion.go2_warp_variants import GO2_WARP_VARIANTS
    getters = [("fast_sac", ep.get_fast_sac_preset), ("flash_sac", ep.get_flash_sac_preset),
               ("fast_td3", ep.get_fast_td3_preset), ("sac", ep.get_sac_preset),
               ("td3", ep.get_td3_preset)]
    for name in GO2_WARP_VARIANTS:
        for algo_name, g in getters:
            cfg, algo_cfg = g(name)
            assert cfg.env_name == name
        ppo_cfg = ep.get_preset(name)        # PPO getter returns a BARE TrainConfig (no tuple)
        assert ppo_cfg.env_name == name
    with pytest.raises(ValueError):
        ep.get_preset("Go2WarpNopeDoesNotExist")
    for _, g in getters:
        with pytest.raises(ValueError):
            g("Go2WarpNopeDoesNotExist")
    # splitbelt names must NOT raise (excluded family)
    ep.get_fast_sac_preset("Go2WarpSplitbelt")


def test_go2_preset_migrated_train_overrides():
    """Explicit expected values for the train deltas migrated out of the
    legacy preset tables (curriculum reset_mode; PPO Go2 recipe → base)."""
    from jax_rl.configs import env_presets as ep
    # Curriculum entries carried reset_mode="per_step" in every legacy table.
    for name in ("Go2WarpJoystickCurriculum", "Go2WarpJoystickCurriculumTorqueSpeed"):
        assert ep.get_fast_sac_preset(name)[0].reset_mode == "per_step"
        assert ep.get_flash_sac_preset(name)[0].reset_mode == "per_step"
        assert ep.get_preset(name).reset_mode == "per_step"
    # Non-curriculum variants keep base defaults at this stage (Task 4 changes some).
    cfg, _ = ep.get_fast_sac_preset("Go2WarpJoystickFlat")
    assert cfg.reset_mode == "legacy" and cfg.eval_every_n_episodes == 5000
    # PPO Go2 recipe moved verbatim into _GO2_PPO_BASE_CFG — pin headline fields.
    ppo_cfg = ep.get_preset("Go2WarpJoystickFlat")
    assert ppo_cfg.total_timesteps == 100_000_000
    assert ppo_cfg.num_envs == 4096
    assert ppo_cfg.gamma == 0.97
    assert ppo_cfg.ppo.policy_hidden_dim == (512, 256, 128)


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
