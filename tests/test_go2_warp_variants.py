"""tests/test_go2_warp_variants.py — variants-as-data registry tests."""
import pytest


def _deep_eq(a, b, path=""):
    """Deep equality (exact values) for plain dicts/lists/scalars.

    int/float cross-type compares by value (so 1 == 1.0 — and bool, being an
    int subclass, lets True == 1 slip through; acceptable for now)."""
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
    rec(a, b, path)


def test_variant_configs_match_snapshot():
    r"""Every variant's config == the frozen snapshot (regression pin).

    The snapshot was generated from the variants table the moment it was
    proven equal to the legacy per-env config factories (since deleted).
    On an INTENTIONAL config change, regenerate with:

        uv run python -c "import json; from jax_rl.envs.locomotion.go2_warp_variants import GO2_WARP_VARIANTS; open('tests/data/go2_warp_variants_snapshot.json', 'w').write(json.dumps({n: v.config().to_dict() for n, v in GO2_WARP_VARIANTS.items()}, indent=1, sort_keys=True, default=list) + '\n')"

    Both sides are JSON-round-tripped so tuples normalize to lists.
    """
    import json
    import pathlib
    from jax_rl.envs.locomotion.go2_warp_variants import GO2_WARP_VARIANTS
    snap_path = (pathlib.Path(__file__).parent / "data"
                 / "go2_warp_variants_snapshot.json")
    snapshot = json.loads(snap_path.read_text())
    assert snapshot.keys() == GO2_WARP_VARIANTS.keys()
    for name, v in GO2_WARP_VARIANTS.items():
        live = json.loads(json.dumps(v.config().to_dict(), default=list))
        _deep_eq(snapshot[name], live, path=name)


def test_registry_uses_variant_cls():
    from mujoco_playground import registry as pg_registry
    import jax_rl.training.env_backends.mjx_backend  # noqa: F401
    from jax_rl.envs.locomotion.go2_warp_variants import GO2_WARP_VARIANTS
    # construction-only; pick the 5 non-default-cls + 2 default-cls names to keep runtime sane
    check = {n: v for n, v in GO2_WARP_VARIANTS.items()
             if v.cls != "WarpJoystick"} | {
        "Go2WarpJoystickFlat": GO2_WARP_VARIANTS["Go2WarpJoystickFlat"],
        "Go2WarpOscFlatSoftPhysical": GO2_WARP_VARIANTS["Go2WarpOscFlatSoftPhysical"]}
    for name, v in check.items():
        env = pg_registry.load(name)
        assert type(env).__name__ == v.cls, f"{name}: {type(env).__name__} != {v.cls}"


def test_go2_presets_resolve_and_unknown_raises():
    from jax_rl.configs import env_presets as ep
    from jax_rl.configs.fast_sac_config import FastSACConfig
    from jax_rl.configs.flash_sac_config import FlashSACConfig
    from jax_rl.configs.fast_td3_config import FastTD3Config
    from jax_rl.configs.sac_config import SACConfig
    from jax_rl.configs.td3_config import TD3Config
    from jax_rl.configs.train_config import TrainConfig
    from jax_rl.envs.locomotion.go2_warp_variants import GO2_WARP_VARIANTS
    getters = [("fast_sac", ep.get_fast_sac_preset, FastSACConfig),
               ("flash_sac", ep.get_flash_sac_preset, FlashSACConfig),
               ("fast_td3", ep.get_fast_td3_preset, FastTD3Config),
               ("sac", ep.get_sac_preset, SACConfig),
               ("td3", ep.get_td3_preset, TD3Config)]
    for name in GO2_WARP_VARIANTS:
        for algo_name, g, algo_cls in getters:
            cfg, algo_cfg = g(name)
            assert cfg.env_name == name
            assert type(algo_cfg) is algo_cls, f"{name}/{algo_name}: {type(algo_cfg)}"
        ppo_cfg = ep.get_preset(name)        # PPO getter returns a BARE TrainConfig (no tuple)
        assert ppo_cfg.env_name == name
        assert type(ppo_cfg) is TrainConfig
    with pytest.raises(ValueError):
        ep.get_preset("Go2WarpNopeDoesNotExist")
    for _, g, _cls in getters:
        with pytest.raises(ValueError):
            g("Go2WarpNopeDoesNotExist")
    # splitbelt names must NOT raise (excluded family)
    ep.get_fast_sac_preset("Go2WarpSplitbelt")


def test_resolve_go2_variant_routes_algo_overrides(monkeypatch):
    """Direct unit test of the algo-override split in _resolve_go2_variant:
    TrainConfig-field keys land on the TrainConfig, the rest on the algo
    config; a getter without an algo config (PPO) raises on leftover
    algo-level keys instead of crashing on replace(None, ...)."""
    from jax_rl.configs import env_presets as ep
    from jax_rl.envs.locomotion.go2_warp_variants import (
        GO2_WARP_VARIANTS, EnvVariant, go2_config)
    monkeypatch.setitem(
        GO2_WARP_VARIANTS, "Go2WarpSyntheticSplit",
        EnvVariant(config=go2_config,
                   algo={"fast_sac": {"batch_size": 1024, "episode_length": 777}}))
    cfg, algo_cfg = ep.get_fast_sac_preset("Go2WarpSyntheticSplit")
    assert cfg.episode_length == 777        # TrainConfig field → cfg
    assert algo_cfg.batch_size == 1024      # algo field → algo config
    monkeypatch.setitem(
        GO2_WARP_VARIANTS, "Go2WarpSyntheticPpoAlgo",
        EnvVariant(config=go2_config, algo={"ppo": {"batch_size": 1024}}))
    with pytest.raises(ValueError, match="no algo config"):
        ep.get_preset("Go2WarpSyntheticPpoAlgo")


def test_go2_preset_migrated_train_overrides():
    """Explicit expected values for the train deltas migrated out of the
    legacy preset tables (curriculum reset_mode; PPO Go2 recipe → base)."""
    from jax_rl.configs import env_presets as ep
    # Curriculum entries carried reset_mode="per_step" in every legacy table.
    for name in ("Go2WarpJoystickCurriculum", "Go2WarpJoystickCurriculumTorqueSpeed"):
        assert ep.get_fast_sac_preset(name)[0].reset_mode == "per_step"
        assert ep.get_flash_sac_preset(name)[0].reset_mode == "per_step"
        assert ep.get_preset(name).reset_mode == "per_step"
    # Benchmark joint-PD family keeps base defaults (OSC/physical variants
    # switched to per_step/500 — see test_osc_physical_variants_default_per_step_dr).
    cfg, _ = ep.get_fast_sac_preset("Go2WarpJoystickFlat")
    assert cfg.reset_mode == "legacy" and cfg.eval_every_n_episodes == 5000
    # PPO Go2 recipe moved verbatim into _GO2_PPO_BASE_CFG — pin headline fields.
    ppo_cfg = ep.get_preset("Go2WarpJoystickFlat")
    assert ppo_cfg.total_timesteps == 100_000_000
    assert ppo_cfg.num_envs == 4096
    assert ppo_cfg.gamma == 0.97
    assert ppo_cfg.ppo.policy_hidden_dim == (512, 256, 128)


def test_osc_physical_variants_default_per_step_dr():
    """Task 4 behavior change: every OSC / physical-motor / rough-terrain
    variant trains with per-step DR + eval every 500 episodes by default.
    EXCEPTION: the two Curriculum variants keep their historical train dict
    (per_step only, no eval_every key)."""
    from jax_rl.configs import env_presets as ep
    from jax_rl.envs.locomotion.go2_warp_variants import GO2_WARP_VARIANTS
    targets = [n for n in GO2_WARP_VARIANTS
               if ("Osc" in n or n.endswith("Physical") or n.endswith("RoughUni"))
               and "Curriculum" not in n]
    assert len(targets) == 43, sorted(targets)
    for name in targets:
        cfg, _ = ep.get_fast_sac_preset(name)
        assert cfg.reset_mode == "per_step", name
        assert cfg.eval_every_n_episodes == 500, name
    # Curriculum keeps its existing train verbatim — no eval_every key added.
    for name in ("Go2WarpJoystickCurriculum", "Go2WarpJoystickCurriculumTorqueSpeed"):
        assert GO2_WARP_VARIANTS[name].train == {"reset_mode": "per_step"}, name


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


# ── Mud force field (analytic foot wrench) ───────────────────────────────────
def test_mud_foot_force_signs_and_gating():
    """Pure-math unit tests for the mud foot force (no env needed)."""
    import jax.numpy as jp
    import numpy as np
    from jax_rl.envs.locomotion.go2_warp_components import mud_foot_force
    kw = dict(mud_height=jp.array(0.22), coeff=jp.array(14.5),
              m=jp.array(9.5), b=jp.array(6.5), circ=jp.array(0.12))
    # foot submerged, moving UP -> suction (z force DOWN, < 0)
    F_up = mud_foot_force(jp.array(0.0), jp.array([0., 0., 0.5]), **kw)
    assert float(F_up[2]) < 0.0
    # foot submerged, moving DOWN -> resistance (z force UP, > 0)
    F_dn = mud_foot_force(jp.array(0.0), jp.array([0., 0., -0.5]), **kw)
    assert float(F_dn[2]) > 0.0
    # above the mud surface -> no force at all
    F_above = mud_foot_force(jp.array(0.5), jp.array([0., 0., -0.5]), **kw)
    assert np.allclose(np.asarray(F_above), 0.0)
    # shear opposes horizontal motion: foot sliding +x (fast) -> shear -x
    F_slide = mud_foot_force(jp.array(0.0), jp.array([1.0, 0., 0.]), **kw)
    assert float(F_slide[0]) < 0.0
    # yield offset b only kicks in above the 0.3 m/s horizontal threshold:
    # slow slide (0.1 m/s) has strictly smaller |shear_x| than the 0.3+ regime
    F_slow = mud_foot_force(jp.array(0.0), jp.array([0.1, 0., 0.]), **kw)
    F_fast = mud_foot_force(jp.array(0.0), jp.array([0.31, 0., 0.]), **kw)
    assert abs(float(F_fast[0])) > abs(float(F_slow[0]))
    # finite at v=0 (log(|v|+1e-3) guarded)
    F_zero = mud_foot_force(jp.array(0.0), jp.array([0., 0., 0.]), **kw)
    assert np.all(np.isfinite(np.asarray(F_zero)))


def test_field_from_config_dispatch():
    from ml_collections import config_dict
    from jax_rl.envs.locomotion.go2_warp_components import (
        field_from_config, NoField, MudField)
    from jax_rl.envs.locomotion.go2_warp_variants import go2_config
    assert isinstance(field_from_config(go2_config()), NoField)
    assert isinstance(field_from_config(go2_config(
        controller="var_impedance", stiffness_granularity="per_axis",
        damping_action=True, motor="physical",
        mud=dict(depth_range=(0.22, 0.22)))), MudField)


import pytest as _pytest


@_pytest.mark.gpu
def test_mud_field_applies_force_and_nofield_identity():
    """GPU: a mud env writes nonzero xfrc on submerged feet and steps NaN-free;
    the matching NoField env keeps xfrc_applied at exactly zero (bit-identity of
    the force path)."""
    import jax, jax.numpy as jp
    import numpy as np
    from jax_rl.envs.locomotion.go2_warp_joystick import WarpJoystick
    from jax_rl.envs.locomotion.go2_warp_variants import go2_config
    base = dict(controller="var_impedance", stiffness_granularity="per_axis",
                damping_action=True, motor="physical")
    mud_env = WarpJoystick(task="flat_terrain",
                           config=go2_config(mud=dict(depth_range=(0.22, 0.22)), **base))
    st = mud_env.reset(jax.random.PRNGKey(0))
    a = jp.zeros(mud_env.action_size)
    for _ in range(3):
        st = mud_env.step(st, a)
    fb = mud_env._force_field._foot_body_ids
    foot_xfrc = np.asarray(st.data.xfrc_applied[fb, :3])
    assert np.abs(foot_xfrc).max() > 1.0          # mud pushes the feet
    assert not np.isnan(np.asarray(st.obs["state"])).any()

    no_env = WarpJoystick(task="flat_terrain", config=go2_config(**base))
    st2 = no_env.reset(jax.random.PRNGKey(0))
    for _ in range(3):
        st2 = no_env.step(st2, jp.zeros(no_env.action_size))
    assert float(np.abs(np.asarray(st2.data.xfrc_applied)).max()) == 0.0
