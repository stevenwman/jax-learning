"""Unit tests for the Terrain component (Flat / RoughHF / terrain_from_config).

CPU-only — MjSpec compile, no Warp/GPU. Locks the contract that Terrain
reproduces the former `_RoughHFMixin` terrain exactly: the rough heightfield is
config-driven (no second scene XML), and the compiled `hfield_data` is
BIT-IDENTICAL to the reference elevation formula despite the MjSpec compiler's
[0,1] userdata renormalization (the post-compile poke overwrites it).
"""
import numpy as np
import pytest
import mujoco

from jax_rl.envs.locomotion import go2_constants as consts
from jax_rl.envs.locomotion.go2_warp_base import get_warp_assets
from jax_rl.envs.locomotion.go2_warp_components import (
    Flat, RoughHF, Terrain, terrain_from_config, _make_heightfield,
)


# ── terrain_from_config dispatch (legacy config-key bridge) ──────────────────
class _Cfg:
    """Minimal stand-in for an ml_collections config (getattr access)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_from_config_no_rough_key_is_flat():
    assert isinstance(terrain_from_config(_Cfg()), Flat)


def test_from_config_with_rough_profile_is_roughhf_with_params():
    t = terrain_from_config(_Cfg(rough_profile="uniform", rough_amplitude=0.05,
                                 rough_seed=7))
    assert isinstance(t, RoughHF)
    assert (t.profile, t.amplitude, t.seed) == ("uniform", 0.05, 7)


def test_from_config_rough_defaults():
    t = terrain_from_config(_Cfg(rough_profile="perlin_hf"))
    assert (t.amplitude, t.seed) == (0.07, 0)   # documented defaults


# ── RoughHF elevation: deterministic + amplitude-scaled ──────────────────────
def test_elevation_is_seed_deterministic():
    a = RoughHF("uniform", 0.07, seed=3)._elevation(160, 160, 0.10)
    b = RoughHF("uniform", 0.07, seed=3)._elevation(160, 160, 0.10)
    np.testing.assert_array_equal(a, b)
    assert RoughHF("uniform", 0.07, seed=3)._elevation(160, 160, 0.10).max() != \
        RoughHF("uniform", 0.07, seed=4)._elevation(160, 160, 0.10).max()


def test_elevation_scales_to_amplitude_over_zmax():
    # stored hfield value = noise01 * amplitude/zmax; physical height = value*zmax.
    amp, zmax = 0.07, 0.10
    data = RoughHF("uniform", amp, seed=0)._elevation(160, 160, zmax)
    ref = _make_heightfield("uniform", 160, 160, 0) * (amp / zmax)
    np.testing.assert_array_equal(data, ref.astype(np.float32))
    assert (data * zmax).max() == pytest.approx(amp * _make_heightfield(
        "uniform", 160, 160, 0).max(), rel=1e-6)


# ── Flat is a pure no-op on the spec + model ─────────────────────────────────
def test_flat_is_noop():
    spec = mujoco.MjSpec.from_string(consts.WARP_SCENE_FLAT_XML.read_text(),
                                     get_warp_assets())
    Flat().apply(spec)
    spec.meshdir = ""
    for k, v in get_warp_assets().items():
        spec.assets[k] = v
    m = spec.compile()
    # floor stays a plane (type 0), no hfield added.
    assert m.geom_type[m.geom("floor").id] == mujoco.mjtGeom.mjGEOM_PLANE
    assert m.nhfield == 0
    Flat().customize_model(m)   # must not raise


# ── RoughHF end-to-end: compiled hfield_data bit-identical to reference ───────
def _build_rough(profile="uniform", amp=0.07, seed=0):
    assets = get_warp_assets()
    spec = mujoco.MjSpec.from_string(consts.WARP_SCENE_FLAT_XML.read_text(), assets)
    spec.meshdir = ""
    for k, v in assets.items():
        spec.assets[k] = v
    terr = RoughHF(profile, amp, seed)
    terr.apply(spec)
    m = spec.compile()
    terr.customize_model(m)
    return m


def test_roughhf_floor_becomes_hfield():
    m = _build_rough()
    assert m.geom_type[m.geom("floor").id] == mujoco.mjtGeom.mjGEOM_HFIELD
    assert m.nhfield == 1
    hid = m.hfield("rough").id
    assert (int(m.hfield_nrow[hid]), int(m.hfield_ncol[hid])) == (160, 160)


def test_roughhf_hfield_data_bit_identical_to_reference():
    # The exact surface the old _RoughHFMixin produced: noise01 * amp/zmax,
    # poked post-compile. Must survive the compiler's userdata renormalization.
    m = _build_rough("uniform", 0.07, 0)
    hid = m.hfield("rough").id
    adr = int(m.hfield_adr[hid])
    n = int(m.hfield_nrow[hid]) * int(m.hfield_ncol[hid])
    got = np.array(m.hfield_data[adr:adr + n])
    zmax = float(m.hfield_size[hid, 2])
    ref = (_make_heightfield("uniform", 160, 160, 0) * (0.07 / zmax)).astype(np.float32)
    np.testing.assert_array_equal(got, ref.flatten())
    # max ≈ 0.7 (= amp/zmax × noise_max), decisively NOT renormalized to 1.0 —
    # which is what an in-spec userdata (without the post-compile poke) would give.
    assert 0.65 < got.max() < 0.72
