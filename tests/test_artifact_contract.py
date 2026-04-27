"""Tests for the checkpoint artifact contract (kind + version + validators).

Phase A stamps `meta["artifact_kind"]` + `meta["artifact_version"]`. Phase B
adds `assert_artifact_kind` for consumer validation. Phase C adds per-kind
on-disk file validators. These tests cover the contract surface
hermetically — no env construction, no GPU.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from jax_rl.training.artifact_contract import (
    KIND_LEGACY_SHARED_ACTOR,
    KIND_SHARED_ACTOR,
    KIND_TDMPC2,
    LATEST_VERSION,
    assert_artifact_kind,
    read_artifact_kind,
    stamp_meta,
    validate_shared_actor_files,
    validate_tdmpc2_files,
)


def test_stamp_meta_adds_kind_and_version():
    m = stamp_meta({}, KIND_SHARED_ACTOR)
    assert m["artifact_kind"] == "shared_actor_v1"
    assert m["artifact_version"] == LATEST_VERSION


def test_stamp_meta_is_idempotent():
    m = stamp_meta({}, KIND_SHARED_ACTOR)
    m2 = stamp_meta(m, KIND_SHARED_ACTOR)
    assert m is m2  # mutates in place
    assert m["artifact_kind"] == "shared_actor_v1"


def test_read_artifact_kind_returns_stamped_value():
    m = stamp_meta({}, KIND_TDMPC2)
    assert read_artifact_kind(m) == "tdmpc2_v1"


def test_read_artifact_kind_falls_back_on_missing(capsys):
    """Pre-Phase-A meta has no kind; reader returns legacy + warns."""
    kind = read_artifact_kind({}, ckpt_path="/tmp/fake")
    assert kind == KIND_LEGACY_SHARED_ACTOR
    captured = capsys.readouterr()
    assert "missing artifact_kind" in captured.out
    assert "/tmp/fake" in captured.out


def test_assert_artifact_kind_passes_on_match():
    m = stamp_meta({}, KIND_SHARED_ACTOR)
    kind = assert_artifact_kind(
        m, allowed=[KIND_SHARED_ACTOR], tool_name="test"
    )
    assert kind == KIND_SHARED_ACTOR


def test_assert_artifact_kind_passes_on_legacy_when_allowed():
    """Legacy ckpts (no field) should pass when KIND_LEGACY_SHARED_ACTOR is in allowed."""
    kind = assert_artifact_kind(
        {}, allowed=[KIND_SHARED_ACTOR, KIND_LEGACY_SHARED_ACTOR], tool_name="test"
    )
    assert kind == KIND_LEGACY_SHARED_ACTOR


def test_assert_artifact_kind_raises_on_mismatch_with_redirect():
    m = stamp_meta({}, KIND_TDMPC2)
    with pytest.raises(ValueError) as exc:
        assert_artifact_kind(
            m,
            allowed=[KIND_SHARED_ACTOR],
            tool_name="record_video.py",
            ckpt_path="/tmp/ckpt",
            redirect="Use record_video_tdmpc2.py instead.",
        )
    msg = str(exc.value)
    assert "record_video.py" in msg
    assert "tdmpc2_v1" in msg
    assert "/tmp/ckpt" in msg
    assert "record_video_tdmpc2.py" in msg


def test_validate_shared_actor_files_passes_on_complete_dir():
    with tempfile.TemporaryDirectory() as td:
        # Create the three required files
        with open(os.path.join(td, "meta.json"), "w") as f:
            json.dump({}, f)
        np.save(os.path.join(td, "actor_params.npy"), {"actor_params": {}})
        os.makedirs(os.path.join(td, "orbax"))
        # Should not raise
        validate_shared_actor_files(td)


def test_validate_shared_actor_files_lists_missing_files():
    with tempfile.TemporaryDirectory() as td:
        # Only meta.json present
        with open(os.path.join(td, "meta.json"), "w") as f:
            json.dump({}, f)
        with pytest.raises(FileNotFoundError) as exc:
            validate_shared_actor_files(td)
        msg = str(exc.value)
        assert "actor_params.npy" in msg
        assert "orbax" in msg


def test_validate_tdmpc2_files_passes_on_complete_dir():
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "meta.json"), "w") as f:
            json.dump({}, f)
        np.savez(os.path.join(td, "actor_params.npz"))
        np.savez(os.path.join(td, "world_model_params.npz"))
        validate_tdmpc2_files(td)


def test_validate_tdmpc2_files_complains_about_npy_when_npz_expected():
    """A shared-actor ckpt should fail tdmpc2 validation (different shape)."""
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "meta.json"), "w") as f:
            json.dump({}, f)
        np.save(os.path.join(td, "actor_params.npy"), {})
        os.makedirs(os.path.join(td, "orbax"))
        # Required for tdmpc2: actor_params.npz + world_model_params.npz, both missing
        with pytest.raises(FileNotFoundError) as exc:
            validate_tdmpc2_files(td)
        msg = str(exc.value)
        assert "actor_params.npz" in msg
        assert "world_model_params.npz" in msg
