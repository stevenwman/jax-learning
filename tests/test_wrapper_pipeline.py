"""Tests for config-driven wrapper pipeline."""
import pytest
from dataclasses import dataclass

from jax_rl.envs.wrappers.pipeline import build_wrapper_pipeline, apply_wrapper_pipeline


@dataclass
class FakeConfig:
    n_frame_stack: int = 1
    action_delay_ms: int = 0
    action_delay_range_ms: tuple[int, int] | None = None


class TestBuildPipeline:
    def test_empty_config_returns_empty(self):
        cfg = FakeConfig()
        pipeline = build_wrapper_pipeline(cfg)
        assert len(pipeline) == 0

    def test_frame_stack_only(self):
        cfg = FakeConfig(n_frame_stack=3)
        pipeline = build_wrapper_pipeline(cfg)
        assert len(pipeline) == 1
        assert pipeline[0][0] == "frame_stack"

    def test_action_delay_only(self):
        cfg = FakeConfig(action_delay_ms=120)
        pipeline = build_wrapper_pipeline(cfg)
        assert len(pipeline) == 1
        assert pipeline[0][0] == "action_delay"

    def test_both_wrappers_ordered(self):
        cfg = FakeConfig(n_frame_stack=3, action_delay_ms=60)
        pipeline = build_wrapper_pipeline(cfg)
        assert len(pipeline) == 2
        assert pipeline[0][0] == "action_delay"
        assert pipeline[1][0] == "frame_stack"

    def test_dict_config_works(self):
        cfg = {"n_frame_stack": 3, "action_delay_ms": 60}
        pipeline = build_wrapper_pipeline(cfg)
        assert len(pipeline) == 2

    def test_delay_range_overrides_fixed(self):
        cfg = FakeConfig(action_delay_ms=0, action_delay_range_ms=(40, 120))
        pipeline = build_wrapper_pipeline(cfg)
        assert len(pipeline) == 1
        assert pipeline[0][0] == "action_delay"
        assert pipeline[0][2]["delay_range_ms"] == (40, 120)
