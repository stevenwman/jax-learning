"""Config-driven wrapper pipeline builder.

Reads TrainConfig and returns an ordered list of wrappers to apply.
Adding a new wrapper = (1) write the class, (2) add config field, (3) add entry here.
No changes to env_setup.py or record_video.py needed.
"""
from __future__ import annotations
from typing import Any


def build_wrapper_pipeline(cfg) -> list[tuple[str, Any, dict]]:
    """Return ordered list of (name, wrapper_cls, kwargs) from config.

    Wrappers are applied in the returned order, before wrap_for_training().
    Order matters: action-modifying wrappers first, obs-modifying wrappers second.

    Args:
        cfg: TrainConfig or dict (from meta.json["train_config"]).
    """
    def _get(key, default=None):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    pipeline = []

    # 1. Action delay (modifies actions going in)
    action_delay_ms = _get("action_delay_ms", 0)
    action_delay_range_ms = _get("action_delay_range_ms", None)
    if action_delay_range_ms is not None or action_delay_ms > 0:
        from jax_rl.envs.wrappers.action_delay import ActionDelayWrapper
        pipeline.append((
            "action_delay",
            ActionDelayWrapper,
            {"delay_ms": action_delay_ms, "delay_range_ms": action_delay_range_ms},
        ))

    # 2. Frame stacking (modifies obs coming out)
    n_frame_stack = _get("n_frame_stack", 1)
    if n_frame_stack > 1:
        from jax_rl.envs.wrappers.frame_stack import FrameStackWrapper
        pipeline.append((
            "frame_stack",
            FrameStackWrapper,
            {"n_frames": n_frame_stack},
        ))

    return pipeline


def apply_wrapper_pipeline(env, cfg) -> Any:
    """Apply all wrappers from config to env. Returns wrapped env."""
    for name, wrapper_cls, kwargs in build_wrapper_pipeline(cfg):
        env = wrapper_cls(env, **kwargs)
    return env
