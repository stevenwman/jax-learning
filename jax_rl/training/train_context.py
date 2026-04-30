from dataclasses import dataclass
from typing import Any, Optional

from jax_rl.configs.train_config import TrainConfig
from jax_rl.training.checkpointing import CheckpointManager


@dataclass
class TrainContext:
    """Bundle of training state threaded through eval/checkpoint functions."""
    cfg: TrainConfig
    algo_cfg: Any
    algo_name: str
    ckpt_dir: str
    obs_dim: int
    action_dim: int
    metrics_log: list[dict]
    ckpt_mgr: CheckpointManager
    resume: str | None = None
    backend_kind: str = "mjx"   # "mjx" | "gym" | "isaaclab" — dispatches eval path
    # Pre-loaded env handle from EnvBundle. Threaded into save_checkpoint so it
    # can read env metadata (DR specs / obs schema / control) without
    # re-importing mujoco_playground.registry. Optional for back-compat —
    # save_checkpoint falls back to the registry probe when this is None.
    env: Optional[Any] = None
