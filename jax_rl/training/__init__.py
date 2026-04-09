"""Shared training infrastructure — env setup, checkpointing, episode tracking, logging, eval.

Replaces the duplicated boilerplate across train_*.py scripts.
Each utility is a stateless function or lightweight class. No Trainer base class.
"""

from jax_rl.training.checkpointing import save_checkpoint, load_checkpoint, load_actor_for_inference
from jax_rl.training.cli_utils import apply_cli_overrides
from jax_rl.training.episode_tracker import EpisodeTracker
from jax_rl.training.env_setup import make_envs, make_identity_norm_state
from jax_rl.training.metrics_logger import log_training_step, make_metrics_row
from jax_rl.training.eval_runner import maybe_eval_and_checkpoint, final_eval_and_checkpoint
from jax_rl.training.obs_pipeline import ObsPipeline
from jax_rl.training.train_context import TrainContext
