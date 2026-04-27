"""TDMPC2 — model-based RL with learned world model + MPPI planner.

Re-exports the public API from focused submodules:
- networks: Encoder, Dynamics, Reward, QEnsemble, PolicyPrior + helpers
- losses: compute_all_latents, compute_td_target, world_model_loss, policy_loss
- mppi: make_plan_batched + planner internals
- agent: TDMPC2State, make_update_step, build_*_optimizer

Test internals (NormedLinear, mppi_rollout, etc.) are also re-exported here
because `tests/test_tdmpc2.py` imports them via this package path.
"""

from jax_rl.algos.tdmpc2.networks import (
    NormedLinear,
    Encoder, Dynamics, Reward, QEnsemble, PolicyPrior,
    bound_log_std, squash_log_prob_correction, gaussian_log_prob,
    compute_scaled_entropy,
)
from jax_rl.algos.tdmpc2.losses import (
    compute_all_latents, compute_td_target,
    world_model_loss, policy_loss,
)
from jax_rl.algos.tdmpc2.mppi import (
    make_plan_batched,
    mppi_rollout, mppi_iteration, sample_pi_trajectories,
    init_mppi_mean, init_mppi_mean_batched,
    gumbel_sample_elite, plan,
)
from jax_rl.algos.tdmpc2.agent import (
    TDMPC2State, make_update_step,
    build_world_model_optimizer, build_policy_optimizer,
)

__all__ = [
    "TDMPC2State",
    "make_plan_batched",
    "make_update_step",
    "Encoder", "Dynamics", "Reward", "QEnsemble", "PolicyPrior",
    "build_world_model_optimizer", "build_policy_optimizer",
    "NormedLinear",
    "bound_log_std", "squash_log_prob_correction", "gaussian_log_prob",
    "compute_scaled_entropy",
    "compute_all_latents", "compute_td_target",
    "world_model_loss", "policy_loss",
    "mppi_rollout", "mppi_iteration", "sample_pi_trajectories",
    "init_mppi_mean", "init_mppi_mean_batched",
    "gumbel_sample_elite", "plan",
]
