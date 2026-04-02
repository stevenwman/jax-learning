"""Unit tests for W&B metric key remapping.

Tests the _WANDB_PREFIX dict transform — no wandb import needed.
"""

from jax_rl.training.metrics_logger import _WANDB_PREFIX


def test_known_keys_get_prefixed():
    """Metrics in _WANDB_PREFIX should be remapped to their prefixed form."""
    row = {"q1_mean": 1.5, "actor_loss": -2.0, "eval_mean": 100.0, "sps": 5000}
    remapped = {_WANDB_PREFIX.get(k, k): v for k, v in row.items()}
    assert remapped == {
        "critic/q1_mean": 1.5,
        "actor/actor_loss": -2.0,
        "perf/eval_mean": 100.0,
        "infra/sps": 5000,
    }


def test_unknown_keys_pass_through():
    """Keys not in _WANDB_PREFIX should pass through unchanged."""
    row = {"total_steps": 10000, "some_future_metric": 42.0}
    remapped = {_WANDB_PREFIX.get(k, k): v for k, v in row.items()}
    assert remapped == {"total_steps": 10000, "some_future_metric": 42.0}


def test_total_steps_not_remapped():
    """total_steps must stay flat — W&B uses it for x-axis via step= arg."""
    assert "total_steps" not in _WANDB_PREFIX


def test_all_sections_present():
    """All 5 dashboard sections should have at least one metric."""
    prefixes = {v.split("/")[0] for v in _WANDB_PREFIX.values()}
    assert prefixes == {"perf", "critic", "actor", "eval", "infra"}


def test_mixed_row_training_metrics():
    """Simulate a real off-policy training row with mixed known/unknown keys."""
    row = {
        "total_steps": 50000,
        "episodes": 100,
        "avg_return": 42.5,
        "q1_mean": 15.0,
        "q2_mean": 14.8,
        "actor_loss": -12.0,
        "grad_steps": 3000,
        "sps": 4500,
        "elapsed": 30.0,
        "entropy": 3.5,
        "alpha": 0.05,
        "alpha_loss": 0.01,
    }
    remapped = {_WANDB_PREFIX.get(k, k): v for k, v in row.items()}

    # total_steps passes through
    assert "total_steps" in remapped
    # known keys are prefixed
    assert "perf/avg_return" in remapped
    assert "critic/q1_mean" in remapped
    assert "actor/entropy" in remapped
    assert "infra/sps" in remapped
    # flat keys should NOT appear for mapped metrics
    assert "avg_return" not in remapped
    assert "q1_mean" not in remapped
    assert "entropy" not in remapped
    assert "sps" not in remapped


def test_eval_metrics_remapped():
    """Eval metrics from evaluate() should land in perf/ and eval/ sections."""
    eval_row = {
        "eval_mean": 233.0,
        "eval_std": 15.0,
        "eval_min": 200.0,
        "eval_max": 260.0,
        "q_bias": 5.2,
        "q_rmse": 6.1,
        "q_corr": 0.95,
        "q_mean": 20.0,
        "mc_mean": 14.8,
    }
    remapped = {_WANDB_PREFIX.get(k, k): v for k, v in eval_row.items()}
    assert "perf/eval_mean" in remapped
    assert "perf/eval_std" in remapped
    assert "eval/q_bias" in remapped
    assert "eval/q_corr" in remapped
