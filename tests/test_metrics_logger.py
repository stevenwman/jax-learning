import numpy as np


def test_log_terrain_metrics_returns_empty_dict_when_no_terrain():
    from jax_rl.training.metrics_logger import log_terrain_metrics
    info = {"some_other_key": np.zeros(10)}
    result = log_terrain_metrics(info, terrain_type_names=["rough"])
    assert result == {}


def test_log_terrain_metrics_computes_per_type_metrics():
    from jax_rl.training.metrics_logger import log_terrain_metrics
    info = {
        "terrain_level": np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        "terrain_type":  np.array([0, 0, 1, 1, 0, 1, 0, 1]),
        "episode_reached_goal": np.array([True,  False, True,  False, True,  True,  False, False]),
        "episode_fallen":       np.array([False, True,  False, False, False, False, True,  True]),
        "episode_promoted":     np.array([True,  False, True,  False, False, True,  False, False]),
        "episode_demoted":      np.array([False, True,  False, False, False, False, True,  True]),
    }
    result = log_terrain_metrics(info, terrain_type_names=["rough", "pyramid_up"])
    # Type 0 (rough): env indices [0,1,4,6], levels [0,1,4,6], mean=2.75
    assert abs(result["terrain/rough/mean_level"] - 2.75) < 1e-5
    assert abs(result["terrain/rough/reach_rate"] - 0.5) < 1e-5
    assert abs(result["terrain/rough/fall_rate"] - 0.5) < 1e-5
    assert abs(result["terrain/rough/promote_rate"] - 0.25) < 1e-5
    assert abs(result["terrain/global/mean_level"] - 3.5) < 1e-5


def test_log_terrain_metrics_uses_default_names():
    from jax_rl.training.metrics_logger import log_terrain_metrics
    info = {
        "terrain_level": np.array([0, 1, 2, 3]),
        "terrain_type":  np.array([0, 1, 2, 3]),
    }
    result = log_terrain_metrics(info)
    # Default names: rough, pyramid_up, pyramid_down, tilted
    assert "terrain/rough/mean_level" in result
    assert "terrain/pyramid_up/mean_level" in result
    assert "terrain/pyramid_down/mean_level" in result
    assert "terrain/tilted/mean_level" in result
