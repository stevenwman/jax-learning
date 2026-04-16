import jax
import jax.numpy as jnp


def test_env_loads():
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    env = WarpJoystickCurriculum()
    assert env.action_size == 12
    assert env._terrain_origins.shape == (10, 4, 3)
    assert env._num_rows == 10
    assert env._num_cols == 4


def test_env_model_has_terrain_geoms():
    """Composed MJCF should have many more geoms than flat env."""
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    env = WarpJoystickCurriculum()
    # Base robot has ~100 geoms; terrain adds ~1400 (40 tiles × avg ~35 geoms)
    assert env.mj_model.ngeom > 500


def test_generated_scene_file_written():
    """Confirm the generated MJCF file exists after env init."""
    from pathlib import Path
    import os
    from jax_rl.envs.locomotion.go2_warp_curriculum import WarpJoystickCurriculum
    env = WarpJoystickCurriculum()
    expected = Path(__file__).parent.parent / "jax_rl" / "envs" / "locomotion" / "xmls" / f"_generated_curriculum_scene_{os.getpid()}.xml"
    assert expected.exists()
