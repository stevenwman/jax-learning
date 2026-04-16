"""Curriculum joystick env for Unitree Go2 (Warp backend).

Extends WarpJoystick with a procedurally generated terrain grid.  The terrain
MJCF is injected into a scene template at init time; the composed XML is
written to a PID-scoped temp file under xmls/ to avoid multiprocess races.

Terrain layout (GO2_DEFAULT_CFG): 10 rows × 4 cols, tile size 9.6×9.6 m.
Row 0 = flat/easy, row 9 = hardest.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax.numpy as jp
from ml_collections import config_dict

from jax_rl.envs.locomotion import go2_warp_base
from jax_rl.envs.locomotion.go2_warp_joystick import (
    WarpJoystick,
    default_config as _warp_default_config,
)
from jax_rl.envs.terrains.config import GO2_DEFAULT_CFG
from jax_rl.envs.terrains.generator import TerrainGenerator

_TEMPLATE_PATH = Path(__file__).parent / "xmls" / "go2_warp_curriculum_scene_template.xml"


def default_config() -> config_dict.ConfigDict:
    cfg = _warp_default_config()
    cfg.torque_speed_model = False
    cfg.terrain_seed = 0
    return cfg


class WarpJoystickCurriculum(WarpJoystick):
    """Joystick velocity tracking on a procedurally generated terrain grid.

    At init, the terrain MJCF is injected into a scene template and the
    composed XML is written to a PID-scoped file.  The grandparent
    ``Go2WarpEnv.__init__`` is called directly to bypass WarpJoystick's
    hardcoded flat-scene path.
    """

    def __init__(
        self,
        task: str = "flat_terrain",
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        if config is None:
            config = default_config()

        # Generate terrain MJCF fragment + spawn origins.
        seed = int(getattr(config, "terrain_seed", 0))
        gen = TerrainGenerator(GO2_DEFAULT_CFG)
        terrain_xml, origins = gen.generate(seed=seed)

        # Compose: inject terrain fragment into scene template.
        scene = _TEMPLATE_PATH.read_text()
        scene = scene.replace("<!-- TERRAIN_INJECT_POINT -->", terrain_xml)

        # Write to PID-scoped file to avoid multiprocess collisions.
        scene_path = _TEMPLATE_PATH.parent / f"_generated_curriculum_scene_{os.getpid()}.xml"
        scene_path.write_text(scene)

        # Call grandparent directly — WarpJoystick.__init__ hardcodes the flat
        # scene path, so we skip it and go straight to Go2WarpEnv.__init__.
        go2_warp_base.Go2WarpEnv.__init__(
            self,
            xml_path=str(scene_path),
            config=config,
            config_overrides=config_overrides,
        )
        # _post_init sets _init_q, joint limits, obs spec, reward spec, etc.
        self._post_init()

        # Terrain grid metadata for curriculum logic (Tasks 2.3+).
        self._terrain_origins = jp.array(origins)   # shape (10, 4, 3)
        self._num_rows = GO2_DEFAULT_CFG.num_rows    # 10
        self._num_cols = GO2_DEFAULT_CFG.num_cols    # 4
