"""Hermetic test that ant.xml loads + has expected geometry."""

from pathlib import Path

import mujoco
import pytest
from mujoco import mjx


ANT_XML = Path("jax_rl/envs/locomotion/xmls/ant.xml")


def test_ant_xml_dims():
    """Cheap CPU-only structural check; no GPU required."""
    m = mujoco.MjModel.from_xml_path(str(ANT_XML))
    assert m.nq == 15  # 7 free joint + 8 hinge
    assert m.nv == 14  # 6 free joint vel + 8 hinge vel
    assert m.nu == 8  # 8 actuators
    assert m.nbody == 14  # worldbody + torso + 12 leg parts


@pytest.mark.gpu
@pytest.mark.warp
def test_ant_xml_warp_putmodel():
    """Warp-backed put_model — requires GPU + Warp deps."""
    m = mujoco.MjModel.from_xml_path(str(ANT_XML))
    mx = mjx.put_model(m, impl="warp")
    assert mx.nq == 15
