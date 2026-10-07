"""Exercise RGB-D observations without importing any hardware backend."""

from __future__ import annotations

import sys
from unittest.mock import Mock

import numpy as np
import pytest
from inspect_robots.scene import Scene

from conftest import FakeDriver, _operator
from inspect_robots_so101.config import SOArmConfig
from inspect_robots_so101.embodiment import SOArmEmbodiment
from inspect_robots_so101.rgbd import RGBDSnapshot


def snapshot() -> RGBDSnapshot:
    return RGBDSnapshot(
        rgb=np.full((4, 4, 3), [11, 22, 33], dtype=np.uint8),
        depth_m=np.full((4, 4), 0.8),
        xyz_depth_m=np.array([[0.1, 0.2, 0.8]]),
        color_uv=np.array([[0.5, 0.5]]),
        xyz_color_m=np.array([[0.1, 0.2, 0.8]]),
        intrinsics={"width": 4, "height": 4, "fx": 2.0, "fy": 2.0},
        extrinsics={"source": "synthetic"},
        timing={"color_number": 7, "depth_number": 9},
        received_at_s=10.0,
    )


def test_rgbd_and_rgb_cameras_preserve_distinct_timestamps(monkeypatch) -> None:
    packet = snapshot()

    class RGBDDriver(FakeDriver):
        def get_observation(self):
            raw = super().get_observation()
            raw["front"] = packet
            raw["wrist"] = np.zeros((4, 4, 3), dtype=np.uint8)
            return raw

        def send_action(self, _action):
            raise AssertionError("Observation proof must not command motors")

    forbidden_factory = Mock(side_effect=AssertionError("No hardware factory allowed"))
    monkeypatch.setattr(
        "inspect_robots_so101.embodiment._default_driver_factory", forbidden_factory
    )
    for module in ("lerobot", "pyrealsense2", "serial", "torch"):
        monkeypatch.setitem(sys.modules, module, None)
    driver = RGBDDriver()
    with SOArmEmbodiment(
        SOArmConfig(cameras=("front", "wrist"), cam_height=4, cam_width=4, home_pose=None),
        driver_factory=lambda _: driver,
        operator=_operator(),
        poll_end=lambda: False,
        clock=lambda: 20.0,
    ) as emb:
        observation = emb.reset(Scene(id="offline", instruction="Observe only"))
        assert observation.images["front"] is packet.rgb
        assert observation.extra["front_depth"] is packet.depth_m
        assert observation.extra["front_rgbd"] is packet
        assert observation.extra["front_timing"]["depth_number"] == 9
        assert observation.image_times == {"front": 10.0, "wrist": 20.0}
        assert observation.state_time == 20.0
        assert "wrist_depth" not in observation.extra
        assert driver.commands == []
        assert driver.observation_reads == 1
    forbidden_factory.assert_not_called()
    assert driver.disconnected


def test_rgbd_observation_still_validates_configured_image_size() -> None:
    driver = FakeDriver()
    raw = driver.get_observation()
    raw["front"] = snapshot()
    driver.get_observation = lambda: raw
    with (
        SOArmEmbodiment(
            SOArmConfig(cam_height=8, cam_width=4),
            driver_factory=lambda _: driver,
            operator=_operator(),
        ) as emb,
        pytest.raises(ValueError, match="returned shape"),
    ):
        emb.reset(Scene(id="bad-size", instruction="Observe only"))
