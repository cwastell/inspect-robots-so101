"""Offline RGB-D validation with deliberately inconsistent display/native depth."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from inspect_robots_so101.rgbd import RGBDSnapshot


def packet(**changes: Any) -> RGBDSnapshot:
    args: dict[str, Any] = {
        "rgb": np.zeros((4, 6, 3), dtype=np.uint8),
        "depth_m": np.full((4, 6), 8.0),
        "xyz_depth_m": np.array([[0.1, 0.2, 1.0], [0.2, 0.3, 2.0]]),
        "color_uv": np.array([[0.25, 0.25], [0.75, 0.75]]),
        "xyz_color_m": np.array([[0.3, 0.4, 1.1], [0.4, 0.5, 2.1]]),
        "intrinsics": {"color": {"fx": 100.0, "coeffs": [0, 0, 0, 0, 0]}},
        "extrinsics": {"rotation": np.eye(3), "translation_m": [0.2, 0.2, 0.1]},
        "timing": {"color_frame": 123, "depth_frame": 122, "domain": "device", "missing": None},
        "received_at_s": 10.0,
    }
    return RGBDSnapshot(**(args | changes))


def test_owns_immutable_data_and_metadata() -> None:
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    depth = np.ones((4, 6))
    coeffs = [0, 0]
    rotation = np.eye(3)
    snapshot = packet(rgb=rgb, depth_m=depth, intrinsics={"coeffs": coeffs, "K": rotation})
    rgb[:] = 255
    depth[:] = 22
    coeffs[0] = 100
    rotation[:] = 20
    assert snapshot.rgb.max() == 0
    assert snapshot.depth_m.max() == 1
    assert snapshot.intrinsics["coeffs"] == (0, 0)
    np.testing.assert_array_equal(snapshot.intrinsics["K"], np.eye(3))
    for array in (
        snapshot.rgb,
        snapshot.depth_m,
        snapshot.xyz_depth_m,
        snapshot.color_uv,
        snapshot.xyz_color_m,
        snapshot.intrinsics["K"],
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        snapshot.intrinsics["bad"] = 1  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        snapshot.received_at_s = 0  # type: ignore[misc]


def test_holes_are_invalid_not_zero_distance() -> None:
    depth = np.array([[0.0, -1, np.nan, np.inf, -np.inf, 1.5]] * 4)
    snapshot = packet(depth_m=depth)
    assert np.isnan(snapshot.depth_m[:, :5]).all()
    assert (snapshot.depth_m[:, 5] == 1.5).all()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"rgb": np.zeros((4, 6, 3))}, "rgb must be uint8"),
        ({"rgb": np.zeros((4, 6), dtype=np.uint8)}, "rgb must be uint8"),
        ({"rgb": np.zeros((4, 6, 4), dtype=np.uint8)}, "rgb must be uint8"),
        ({"rgb": np.zeros((0, 6, 3), dtype=np.uint8)}, "nonempty"),
        ({"depth_m": np.zeros((4, 6), dtype=np.uint16)}, "floating-point metres"),
        ({"depth_m": np.zeros((4, 6, 1))}, "floating-point metres"),
        ({"xyz_depth_m": np.ones(3)}, "xyz_depth_m"),
        ({"xyz_depth_m": np.ones((2, 4))}, "xyz_depth_m"),
        ({"color_uv": np.ones((2, 3))}, "color_uv"),
        ({"xyz_color_m": np.ones((2, 2))}, "xyz_color_m"),
        ({"color_uv": np.ones((1, 2))}, "one-for-one"),
        ({"xyz_color_m": np.ones((1, 3))}, "one-for-one"),
        ({"received_at_s": np.nan}, "host timestamp"),
        ({"intrinsics": {1: "bad"}}, "keys must be strings"),
        ({"timing": {"bad": object()}}, "unsupported metadata"),
        ({"intrinsics": {"bad": np.array([object()])}}, "Python objects"),
        ({"robot_from_camera": np.eye(3)}, "finite 4x4"),
        ({"robot_from_camera": np.full((4, 4), np.nan)}, "finite 4x4"),
        ({"robot_from_camera": np.zeros((4, 4))}, "without scale/reflection"),
        ({"robot_from_camera": np.diag([2, 1, 1, 1])}, "without scale/reflection"),
        ({"robot_from_camera": np.diag([-1, 1, 1, 1])}, "without scale/reflection"),
    ],
)
def test_rejects_invalid_packet(changes: dict[str, Any], message: str) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        packet(**changes)


def test_extra_is_eager_and_bound_to_one_packet() -> None:
    snapshot = packet()
    extra = snapshot.observation_extra("realsense")
    assert extra["realsense_depth"] is snapshot.depth_m
    assert not callable(extra["realsense_depth"])
    assert extra["realsense_intrinsics"] is snapshot.intrinsics
    assert extra["realsense_rgbd"] is snapshot
    assert extra["realsense_calibration"]["robot_from_camera"] is None
    assert extra["realsense_timing"]["color_frame"] == 123
    assert extra["realsense_timing"]["received_at_s"] == 10
    with pytest.raises(ValueError, match="camera_name"):
        snapshot.observation_extra("")


def test_native_geometry_not_display_depth_and_roi_is_half_open() -> None:
    snapshot = packet()
    result = snapshot.measure_region((0, 0, 4.5, 4))
    assert result["valid_point_count"] == 1
    assert result["camera_surface_median_m"] == [0.3, 0.4, 1.1]
    assert result["robot_surface_median_m"] is None
    assert result["height_above_plane_m"] is None
    assert result["measurement"] == "visible_surface_not_object_center"
    result = snapshot.measure_region((0, 0, 6, 4))
    np.testing.assert_allclose(result["camera_surface_median_m"], [0.35, 0.45, 1.6])


def test_rejects_invalid_native_points_before_transformed_xyz() -> None:
    count = 9
    native = np.ones((count, 3))
    native[0] = 0  # Transformed zero-depth points look valid after translation.
    native[1, 2] = -1
    native[2, 0] = np.nan
    uv = np.full((count, 2), 0.5)
    uv[3, 0] = -0.1
    uv[4, 0] = 1
    uv[5, 1] = np.nan
    color = np.ones((count, 3))
    color[6, 2] = 0
    color[7, 1] = np.inf
    snapshot = packet(xyz_depth_m=native, color_uv=uv, xyz_color_m=color)
    result = snapshot.measure_region((0, 0, 6, 4))
    assert result["valid_point_count"] == 1
    assert result["camera_surface_median_m"] == [1, 1, 1]


def test_empty_region_has_no_fabricated_coordinates() -> None:
    result = packet().measure_region((0, 0, 1, 1), plane=([0, 0, -1], 1))
    assert result["valid_point_count"] == 0
    assert result["camera_surface_median_m"] is None
    assert result["robot_surface_median_m"] is None
    assert result["height_above_plane_m"] is None
    snapshot = packet(
        xyz_depth_m=np.empty((0, 3)),
        color_uv=np.empty((0, 2)),
        xyz_color_m=np.empty((0, 3)),
    )
    assert snapshot.measure_region((0, 0, 6, 4))["valid_point_count"] == 0


def test_calibrated_robot_transform_and_signed_normalized_plane_height() -> None:
    transform = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1.0]])
    snapshot = packet(robot_from_camera=transform)
    transform[:] = 0  # The snapshot owns the calibration copy.
    result = snapshot.measure_region((0, 0, 6, 4), plane=([0, 0, -2], 4))
    np.testing.assert_allclose(result["robot_surface_median_m"], [0.55, 2.35, 4.6])
    heights = result["height_above_plane_m"]
    assert heights["p50"] == pytest.approx(0.4)
    assert heights["p05"] == pytest.approx(-0.05)
    assert heights["p95"] == pytest.approx(0.85)
    assert heights["p99"] == pytest.approx(0.89)
    assert heights["max"] == pytest.approx(0.9)
    assert snapshot.robot_from_camera is not None
    with pytest.raises(ValueError):
        snapshot.robot_from_camera.setflags(write=True)


@pytest.mark.parametrize(
    "rect",
    [(0, 0, 1), (0, 0, np.nan, 4), (0, 0, 7, 4), (2, 0, 1, 4), (0, -1, 6, 4)],
)
def test_invalid_regions(rect: Any) -> None:
    with pytest.raises(ValueError, match="rect"):
        packet().measure_region(rect)


@pytest.mark.parametrize(
    "plane",
    [([0, 1], 0), ([0, np.nan, 1], 0), ([0, 0, 1], np.inf), ([0, 0, 0], 0)],
)
def test_invalid_planes(plane: Any) -> None:
    with pytest.raises(ValueError, match="plane"):
        packet().measure_region((0, 0, 6, 4), plane=plane)
