"""Immutable RGB-D packets and native-point measurements without a camera SDK.

The caller owns acquisition: all inputs must describe one captured frame pair.
Aligned depth is for the agent's image display. Numerical measurements instead
select SDK-native points projected into the color image; aligned Z16 values must
not be treated as color-camera axial depth. Coordinates are optical-frame metres
(x right, y down, z forward), not robot coordinates or object-center estimates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

_Scalar = TypeVar("_Scalar", bound=np.generic)
FloatArray = npt.NDArray[np.float64]
Plane = tuple[npt.ArrayLike, float]


def _readonly(array: npt.NDArray[_Scalar]) -> npt.NDArray[_Scalar]:
    # A bytes-backed copy cannot be made writable again via setflags().
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("metadata mapping keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("metadata arrays must not contain Python objects")
        return _readonly(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported metadata value: {type(value).__name__}")


def _points(value: npt.ArrayLike, columns: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns})")
    return _readonly(array)


def _rigid_transform(value: npt.ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise ValueError("robot_from_camera must be a finite 4x4 rigid transform")
    rotation = array[:3, :3]
    valid = (
        np.allclose(array[3], (0, 0, 0, 1), atol=1e-8, rtol=0)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
        and np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)
    )
    if not valid:
        raise ValueError("robot_from_camera must be a rigid transform without scale/reflection")
    return _readonly(array)


@dataclass(frozen=True, eq=False, kw_only=True)
class RGBDSnapshot:
    """Own one RGB/depth pair, its native points, calibration and capture metadata.

    ``depth_m`` must be a floating-point, color-aligned metric depth image; zero,
    negative and nonfinite values become NaN holes. ``color_uv`` contains normalized
    SDK texture coordinates corresponding one-for-one to both XYZ arrays.
    ``xyz_color_m`` is the native cloud transformed to the color optical frame.
    Native depth validity is checked before these transformed points are used.

    ``timing`` preserves source frame IDs, device timestamps and their clock
    domains. ``received_at_s`` is the host receipt timestamp, not exposure time.
    Passing a robot transform asserts that the caller has calibrated it. Omission
    never implies identity. Metadata is frozen but is not automatically exposed
    to an LLM: an integration must explicitly render the desired fields.
    """

    rgb: npt.NDArray[np.uint8]
    depth_m: npt.NDArray[Any]
    xyz_depth_m: npt.NDArray[Any]
    color_uv: npt.NDArray[Any]
    xyz_color_m: npt.NDArray[Any]
    intrinsics: Mapping[str, Any]
    extrinsics: Mapping[str, Any]
    timing: Mapping[str, Any]
    received_at_s: float
    robot_from_camera: npt.NDArray[Any] | None = None

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must be uint8 with shape (H, W, 3), in RGB channel order")
        if min(rgb.shape[:2]) == 0:
            raise ValueError("rgb must have nonempty height and width")
        depth = np.asarray(self.depth_m)
        if depth.dtype.kind != "f" or depth.shape != rgb.shape[:2]:
            raise ValueError("depth_m must be floating-point metres with shape (H, W)")
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan).astype(np.float64)
        native = _points(self.xyz_depth_m, 3, "xyz_depth_m")
        uv = _points(self.color_uv, 2, "color_uv")
        color = _points(self.xyz_color_m, 3, "xyz_color_m")
        if len(native) != len(uv) or len(native) != len(color):
            raise ValueError("native XYZ, color UV and color XYZ must correspond one-for-one")
        if not np.isfinite(self.received_at_s):
            raise ValueError("received_at_s must be a finite host timestamp")
        object.__setattr__(self, "rgb", _readonly(rgb))
        object.__setattr__(self, "depth_m", _readonly(depth))
        object.__setattr__(self, "xyz_depth_m", native)
        object.__setattr__(self, "color_uv", uv)
        object.__setattr__(self, "xyz_color_m", color)
        object.__setattr__(self, "intrinsics", _freeze(self.intrinsics))
        object.__setattr__(self, "extrinsics", _freeze(self.extrinsics))
        object.__setattr__(self, "timing", _freeze(self.timing))
        if self.robot_from_camera is not None:
            object.__setattr__(self, "robot_from_camera", _rigid_transform(self.robot_from_camera))

    def observation_extra(self, camera_name: str) -> dict[str, Any]:
        """Return eager depth for Inspect Robots plus explicitly named metadata.

        The ``<camera>_rgbd`` snapshot is an in-process geometry handle, not a
        JSON record. Dataset writers must persist arrays and metadata explicitly.
        All returned payload values remain bound to this packet, never a future
        camera frame, even if the dictionary itself is subsequently modified.
        """
        if not camera_name:
            raise ValueError("camera_name must not be empty")
        return {
            f"{camera_name}_depth": self.depth_m,
            f"{camera_name}_intrinsics": self.intrinsics,
            f"{camera_name}_rgbd": self,
            f"{camera_name}_calibration": MappingProxyType(
                {
                    "intrinsics": self.intrinsics,
                    "extrinsics": self.extrinsics,
                    "robot_from_camera": self.robot_from_camera,
                }
            ),
            f"{camera_name}_timing": MappingProxyType(
                {**self.timing, "received_at_s": self.received_at_s}
            ),
        }

    def measure_region(
        self, rect: tuple[float, float, float, float], plane: Plane | None = None
    ) -> dict[str, Any]:
        """Measure visible native surface points inside a half-open color-pixel ROI.

        ``rect=(x0,y0,x1,y1)`` must lie inside the RGB image. A supplied plane is
        ``(normal_xyz, offset_m)`` in the color frame, with its normal pointing
        above the table. Height is signed plane distance after normalization.
        ROI medians describe visible surfaces, not the center or pose of an
        object. Empty regions report null measurements instead of fabricated zeros.
        """
        bounds = np.asarray(rect, dtype=np.float64)
        if bounds.shape != (4,) or not np.isfinite(bounds).all():
            raise ValueError("rect must contain four finite pixel coordinates")
        x0, y0, x1, y1 = bounds
        height, width = self.rgb.shape[:2]
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("rect must be nonempty and inside the color image")
        normalized_plane: tuple[FloatArray, float] | None = None
        if plane is not None:
            normal = np.asarray(plane[0], dtype=np.float64)
            offset = float(plane[1])
            if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isfinite(offset):
                raise ValueError("plane must have a finite 3-D normal and offset")
            norm = float(np.linalg.norm(normal))
            if norm == 0:
                raise ValueError("plane normal must not be zero")
            normalized_plane = (normal / norm, offset / norm)

        native, uv, color = self.xyz_depth_m, self.color_uv, self.xyz_color_m
        valid = np.isfinite(native).all(axis=1) & (native[:, 2] > 0)
        valid &= np.isfinite(uv).all(axis=1) & (uv >= 0).all(axis=1) & (uv < 1).all(axis=1)
        valid &= np.isfinite(color).all(axis=1) & (color[:, 2] > 0)
        x, y = uv[:, 0] * width, uv[:, 1] * height
        valid &= (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        points = color[valid]
        result: dict[str, Any] = {
            "rect_pixels": bounds.tolist(),
            "source": "native_points_projected_into_color",
            "coordinate_frame": "color_optical",
            "units": "metres",
            "measurement": "visible_surface_not_object_center",
            "valid_point_count": len(points),
            "camera_surface_median_m": None,
            "robot_surface_median_m": None,
            "height_above_plane_m": None,
        }
        if len(points) == 0:
            return result
        result["camera_surface_median_m"] = np.median(points, axis=0).tolist()
        if self.robot_from_camera is not None:
            robot_points = points @ self.robot_from_camera[:3, :3].T
            robot_points += self.robot_from_camera[:3, 3]
            result["robot_surface_median_m"] = np.median(robot_points, axis=0).tolist()
        if normalized_plane is not None:
            normal, offset = normalized_plane
            distances = points @ normal + offset
            percentiles = np.percentile(distances, [5, 50, 95, 99])
            result["height_above_plane_m"] = {
                "p05": float(percentiles[0]),
                "p50": float(percentiles[1]),
                "p95": float(percentiles[2]),
                "p99": float(percentiles[3]),
                "max": float(np.max(distances)),
            }
        return result
