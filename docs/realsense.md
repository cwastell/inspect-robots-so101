# RGBD observations and offline proof

An injected SO-ARM driver can now return a `RGBDSnapshot` in place of a camera's
RGB array. The embodiment forwards its RGB, metric depth and metadata without
reading the camera again. This supports saved RealSense captures without adding
the SDK, LeRobot, torch or motor access to the package's base dependencies.

## Observation contract

Import `RGBDSnapshot` from `inspect_robots_so101.rgbd`. Its keyword-only fields
are:

| Field | Meaning |
| --- | --- |
| `rgb` | uint8 H by W by 3, RGB channel order |
| `depth_m` | Floating-point H by W color-aligned depth in metres |
| `xyz_depth_m` | N by 3 SDK-native depth optical points in metres |
| `color_uv` | N by 2 normalized SDK texture coordinates |
| `xyz_color_m` | The same native points transformed into the color optical frame |
| `intrinsics`, `extrinsics` | Calibration metadata for this capture |
| `timing` | Original frame IDs, timestamps and clock domains |
| `received_at_s` | Host monotonic receipt time, not exposure time |
| `robot_from_camera` | Optional calibrated 4 by 4 rigid transform; unknown by default |

The caller must supply one coherent frame pair. The class copies and freezes
arrays and metadata so later camera-buffer reuse cannot change an observation.
It cannot detect a caller passing individually valid arrays from different
frames. A live reader must construct the packet from a single SDK frameset and
retain pairing evidence.

Zero, negative and nonfinite depth samples become NaN holes. Raw uint16 depth
is rejected: scale Z16 values to metres explicitly before constructing a packet.
RGB channel order must be established at acquisition or file decoding.

The injected driver's observation mapping still contains all six
`<motor>.pos` values. Its camera entry contains the packet:

```python
raw = {f"{motor}.pos": value for motor, value in zip(motor_names, joint_values)}
raw["front"] = packet
```

`SOArmEmbodiment._observe()` keeps `packet.rgb` in `Observation.images` and
`packet.received_at_s` in `image_times`. Source exposure/device timestamps and
clock domains remain in `extra["front_timing"]`. The joint state retains the
driver-read timestamp; the camera timestamp does not claim synchronization
between encoders and camera exposures.

The extras are `front_depth`, `front_intrinsics`, `front_calibration`,
`front_timing` and `front_rgbd`. The last is an in-process geometry handle,
not a JSON dataset record. A future recorder must persist arrays and metadata
explicitly. The existing Inspect Robots agent can render `front_depth` when
depth rendering is enabled. It does not automatically read calibration metadata
or acquire a numerical geometry tool from these keys.

## Numerical measurements

```python
surface = packet.measure_region((x0, y0, x1, y1))
height = packet.measure_region(
    (x0, y0, x1, y1), plane=(table_normal_in_color_frame, table_offset_m)
)
```

Regions are half-open color-image pixel rectangles. Measurements select native
points through SDK texture coordinates, rejecting invalid native depth before
using the transformed XYZ. They do not deproject remapped Z16 using color
intrinsics, which would confuse depth-camera axial depth with color-camera Z.

Results describe visible surfaces, not an object's hidden center or a grasp
pose. Signed plane heights require a supplied table plane with the normal
pointing above the table. Robot-frame medians remain null unless a calibrated
robot transform is supplied. A missing transform never means identity.

## Offline checks

Use an isolated environment. Do not install the `lerobot` extra for this proof
or into an existing robot environment. The upstream hardware extra currently
requires LeRobot below 0.6 and is separate from the SDK-free packet path.

```bash
python -m venv .venv
# Activate this environment before the following commands.
python -m pip install -e ".[dev]"
python -m pytest --cov
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

Tests cover immutable storage, invalid samples, units and shapes, region bounds,
plane distances, rigid transforms and RGB-only compatibility. The embodiment
integration test injects a fake driver, blocks hardware imports, rejects motor
commands and verifies reset with `home_pose=None` issues no command.

The saved-data proof is maintained in the parent robot workspace as
`scripts/prove_inspect_rgbd.py`, because its recordings and manual annotations
are private. It passes a saved pair through this embodiment and upstream agent
formatting, reproduces native-point measurements, and writes local provenance,
measurement and image artifacts. It does not call an LLM API. Synthetic joint
positions in that proof exercise the interface only; they are not recorded
demonstrations and must not be used for training.

Reference revisions used by this integration:

- SO-ARM adapter base: `096d89e12a9e31228fa57a7e5b7d236b09f3db6e`.
- Inspect Robots core and agent source: `d08442a9d1f43af4658c8d71e02d461e780286e1`.

## Next integration stage

Add a passive live frameset reader, verify camera-to-robot registration, and
provide the teacher with an explicit numerical measurement tool. Cartesian
commands also need a calibrated SO-101 kinematics/controller adapter; the
upstream embodiment remains joint-position-only. A timestamped episode recorder
and successful supervised real attempts are prerequisites for student training.
This proof changes neither motor control nor trained policy weights.
