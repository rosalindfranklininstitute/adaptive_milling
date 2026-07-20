from __future__ import annotations
import typing
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from itertools import product
from traceback import format_exc

import numpy as np
from skimage import transform
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import (
    BitmapPatternDefinition,
    StreamPatternDefinition,
)

if typing.TYPE_CHECKING:
    from os import PathLike
    from numpy.typing import NDArray


@dataclass
class PatternSettings:
    centre_x: float
    centre_y: float
    width: float
    height: float
    depth: float
    bitmap: NDArray[np.float64]
    points: NDArray[typing.Any] | None = None
    scan_direction: str = "TopToBottom"
    passes: int = 0
    interpolate: typing.Literal["bicubic", "bilinear", "nearest"] | None = None


def create_filled_bitmap(
    width: float, height: float, density: float, fill_value: float = 1.0
) -> NDArray[np.float64]:
    return np.full(
        (int(round(height * density)), int(round(width * density))), fill_value
    )


def _resize_bitmap_to_pattern(
    connection, pattern_settings: PatternSettings
) -> NDArray[np.float64 | np.uint8]:
    points = pattern_settings.bitmap

    if points is None:
        raise ValueError(
            "Unable to resize bitmap as FibsemBitmapSettings.bitmap is None"
        )

    # Get pitch to calculate expected pixel size
    rectangle = connection.patterning.create_rectangle(
        center_x=pattern_settings.centre_x,
        center_y=pattern_settings.centre_y,
        width=pattern_settings.width,
        height=pattern_settings.height,
        depth=pattern_settings.depth,
    )

    pitch_y = rectangle.pitch_y
    pitch_x = rectangle.pitch_x

    new_shape = (
        int(round(pattern_settings.height / pitch_y)),
        int(round(pattern_settings.width / pitch_x)),
    )

    # Disable after calculations just in case values are cleared
    rectangle.enabled = False

    if pattern_settings.interpolate == "bicubic":
        order = 3
    elif pattern_settings.interpolate == "bilinear":
        order = 1
    elif pattern_settings.interpolate == "nearest":
        order = 0
    else:
        raise ValueError(f"Invalid interpolate option '{pattern_settings.interpolate}'")

    resized_points = np.empty((*new_shape, 2), dtype=object)

    resized_points[:, :, 0] = transform.resize(
        points[:, :, 0].reshape(points.shape[0], points.shape[1]).astype(np.float64),
        output_shape=new_shape,
        order=order,
        preserve_range=True,
    ).astype(np.float64)
    resized_points[:, :, 1] = transform.resize(
        points[:, :, 1].reshape(points.shape[0], points.shape[1]).astype(np.uint8),
        output_shape=new_shape,
        order=0,
        preserve_range=True,
    ).astype(np.uint8)

    return resized_points


def draw_bitmap_pattern(connection, pattern_settings: PatternSettings):
    connection.patterning.set_default_application_file("Si")

    # Get bitmap from pattern settings
    bitmap_pattern = BitmapPatternDefinition()

    points = pattern_settings.bitmap

    if pattern_settings.interpolate is not None:
        points = _resize_bitmap_to_pattern(
            connection,
            pattern_settings,
        )
    bitmap_pattern.points = points
    pattern = connection.patterning.create_bitmap(
        center_x=pattern_settings.centre_x,
        center_y=pattern_settings.centre_y,
        width=pattern_settings.width,
        height=pattern_settings.height,
        depth=pattern_settings.depth,
        bitmap_pattern_definition=bitmap_pattern,
    )

    pattern.scan_direction = pattern_settings.scan_direction

    # set passes
    if pattern_settings.passes:  # not zero
        original_dwell_time = float(pattern.dwell_time)
        new_dwell_time = pattern.dwell_time * (
            pattern.pass_count / pattern_settings.passes
        )
        pattern.dwell_time = new_dwell_time
        # assert pattern.dwell_time == new_dwell_time, (
        #     "Failed to update pattern.dwell_time"
        # )
        # pattern.pass_count = int(pattern_settings.passes)
        # assert pattern.dwell_time == original_dwell_time, (
        #     "Updating pattern.passes affected pattern.dwell_time"
        # )

        # NB: passes, time, dwell time are all interlinked, therefore can only adjust passes indirectly
        # if we adjust passes directly, it just reduces the total time to compensate, rather than increasing the dwell_time
        # NB: the current must be set before doing this, otherwise it will be out of range

    return pattern


def _update_pattern_settings_from_rectangle(
    connection, pattern_settings: PatternSettings, cleaning: bool = False
) -> PatternSettings:
    points = pattern_settings.bitmap

    if points is None:
        raise ValueError(
            "Unable to resize bitmap as FibsemBitmapSettings.bitmap is None"
        )

    if cleaning:
        # Get pitch to calculate expected pixel size
        pattern = connection.patterning.create_cleaning_cross_section(
            center_x=pattern_settings.centre_x,
            center_y=pattern_settings.centre_y,
            width=pattern_settings.width,
            height=pattern_settings.height,
            depth=pattern_settings.depth,
        )
    else:
        # Get pitch to calculate expected pixel size
        pattern = connection.patterning.create_rectangle(
            center_x=pattern_settings.centre_x,
            center_y=pattern_settings.centre_y,
            width=pattern_settings.width,
            height=pattern_settings.height,
            depth=pattern_settings.depth,
        )
    if pattern_settings.passes:
        pattern.pass_count = pattern_settings.passes

    pattern_settings.passes = pattern.pass_count

    pitch_y = pattern.pitch_y
    pitch_x = pattern.pitch_x

    new_shape = (
        int(round(pattern_settings.height / pitch_y)),
        int(round(pattern_settings.width / pitch_x)),
    )
    # Disable after calculations just in case values are cleared
    pattern.enabled = False

    if pattern_settings.interpolate == "bicubic":
        order = 3
    elif pattern_settings.interpolate == "bilinear":
        order = 1
    elif pattern_settings.interpolate == "nearest":
        order = 0
    else:
        raise ValueError(f"Invalid interpolate option '{pattern_settings.interpolate}'")

    resized_points = np.empty((*new_shape, 2), dtype=object)

    resized_points[:, :, 0] = transform.resize(
        points[:, :, 0].reshape(points.shape[0], points.shape[1]).astype(np.float64),
        output_shape=new_shape,
        order=order,
        preserve_range=True,
    ).astype(np.float64)
    resized_points[:, :, 1] = transform.resize(
        points[:, :, 1].reshape(points.shape[0], points.shape[1]).astype(np.uint8),
        output_shape=new_shape,
        order=0,
        preserve_range=True,
    ).astype(np.uint8)

    # Convert dwell multiplier to dwell time
    resized_points[:, :, 0] *= pattern.dwell_time

    pattern_settings = deepcopy(pattern_settings)
    pattern_settings.bitmap = resized_points

    return pattern_settings


def _bitmap_settings_to_stream_points(
    connection, pattern_settings: PatternSettings, max_passes: int | None = None
) -> NDArray[typing.Any]:
    hfw = connection.beams.ion_beam.horizontal_field_width.value

    # Assuming 16 bit depth
    single_step_size = hfw / 65536

    centre_x = pattern_settings.centre_x * single_step_size
    centre_y = pattern_settings.centre_y * single_step_size  # assuming square steps

    width = pattern_settings.width * single_step_size
    height = pattern_settings.height * single_step_size  # assuming square steps

    y_pixels, x_pixels = pattern_settings.bitmap.shape[:2]
    xy_array = np.asarray(
        [
            [
                (
                    int(round(centre_x / 2 + ((x / x_pixels) * width - (width / 2)))),
                    int(round(centre_y / 2 + ((y / y_pixels) * height - (height / 2)))),
                )
                for x in range(x_pixels)
            ]
            for y in range(y_pixels)
        ],
        dtype=np.uint16,
    )

    stream_points = np.concatenate(
        (xy_array, pattern_settings.bitmap), axis=-1, dtype=object
    )

    # Reorder points to match scan direction
    if pattern_settings.scan_direction == "BottomToTop":
        stream_points = np.flipud(stream_points)
    elif pattern_settings.scan_direction == "TopToBottom":
        pass
    else:
        raise ValueError("Unsupported scan direction")

    if max_passes:
        stream_points = np.repeat(
            stream_points,
            np.round(
                np.linspace(
                    max_passes / 3,
                    max_passes,
                    num=stream_points.shape[0],
                    endpoint=True,
                )
            ).astype(np.uint16),
            axis=0,
        )

    # Snake through points
    stream_points[1::2] = stream_points[1::2, ::-1]

    return stream_points.reshape((-1, 4))


def draw_stream_pattern(
    connection, pattern_settings: PatternSettings, cleaning: bool = False
):
    connection.patterning.set_default_application_file("Si")

    pattern_settings = _update_pattern_settings_from_rectangle(
        connection=connection,
        pattern_settings=pattern_settings,
        cleaning=cleaning,
    )

    # Get bitmap from pattern settings
    stream_pattern = StreamPatternDefinition()
    stream_pattern.bit_depth = 16

    if cleaning:
        max_passes = pattern_settings.passes
        stream_pattern.repeat_count = 1
    else:
        max_passes = None
        stream_pattern.repeat_count = pattern_settings.passes

    points = _bitmap_settings_to_stream_points(
        connection, pattern_settings, max_passes=max_passes
    )

    print(
        f"Estimated stream time: {np.sum(points[:, 2]) * stream_pattern.repeat_count} s"
    )

    stream_pattern.points = points
    pattern = connection.patterning.create_stream(
        center_x=pattern_settings.centre_x,
        center_y=pattern_settings.centre_y,
        stream_pattern_definition=stream_pattern,
    )
    return pattern


def setup_microscope(
    host: str = "127.0.0.1",
    current: float = 20e-9,
    voltage: int = 30000,
    hfw: float = 40e-6,
) -> SdbMicroscopeClient:
    connection = SdbMicroscopeClient()
    connection.connect(host=host, port=7520)
    # Set ion beam (2)
    connection.imaging.set_active_view(2)
    connection.imaging.set_active_device(2)

    connection.patterning.clear_patterns()  # clear any existing patterns

    beam = connection.beams.ion_beam

    # Set HFW
    hfw_limits = beam.horizontal_field_width.limits
    hfw = np.clip(hfw, hfw_limits.min, hfw_limits.max - 10e-6)
    beam.horizontal_field_width.value = hfw

    # Note: can compare to available values via `beam.beam_current.available_values` if necessary
    if current in beam.beam_current.available_values:
        beam.beam_current.value = current
    else:
        raise ValueError(f"Invalid FIB current, please select from {beam.beam_current.available_values}")

    print(f"FIB voltage limits: {beam.high_voltage.limits}")
    # Note: can compare to limits via `beam.high_voltage.limits` if necessary
    beam.high_voltage.value = voltage

    return connection


def run(
    connection: SdbMicroscopeClient,
    mill: bool = False,
    width: float = 10e-6,
    height: float = 0.35e-6,
    depth: float = 5e-6,
    parallel: bool | typing.Literal["single"] = True,
    fill_value: float | typing.Literal["rand"] = 1.0,
    pattern_type: typing.Literal["bitmap", "stream", "stream_cleaning"] = "bitmap",
) -> None:
    connection.patterning.mode = "Parallel" if parallel else "Serial"

    try:
        # Get pitch to calculate expected pixel size
        rectangle = connection.patterning.create_rectangle(
            center_x=0,
            center_y=0,
            width=width,
            height=height,
            depth=depth,
        )

        pitch_x = float(rectangle.pitch_x)
        pitch_y = float(rectangle.pitch_y)
        print(f"Single rectangle time: {rectangle.time} s")
    finally:
        connection.patterning.clear_patterns()

    bitmap_shape = int(round(height / pitch_y)), int(round(width / pitch_x))
    bitmap = np.full((*bitmap_shape, 2), fill_value=np.nan, dtype=object)
    if fill_value == "rand":
        bitmap[:, :, 0] = np.random.rand(*bitmap_shape).astype(np.float64)
    elif isinstance(fill_value, (int, float)):
        bitmap[:, :, 0] = np.full(bitmap_shape, fill_value=fill_value, dtype=np.float64)
    else:
        raise TypeError("Invalid fill_value")
    bitmap[:, :, 1] = np.zeros(bitmap_shape, dtype=np.uint8)

    pattern_settings = PatternSettings(
        0,
        0,
        width=width,
        height=height,
        depth=depth,
        bitmap=bitmap,
        interpolate="bilinear",
    )

    patterns = []
    try:
        if pattern_type == "stream":
            pattern_drawing_function = draw_stream_pattern
        elif pattern_type == "stream_cleaning":
            pattern_drawing_function = partial(draw_stream_pattern, cleaning=True)
        elif pattern_type == "bitmap":
            pattern_drawing_function = draw_bitmap_pattern
        else:
            raise ValueError(f"Invalid pattern type {pattern_type}")
        patterns.append(
            pattern_drawing_function(
                connection=connection, pattern_settings=pattern_settings
            )
        )

        if parallel != "single":
            pattern_settings_2 = deepcopy(pattern_settings)
            pattern_settings_2.scan_direction = "BottomToTop"
            patterns.append(
                pattern_drawing_function(
                    connection=connection, pattern_settings=pattern_settings_2
                )
            )
        if not pattern_type.startswith("stream"):
            # StreamPattern doesn't have a time attribute (how will we deal with this?)
            estimated_milling_time = sum(_.time for _ in patterns)
            print(f"Total milling time estimated at {estimated_milling_time} s")
    except Exception:
        print(f"Failed to create {pattern_type} pattern: {format_exc()}")
    else:
        if mill:
            # for pattern in patterns:
            #     assert pattern.pass_count == passes, (
            #         "Pattern passes were not properly set"
            #     )
            print(f"Starting milling {pattern_type}")
            connection.patterning.run()
            print(f"Completed milling {pattern_type}")
        else:
            print(f"Not milling {pattern_type}")
    finally:
        connection.patterning.clear_patterns()


if __name__ == "__main__":
    connection = setup_microscope(
        host="localhost", current=20e-12, voltage=30000, hfw=40e-6
    )

    mill = False

    for fill_value, parallel, pattern_type in product(
        (1, "rand"), (True, False, "single"), ("bitmap", "stream", "stream_cleaning")
    ):
        experiment_name = f"{pattern_type}_{fill_value}"
        if isinstance(parallel, str):
            experiment_name += f"_{parallel}"
        elif parallel:
            experiment_name += "_parallel"
        else:
            experiment_name += "_serial"

        print(f"Running {experiment_name}")
        run(
            connection=connection,
            mill=mill,
            depth=0.5e-6,
            parallel=parallel,
            fill_value=fill_value,
            pattern_type=pattern_type,
        )
