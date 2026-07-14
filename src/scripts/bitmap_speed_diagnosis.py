from __future__ import annotations
import typing
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from random import random

import numpy as np
from skimage import transform
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import BitmapPatternDefinition

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

    new_shape = (
        int(round(pattern_settings.height / rectangle.pitch_y)),
        int(round(pattern_settings.width / rectangle.pitch_x)),
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
        print("Resizing")
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
        new_dwell_time = pattern.dwell_time * (
            pattern.pass_count / pattern_settings.passes
        )
        pattern.dwell_time = new_dwell_time
        assert pattern.dwell_time == new_dwell_time, (
            "Failed to update pattern.dwell_time"
        )
        pattern.pass_count = int(pattern_settings.passes)
        assert pattern.dwell_time == new_dwell_time, (
            "Updating pattern.passes affected pattern.dwell_time"
        )

        # NB: passes, time, dwell time are all interlinked, therefore can only adjust passes indirectly
        # if we adjust passes directly, it just reduces the total time to compensate, rather than increasing the dwell_time
        # NB: the current must be set before doing this, otherwise it will be out of range

    return pattern


def plot_3d(data: NDArray[np.float64], path: str | PathLike[str]) -> None:
    def _log_tick_formatter(val, pos=None):
        return f"$10^{{{int(val)}}}$"

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    surface = ax.plot_surface(
        np.log10(data[:, :, 0]),
        data[:, :, 1],
        data[:, :, 2],
        cmap="coolwarm",
        rcount=data.size,
        alpha=0.5,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_log_tick_formatter))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.colorbar(surface, shrink=0.5, aspect=5)
    ax.set_xlabel("Pass Count")
    ax.set_ylabel("Pixel Density Multiplier")
    ax.set_zlabel("Plot Time (s)")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_density(data: NDArray[np.float64], path: str | PathLike[str]) -> None:
    fig, ax = plt.subplots()
    closest_to_one = np.absolute(data[:, 0, 0] - 1).argmin()
    ax.plot(
        data[closest_to_one, :, 1],
        data[closest_to_one, :, 2],
    )
    ax.set_xlabel("Pixel Density Multiplier")
    ax.set_ylabel("Plot Time (s)")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def setup_microscope(
    host: str = "127.0.0.1", current: float = 20e-9
) -> SdbMicroscopeClient:
    connection = SdbMicroscopeClient()
    connection.connect(host=host, port=7520)
    # Set ion beam (2)
    connection.imaging.set_active_view(2)
    connection.imaging.set_active_device(2)

    beam = connection.beams.ion_beam

    # Note: can compare to available values via `beam.beam_current.available_values` if necessary
    beam.beam_current.value = current
    return connection


def run(
    connection: SdbMicroscopeClient,
    output_dir: str | PathLike[str],
    experiment_name: str = "",
    width: float = 10e-6,
    height: float = 0.35e-6,
    depth: float = 5e-6,
    parallel: bool | typing.Literal["single"] = True,
    fill_value: float | typing.Literal["rand"] = 1.0,
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
        rectangle_passes = int(rectangle.pass_count)
    finally:
        connection.patterning.clear_patterns()

    passes_multipliers = np.logspace(-30, 0, num=31)
    passes_multipliers = np.append(passes_multipliers, (2, 3, 4, 5))
    density_multipliers = np.arange(0.05, 3.5, 0.05)

    data = np.empty(
        [len(passes_multipliers), len(density_multipliers), 3], dtype=np.float64
    )
    for i, passes_multiplier in enumerate(passes_multipliers):
        for j, density_multiplier in enumerate(density_multipliers):
            new_shape = (
                int(round((height * density_multiplier) / pitch_y)),
                int(round((width * density_multiplier) / pitch_x)),
            )
            passes = rectangle_passes * passes_multiplier

            bitmap = np.empty((*new_shape, 2), dtype=object)
            if fill_value == "rand":
                bitmap[:, :, 0] = np.random.rand(*new_shape).astype(np.float64)
            elif isinstance(fill_value, (int, float)):
                bitmap[:, :, 0] = np.full(new_shape, fill_value=fill_value, dtype=np.float64)
            else:
                raise TypeError("Invalid fill_value")
            bitmap[:, :, 1] = np.zeros(new_shape, dtype=np.uint8)

            pattern_settings = PatternSettings(
                0,
                0,
                width=width,
                height=height,
                depth=depth,
                bitmap=bitmap,
                passes=passes,
            )

            patterns = []
            try:
                patterns.append(draw_bitmap_pattern(
                    connection=connection, pattern_settings=pattern_settings
                ))

                if parallel != "single":
                    pattern_settings_2 = deepcopy(pattern_settings)
                    pattern_settings_2.scan_direction = "BottomToTop"
                    patterns.append(draw_bitmap_pattern(
                        connection=connection, pattern_settings=pattern_settings_2
                    ))
            except Exception as e:
                print(f"Failed to create bitmap pattern with passes multiplier {passes_multiplier} and density multiplier {density_multiplier}: {e}")
            else:
                # for pattern in patterns:
                #     assert pattern.pass_count == passes, (
                #         "Pattern passes were not properly set"
                #     )
                data[i, j, :] = (patterns[0].pass_count, density_multiplier, sum(_.time for _ in patterns))
            finally:
                connection.patterning.clear_patterns()

    output_dir = Path(output_dir)
    np.savez(
        output_dir / f"bitmap_data_{experiment_name}",
        data=data,
    )


if __name__ == "__main__":
    connection = setup_microscope(host="localhost", current=20e-9)

    for fill_value, parallel in product((1, "rand"), (True, False, "single")):
        experiment_name = f"{fill_value}"
        if isinstance(parallel, str):
            experiment_name += f"_{parallel}"
        elif parallel:
            experiment_name += "_parallel"
        else:
            experiment_name += "_serial"
        print(f"Running {experiment_name}")
        output_dir = Path(__file__).parent
        # run(output_dir=output_dir, experiment_name=experiment_name, fill_value=fill_value, connection=connection)

        with np.load(output_dir / f"bitmap_data_{experiment_name}.npz") as data:
            plot_density(data["data"], output_dir / f"bitmap_density_{experiment_name}.png")
            plot_3d(data["data"], output_dir / f"bitmap_3d_{experiment_name}.png")
