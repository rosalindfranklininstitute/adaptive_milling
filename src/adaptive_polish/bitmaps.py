# general
from __future__ import annotations
import typing

# for display
from scipy import ndimage as ndi
import numpy as np


if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


def create_dwell_and_blanking_arrays(
    gis_m: NDArray[np.floating],
    gis_minimum_m: float,
    gis_maximum_m: float,
    gis_blanking_m: typing.Union[float, None] = None,
    gis_resolution_m: typing.Union[float, None] = None,
    blanking_width_m: typing.Union[float, None] = None,
    bitmap_resolution_m: typing.Union[float, None] = None,
    max_output_range: tuple[float, float] = (1, 255),
    nan_value: float = 0,
) -> tuple[NDArray[typing.Any], typing.Union[NDArray[np.bool_], None]]:
    """
    Strength multiplier should be between 0 and 1

    If `bitmap_resolution_m` and `gis_resolution_m` are `None`, no pixel interpolation is used.

    `blanking_width_m` is only used if both `gis_blanking_m` and `bitmap_resolution_m` are not `None`
    """

    n = len(gis_m)
    if gis_resolution_m is not None and bitmap_resolution_m is not None:
        factor = bitmap_resolution_m / gis_resolution_m
    else:
        factor = 1

    x = np.linspace(0, n - 1, (n - 1) * factor + 1)
    gis_m = gis_m.copy()
    gis_m[np.isnan(gis_m)] = nan_value
    # Interpolate over to ensure pixel size
    interpolated = np.interp(x, range(n), gis_m)

    # Rescale values to from (gis_minimum_m, gis_maximum_m) to max_output_range
    rescaled = np.interp(
        interpolated,
        (gis_minimum_m, gis_maximum_m),
        max_output_range,
    ).reshape(1, -1)

    if gis_blanking_m is not None:
        blanking_array = (interpolated < gis_blanking_m).reshape(1, -1)
        if blanking_width_m is not None and bitmap_resolution_m is not None:
            blanking_half_width_px = int(
                np.ceil(blanking_width_m / (2 * bitmap_resolution_m))
            )
            ndi.binary_dilation(
                blanking_array,
                iterations=blanking_half_width_px - 1,
                output=blanking_array,
            )

    else:
        # Don't blank anything
        blanking_array = None

    return rescaled, blanking_array


def bitmap_to_points(bitmap_image: NDArray[np.uint8]) -> NDArray[typing.Any]:
    points_array = np.empty((*bitmap_image.shape[:2], 2), dtype=object)
    points_array[:, :, 0] = np.interp(bitmap_image[:, :, 2], (0, 255), (0, 1))
    points_array[:, :, 1] = 1 - bitmap_image[:, :, 1]
    return points_array


@typing.overload
def create_bitmap_array(
    gis_thickness_m: float,
    xlims: tuple[int, int],
    window_size_m: float,
    pixel_size_m: float,
    max_dwell_thickness_m: float,
    min_dwell_thickness_m: float,
    blanking_thickness_m: float,
    blanking_width_m: float,
    nan_value: float,
    as_image: typing.Literal[True],
) -> NDArray[np.uint8]: ...


@typing.overload
def create_bitmap_array(
    gis_thickness_m: float,
    xlims: tuple[int, int],
    window_size_m: float,
    pixel_size_m: float,
    max_dwell_thickness_m: float,
    min_dwell_thickness_m: float,
    blanking_thickness_m: float,
    blanking_width_m: float,
    nan_value: float,
    as_image: typing.Literal[False],
) -> NDArray[typing.Any]: ...


def create_bitmap_array(
    gis_thickness_m: float,
    xlims: tuple[int, int],
    window_size_m: float,
    pixel_size_m: float,
    max_dwell_thickness_m: float,
    min_dwell_thickness_m: float,
    blanking_thickness_m: float = -1,  # Blanking won't be applied
    blanking_width_m: float = 5e-7,
    nan_value: float = 0,
    as_image: bool = True,
) -> NDArray[typing.Union[np.uint8, typing.Any]]:
    """
    Creates bitmap array for TFS AutoScript API.
    This has two behaviours:

    - Bitmap image (for loading with `BitmapPatternDefinition.load(bitmap_path)`)
        - 3-channel (RGB) images of type `np.uint8` in the shape (y, x, c)
        - Channel 0 (R) is not used
        - Channel 1 (G) is a flag (0 blanks the point, 1 means no flags)
        - Channel 2 (B) is the dwell time multiplier (255 translates to 1x the pattern value)

    - Bitmap points (for setting a numpy array to `bpd = BitmapPatternDefinition(); bpd.points = bitmap_array`)
        - 2-channel numpy arrays of type `object` in the shape (y, x, c)
        - Channel 0 is the dwell time multiplier (now a float with the range of 0-1)
        - Channel 1 is a flag (0 means no flags, 1 blanks the point)
        - There is no Channel 2 (this was channel 0)

    The key changes, which are internally made by `BitmapPatternDefinition.load(bitmap_path)`, are:
    - The channels are flipped around.
    - The dwell time are floats between 0 and 1.
    - The flag values have been inverted.
    - What was previously Channel 0 has been removed.
    """
    #

    # The bitmap needs to be offset by the window size / 2 (in pixels) as the medium rank filter isn't centred around the pixel
    bitmap_offset = int(np.floor(window_size_m / (2 * pixel_size_m)))

    if as_image:
        dwell_time_range = (1, 255)
        dwell_time_channel = 2
        blanking_flag_value = 0
        bitmap_array = np.ones(  # If flags channel were all 0s, it would blank everything
            (
                1,  # This could be expanded if some falloff is wanted
                len(
                    gis_thickness_m[xlims[0] - bitmap_offset : xlims[1] - bitmap_offset]
                ),
                3,
            ),
            dtype=np.uint8,
        )
    else:
        # TODO: find out what the lowest value can be. Docs say 'If the value is 0, the pixel is skipped' but what if it is smaller than 1/255?
        dwell_time_range = (1 / 255, 1)
        dwell_time_channel = 0
        blanking_flag_value = 1
        bitmap_array = np.zeros(
            (
                1,  # This could be expanded if some falloff is wanted
                len(
                    gis_thickness_m[xlims[0] - bitmap_offset : xlims[1] - bitmap_offset]
                ),
                2,
            ),
            dtype=object,
        )

    dwell_time_array, blanking_array = create_dwell_and_blanking_arrays(
        gis_m=gis_thickness_m[xlims[0] - bitmap_offset : xlims[1] - bitmap_offset],
        gis_minimum_m=min_dwell_thickness_m,
        gis_maximum_m=max_dwell_thickness_m,
        gis_blanking_m=blanking_thickness_m,
        max_output_range=dwell_time_range,
        bitmap_resolution_m=pixel_size_m,
        blanking_width_m=blanking_width_m,
        nan_value=nan_value,
    )
    bitmap_array[:, :, dwell_time_channel] = dwell_time_array

    if blanking_array is not None:
        # Flags are channel 1 for both types
        bitmap_array[:, :, 1][blanking_array] = blanking_flag_value

    return bitmap_array
