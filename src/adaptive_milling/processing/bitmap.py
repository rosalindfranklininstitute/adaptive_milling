# general
from __future__ import annotations
import typing

import numpy as np
from scipy import ndimage as ndi

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


def create_dwell_and_blanking_arrays(
    input_signal: NDArray[
        np.float128 | np.float96 | np.float64 | np.float32 | np.float16
    ],
    min_dwell_threshold: float,
    max_dwell_threshold: float,
    blanking_threshold: float | None = None,
    input_resolution_m: float | None = None,
    blanking_width_m: float | None = None,
    bitmap_resolution_m: float | None = None,
    max_output_range: tuple[float, float] = (0, 255),
    nan_value: float = 0,
) -> tuple[NDArray[typing.Any], NDArray[np.bool_] | None]:
    """
    If `bitmap_resolution_m` and `input_resolution_m` are `None`, no pixel interpolation is used.

    `blanking_width_m` is only used if both `blanking_threshold` and `bitmap_resolution_m` are not `None`
    """

    n = len(input_signal)
    if input_resolution_m is not None and bitmap_resolution_m is not None:
        factor = bitmap_resolution_m / input_resolution_m
    else:
        factor = 1

    x = np.linspace(0, n - 1, int(round((n - 1) * factor)) + 1)
    input_signal = input_signal.copy()
    input_signal[np.isnan(input_signal)] = nan_value
    # Interpolate over to ensure pixel size
    interpolated = np.interp(x, range(n), input_signal)

    # Rescale values to from (min_dwell_threshold, max_dwell_threshold) to max_output_range
    rescaled = np.interp(
        interpolated,
        (min_dwell_threshold, max_dwell_threshold),
        max_output_range,
    ).reshape(1, -1)

    if blanking_threshold is not None:
        blanking_array = (interpolated < blanking_threshold).reshape(1, -1)
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
    input_signal: NDArray[
        np.float128 | np.float96 | np.float64 | np.float32 | np.float16
    ],
    xlims: tuple[int, int],
    max_dwell_threshold: float,
    min_dwell_threshold: float,
    input_resolution_m: float | None = ...,
    bitmap_resolution_m: float | None = ...,
    blanking_threshold: float | None = ...,
    blanking_width_m: float = ...,
    nan_value: float = ...,
    as_image: typing.Literal[True] = ...,
) -> NDArray[np.uint8]: ...


@typing.overload
def create_bitmap_array(
    input_signal: NDArray[
        np.float128 | np.float96 | np.float64 | np.float32 | np.float16
    ],
    xlims: tuple[int, int],
    max_dwell_threshold: float,
    min_dwell_threshold: float,
    input_resolution_m: float | None = ...,
    bitmap_resolution_m: float | None = ...,
    blanking_threshold: float | None = ...,
    blanking_width_m: float = ...,
    nan_value: float = ...,
    as_image: typing.Literal[False] = ...,
) -> NDArray[typing.Any]: ...


def create_bitmap_array(
    input_signal: NDArray[
        np.float128 | np.float96 | np.float64 | np.float32 | np.float16
    ],
    xlims: tuple[int, int],
    max_dwell_threshold: float,
    min_dwell_threshold: float,
    input_resolution_m: float | None = None,
    bitmap_resolution_m: float | None = None,
    blanking_threshold: float | None = None,  # Blanking won't be applied
    blanking_width_m: float = 5e-7,
    nan_value: float = 0,
    as_image: bool = True,
) -> NDArray[np.uint8 | typing.Any]:
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
    if as_image:
        dwell_time_range = (0, 255)
        dwell_time_channel = 2
        blanking_flag_value = 0
        bitmap_array = (
            np.ones(  # If flags channel were all 0s, it would blank everything
                (
                    1,  # This could be expanded if some falloff is wanted
                    xlims[1] - xlims[0] + 1,
                    3,
                ),
                dtype=np.uint8,
            )
        )
    else:
        # Pixel value of 0 means as little milling as possible
        dwell_time_range = (0, 1)
        dwell_time_channel = 0
        blanking_flag_value = 1
        bitmap_array = np.zeros(
            (
                1,  # This could be expanded if some falloff is wanted
                xlims[1] - xlims[0] + 1,
                2,
            ),
            dtype=object,
        )

    dwell_time_array, blanking_array = create_dwell_and_blanking_arrays(
        input_signal=input_signal[xlims[0] : xlims[1] + 1],
        min_dwell_threshold=min_dwell_threshold,
        max_dwell_threshold=max_dwell_threshold,
        blanking_threshold=blanking_threshold,
        max_output_range=dwell_time_range,
        input_resolution_m=input_resolution_m,
        bitmap_resolution_m=bitmap_resolution_m,
        blanking_width_m=blanking_width_m,
        nan_value=nan_value,
    )
    bitmap_array[:, :, dwell_time_channel] = dwell_time_array

    if blanking_array is not None:
        # Flags are channel 1 for both types
        bitmap_array[:, :, 1][blanking_array] = blanking_flag_value

    return bitmap_array


def filter_bitmap_signal(
    array_1d: NDArray[np.float32 | np.float64],
    erosion_px: int,
    gaussian_sigma: float,
    minimum_value: float | None = None,
    maximum_value: float | None = None,
) -> NDArray[np.float32 | np.float64]:
    array_1d = np.clip(array_1d, 0, maximum_value)

    if minimum_value is not None:
        min_mask = array_1d <= minimum_value

        if erosion_px > 0:
            # Expand the minimum regions according to bitmap_erosion_px and ensure
            # that the blur doesn't cut into them
            expanded_mask = ndi.binary_dilation(
                min_mask,
                iterations=erosion_px,
            )
        else:
            expanded_mask = min_mask
    else:
        expanded_mask = False

    # Use grey erosion to filter
    array_1d = ndi.grey_erosion(array_1d, size=erosion_px)

    if gaussian_sigma > 0:
        blurred = ndi.gaussian_filter1d(
            array_1d, sigma=gaussian_sigma, mode="constant", cval=0, truncate=6
        )
        array_1d = blurred

    # Set to 0 to ensure the masked area is definitely at/below the minimum
    array_1d[expanded_mask] = 0

    return array_1d


def get_angle_dwell_multiplier(
    gis_lower_edge_coordinates: NDArray[
        np.float64 | np.float32 | np.integer[typing.Any]
    ],
    pixel_size: tuple[float, float],
    sputter_ratio: float = 4.3,
) -> NDArray[np.float64]:
    """Create a multiplier to adjust the bitmap signal with.

    This is based on the sputter yield distribution in Fig. 7 of https://doi.org/10.1016/j.matdes.2022.110563

    sputter_ratio sets the ratio between 0 degrees and the maximum sputter rate (at ~80 degrees).
    """
    gradient = np.gradient(
        gis_lower_edge_coordinates[:, 0] * pixel_size[0],
        gis_lower_edge_coordinates[:, 1] * pixel_size[1],
    )

    incident_angles = np.arctan(gradient)

    # Yield multiplier is based on fitting a Beta distribution to match
    # Fig. 7 of https://doi.org/10.1016/j.matdes.2022.110563
    A = 7
    p = 5.4
    q = 1.65
    yield_multiplier = 1 + sputter_ratio * A * (
        np.abs(incident_angles) * 2 / np.pi
    ) ** (p - 1) * (1 - np.abs(incident_angles) * 2 / np.pi) ** (q - 1)

    return 1 / yield_multiplier
