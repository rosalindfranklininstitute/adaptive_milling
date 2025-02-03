# general
from __future__ import annotations
import os
import logging
from contextlib import contextmanager
from copy import deepcopy
import tempfile
from collections.abc import Generator, Iterable
import typing

# for display
import PIL
from scipy.signal import find_peaks
from skimage import filters, morphology, measure, transform, feature, exposure, util
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib as mpl
import tifffile as tff
import numpy as np
from pathlib import Path

# for the fibsem structures
from fibsem.structures import FibsemImage, MicroscopeSettings, Point
from fibsem.patterning import get_milling_stages, FibsemMillingStage
from fibsem.microscope import FibsemMicroscope
from fibsem.patterns.ui import (
    drawing_functions,
    COLOURS,
    PROPERTIES,
)

# Set up test microscope
from fibsem import utils, acquire

# Adaptive polishing
from adaptive_polish import gis_measurement as gm

if typing.TYPE_CHECKING:
    from os import PathLike
    from numpy.typing import NDArray
    from matplotlib.figure import Axes

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logging.basicConfig(handlers=[stream_handler])


mpl.use("agg")


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


def measure_gis_thickness_modified(
    mask_gis_clean: np.array, window_size_m: int, pixel_size_m: float
) -> tuple[NDArray[np.uint16], NDArray[np.uint16]]:
    # Get window size in px
    window_size_px = int(round(window_size_m / pixel_size_m))
    logging.info(f"window_size_px: {window_size_px}")
    opened_gis = morphology.binary_opening(
        mask_gis_clean,
        footprint=[
            (np.ones((window_size_px, 1)), 1),
            (np.ones((1, window_size_px)), 1),
        ],
        mode="ignore",
    )

    gis_thickness = np.sum(mask_gis_clean, axis=0, dtype=np.uint16)
    opened_gis_thickness = np.sum(opened_gis, axis=0, dtype=np.uint16)

    # Sum to get thickness along x in pixels
    return gis_thickness, opened_gis_thickness


def filter_gis_thickness(
    gis_thickness_px: NDArray[np.integer], window_size_m: int, pixel_size_m: float
) -> tuple[NDArray[np.float64], tuple[int, int]]:
    # Get window size in px
    window_size_px = int(window_size_m / pixel_size_m)
    logging.info(f"window_size_px: {window_size_px}")

    # Get left and right limits from where the mean should be calculated from
    window_size_px_where_above_zero = np.where(gis_thickness_px > 0)
    xlim_min = window_size_px_where_above_zero[0][0]  # first occurrence along x
    if len(window_size_px_where_above_zero[0]) > 1:
        xlim_max = window_size_px_where_above_zero[0][-1]  # last occurrence along x
    else:
        xlim_max = len(gis_thickness_px) - 1

    # make the L and R limits of the lamella x% smaller
    lamella_width_px = xlim_max - xlim_min
    lamella_width_to_cut = int(0.05 * lamella_width_px)  # cut 5% from each end
    xlim_min = lamella_width_to_cut + xlim_min
    xlim_max = xlim_max - lamella_width_to_cut

    # Use median filter to reduce spikiness/remove outliers
    filtered_gis_thickness_px = ndi.median_filter(
        gis_thickness_px, size=window_size_px, mode="nearest"
    )

    # Pad with half the window size to ensure the mean values are centred
    pad_size = int(np.ceil(window_size_px / 2))
    padded_filtered_gis_thickness_px = np.pad(
        filtered_gis_thickness_px, pad_width=pad_size, mode="edge"
    )
    # Create stack for windowed averaging
    stack = [
        padded_filtered_gis_thickness_px[_ : _ + window_size_px]
        for _ in range(
            padded_filtered_gis_thickness_px.size - pad_size * 2
        )  # Ensure the same length as `gis_thickness_px` in the case of odd `window_size_px`
    ]

    averaged_gis_px = np.nanmean(stack, axis=1, dtype=np.float64)

    return averaged_gis_px, (xlim_min, xlim_max)


@typing.overload
def create_bitmap_array(
    gis_thickness_m: float,
    xlims: tuple[int, int],
    window_size_m: float,
    pixel_size_m: float,
    settings: MicroscopeSettings,
    as_image: typing.Literal[True],
) -> NDArray[np.uint8]: ...


@typing.overload
def create_bitmap_array(
    gis_thickness_m: float,
    xlims: tuple[int, int],
    window_size_m: float,
    pixel_size_m: float,
    settings: MicroscopeSettings,
    as_image: typing.Literal[False],
) -> NDArray[typing.Any]: ...


def create_bitmap_array(
    gis_thickness_m: float,
    xlims: tuple[int, int],
    window_size_m: float,
    pixel_size_m: float,
    settings: MicroscopeSettings,
    as_image: bool = True,
) -> NDArray[np.uint8] | NDArray[typing.Any]:
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
        gis_minimum_m=float(settings.protocol["adaptive_polish"]["gis_minimum_m"]),
        gis_maximum_m=float(settings.protocol["adaptive_polish"]["gis_maximum_m"]),
        gis_blanking_m=float(settings.protocol["adaptive_polish"]["gis_stop_m"]),
        max_output_range=dwell_time_range,
        bitmap_resolution_m=pixel_size_m,
        blanking_width_m=1.6e-6,
        nan_value=-50000,
    )
    bitmap_array[:, :, dwell_time_channel] = dwell_time_array

    if blanking_array is not None:
        # Flags are channel 1 for both types
        bitmap_array[:, :, 1][blanking_array] = blanking_flag_value

    return bitmap_array


def setup(
    model_path: str | PathLike[str],
    config_path: str | PathLike[str],
    protocol_path: str | PathLike[str],
) -> tuple[FibsemMicroscope, MicroscopeSettings]:
    gm.init_model_with_path(model_path=model_path)
    # Set up test microscope
    microscope, settings = utils.setup_session(
        config_path=config_path, protocol_path=protocol_path
    )
    return microscope, settings


def get_next_images(
    microscope: FibsemMicroscope, settings: MicroscopeSettings
) -> tuple[FibsemImage, FibsemImage]:
    # "Acquire" images
    return acquire.take_reference_images(microscope, settings.image)


def make_predictions(
    electron_beam_image: FibsemImage,
) -> tuple[NDArray[typing.Any], NDArray[typing.Any]]:
    # Find the lamella width based on SEM image segmentation
    prediction = gm.segment(electron_beam_image.data)
    clean_prediction = gm.clean_prediction(
        prediction=prediction, pixel_size_m=electron_beam_image.metadata.pixel_size.x
    )
    return prediction, clean_prediction


def measure_gis(
    electron_beam_image: FibsemImage,
    clean_prediction: NDArray[typing.Any],
    window_size_m: float,
) -> tuple[dict[str, NDArray[np.floating]], tuple[int, int]]:
    pixel_size_m = electron_beam_image.metadata.pixel_size.x
    gis_thickness_px, opened_gis_thickness_px = measure_gis_thickness_modified(
        mask_gis_clean=clean_prediction,
        window_size_m=window_size_m,
        pixel_size_m=pixel_size_m,
    )
    filtered_gis_thickness_px, xlims = filter_gis_thickness(
        gis_thickness_px=opened_gis_thickness_px,
        window_size_m=window_size_m,
        pixel_size_m=pixel_size_m,
    )
    lamella_width = (xlims[1] - xlims[0]) * pixel_size_m
    print(f"Lamella width is {lamella_width} m")

    gis_thicknesses = {
        "basic": gis_thickness_px * pixel_size_m,
        "opened": opened_gis_thickness_px * pixel_size_m,
        "filtered": filtered_gis_thickness_px * pixel_size_m,
    }

    return gis_thicknesses, xlims


def draw_milling_patterns(
    ax: Axes,
    image: FibsemImage,
    milling_stages: list[FibsemMillingStage],
    crosshair: bool = True,
    scalebar: bool = True,
) -> None:
    ax.imshow(image.data, cmap="gray")

    patch_collections = []
    for i, stage in enumerate(milling_stages):
        colour = COLOURS[i % len(COLOURS)]
        p = stage.pattern

        patch_collections.extend(
            drawing_functions[type(p)](image, p, colour=colour, name=stage.name)
        )

    for pc in patch_collections:
        ax.add_collection(pc)
    ax.legend()

    # draw crosshair at centre of image
    if crosshair:
        cy, cx = image.data.shape[0] // 2, image.data.shape[1] // 2
        ax.plot(cx, cy, "y+", markersize=PROPERTIES["crosshair_size"])

    # draw scalebar
    if scalebar:
        try:
            # optional dependency, best effort
            from matplotlib_scalebar.scalebar import ScaleBar

            scalebar = ScaleBar(
                dx=image.metadata.pixel_size.x,
                color="black",
                box_color="white",
                box_alpha=0.5,
                location="lower right",
            )

            plt.gca().add_artist(scalebar)
        except ImportError:
            logging.debug("Scalebar not available, skipping")


@contextmanager
def _temp_file_wrapper(bitmap_array: NDArray[np.uint8]) -> Generator[str, None, None]:
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".bmp", mode="w+b", delete=False
        ) as tmp_f:
            if bitmap_array.dtype is np.dtype(np.uint8):
                PIL.Image.fromarray(bitmap_array).save(tmp_f, format="BMP")
                yield tmp_f.name
    finally:
        os.unlink(tmp_f.name)


@contextmanager
def _passthrough_wrapper(arg: typing.Any) -> Generator[typing.Any, None, None]:
    yield arg


def plot_bitmap_trench_pattern(
    ax: Axes,
    bitmap_array: NDArray[np.uint8],
    lamella_width: float,
    settings: MicroscopeSettings,
    ion_beam_image: FibsemImage,
    centre: tuple[float, float],
) -> None:
    # Use TrenchBitmapPattern via protocol
    try:
        if bitmap_array.dtype is np.dtype(np.uint8):
            bitmap_wrapper = _temp_file_wrapper
        else:
            bitmap_wrapper = _passthrough_wrapper
        with bitmap_wrapper(bitmap_array) as bitmap:
            previous_protocol = settings.protocol["milling"]["lamella"]["stages"][0]

            # Place pattern on image from edited pattern
            ap_milling_protocol = deepcopy(previous_protocol)
            ap_milling_protocol["lamella_width"] = lamella_width
            # ap_milling_protocol["hfw"] = 4e-05
            # ap_milling_protocol["trench_height"] = 5e-7
            ap_milling_protocol["type"] = "TrenchBitmapPattern"
            ap_milling_protocol["bitmap"] = bitmap
            ap_milling_protocol["name"] = "Adjusted adaptive polishing"
            settings.protocol["milling"]["lamella"]["stages"][0] = ap_milling_protocol
            settings.protocol["milling"]["lamella"]["stages"][-1]["patterning_mode"] = (
                "Serial"
            )
            stages = get_milling_stages(
                "lamella",
                settings.protocol["milling"],
                Point(*centre),
            )
            draw_milling_patterns(ax, ion_beam_image, stages)
            ax.set_title(
                f"Bitmap trench milling pattern\n(bitmap min, max = {bitmap_array.min()}, {bitmap_array.max()})"
            )
    except Exception:
        logging.error("Plotting bitmap pattern raised an exception", exc_info=True)


def calculate_coordinates_from_x_and_Hough_peaks(
    peaks: tuple[NDArray[np.uint64], NDArray[np.float64], NDArray[np.float64]],
    *xs: float,
) -> NDArray[np.float64]:
    xy_points = np.full((len(peaks[0]), len(xs), 2), np.nan, dtype=np.float64)
    for i, (_, angle, dist) in enumerate(zip(*peaks)):
        xy_points[i, :, 0] = xs
        sin_angle = np.sin(angle)
        if sin_angle == 0:
            continue
        y0 = dist / sin_angle  # x = 0
        for j, x in enumerate(xs):
            if x == 0:
                xy_points[i, j, 1] = y0
            else:
                xy_points[i, j, 1] = y0 - x / np.tan(angle)
    return xy_points


def calculate_coordinates_from_y_and_Hough_peaks(
    peaks: tuple[NDArray[np.uint64], NDArray[np.float64], NDArray[np.float64]],
    *ys: float,
) -> NDArray[np.float64]:
    xy_points = np.full((len(peaks[0]), len(ys), 2), np.nan, dtype=np.float64)
    for i, (_, angle, dist) in enumerate(zip(*peaks)):
        xy_points[i, :, 1] = ys
        cos_angle = np.cos(angle)
        if cos_angle == 0:
            continue
        x0 = dist / cos_angle  # x = 0
        for j, y in enumerate(ys):
            if y == 0:
                xy_points[i, j, 0] = x0
            else:
                xy_points[i, j, 0] = x0 - y * np.tan(angle)
    return xy_points


def find_trough_peaks_from_thresholded_image(
    image: NDArray[typing.Any], distance_px: int, width_px: float = 1
) -> tuple[float, float]:
    radius_px = distance_px / 2
    y_sum = image.sum(axis=0)
    filtered_y_sum = ndi.gaussian_filter1d(y_sum, sigma=2)
    # TODO: Stop filtering and instead use distance=1 and then check for all the peaks and select the most prominent one within distance
    peaks, properties = find_peaks(
        filtered_y_sum,
        distance=1,
        width=width_px,
        rel_height=image.shape[0] // 5,
        prominence=1,
        # threshold=1,w
        # wlen=1,
    )

    sorted_idxs = np.argsort(properties["prominences"])
    peaks_mask = np.ones_like(peaks, dtype=np.bool_)

    # Group peaks to most prominent within distance_px
    for i in sorted_idxs[::-1]:  # largest first
        if not peaks_mask[i]:
            continue
        peaks_mask &= np.abs(peaks - peaks[i]) > radius_px
        peaks_mask[i] = True

    # plt.plot(y_sum, color="C1")
    # plt.plot(filtered_y_sum, color="C2")
    # plt.scatter(peaks, filtered_y_sum[peaks], color="C2")
    # plt.scatter(peaks[peaks_mask], filtered_y_sum[peaks[peaks_mask]], color="C3")
    # fig = plt.gcf()
    # plt.clf()

    # TODO: use widths to find peak inner edges
    sorted_valid_idxs = np.argsort(
        properties["prominences"][peaks_mask]
    )  # sort by confidence
    sorted_valid_peaks = peaks[peaks_mask][sorted_valid_idxs]
    if len(sorted_valid_peaks) < 2:
        raise ValueError("Not enough valid peaks found")
    return sorted(
        (
            sorted_valid_peaks[-2],
            sorted_valid_peaks[-1],
        )
    )


def find_centre(
    image: FibsemImage,
    lamella_width_m: float,
    lamella_height_m: float,
    plot_path: str | PathLike[str] | None = None,
) -> tuple[float, float]:
    """Finds the centre of the lamella based on the positions of the stress relief cuts. Definitely room for improvement."""
    pixel_size = image.metadata.pixel_size.x
    lamella_width_px = lamella_width_m / pixel_size
    lamella_height_px = lamella_height_m / pixel_size

    # TODO: add "lamella_angle" and "lamalla_depth" arguments and do some trig to figure out the min distance for the lamella (or if only 1 line should be found)

    data = exposure.equalize_adapthist(image.data)
    # gauss_1 = (30, 12)
    # gauss_2 = (80, 80)

    large_padding = int(lamella_width_px * 0.25)
    small_padding = int(lamella_width_px * 0.05)

    sigma_large = 30
    sigma_small = 2

    gaussian = ndi.gaussian_filter(data, (sigma_large, sigma_small))
    np.min(gaussian, axis=0)
    gaussian_triangle_threshold = filters.threshold_triangle(gaussian)
    # gaussian_clipped = gaussian.copy()
    # gaussian_clipped[gaussian < gaussian_triangle_threshold] = 0
    gaussian_clipped = np.clip(gaussian, None, gaussian_triangle_threshold)
    sobel_v = np.abs(filters.sobel_v(gaussian_clipped))
    sobel_thresholded = sobel_v > filters.threshold_otsu(sobel_v)
    # sobel_thresholded2 = sobel_v > filters.threshold_yen(sobel_v)
    # sobel_thresholded3 = sobel_v > filters.threshold_isodata(sobel_v)
    # sobel_thresholded4 = sobel_v > filters.threshold_niblack(sobel_v)
    # sobel_thresholded5 = sobel_v > filters.threshold_sauvola(sobel_v)
    final = morphology.remove_small_objects(
        sobel_thresholded, min_size=data.shape[0] // 3
    )

    # threshold_functions = (
    #     filters.threshold_isodata,
    #     filters.threshold_li,
    #     filters.threshold_mean,
    #     filters.threshold_minimum,
    #     filters.threshold_otsu,
    #     filters.threshold_triangle,
    #     filters.threshold_yen,
    # )
    # for threshold_fn in threshold_functions:
    #     print(f"{threshold_fn.__name__} {threshold_fn(sobel_v)}")

    # for i in np.arange(1.5, 4, 0.5):
    #     for j in np.arange(4, 7, 1):
    #         dog_values = (i, i + j)
    #         data = image.data
    #         # data = filters.rank.median(
    #         #     util.img_as_uint(data), footprint=morphology.disk(2)
    #         # )
    #         # data = ndi.gaussian_filter(data, sigma=(5, 2))
    #         dog = filters.difference_of_gaussians(
    #             data,
    #             (5, dog_values[0]),
    #             (5, dog_values[1]),
    #             mode="nearest",
    #         )
    #         # median_dog = filters.rank.median(
    #         #     util.img_as_uint(dog), footprint=np.ones((20, 3), dtype=np.bool_)
    #         # )
    #         # median_div = util.img_as_uint(dog) / (1 + median_dog)
    #         # dog = ndi.gaussian_filter(dog, sigma=(5, 2))

    #         dog = np.clip(dog, None, 0)
    #         final = util.invert(dog, signed_float=True)
    #         threshold = filters.threshold_triangle(final)
    #         threshold_otsu = filters.threshold_otsu(final)
    #         threshold2 = filters.threshold_local(dog, method="median", block_size=9)
    #         threshold3 = filters.threshold_local(
    #             dog, method="gaussian", param=3, block_size=9
    #         )

    #         # TODO: look into thresholding or directly hough transforming the sobel filter
    #         # Seems to work well with i = 2, j = 4
    #         sobel = filters.sobel_v(dog)
    #         sobel2 = filters.sobel_v(final)

    #         thresholded_dog = final > threshold + np.std(final) * 2
    #         thresholded2_dog = final > (threshold2 + np.std(final) * 2)
    #         thresholded3_dog = final > (threshold3 + np.std(final) * 2)
    #         thresholded_sobel = sobel > filters.threshold_otsu(sobel)
    #         thresholded_sobel_removed_objects = morphology.remove_small_objects(
    #             thresholded_sobel, min_size=data.shape[0] // 3
    #         )

    #         ideal_sobel_threshold = np.mean(sobel) + np.std(sobel) * 1.5

    #         threshold_functions = (
    #             filters.threshold_isodata,
    #             filters.threshold_li,
    #             filters.threshold_mean,
    #             filters.threshold_minimum,
    #             filters.threshold_otsu,
    #             filters.threshold_triangle,
    #             filters.threshold_yen,
    #         )
    #         for threshold_fn in threshold_functions:
    #             print(f"{threshold_fn.__name__} {threshold_fn(sobel)}")

    #         lowt = (sobel > threshold).astype(int)
    #         hight = (sobel > threshold_otsu).astype(int)
    #         hyst = filters.apply_hysteresis_threshold(sobel, threshold, threshold_otsu)

    #         for k in np.arange(0.5, 4, 0.25):
    #             thresholded_dog = final > (threshold + np.std(final) * k)
    #             thresholded_dog2 = final > (threshold2 + np.std(final) * k)
    #             thresholded_dog3 = final > (threshold3 + np.std(final) * k)
    #             thresholded_dog4 = final > (np.mean(final) + np.std(final) * k)

    #             thresholded_sobel2 = sobel > (np.mean(sobel) + np.std(sobel) * k)
    #             print("next")

    # dog = ndi.gaussian_filter(dog, sigma=(2, 5))

    # threshold = filters.threshold_triangle(dog)
    # thresholded_dog = dog > threshold

    # footprint_dilate = np.ones((int(dog_values[1]), 2), dtype=np.bool_)
    # dilate_seq = morphology.binary_dilation(thresholded_dog, footprint=footprint_dilate)
    # footprint_erode = np.ones((dog_values[0] * 2, 5), dtype=np.bool_)
    # erode_seq = morphology.binary_erosion(dilate_seq, footprint=footprint_erode)
    # # mask_sequence = np.logical_and(dilate_seq, ~erode_seq)

    # canny_thresholded_dog = feature.canny(thresholded_dog, sigma=0)

    # Canny Hough
    # lamella_angles = np.linspace(
    #     17 * np.pi / 36, 19 * np.pi / 36, 5, endpoint=True, dtype=np.float64
    # )

    distances_x = find_trough_peaks_from_thresholded_image(
        final, distance_px=lamella_width_px * 0.75, width_px=1
    )
    centre_x = np.mean(distances_x)
    centre_x_int = int(centre_x)

    # h, theta, d = transform.hough_line(
    #     final,
    #     theta=np.asarray((0.0,), dtype=np.float64),
    # )

    # zero_deg_peaks = transform.hough_line_peaks(
    #     h,
    #     theta,
    #     d,
    #     min_distance=int(lamella_width_px * 0.75),
    #     num_peaks=2,
    # )

    # centre_x = np.mean(zero_deg_peaks[2])
    # centre_x_int = int(centre_x)

    # lamella_edges_x = zero_deg_peaks[2]
    # lamella_edges_x.sort()
    # distances_x = lamella_edges_x.astype(int)

    indexes_1 = [
        (0, data.shape[0]),
        (
            # Widen the padding for this as we are looking for X
            max(centre_x_int - large_padding, 0),
            min(centre_x_int + large_padding, data.shape[1]),
        ),
    ]

    trench = data[
        indexes_1[0][0] : indexes_1[0][1],
        indexes_1[1][0] : indexes_1[1][1],
    ]

    blurred_trench = ndi.gaussian_filter(
        trench, lamella_height_px / 10, axes=0, mode="reflect"
    )
    blurred_trench[blurred_trench < gaussian_triangle_threshold] = 0
    print(blurred_trench.shape)
    print(indexes_1)
    # fig = filters.try_all_threshold(blurred_trench)[0]
    mean_trench = blurred_trench.mean(axis=1).squeeze()
    trench_line = mean_trench > filters.threshold_triangle(mean_trench)

    peaks, peak_properties = find_peaks(
        trench_line, distance=lamella_width_px / 2, width=1
    )
    if len(peaks) > 1:
        # find the peak closest to the centre of the image
        peak_idx = np.argmin(np.abs(peaks - trench_line.size // 2))
    else:
        peak_idx = 0

    lamella_peak = peaks[peak_idx]
    lamella_edges_y = (
        peak_properties["left_ips"][peak_idx],
        peak_properties["right_ips"][peak_idx],
    )

    # Find centre peak or peak nearest centre in Y?

    # dog[
    #     indexes_1[0][0] : indexes_1[0][1],
    #     indexes_1[1][0] + 2 * large_padding : indexes_1[1][1] - 2 * large_padding,
    # ]

    # h, theta, d = transform.hough_line(
    #     canny_thresholded_dog[
    #         indexes_1[0][0] : indexes_1[0][1], indexes_1[1][0] : indexes_1[1][1]
    #     ],
    #     theta=lamella_angles,
    # )

    # lamella_peaks = transform.hough_line_peaks(h, theta, d, min_distance=dog_values[0])

    # # Calculate the negative to account for potential bounding box cropping due to image size
    # x_min = -(distances_x[0] + large_padding) if indexes_1[1][0] == 0 else 0
    # # Calculate the x_max, ignoring the maximum image size
    # x_max = distances_x[1] - distances_x[0] + large_padding + x_min

    # mask = np.full_like(data, False, dtype=np.bool_)
    # mask[:, indexes_1[1][0] : indexes_1[1][1]] = True

    # lamella_box_edge_coordinates = calculate_coordinates_from_x_and_Hough_peaks(
    #     lamella_peaks, x_min, x_max
    # )

    # min_order_idxs = np.argsort(
    #     np.take_along_axis(
    #         lamella_box_edge_coordinates[:, :, 1],
    #         np.expand_dims(lamella_box_edge_coordinates[:, :, 1].argmin(axis=1), 1),
    #         axis=1,
    #     ),
    #     axis=0,
    # )

    # y_min_idxs =

    # y_min_idxs = lamella_box_edge_coordinates[:, :, 1].argmin(axis=1)
    # y_max_idxs = lamella_box_edge_coordinates[:, :, 1].argmax(axis=1)

    # sorted_by_y_idxs = np.argsort(
    #     np.take_along_axis(
    #         lamella_box_edge_coordinates[:, :, 1],
    #         np.expand_dims(y_max_idxs, 1),
    #         axis=1,
    #     ),
    #     axis=0,
    # ).squeeze()
    # print(lamella_box_edge_coordinates[:, :, 1].shape)

    # sorted_ys = lamella_box_edge_coordinates[sorted_by_y_idxs, :, 1].astype(np.uint32)
    # area_medians = []
    # for i in range(lamella_box_edge_coordinates.shape[0] - 1):
    #     ymin = sorted_ys[i, y_min_idxs[i]] - small_padding
    #     ymax = sorted_ys[i + 1, y_max_idxs[i + 1]] + small_padding
    #     area_medians.append(
    #         np.median(
    #             data[
    #                 max(
    #                     ymin,
    #                     0,
    #                 ) : min(
    #                     ymax,
    #                     data.shape[1],
    #                 )
    #                 + small_padding,
    #                 x_min:x_max,
    #             ]
    #         )
    #     )

    # lamella_index = np.argmax(area_medians)
    # lamella_peak_indexes = (
    #     sorted_by_y_idxs[lamella_index],
    #     sorted_by_y_idxs[lamella_index + 1],
    # )

    # final_lamella_peaks = tuple(
    #     _[lamella_index : lamella_index + 2] for _ in lamella_peaks
    # )

    # distances_y = (
    #     int(
    #         np.floor(
    #             lamella_box_edge_coordinates[
    #                 lamella_peak_indexes[0], y_min_idxs[lamella_index], 1
    #             ]
    #         )
    #     ),
    #     int(
    #         np.ceil(
    #             lamella_box_edge_coordinates[
    #                 lamella_peak_indexes[1], y_max_idxs[lamella_index + 1], 1
    #             ]
    #         )
    #     ),
    # )

    half_height = int(lamella_height_px / 2)
    indexes_2 = [
        (
            # Crop by padding to focus on the sides of the lamella only
            max(lamella_peak - half_height, 0),
            min(lamella_peak + half_height, data.shape[0]),
        ),
        (
            # Widen the padding for this as we are looking for X
            max(int(centre_x - lamella_width_px / 2 - large_padding), 0),
            min(int(centre_x + lamella_width_px / 2 + large_padding), data.shape[1]),
        ),
    ]

    # gaussian_clipped2 = np.clip(
    #     gaussian[indexes_2[0][0] : indexes_2[0][1], indexes_2[1][0] : indexes_2[1][1]],
    #     None,
    #     filters.threshold_triangle(
    #         gaussian[
    #             indexes_2[0][0] : indexes_2[0][1], indexes_2[1][0] : indexes_2[1][1]
    #         ]
    #     ),
    # )
    # sobel_v2 = np.abs(filters.sobel_v(gaussian_clipped2))
    # sobel_thresholded2 = sobel_v2 > filters.threshold_otsu(sobel_v2)
    # # sobel_thresholded22 = sobel_v2 > filters.threshold_yen(sobel_v2)
    # # sobel_thresholded23 = sobel_v2 > filters.threshold_isodata(sobel_v2)
    # # sobel_thresholded24 = sobel_v2 > filters.threshold_triangle(sobel_v2)
    # # sobel_thresholded25 = sobel_v2 > filters.threshold_li(sobel_v2)
    # final2 = morphology.remove_small_objects(
    #     sobel_thresholded2, min_size=sobel_v2.shape[0] * 2
    # )

    # # look for finer features
    # dog2 = filters.difference_of_gaussians(
    #     image.data[
    #         indexes_2[0][0] : indexes_2[0][1], indexes_2[1][0] : indexes_2[1][1]
    #     ],
    #     (0, dog_values[0]),
    #     (0, dog_values[1]),
    #     mode="nearest",
    # )
    # dog2 = ndi.gaussian_filter(dog2, dog_values[0], axes=0)

    # threshold2 = filters.threshold_triangle(dog2)
    # thresholded_dog2 = dog2 > threshold2
    # canny_thresholded_dog2 = feature.canny(thresholded_dog2, sigma=0)

    distances_x2 = find_trough_peaks_from_thresholded_image(
        final[indexes_2[0][0] : indexes_2[0][1], indexes_2[1][0] : indexes_2[1][1]],
        distance_px=lamella_width_px * 0.75,
        width_px=1,
    )
    distances_x2 = np.asarray(distances_x2) + indexes_2[1][0]
    centre_x2 = np.mean(distances_x2)
    # centre_x_int = int(centre_x)

    # h, theta, d = transform.hough_line(
    #     final2,
    #     theta=np.asarray((0.0,), dtype=np.float64),
    # )

    # second_zero_deg_peaks = transform.hough_line_peaks(
    #     h,
    #     theta,
    #     d,
    #     min_distance=int(lamella_width_px * 0.75),
    #     num_peaks=2,
    # )
    # # if len(second_zero_deg_peaks[0]) < 2:
    # #     logging.warning("Failed to refine lamella edges")
    # #     indexes_3 = [
    # #         [0, canny_thresholded_dog.shape[0]],
    # #         [0, canny_thresholded_dog.shape[1]],
    # #     ]
    # # else:

    # lamella_edges_x.sort()
    # lamella_centre_x = np.mean(lamella_edges_x)
    # distances_x2 = lamella_edges_x.astype(int)
    indexes_3 = [
        [
            lamella_edges_y[0] - large_padding,
            lamella_edges_y[1] + large_padding,
        ],
        (
            max(int(distances_x2[0]) - large_padding, 0),
            min(
                int(distances_x2[1]) + large_padding,
                data.shape[1],
            ),
        ),
    ]

    if plot_path is not None:
        dpi = 300
        fig, axs = plt.subplots(
            2, 2, figsize=(20, 16), dpi=dpi, tight_layout=True, sharex=True, sharey=True
        )
        linewidth = 3
        axs = axs.ravel()

        axs[0].imshow(sobel_v, cmap="Greys")
        axs[0].set_title("Vertical sobel filter")
        # axs[0].set_axis_off()

        axs[1].imshow(final, cmap="Greys")
        axs[1].set_title("Final lines images")
        # axs[0].set_axis_off()

        axs[1].axvline(
            distances_x[0],
            alpha=0.5,
            linewidth=linewidth,
            color="C1",
        )
        axs[1].axvline(
            distances_x[1],
            alpha=0.5,
            linewidth=linewidth,
            color="C1",
            label="Detected trough edges",
        )

        # for _, angle, dist in zip(*zero_deg_peaks):
        #     (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        #     _ = axs[1].axline(
        #         (x0, y0),
        #         slope=np.tan(angle + np.pi / 2),
        #         alpha=0.5,
        #         linewidth=linewidth,
        #         color="C1",
        #     )
        # _.set_label("Detected trough edges")

        axs[1].axvline(
            indexes_1[1][0],
            alpha=0.5,
            linewidth=linewidth,
            color="C2",
            linestyle=":",
        )
        axs[1].axvline(
            indexes_1[1][1],
            alpha=0.5,
            linewidth=linewidth,
            color="C2",
            linestyle=":",
            label="Lamella Y search area",
        )
        axs[1].imshow(
            np.expand_dims(trench_line[::-1], axis=1),
            cmap="Greys",
            extent=[indexes_1[1][0], indexes_1[1][1], indexes_1[0][0], indexes_1[0][1]],
        )

        axs[1].legend()

        axs[2].imshow(data, cmap="Greys")
        # axs[2].set_title("Initial lamella finding")
        # axs[2].set_axis_off()

        axs[2].axvline(
            distances_x[0],
            alpha=0.5,
            linewidth=linewidth,
            color="C1",
        )
        axs[2].axvline(
            distances_x[1],
            alpha=0.5,
            linewidth=linewidth,
            color="C1",
            label="Detected trough edges",
        )

        # for _, angle, dist in zip(*zero_deg_peaks):
        #     (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        #     _ = axs[2].axline(
        #         (x0, y0),
        #         slope=np.tan(angle + np.pi / 2),
        #         alpha=0.5,
        #         linewidth=linewidth,
        #         color="C1",
        #     )
        # _.set_label("Detected trough edges")
        axs[2].axhline(
            lamella_edges_y[0],
            alpha=0.5,
            linewidth=linewidth,
            color="C2",
            label="Detected lamella min/max",
        )
        axs[2].axhline(
            lamella_edges_y[1],
            alpha=0.5,
            linewidth=linewidth,
            color="C2",
        )

        # for _, angle, dist in zip(*lamella_peaks):
        #     (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        #     axs[2].axline(
        #         (x0 + indexes_1[1][0], y0),
        #         slope=np.tan(angle + np.pi / 2),
        #         alpha=0.5,
        #         linewidth=linewidth,
        #         color="C2",
        #     )
        axs[2].plot(
            (indexes_2[1][0], indexes_2[1][1]),
            (indexes_2[0][0], indexes_2[0][0]),
            linewidth=linewidth,
            linestyle=":",
            color="C3",
        )
        axs[2].plot(
            (indexes_2[1][0], indexes_2[1][1]),
            (indexes_2[0][1], indexes_2[0][1]),
            linewidth=linewidth,
            linestyle=":",
            color="C3",
        )
        axs[2].plot(
            (indexes_2[1][0], indexes_2[1][0]),
            (indexes_2[0][0], indexes_2[0][1]),
            linewidth=linewidth,
            linestyle=":",
            color="C3",
        )
        axs[2].plot(
            (indexes_2[1][1], indexes_2[1][1]),
            (indexes_2[0][0], indexes_2[0][1]),
            linewidth=linewidth,
            linestyle=":",
            color="C3",
            label="Bounding box for second trench detection",
        )
        axs[2].legend()

        axs[3].imshow(data, cmap="Greys")
        # axs[3].set_axis_off()

        axs[3].axvline(
            distances_x2[0],
            alpha=0.5,
            linewidth=linewidth,
            color="C3",
        )
        axs[3].axvline(
            distances_x2[1],
            alpha=0.5,
            linewidth=linewidth,
            color="C3",
            label="Detected trough edges",
        )

        # for _, angle, dist in zip(*second_zero_deg_peaks):
        #     (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        #     _ = axs[3].axline(
        #         (x0 + indexes_2[1][0], y0 + indexes_2[0][0]),
        #         slope=np.tan(angle + np.pi / 2),
        #         alpha=0.5,
        #         linewidth=linewidth,
        #         color="C3",
        #     )
        # _.set_label("Detected trough edges")

        axs[3].plot(
            (indexes_3[1][0], indexes_3[1][1]),
            (indexes_3[0][0], indexes_3[0][0]),
            linewidth=2,
            linestyle=":",
            color="C4",
        )
        axs[3].plot(
            (indexes_3[1][0], indexes_3[1][1]),
            (indexes_3[0][1], indexes_3[0][1]),
            linewidth=linewidth,
            linestyle=":",
            color="C4",
        )
        axs[3].plot(
            (indexes_3[1][0], indexes_3[1][0]),
            (indexes_3[0][0], indexes_3[0][1]),
            linewidth=linewidth,
            linestyle=":",
            color="C4",
        )
        axs[3].plot(
            (indexes_3[1][1], indexes_3[1][1]),
            (indexes_3[0][0], indexes_3[0][1]),
            linewidth=linewidth,
            linestyle=":",
            color="C4",
            label="Bounding box for training",
        )

        axs[3].plot(
            centre_x2,
            lamella_peak,
            color="C4",
            marker="+",
            linewidth=linewidth,
            markersize=12 * linewidth,
        )
        axs[3].legend()

        # for _, angle, dist in zip(*lamella_peaks):
        #     (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        #     axs[3].axline(
        #         (x0 + indexes_1[1][0], y0),
        #         slope=np.tan(angle + np.pi / 2),
        #         alpha=0.5,
        #         linewidth=linewidth,
        #         color="C2",
        #     )
        # for _, angle, dist in zip(*final_lamella_peaks):
        #     (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        #     axs[3].axline(
        #         (x0 + indexes_1[1][0], y0),
        #         slope=np.tan(angle + np.pi / 2),
        #         alpha=0.5,
        #         linewidth=linewidth,
        #         color="C5",
        #     )

        fig.savefig(plot_path)
        plt.close()

    return (centre_x2 * pixel_size, lamella_peak * pixel_size)


def milling_cycle_plot(
    sem_image: FibsemImage,
    first_prediction: NDArray[typing.Any],
    clean_prediction: NDArray[typing.Any],
    fib_image: FibsemImage,
    settings: MicroscopeSettings,
    gis_thicknesses_m: dict[str, NDArray[typing.Any]],
    xlims: tuple[float, float],
    bitmap_array: NDArray[np.uint8],
    image_name: str = None,
    save_path: Path = None,
):
    logging.info("milling_cycle_plot()")
    fig, axs = plt.subplots(
        nrows=2, ncols=3, figsize=(24, 16), dpi=300, tight_layout=True
    )
    fig.suptitle(image_name)

    # SEM
    axs[0, 0].imshow(sem_image.data, cmap="Greys_r")
    axs[0, 0].axis("off")
    axs[0, 0].set_title("SEM")

    # SEM + 1st pass prediction
    axs[0, 1].imshow(sem_image.data, cmap="Greys_r")
    axs[0, 1].imshow(
        first_prediction,
        alpha=0.4,
        cmap="tab10",
        vmin=0,
        vmax=10,
        interpolation="nearest",
    )
    axs[0, 1].axis("off")
    axs[0, 1].set_title("SEM, 1st prediction")

    # SEM + clean prediction
    axs[0, 2].imshow(sem_image.data, cmap="Greys_r")
    axs[0, 2].imshow(
        clean_prediction,
        alpha=0.4,
        cmap="tab10",
        vmin=0,
        vmax=10,
        interpolation="nearest",
    )
    axs[0, 2].axvline(x=xlims[0])
    axs[0, 2].axvline(x=xlims[1])
    axs[0, 2].axis("off")
    axs[0, 2].set_title(
        f"SEM, clean, crack area $\mu$m2 = {float(settings.protocol['adaptive_polish']['max_crack_area_m2']) * 1e12:.2f}"
    )

    # FIB image
    axs[1, 0].imshow(fib_image.data, cmap="Greys_r")
    axs[1, 0].axis("off")
    axs[1, 0].set_title("FIB")

    # Calculate offset in x
    xlims_centre = xlims[0] + (xlims[1] - xlims[0]) / 2

    lamella_centre_x = xlims_centre * sem_image.metadata.pixel_size.x

    im_centre_x = (
        fib_image.metadata.image_settings.resolution[0] / 2
    ) * fib_image.metadata.pixel_size.x

    centre = (
        float(-im_centre_x + lamella_centre_x),
        0,
    )
    # # Get and apply shifts from tiffs
    # with tff.TiffFile(fib_image.get_save_path(), mode="r") as tiff:
    #     fib_shift = tiff.shaped_metadata[0]["microscope_state"]["ion_beam"]["shift"]

    # with tff.TiffFile(sem_image.get_save_path(), mode="r") as tiff:
    #     sem_shift = tiff.shaped_metadata[0]["microscope_state"]["electron_beam"][
    #         "shift"
    #     ]

    # centre = (centre[0] - sem_shift["x"], centre[1] - sem_shift["y"])
    # centre = (centre[0] + fib_shift["x"], centre[1] + fib_shift["y"])

    bitmap_as_image = bitmap_array.dtype is np.dtype(np.uint8)

    plot_bitmap_trench_pattern(
        axs[1, 1],
        bitmap_array=bitmap_array,
        lamella_width=(xlims[1] - xlims[0]) * sem_image.metadata.pixel_size.x,
        settings=settings,
        ion_beam_image=fib_image,
        centre=centre,
    )

    axs[1, 2].axvline(x=xlims[0], linestyle="--", label="Lamella boundaries")
    axs[1, 2].axvline(x=xlims[1], linestyle="--")

    axs[1, 2].plot(
        gis_thicknesses_m["basic"] * 1e6,
        "-",
        zorder=2,
        label="GIS thickness",
        alpha=0.33,
        color="blue",
    )
    axs[1, 2].plot(
        gis_thicknesses_m["opened"] * 1e6,
        "-",
        zorder=3,
        label="Opened GIS thickness",
        alpha=0.33,
        color="red",
    )
    axs[1, 2].plot(
        gis_thicknesses_m["filtered"] * 1e6,
        "-",
        zorder=4,
        label="Averaged opened GIS thickness",
        alpha=0.33,
        color="green",
    )

    xmax = len(gis_thicknesses_m["basic"])

    axs[1, 2].set_xlabel("Distance along x (px)")
    axs[1, 2].set_ylabel("GIS thickness ($\mu$m)")
    axs[1, 2].hlines(
        y=float(settings.protocol["adaptive_polish"]["gis_minimum_m"]) * 1e6,
        xmin=0,
        xmax=xmax,
        label="Minimum GIS thickness",
        linestyles="dashed",
        colors="orange",
    )
    axs[1, 2].hlines(
        y=float(settings.protocol["adaptive_polish"]["gis_maximum_m"]) * 1e6,
        xmin=0,
        xmax=xmax,
        label="Full power GIS thickness",
        linestyles="dashed",
        colors="green",
    )
    axs[1, 2].hlines(
        y=float(settings.protocol["adaptive_polish"]["gis_stop_m"]) * 1e6,
        xmin=0,
        xmax=xmax,
        label="Stop GIS thickness",
        linestyles="dashed",
        colors="red",
    )
    axs[1, 2].set_xlim(0, xmax)
    axs[1, 2].set_ylim(
        0,
    )
    axs[1, 2].set_title(
        f"GIS thickness, min={np.nanmin(gis_thicknesses_m['filtered']) * 1e6:.2f} $\mu$m"
    )
    axs[1, 2].legend()

    blanking_array = bitmap_array[:, :, 1].astype(np.bool_)
    if bitmap_as_image:
        blanking_array = ~blanking_array
    axs[1, 2].imshow(
        blanking_array,
        cmap="Reds",
        extent=(
            xlims[0],
            xlims[1],
            0,
            float(settings.protocol["adaptive_polish"]["gis_stop_m"]) * 1e6,
        ),
        interpolation="none",
        aspect="auto",
        zorder=1,
    )

    if bitmap_as_image:
        vmax = 255
        dwell_time_array = bitmap_array[:, :, 2]
    else:
        vmax = 1
        dwell_time_array = bitmap_array[:, :, 0].astype(np.float64)

    _ = axs[1, 2].imshow(
        dwell_time_array,
        vmin=0,
        vmax=vmax,
        cmap="cool",
        extent=(
            xlims[0],
            xlims[1],
            float(settings.protocol["adaptive_polish"]["gis_minimum_m"]) * 1e6,
            float(settings.protocol["adaptive_polish"]["gis_maximum_m"]) * 1e6,
        ),
        interpolation="bilinear",
        aspect="auto",
        zorder=1,
    )

    fig.colorbar(
        _,
        ax=axs[1, 2],
        orientation="horizontal",
        label="Bitmap pixel value" if bitmap_as_image else "Dwell time multiplier",
    )

    if save_path is not None:
        print(f"Saving {save_path}")
        fig.savefig(save_path)
        plt.close(fig)


def create_next_plots(
    electron_beam_image: FibsemImage,
    ion_beam_image: FibsemImage,
    settings: MicroscopeSettings,
    save_directory: Path,
    window_size_m: float,
) -> None:
    prediction, clean_prediction = make_predictions(electron_beam_image)

    gis_thicknesses_m, xlims = measure_gis(
        electron_beam_image=electron_beam_image,
        clean_prediction=clean_prediction,
        window_size_m=window_size_m,
    )

    bitmap_array = create_bitmap_array(
        gis_thickness_m=gis_thicknesses_m["filtered"],
        xlims=xlims,
        window_size_m=window_size_m,
        pixel_size_m=ion_beam_image.metadata.pixel_size.x,
        settings=settings,
        as_image=False,
    )

    file_stem = Path(electron_beam_image.get_save_path()).stem
    milling_cycle_plot(
        electron_beam_image,
        prediction,
        clean_prediction,
        ion_beam_image,
        gis_thicknesses_m=gis_thicknesses_m,
        settings=settings,
        xlims=xlims,
        bitmap_array=bitmap_array,
        image_name=file_stem,
        save_path=save_directory / f"{file_stem}.png",
    )


if __name__ == "__main__":
    base_path = (
        Path.home()
        / "OneDrive - The Rosalind Franklin Institute"
        / "Documents"
        / "test data"
        / "adaptive milling"
        / "2024segmentation_testdata"
    )
    model_path = (
        base_path.parent / "2024-02-24_0013_gis_lamela_crack_pytorch_AUnet.ptchkp"
    )
    config_path = (
        Path(__file__).parent.parent.parent.parent.parent
        / "fibsem"
        / "fibsem"
        / "config"
        / "microscope-configuration-demo2.yaml"
    )
    plot_centres = True

    protocol_path = Path(__file__).parent / "spoof_microscope_protocol.yaml"

    window_size_m = 2e-7

    microscope, settings = setup(
        model_path=model_path, config_path=config_path, protocol_path=protocol_path
    )

    plots_path = base_path / "bitmap_plots"
    plots_path.mkdir(exist_ok=True)

    if plot_centres:
        centre_plots_dir = plots_path / "centres"
        centre_plots_dir.mkdir(exist_ok=True)

        centre_diffs_path = centre_plots_dir / "diffs.txt"
        centre_diffs_path.unlink(missing_ok=True)

    for _ in range(8):
        # skip
        get_next_images(microscope=microscope, settings=settings)

    i = 1
    while True:
        try:
            electron_beam_image, ion_beam_image = get_next_images(
                microscope=microscope, settings=settings
            )

            try:
                if plot_centres:
                    elecron_centre_plot_path = (
                        centre_plots_dir
                        / f"{Path(electron_beam_image.get_save_path()).stem}.png"
                    )
                    ion_centre_plot_path = (
                        centre_plots_dir
                        / f"{Path(ion_beam_image.get_save_path()).stem}.png"
                    )
                else:
                    elecron_centre_plot_path = None
                    ion_centre_plot_path = None
                electron_centre = np.asarray(
                    find_centre(
                        electron_beam_image,
                        plot_path=elecron_centre_plot_path,
                        lamella_width_m=settings.protocol["milling"]["lamella"][
                            "stages"
                        ][0]["lamella_width"],
                        lamella_height_m=7e-6,
                    )
                )
                ion_centre = np.asarray(
                    find_centre(
                        ion_beam_image,
                        plot_path=ion_centre_plot_path,
                        lamella_width_m=settings.protocol["milling"]["lamella"][
                            "stages"
                        ][0]["lamella_width"],
                        lamella_height_m=settings.protocol["milling"]["lamella"][
                            "stages"
                        ][0]["lamella_height"],
                    )
                )
                if plot_centres:
                    x_diff = ion_centre - electron_centre
                    with (centre_diffs_path).open("a+") as f:
                        f.write(
                            f"{Path(electron_beam_image.get_save_path()).stem}: {x_diff}\n"
                        )

            except AssertionError:
                print("Failed to get centres")

        #     print(f"Starting plot {i}")
        #     create_next_plots(
        #         electron_beam_image=electron_beam_image,
        #         ion_beam_image=ion_beam_image,
        #         settings=settings,
        #         save_directory=plots_path,
        #         window_size_m=window_size_m,
        #     )
        #     print(f"Completed plot {i}")
        #     i += 1
        except Exception:
            logging.error("Exception occurred", exc_info=True)
            break
