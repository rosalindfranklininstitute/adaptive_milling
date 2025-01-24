# general
from __future__ import annotations
import os
import logging
from contextlib import contextmanager
from copy import deepcopy
import tempfile
from collections.abc import Generator
import typing

# for display
import PIL
from scipy.signal import find_peaks
from skimage import filters, morphology
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
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


def find_centre(
    image: FibsemImage, plot_path: str | PathLike[str] | None = None
) -> tuple[float, float]:
    """Finds the centre of the lamella based on the positions of the stress relief cuts. Definitely room for improvement."""
    # TODO: link sigma and peaks widths to physical scale via pixel size
    # TODO: handle waffle cuts etc by using the edges of the central void?
    data = image.data.astype(np.float64)
    gaussian1 = ndi.gaussian_filter(data, (50, 4), axes=(0, 1))
    gaussian2 = ndi.gaussian_filter(data, (5, 50), axes=(0, 1))
    dog = gaussian1 - gaussian2
    median_flattened2 = np.median(dog, axis=0).flatten()  # get median across y-axis
    thresholed_value = filters.threshold_li(median_flattened2)
    thresholded_1d = median_flattened2 < thresholed_value
    peaks, peak_properties = find_peaks(thresholded_1d, width=(20, 100))
    lamella_centre_px = peaks[-1] - (peaks[-1] - peaks[0]) / 2
    assert len(peaks) >= 2, "Too few peaks found"

    if plot_path is not None:
        fig, ax = plt.subplots(1, 1)
        plt.imshow(image.data, cmap="Greys_r")
        ax.axvline(x=peaks[0])
        ax.axvline(x=peaks[-1])
        if len(peaks) > 2:
            for peak in peaks[1:-1]:
                ax.axvline(x=peak, ls="--")
        ax.axvline(x=lamella_centre_px, color="C2")
        ax.hlines(
            y=[image.data.shape[0] / 2] * len(peaks),
            xmin=peak_properties["left_ips"],
            xmax=peak_properties["right_ips"],
            color="C3",
        )
        fig.savefig(plot_path)
        plt.close()

    return lamella_centre_px * image.metadata.pixel_size.x


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
    plot_centres = False

    protocol_path = Path(__file__).parent / "spoof_microscope_protocol.yaml"

    window_size_m = 1e-7

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
                electron_centre = find_centre(
                    electron_beam_image,
                    plot_path=elecron_centre_plot_path,
                )
                ion_centre = find_centre(
                    ion_beam_image,
                    plot_path=ion_centre_plot_path,
                )
                if plot_centres:
                    x_diff = ion_centre - electron_centre
                    with (centre_diffs_path).open("a+") as f:
                        f.write(
                            f"{Path(electron_beam_image.get_save_path()).stem}: {x_diff}\n"
                        )

            except AssertionError:
                print("Failed to get centres")

            print(f"Starting plot {i}")
            create_next_plots(
                electron_beam_image=electron_beam_image,
                ion_beam_image=ion_beam_image,
                settings=settings,
                save_directory=plots_path,
            )
            print(f"Completed plot {i}")
            i += 1
        except Exception:
            logging.error("Exception occurred", exc_info=True)
            break
