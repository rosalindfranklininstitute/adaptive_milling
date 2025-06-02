# general
from __future__ import annotations
import os
import logging
import yaml
from contextlib import contextmanager
from copy import deepcopy
import tempfile
from collections.abc import Generator
import typing

# for display
import PIL
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Adaptive polishing
from adaptive_polish import gis_measurement as gm
from adaptive_polish import bitmaps


# for the fibsem structures
from fibsem.structures import Point
from fibsem.milling import get_milling_stages, FibsemMillingStage
from fibsem.patterns.ui import (
    drawing_functions,
    COLOURS,
    PROPERTIES,
)

from adaptive_polish.strategy import AdaptivePolishMillingStrategy

# Set up test microscope
from fibsem import utils, acquire


if typing.TYPE_CHECKING:
    from os import PathLike
    from numpy.typing import NDArray
    from matplotlib.figure import Axes
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import FibsemImage, MicroscopeSettings

mpl.use("agg")

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logging.basicConfig(handlers=[stream_handler])


_logger = logging.getLogger(__name__)


def get_next_images(
    microscope: FibsemMicroscope, settings: MicroscopeSettings
) -> tuple[FibsemImage, FibsemImage]:
    # "Acquire" images
    return acquire.take_reference_images(microscope, settings.image)


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
    centre: tuple[float, float] | None = None,
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
            get_milling_stages_kwargs = {}
            if centre is not None:
                get_milling_stages_kwargs["point"] = Point(*centre)
            stages = get_milling_stages(
                "lamella", settings.protocol["milling"], **get_milling_stages_kwargs
            )
            draw_milling_patterns(ax, ion_beam_image, stages)
    except Exception:
        logging.error("Plotting bitmap pattern raised an exception", exc_info=True)


def milling_cycle_plot(
    sem_image: FibsemImage,
    first_prediction: NDArray[typing.Any],
    clean_prediction: NDArray[typing.Any],
    fib_image: FibsemImage,
    settings: MicroscopeSettings,
    gis_thicknesses_m: dict[str, NDArray[typing.Any]],
    xlims: tuple[float, float],
    bitmap_array: NDArray[np.uint8],
    centre: tuple[float, float],
    image_name: str = None,
    save_path: Path = None,
):
    logging.info("milling_cycle_plot()")
    fig, axs = plt.subplots(
        nrows=2, ncols=3, figsize=(18, 12), dpi=300, tight_layout=True
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
    axs[0, 1].set_title("SEM Segmentation")

    # # SEM + clean prediction
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
    # xlims_centre = xlims[0] + (xlims[1] - xlims[0]) / 2

    # lamella_centre_x = xlims_centre * sem_image.metadata.pixel_size.x

    # im_centre_x = (
    #     fib_image.metadata.image_settings.resolution[0] / 2
    # ) * fib_image.metadata.pixel_size.x

    # centre = (
    #     float(-im_centre_x + lamella_centre_x),
    #     0,
    # )
    centre = (
        centre[1] * fib_image.metadata.pixel_size.x,
        centre[0] * fib_image.metadata.pixel_size.y,
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

    lamella_width = 10e-6

    bitmap_physical_width = len(bitmap_array) * sem_image.metadata.pixel_size.x

    bitmap_array_padded = np.pad(
        bitmap_array,
        pad_width=int((lamella_width - bitmap_physical_width) / 2),
        constant_values=0,
    )

    plot_bitmap_trench_pattern(
        axs[1, 1],
        bitmap_array=bitmap_array_padded,
        lamella_width=lamella_width,  # (xlims[1] - xlims[0]) * sem_image.metadata.pixel_size.x,
        settings=settings,
        ion_beam_image=fib_image,
        centre=centre,
    )
    axs[1, 1].axis("off")
    axs[1, 1].set_title("Bitmap milling pattern")

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
    # axs[1, 2].plot(
    #     gis_thicknesses_m["opened"] * 1e6,
    #     "-",
    #     zorder=3,
    #     label="Opened GIS thickness",
    #     alpha=0.33,
    #     color="red",
    # )
    # axs[1, 2].plot(
    #     gis_thicknesses_m["filtered"] * 1e6,
    #     "-",
    #     zorder=4,
    #     label="Averaged opened GIS thickness",
    #     alpha=0.33,
    #     color="green",
    # )

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
        f"GIS thickness, min={np.nanmin(gis_thicknesses_m['filtered'][xlims[0] : xlims[1]]) * 1e6:.2f} $\mu$m"
    )
    # axs[1, 2].legend()

    # blanking_array = bitmap_array[:, :, 1].astype(np.bool_)
    # if bitmap_as_image:
    #     blanking_array = ~blanking_array
    # axs[1, 2].imshow(
    #     blanking_array,
    #     cmap="Reds",
    #     extent=(
    #         xlims[0],
    #         xlims[1],
    #         0,
    #         float(settings.protocol["adaptive_polish"]["gis_stop_m"]) * 1e6,
    #     ),
    #     interpolation="none",
    #     aspect="auto",
    #     zorder=1,
    # )

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
    model,
    electron_beam_image: FibsemImage,
    ion_beam_image: FibsemImage,
    settings: MicroscopeSettings,
    save_directory: Path,
    window_size_px: int,
) -> None:
    # Segmentation
    _logger.info("Starting segmentation")
    prediction = model.predict(gm.electron_beam_image.data, full_size=False)
    _logger.info("Segmentation complete")

    prediction_pixel_size_um = (
        electron_beam_image.metadata.pixel_size.x
        * 1e6  # m to um
        * electron_beam_image.data.shape[1]
        / prediction.shape[1]
    )

    mask_lamella_clean, mask_gis_clean, mask_crack_clean = gm.clean_prediction(
        prediction,
        additional_labels=(
            gm.SegmentationLabels.GIS,
            gm.SegmentationLabels.CRACK,
        ),
    )
    lamella_area_um2 = gm.get_mask_area_um2(
        mask_lamella_clean, pixel_size_um=prediction_pixel_size_um
    )
    if lamella_area_um2 < 30:
        _logger.warning(
            "Lamella found was only %.4e um2, below the threshold of 30 um2",
            lamella_area_um2,
        )
    if mask_gis_clean is None:
        _logger.warning("No GIS found beneath the lamella")

    # Get lamella position
    centre_m, mask_centre_px, lamella_bbox = (
        AdaptivePolishMillingStrategy._get_lamella_position(
            electron_beam_image, lamella_mask=mask_lamella_clean
        )
    )

    # Measure GIS
    gis_thickness_um = np.sum(
        gm.resize_image(mask_gis_clean, new_shape=electron_beam_image.data.shape),
        axis=0,
    )

    # Use lamella bounds to determine gis edges
    xlims_px = np.round((lamella_bbox[1], lamella_bbox[3])).astype(np.uint32)

    window_size_m = window_size_px * electron_beam_image.metadata.pixel_size.x

    gis_thickness_filtered_um = gm.filter_gis_thickness(
        gis_thickness_um[xlims_px[0] : xlims_px[1] + 1],
        window_size_m=window_size_m,
        pixel_size_m=electron_beam_image.metadata.pixel_size.x,
    )
    min_gis_um = np.nanmin(gis_thickness_filtered_um)
    _logger.info(f"Took {len(gis_thickness_filtered_um)} GIS measurements along x")

    if mask_crack_clean is None:
        crack_area_um2 = 0
    else:
        crack_area_um2 = gm.get_mask_area_um2(
            mask_crack_clean, pixel_size_um=prediction_pixel_size_um
        )

    if crack_area_um2 > 2:
        _logger.warning(f"Crack area (um2) {crack_area_um2:.4e} > threshold 2 um2")

    bitmap_array = bitmaps.create_bitmap_array(
        gis_thickness_m=gis_thickness_filtered_um,
        xlims=xlims_px,
        window_size_m=window_size_m,
        pixel_size_m=ion_beam_image.metadata.pixel_size.x,
        max_dwell_thickness_m=float(
            settings.protocol["adaptive_polish"]["gis_maximum_m"]
        ),
        min_dwell_thickness_m=float(
            settings.protocol["adaptive_polish"]["gis_minimum_m"]
        ),
        blanking_thickness_m=-1,  # float(settings.protocol["adaptive_polish"]["gis_stop_m"]),
        blanking_width_m=1.6e-6,
        as_image=False,
    )

    file_stem = Path(electron_beam_image.get_save_path()).stem
    milling_cycle_plot(
        electron_beam_image,
        prediction,
        gm.masks_to_labels(
            lamella_mask=mask_lamella_clean,
            gis_mask=mask_gis_clean,
            crack_mask=mask_crack_clean,
        ),
        ion_beam_image,
        gis_thicknesses_m=gis_thickness_filtered_um * 1e-6,  # um to m
        settings=settings,
        xlims=xlims_px,
        bitmap_array=bitmap_array,
        image_name=file_stem,
        save_path=save_directory / f"{file_stem}.png",
    )

    # bitmap_image = bitmaps.create_bitmap_array(
    #     gis_thickness_m=gis_thickness_filtered_um * 1e-6,
    #     xlims=xlims_px,
    #     window_size_m=window_size_m,
    #     pixel_size_m=ion_beam_image.metadata.pixel_size.x,
    #     max_dwell_thickness_m=float(
    #         settings.protocol["adaptive_polish"]["gis_maximum_m"]
    #     ),
    #     min_dwell_thickness_m=float(
    #         settings.protocol["adaptive_polish"]["gis_minimum_m"]
    #     ),
    #     blanking_thickness_m=-1,  # float(settings.protocol["adaptive_polish"]["gis_stop_m"]),
    #     blanking_width_m=1.6e-6,
    #     as_image=True,
    # )
    # bitmap_image_total_diameter = np.tile(bitmap_image, (4, 1, 1))
    # PIL.Image.fromarray(bitmap_image_total_diameter).save(
    #     save_directory / f"{file_stem}_total_diameter.bmp", format="BMP"
    # )
    # bitmap_image_pixel_size = np.tile(bitmap_image, (27, 1, 1))
    # PIL.Image.fromarray(bitmap_image_pixel_size).save(
    #     save_directory / f"{file_stem}_pixel_size.bmp", format="BMP"
    # )


if __name__ == "__main__":
    base_path = (
        Path.home()
        / "OneDrive - The Rosalind Franklin Institute"
        / "Documents"
        / "test data"
        / "adaptive milling"
    )
    model_path = (
        base_path
        / "sem_models"
        / "Gen0"
        / "2024-02-24_0013_gis_lamela_crack_pytorch_AUnet.ptchkp"
    )
    config_path = (
        Path(__file__).parent.parent.parent.parent.parent
        / "fibsem"
        / "fibsem"
        / "config"
        / "microscope-configuration-demo2.yaml"
    )

    experiment_paths = (
        # base_path / "2024segmentation_testdata",
        base_path / "09-pretty-dingo",
        # base_path / "18-whole-piglet",
    )

    plot_centres = True

    protocol_path = Path(__file__).parent / "spoof_microscope_protocol.yaml"

    window_size_m = 2e-7

    plots_path = base_path / "bitmap_plots"
    plots_path.mkdir(exist_ok=True)

    model = gm.load_sem_model(model_path=model_path, generation=0)

    for experiment_path in experiment_paths:
        experiment_path = Path(experiment_path)
        # get fib path:
        fib_dir = experiment_path / "FIB"
        if not fib_dir.is_dir():
            fib_dir = experiment_path / "fib"
            if not fib_dir.is_dir():
                raise FileNotFoundError(f"No FIB directory in '{experiment_path}'")
        sem_dir = experiment_path / "SEM"
        if not sem_dir.is_dir():
            sem_dir = experiment_path / "sem"
            if not sem_dir.is_dir():
                raise FileNotFoundError(f"No SEM directory in '{experiment_path}'")

        with config_path.open("r") as f:
            tmp_microscope_config = yaml.safe_load(f)
        tmp_microscope_config["demo2"]["SEM_folder_path_str"] = str(sem_dir)
        tmp_microscope_config["demo2"]["FIB_folder_path_str"] = str(fib_dir)

        tmp_cfg_path = None
        try:
            # tempfile doesn't work properly with Windows if delete=True (raises PermissionError)
            with tempfile.TemporaryFile("w", delete=False) as tmp_cfg:
                yaml.safe_dump(tmp_microscope_config, tmp_cfg)
                tmp_cfg_path = tmp_cfg.name

                microscope, settings = microscope, settings = utils.setup_session(
                    config_path=config_path, protocol_path=protocol_path
                )
                plots_path = experiment_path / "bitmap_plots"
                plots_path.mkdir(exist_ok=True)
                i = 1
                while True:
                    try:
                        electron_beam_image, ion_beam_image = get_next_images(
                            microscope=microscope, settings=settings
                        )
                        if electron_beam_image is None:
                            logging.info(
                                "No further images for experiment %s", experiment_path
                            )
                            break

                        print(f"Starting plot {i}")
                        create_next_plots(
                            model=model,
                            electron_beam_image=electron_beam_image,
                            ion_beam_image=ion_beam_image,
                            settings=settings,
                            save_directory=plots_path,
                            window_size_m=window_size_m,
                        )
                        print(f"Completed plot {i}")
                        i += 1
                    except Exception:
                        logging.error("Exception occurred", exc_info=True)
                        break
        finally:
            if tmp_cfg_path is not None:
                os.unlink(tmp_cfg_path)
