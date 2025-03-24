# Copyright 2024 Rosalind Franklin Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import annotations
import logging
import typing
import xml.etree.ElementTree as ET  # to handle metadata as xml
from pathlib import Path

import skimage
import pandas as pd
import numpy as np
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt

from fibsem import constants

from adaptive_polish.dl_segmentation import sem_lamella_segmentor as sgm

if typing.TYPE_CHECKING:
    from os import PathLike
    from numpy.typing import NDArray, ArrayLike


load_sem_model = sgm.load_model


def get_pixel_width(img):
    """Gets the pixel width of an AdornedImage, or if not possible, returns None

    Args:
        img (AdornedImage): TFS AutoScript AdornedImage
    """
    try:
        xml_sett = img.metadata.metadata_as_xml
        xml_parse = ET.fromstring(xml_sett)
        pixel_size_x = xml_parse.find("BinaryResult/PixelSize/X")
        pixel_width_m = float(pixel_size_x.text)
        return pixel_width_m
    except Exception:
        # if there is no metadata with 'PixelWidth' defaults to the value above
        return None


def keep_only_largest_object(mask: NDArray[np.integer]) -> NDArray[np.bool_]:
    """Find the largest object in a mask and sets everything in that object to
    `fill_value`, background is 0.

    Args:
        mask (np.array): Segmentation mask

    Returns:
        np.array: Boolean mask with all but the largest object set to False
    """
    instances = skimage.measure.label(mask)
    largest_area = max([_.area for _ in skimage.measure.regionprops(instances)])
    mask_largest_only = skimage.morphology.remove_small_objects(
        instances, min_size=largest_area - 1
    )
    return mask_largest_only > 0


def clean_prediction(
    prediction: NDArray[np.integer],
    pixel_size_m: float,
    minimum_lamella_size_um2: float = 0,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_] | None]:
    logging.debug(
        "Cleaning prediction with shape: %s, pixel_size_m: %.4e",
        str(prediction.shape),
        pixel_size_m,
    )
    # Generate bool masks for GIS and lamella
    mask_gis = prediction == sgm.SegmentationLabels.GIS.value()
    mask_lamella = prediction == sgm.SegmentationLabels.LAMELLA.value()
    mask_crack = prediction == sgm.SegmentationLabels.CRACK.value()

    # Doing these all together could be an issue if the model fails too hard
    # (e.g. layers of gis and crack would mess up the GIS reading) but this
    # seems unlikely.
    # TODO: check whether the crack need to be touching lamella to be a crack
    mask_largest_foreground = keep_only_largest_object(
        mask_lamella + mask_gis + mask_crack
    )

    mask_connected_gis = np.logical_and(
        mask_largest_foreground, mask_gis, dtype=np.bool_
    )  # Gets the GIS that's connected to the lamella

    mask_connected_lamella = np.logical_and(
        mask_largest_foreground, mask_lamella, dtype=np.bool_
    )  # Gets the just the lamella from the foreground element

    mask_connected_crack = np.logical_and(
        mask_largest_foreground, mask_crack, dtype=np.bool_
    )  # Gets the crack that's connected to the lamella

    # Ensure lamella and GIS object is bigger than minimum size
    if (
        np.sum(mask_connected_lamella) * pixel_size_m * constants.SI_TO_MICRO
        <= minimum_lamella_size_um2
    ):
        raise ValueError(
            f"Failed to find lamella larger than {minimum_lamella_size_um2:.4e}um2",
        )

    # Throw away GIS above the lamella mask
    coords_lamella_bottom = mask_connected_lamella.shape[0] - np.argmax(
        mask_connected_lamella[::-1, :], axis=0
    )
    for x, coord in enumerate(coords_lamella_bottom):
        mask_connected_gis[:coord, x] = False

    # Ensure GIS found beneath the lamella
    if not sum(mask_connected_gis):
        raise ValueError("No GIS found beneath the lamella")

    if not np.sum(mask_connected_crack):
        # No need to keep an array of 0s
        mask_connected_crack = None

    return mask_connected_gis, mask_connected_lamella, mask_connected_crack


def apply_binary_opening(
    array: NDArray[np.bool_], window_size_m: float, pixel_size_m: float
):
    # Get window size in px
    if window_size_m > 0:
        window_size_px = int(round(window_size_m / pixel_size_m))
        return skimage.morphology.binary_opening(
            array,
            footprint=[
                (np.ones((window_size_px, 1)), 1),
                (np.ones((1, window_size_px)), 1),
            ],
            mode="ignore",
        )


def filter_gis_thickness(
    gis_thickness_px: NDArray[np.integer | np.floating],
    window_size_m: int,
    pixel_size_m: float,
) -> tuple[NDArray[np.float64], tuple[int, int]]:
    # TODO: Change window_size_m to beam FWHM
    # FWHM is 2 * sqrt(2 * np.log(2)) * sigma, which is approx 2.355 * sigma
    # The number of points in the gaussian curve should be approx 6 * std for convolution
    window_size_px = int(window_size_m / pixel_size_m)
    if window_size_px % 2 == 0:
        # An even window size means we won't get central point of the curve
        window_size_px += 1
    sigma = window_size_px / 6
    gaussian_curve = gaussian(window_size_m, std=sigma)
    gaussian_curve /= gaussian_curve.sum()
    return np.convolve(
        np.pad(gis_thickness_px, int(gaussian_curve.size / 2)),
        gaussian_curve,
        mode="valid",
    )


def filter_gis_thickness_fast(
    gis_thickness_px: NDArray[np.integer | np.floating],
    window_size_m: int,
    pixel_size_m: float,
) -> tuple[NDArray[np.float64], tuple[int, int]]:
    # Get window size in px
    window_size_px = int(window_size_m / pixel_size_m)
    logging.info(f"window_size_px: {window_size_px}")
    cumsum_vec = np.cumsum(np.pad(gis_thickness_px, int(window_size_px / 2)))
    return (cumsum_vec[window_size_px:] - cumsum_vec[:-window_size_px]) / window_size_px


def resize_image(image, new_shape: tuple[int, int]) -> NDArray[np.float32]:
    return skimage.transform.resize(
        # Needs to be floating type if we want interpolation
        image.astype(np.float32),
        output_shape=new_shape,
    )


def measure_gis(
    gis_mask: NDArray[typing.Any],
    window_size_m: float,
    pixel_size_m: float,
) -> NDArray[np.floating]:
    # Sum to get thickness along x in pixels
    gis_thickness_px = np.sum(gis_mask, axis=0)

    return filter_gis_thickness(
        gis_thickness_px=gis_thickness_px,
        window_size_m=window_size_m,
        pixel_size_m=pixel_size_m,
    )


def cleanup_crack_segmentation(prediction: np.array) -> float:
    """Measure the area in the prediction for cracks in um2.

    Cracks are only considered if they are within the largest combined lamella+
    GIS+crack object in the prediction.

    Args:
        prediction (np.array): Segmentation mask

    Returns:
        NDArray[np.bool_]: Segmentation of cracks that are connected to the lamella and/or GIS
    """
    # Restrict crack search area to the largest object which is not background
    mask_gis_lamella = np.isin(
        prediction,
        (
            sgm.SegmentationLabels.LAMELLA.value(),
            sgm.SegmentationLabels.GIS.value(),
        ),
    )
    mask_crack = prediction == 3

    mask_foreground_largest_only = keep_only_largest_object(
        mask_gis_lamella + mask_crack
    )

    return np.logical_and(mask_crack, mask_foreground_largest_only)


def get_mask_area_um2(mask: NDArray[np.bool_], pixel_size_m: float) -> float:
    return np.sum(mask) * (pixel_size_m**2) * (constants.SI_TO_MICRO**2)


def masks_to_labels(
    lamella_mask: NDArray[np.bool_] | None = None,
    gis_mask: NDArray[np.bool_] | None = None,
    crack_mask: NDArray[np.bool_] | None = None,
    background_mask: NDArray[np.bool_] | None = None,
    vacuum_mask: NDArray[np.bool] | None = None,
    default_value: int | float = np.nan,
) -> NDArray[typing.Any]:
    mask_label_pairs = [
        (lamella_mask, sgm.SegmentationLabels.LAMELLA),
        (gis_mask, sgm.SegmentationLabels.GIS),
        (crack_mask, sgm.SegmentationLabels.CRACK),
        (background_mask, sgm.SegmentationLabels.BACKGROUND),
        (vacuum_mask, sgm.SegmentationLabels.VACUUM),
    ]

    masks = []
    label_values = []
    for mask, label in mask_label_pairs:
        if mask is not None:
            masks.append(mask)
            label_values.append(label.value())
    return np.select(masks, label_values, default=default_value)


def milling_cycle_plot(
    sem_image: NDArray[typing.Any],
    first_prediction: NDArray[np.integer],
    clean_prediction: NDArray[typing.Any],
    fib_image: NDArray[typing.Any],
    gis_thickness_um: ArrayLike,
    gis_stop_um: float,
    crack_area_um2: float,
    xlims: tuple[int, int] | None = None,
    fib_screenshot: NDArray[typing.Any] | None = None,
    img_name: str | None = None,
    save_path: str | PathLike[str] | None = None,
):
    logging.debug("milling_cycle_plot()")
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), tight_layout=True)
    fig.suptitle(img_name)

    # SEM
    _ = axs[0, 0].imshow(sem_image, cmap="Greys_r")
    sem_image_extent = _.get_extent()
    axs[0, 0].axis("off")
    axs[0, 0].set_title("SEM")

    # SEM + 1st pass prediction
    axs[0, 1].imshow(sem_image, cmap="Greys_r")
    axs[0, 1].imshow(
        first_prediction,
        alpha=0.4,
        cmap="tab10",
        vmin=0,
        vmax=10,
        extent=sem_image_extent,
        interpolation="nearest",
    )
    axs[0, 1].axis("off")
    axs[0, 1].set_title("SEM, 1st prediction")

    # SEM + clean prediction
    axs[0, 2].imshow(sem_image, cmap="Greys_r")
    axs[0, 2].imshow(
        clean_prediction,
        alpha=0.4,
        cmap="tab10",
        vmin=0,
        vmax=10,
        extent=sem_image_extent,
        interpolation="nearest",
    )
    if xlims is not None:
        axs[0, 2].axvline(x=xlims[0])
        axs[0, 2].axvline(x=xlims[1])
    axs[0, 2].axis("off")
    axs[0, 2].set_title(f"SEM, clean, crack area $\mum^2$ = {crack_area_um2:.2f}")

    # FIB image
    axs[1, 0].imshow(fib_image, cmap="Greys_r")
    axs[1, 0].axis("off")
    axs[1, 0].set_title("FIB")

    # FIB + milling box
    if fib_screenshot is not None:
        # Doesn't work -> for some reason I can't open a new napari.Viewer()
        # pattern_viewer = napari.Viewer()
        # pattern_viewer.add_image(fib_image, name="fib_image")
        # _draw_patterns_in_napari(
        #     viewer=pattern_viewer,
        #     ib_image=FibsemImage(data=fib_image),
        #     eb_image=None,
        #     milling_stages=list(adaptive_polish_stage)
        # )
        # screenshot = pattern_viewer.screenshot()
        # axs[1, 1].imshow(screenshot)
        # pattern_viewer.close()
        axs[1, 1].imshow(fib_screenshot[:, int(fib_screenshot.shape[1] / 2) :, :])
    axs[1, 1].axis("off")

    # GIS thickness
    axs[1, 2].plot(gis_thickness_um, ".-")
    axs[1, 2].set_xlabel("Distance along x $px$")
    axs[1, 2].set_ylabel("GIS thickness ($\mum$)")
    axs[1, 2].set_xlim(0, len(gis_thickness_um))
    axs[1, 2].set_ylim(
        0,
    )
    axs[1, 2].hlines(
        y=gis_stop_um,
        xmin=0,
        xmax=len(gis_thickness_um),
        label="Target GIS",
        linestyles="dashed",
        colors="C1",
    )
    axs[1, 2].set_title(f"GIS thickness, min={np.nanmin(gis_thickness_um):.2f} $\mum$")
    axs[1, 2].legend()

    fig.savefig(save_path)
    plt.close(fig)


def summary_gis_plot(results: pd.DataFrame, save_path: str | PathLike[str]):
    save_path = Path(save_path)
    fig, ax = plt.subplot(1, 1)
    ax.plot(
        results.milling_time_s, results.min_GIS_um, label="Minimum GIS thickness $\mum$"
    )
    ax.set_xlabel("Milling Time $s$")
    ax.set_ylabel("GIS Thickness $\mum$")
    fig.suptitle(save_path.stem)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
