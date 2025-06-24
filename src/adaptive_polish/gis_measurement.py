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
from math import ceil, floor

import numpy as np
from scipy.signal.windows import gaussian
from skimage import measure, transform

from adaptive_polish.edges import get_mask_edge
from adaptive_polish.dl_segmentation import sem_lamella_segmentor as sgm

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


_logger = logging.getLogger(__name__)

DEFAULT_SEM_MODEL_GENERATION = sgm._get_newest_generation_key()

load_sem_model = sgm.load_model


def keep_only_largest_object(
    mask: NDArray[typing.Union[np.integer[typing.Any], np.bool_]],
    connectivity: int = 2,
) -> NDArray[np.bool_]:
    """Find the largest object in a mask and sets everything in that object to
    `fill_value`, background is 0.

    Args:
        mask (NDArray[typing.Union[np.integer[typing.Any], np.bool_]]): Segmentation mask
        connectivity (int): connectivity when finding objects (1 is edges only, 2 includes corners)

    Returns:
        NDArray[np.bool_]: Boolean mask with all but the largest object set to False
    """
    labels, num = measure.label(mask, return_num=True, connectivity=connectivity)
    if num == 1:
        return labels.astype(np.bool_)
    prop = max(measure.regionprops(labels), key=lambda x: x.area)
    return labels == prop.label


def clean_lamella(prediction: NDArray[np.integer]) -> NDArray[np.bool_]:
    # TODO: check whether the crack need to be touching lamella to be a crack
    mask_lamella = prediction == sgm.SegmentationLabels.LAMELLA.value
    mask_gis_crack = np.isin(
        prediction,
        (
            sgm.SegmentationLabels.GIS.value,
            sgm.SegmentationLabels.CRACK.value,
        ),
    )

    mask_largest_foreground = keep_only_largest_object(mask_lamella + mask_gis_crack)
    return mask_largest_foreground & mask_lamella


def prediction_to_masks(
    prediction: NDArray[np.integer[typing.Any]],
) -> dict[sgm.SegmentationLabels, NDArray[np.bool_]]:
    return {label: prediction == label.value for label in sgm.SegmentationLabels}


def clean_prediction(
    prediction: NDArray[np.integer[typing.Any]],
) -> NDArray[np.uint8]:
    _logger.debug(
        "Cleaning prediction with shape: %s",
        str(prediction.shape),
    )

    # Generate bool masks
    masks = prediction_to_masks(prediction=prediction)

    # Doing these all together could be an issue if the model fails too hard
    # (e.g. layers of gis and crack would mess up the GIS reading) but this
    # seems unlikely.
    # TODO: check whether the crack need to be touching lamella to be a crack
    mask_largest_foreground = keep_only_largest_object(
        masks[sgm.SegmentationLabels.LAMELLA]
        + masks[sgm.SegmentationLabels.GIS]
        + masks[sgm.SegmentationLabels.CRACK]
    )

    non_foreground_crack = (
        masks[sgm.SegmentationLabels.CRACK] & ~mask_largest_foreground
    )
    non_foreground_gis = masks[sgm.SegmentationLabels.GIS] & ~mask_largest_foreground
    return masks_to_labels(
        masks={
            sgm.SegmentationLabels.LAMELLA: mask_largest_foreground
            & masks[sgm.SegmentationLabels.LAMELLA],
            sgm.SegmentationLabels.GIS: mask_largest_foreground
            & masks[sgm.SegmentationLabels.GIS],
            sgm.SegmentationLabels.CRACK: mask_largest_foreground
            & masks[sgm.SegmentationLabels.CRACK],
            sgm.SegmentationLabels.VACUUM: masks[sgm.SegmentationLabels.VACUUM]
            + non_foreground_crack,
            sgm.SegmentationLabels.BACKGROUND: masks[sgm.SegmentationLabels.BACKGROUND]
            + non_foreground_gis,
        },
        # Use an unused value so anything that isn't filled in is visible
        default_value=len(sgm.SegmentationLabels),
    ).astype(np.uint8)


def filter_gis_thickness(
    gis_thickness_px: NDArray[typing.Union[np.integer, np.floating]],
    window_size_m: float,
    pixel_size_m: float,
) -> NDArray[np.floating[typing.Any]]:
    # TODO: Change window_size_m to beam FWHM
    # FWHM is 2 * sqrt(2 * np.log(2)) * sigma, which is approx 2.355 * sigma
    # The number of points in the gaussian curve should be approx 6 * std for convolution
    window_size_px = int(window_size_m / pixel_size_m)
    if window_size_px % 2 == 0:
        # An even window size means we won't get central point of the curve
        window_size_px += 1
    sigma = window_size_px / 6
    gaussian_curve = gaussian(window_size_px, std=sigma)
    gaussian_curve /= gaussian_curve.sum()
    return np.convolve(
        np.pad(gis_thickness_px, int(gaussian_curve.size / 2), mode="constant"),
        gaussian_curve,
        mode="valid",
    )


def filter_gis_thickness_fast(
    gis_thickness_px: NDArray[typing.Union[np.integer, np.floating]],
    window_size_m: int,
    pixel_size_m: float,
) -> NDArray[np.float64]:
    # Get window size in px
    window_size_px = int(window_size_m / pixel_size_m)
    _logger.info(f"window_size_px: {window_size_px}")
    cumsum_vec = np.cumsum(np.pad(gis_thickness_px, int(window_size_px / 2)))
    return (cumsum_vec[window_size_px:] - cumsum_vec[:-window_size_px]) / window_size_px


def resize_image(
    image: NDArray[typing.Any], new_shape: typing.Tuple[int, int]
) -> NDArray[np.float_]:
    if not isinstance(image.dtype, np.floating):
        # Needs to be floating type if we want interpolation
        image = image.astype(np.float_)
    return transform.resize(
        image,
        output_shape=new_shape,
        preserve_range=True,
    )


def get_mask_area_um2(mask: NDArray[np.bool_], pixel_size_um: float) -> float:
    return float(np.sum(mask) * (pixel_size_um**2))


def check_minimum_area(
    mask: NDArray[np.bool_], pixel_size: float, minimum_size: float
) -> bool:
    # Ensure lamella and GIS object is bigger than minimum size
    return bool(np.sum(mask) * pixel_size <= minimum_size)


def masks_to_labels(
    masks: typing.Dict[sgm.SegmentationLabels, NDArray[np.bool_]],
    default_value: typing.Union[int, float] = np.nan,
) -> NDArray[typing.Any]:
    return np.select(
        list(masks.values()),
        [_.value for _ in masks.keys()],
        default=default_value,
    )


def get_gis_thickness(
    prediction: NDArray[np.integer[typing.Any]],
    lamella_mask_bbox: tuple[float, float, float, float],
    image_shape: typing.Optional[tuple[int, int]],
) -> NDArray[np.float32]:
    """Get an array of GIS thickness values across the width specified by image_shape (or by the masks not given)

    Note: undefined pixels will be treated as if they are vacuum/crack."""
    prediction_shape = prediction.shape

    # Only include mask that is lamella and below
    slicer = (
        slice(int(floor(lamella_mask_bbox[0])), None),
        slice(int(floor(lamella_mask_bbox[1])), int(ceil(lamella_mask_bbox[3])) + 1),
    )
    prediction = prediction[slicer]

    mask_lamella = prediction == sgm.SegmentationLabels.LAMELLA.value
    mask_gis = prediction == sgm.SegmentationLabels.GIS.value
    mask_background = prediction == sgm.SegmentationLabels.BACKGROUND.value
    mask_bad = np.isin(
        prediction,
        (
            sgm.SegmentationLabels.CRACK.value,
            sgm.SegmentationLabels.VACUUM.value,
        ),
    )

    mask_gis_background = mask_gis + mask_background

    mask_good = mask_gis + mask_lamella

    good_bottom = get_mask_edge(mask_good, axis=0, side="max")

    # Set everything above the good bottom to False for mask_bad
    for y, x in good_bottom:
        mask_bad[: y + 1, x] = False

    # Ignore GIS/background below the top of the lower crack/vacuum area
    for y, x in get_mask_edge(mask_bad, axis=0, side="min"):
        mask_gis_background[y:, x] = False

    # Ignore GIS/background above the bottom of the lamella
    for y, x in get_mask_edge(mask_lamella, axis=0, side="max"):
        mask_gis_background[: y + 1, x] = False

    new_mask_gis: NDArray[typing.Union[np.bool_, np.float_]]
    new_mask_gis = np.zeros(prediction_shape, dtype=np.bool_)
    new_mask_gis[slicer] = mask_gis_background

    if image_shape is not None:
        new_mask_gis = resize_image(
            new_mask_gis,
            new_shape=(image_shape[0], image_shape[1]),
        )
    return np.sum(
        new_mask_gis,
        axis=0,
        dtype=np.float32,
    )
