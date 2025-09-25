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
from scipy.ndimage import median_filter, gaussian_filter1d

from adaptive_polish.processing.image import (
    resize_image,
    keep_only_largest_object,
    get_mask_edge,
    get_mask_edges,
    bbox_to_xlims,
    bbox_to_ylims,
    get_bounding_box_from_edges,
    get_centre_from_bounding_box,
)
from adaptive_polish.processing.sem_segmentation import (
    SegmentationLabels as SemSegmentationLabels,
)

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


_logger = logging.getLogger(__name__)


def clean_lamella(prediction: NDArray[np.integer]) -> NDArray[np.bool_]:
    mask_lamella = prediction == SemSegmentationLabels.LAMELLA.value
    mask_gis_crack = np.isin(
        prediction,
        (
            SemSegmentationLabels.GIS.value,
            SemSegmentationLabels.CRACK.value,
        ),
    )

    mask_largest_foreground = keep_only_largest_object(mask_lamella + mask_gis_crack)
    return mask_largest_foreground & mask_lamella


def prediction_to_masks(
    prediction: NDArray[np.integer[typing.Any]],
) -> dict[SemSegmentationLabels, NDArray[np.bool_]]:
    return {label: prediction == label.value for label in SemSegmentationLabels}


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
    mask_largest_foreground = keep_only_largest_object(
        masks[SemSegmentationLabels.LAMELLA]
        + masks[SemSegmentationLabels.GIS]
        + masks[SemSegmentationLabels.CRACK]
    )

    non_foreground_crack = masks[SemSegmentationLabels.CRACK] & ~mask_largest_foreground
    non_foreground_gis = masks[SemSegmentationLabels.GIS] & ~mask_largest_foreground
    return masks_to_labels(
        masks={
            SemSegmentationLabels.LAMELLA: mask_largest_foreground
            & masks[SemSegmentationLabels.LAMELLA],
            SemSegmentationLabels.GIS: mask_largest_foreground
            & masks[SemSegmentationLabels.GIS],
            SemSegmentationLabels.CRACK: mask_largest_foreground
            & masks[SemSegmentationLabels.CRACK],
            SemSegmentationLabels.VACUUM: masks[SemSegmentationLabels.VACUUM]
            + non_foreground_crack,
            SemSegmentationLabels.BACKGROUND: masks[SemSegmentationLabels.BACKGROUND]
            + non_foreground_gis,
        },
        # Use an unused value so anything that isn't filled in is visible
        default_value=len(SemSegmentationLabels),
    ).astype(np.uint8)


def get_lamella_centre(
    array: NDArray[np.bool_],
    subpixel_accuracy: bool = False,
    edge_finding: typing.Literal["median", "mean", "percentile"] = "median",
    percentile: float | None = None,
) -> tuple[int, int] | tuple[float, float]:
    bbox = get_bounding_box_from_edges(
        get_mask_edges(mask=array), edge_finding=edge_finding, percentile=percentile
    )
    return get_centre_from_bounding_box(bbox, subpixel_accuracy=subpixel_accuracy)


def filter_gis_thickness_OLD(
    gis_thickness_px: NDArray[np.integer | np.float64 | np.float32],
    window_size_m: float,
    pixel_size_m: float,
) -> NDArray[np.float64 | np.float32]:
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


def filter_gis_thickness(
    gis_thickness_px: NDArray[np.integer | np.float64 | np.float32],
    xlims_px: tuple[int, int],
    sigma: float,
) -> NDArray[np.float_]:
    # Filter GIS thickness within xlims to a avoid edge artifacts
    gis_thickness_filtered_px = np.zeros_like(gis_thickness_px, dtype=float)
    gis_thickness_filtered_px[xlims_px[0] : xlims_px[1] + 1] = gaussian_filter1d(
        gis_thickness_px[xlims_px[0] : xlims_px[1] + 1],
        sigma=sigma,
        mode="constant",
        cval=0,
        truncate=6,
    )
    return gis_thickness_filtered_px


def get_mask_area_um2(mask: NDArray[np.bool_], pixel_size_um: float) -> float:
    return float(np.sum(mask) * (pixel_size_um**2))


def check_minimum_area(
    mask: NDArray[np.bool_], pixel_size: float, minimum_size: float
) -> bool:
    # Ensure lamella and GIS object is bigger than minimum size
    return bool(np.sum(mask) * pixel_size <= minimum_size)


def masks_to_labels(
    masks: dict[SemSegmentationLabels, NDArray[np.bool_]],
    default_value: int | float = np.nan,
) -> NDArray[np.float64]:
    return np.select(
        list(masks.values()),
        [_.value for _ in masks.keys()],
        default=default_value,
    )


def get_gis_thickness_old(
    prediction: NDArray[np.integer[typing.Any]],
    lamella_mask_bbox: tuple[float, float, float, float],
    image_shape: typing.Optional[tuple[int, int]],
) -> NDArray[np.float32]:
    """Get an array of GIS thickness values across the width specified by image_shape (or by the masks not given)

    Note: undefined pixels will be treated as if they are vacuum/crack."""
    # Only include mask that is lamella and below
    prediction_xlims = bbox_to_xlims(lamella_mask_bbox, (0, prediction.shape[0] - 1))
    prediction_ylims = bbox_to_ylims(lamella_mask_bbox, (0, prediction.shape[1] - 1))

    slicer = (
        slice(prediction_ylims[0], None),
        slice(prediction_xlims[0], prediction_xlims[1] + 1),
    )
    prediction_slice = prediction[slicer]

    mask_lamella = prediction_slice == SemSegmentationLabels.LAMELLA.value
    mask_gis = prediction_slice == SemSegmentationLabels.GIS.value
    mask_background = prediction_slice == SemSegmentationLabels.BACKGROUND.value
    mask_bad = np.isin(
        prediction_slice,
        (
            SemSegmentationLabels.CRACK.value,
            SemSegmentationLabels.VACUUM.value,
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
    new_mask_gis = np.zeros(prediction.shape, dtype=np.bool_)
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


def get_gis_thickness(
    prediction: NDArray[np.integer[typing.Any]],
    xlims: tuple[int | None, int | None] = (None, None),
    ylims: tuple[int | None, int | None] = (None, None),
    image_shape: typing.Optional[tuple[int, int]] = None,
    clean_edges: bool = True,
) -> tuple[NDArray[np.float32], tuple[int, int]]:
    """Get an array of GIS thickness values across the width specified by image_shape (or by the masks not given)

    Note: undefined pixels will be treated as if they are vacuum/crack."""

    # Pad slices by 1 on each side (unless at limits) to avoid scaling edge issues
    slicer = (
        slice(None if ylims[0] is None else max(0, ylims[0]), None),
        slice(
            None
            if xlims[0] is None
            else max(
                0,
                xlims[0] - 1,
            ),
            None if xlims[1] is None else min(xlims[1] + 2, prediction.shape[1]),
        ),
    )
    prediction_slice = prediction[slicer]

    mask_lamella = prediction_slice == SemSegmentationLabels.LAMELLA.value
    mask_gis = prediction_slice == SemSegmentationLabels.GIS.value
    mask_background = prediction_slice == SemSegmentationLabels.BACKGROUND.value
    mask_bad = np.isin(
        prediction_slice,
        (
            SemSegmentationLabels.CRACK.value,
            SemSegmentationLabels.VACUUM.value,
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
    new_mask_gis = np.full(
        prediction.shape, fill_value=np.nan if clean_edges else 0, dtype=np.float64
    )
    new_mask_gis[slicer] = mask_gis_background

    if image_shape is None or image_shape == (prediction.shape[0], prediction.shape[1]):
        xlims_out = (
            0 if xlims[0] is None else xlims[0],
            prediction.shape[1] - 1 if xlims[1] is None else xlims[1],
        )
    else:
        x_scaling = image_shape[1] / prediction.shape[1]
        xlims_out = (
            0 if xlims[0] is None else floor(xlims[0] * x_scaling),
            image_shape[1] if xlims[1] is None else ceil(xlims[1] * x_scaling),
        )
        new_mask_gis = resize_image(
            new_mask_gis,
            new_shape=image_shape,
        )

    sliced_gis_thickness = np.nansum(
        new_mask_gis,
        axis=0,
        dtype=np.float32,
    )

    return sliced_gis_thickness, xlims_out


def crop_xlims_convolve(
    gis_thickness: NDArray[np.float32 | np.float64],
    xlims: tuple[int, int],
    lamella_width: int,
    edge_size: int = 17,
    sigma: float = 0.7,
) -> tuple[int, int]:
    """This function uses convolution to search for indents caused by the beam
    stopping briefly at the edges of the milling pattern. As such, it assumes a
    pattern with the same width has previously been milled in the position
    being searched for.

    After testing using the pattern_centring.py script, the following were
    found to be optimal:
    edge_size=17
    sigma=0.7
    """
    min_diffs = _get_milled_area_convolve_gaussian(
        np.log(gis_thickness[xlims[0] : xlims[1] + 1]),
        width=lamella_width,
        edge_size=edge_size,
        sigma=sigma,
    )
    return (xlims[0] + min_diffs[0], xlims[0] + min_diffs[1])


def crop_xlims_convolve_filtered(
    gis_thickness: NDArray[np.float32 | np.float64],
    xlims: tuple[int, int],
    lamella_width: int,
    edge_size: int = 15,
    sigma: float = 0.8,
    filter_size: int = 136,
) -> tuple[int, int]:
    trimmed_thickness = gis_thickness[xlims[0] : xlims[1] + 1]
    trimmed_filtered_thickness = median_filter(gis_thickness, size=filter_size)[
        xlims[0] : xlims[1] + 1
    ]
    min_diffs = _get_milled_area_convolve_gaussian(
        np.log(trimmed_thickness / trimmed_filtered_thickness),
        width=lamella_width,
        edge_size=edge_size,
        sigma=sigma,
    )
    return (xlims[0] + min_diffs[0], xlims[0] + min_diffs[1])


def _get_milled_area_convolve_gaussian(
    data: NDArray[np.float32 | np.float64],
    width: int,
    edge_size: int = 17,
    sigma: float = 0.7,
) -> tuple[int, int]:
    kernel = np.zeros((width,))
    if edge_size <= 0:
        raise ValueError("edge_size must be above 0")

    window_size_px = edge_size
    gaussian_curve = gaussian(window_size_px, std=sigma)
    # The gaussian peaks should be 6 sigma pixels inside kernel
    trim = ceil((window_size_px - 1) / 2 + sigma * 6)
    kernel = np.concatenate(
        [gaussian_curve, kernel[trim : width - trim], gaussian_curve],
        axis=0,
    )

    conv = np.convolve(data, kernel, mode="valid")
    idx = int(np.argmin(conv))
    idx += trim - window_size_px
    return (idx, idx + width)


def crop_xlims_centre(
    lamella_width: float,
    xlims: tuple[int, int],
) -> tuple[int, int]:
    crop_amount = round((1 + xlims[1] - xlims[0] - lamella_width) / 2)
    return (
        xlims[0] + floor(crop_amount),
        xlims[1] - ceil(crop_amount),
    )
