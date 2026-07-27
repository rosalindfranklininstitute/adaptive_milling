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
import enum
import logging
import typing
from math import ceil, floor

import numpy as np
from scipy.signal.windows import gaussian
from scipy.ndimage import median_filter, gaussian_filter1d

from adaptive_milling.processing.image import (
    resize_image,
    keep_only_largest_object,
    filter_connected,
    get_mask_edge,
    get_mask_edges,
    get_bounding_box_from_edges,
    get_centre_from_bounding_box,
)

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


_logger = logging.getLogger(__name__)


class SEMSegmentationLabels(enum.IntEnum):
    BACKGROUND = 0
    GIS = 1
    LAMELLA = 2
    CRACK = 3
    VACUUM = 4


def clean_lamella(prediction: NDArray[np.integer]) -> NDArray[np.bool_]:
    mask_lamella = prediction == SEMSegmentationLabels.LAMELLA.value
    mask_gis_crack = np.isin(
        prediction,
        (
            SEMSegmentationLabels.GIS.value,
            SEMSegmentationLabels.CRACK.value,
        ),
    )

    mask_largest_foreground = keep_only_largest_object(mask_lamella + mask_gis_crack)
    return mask_largest_foreground & mask_lamella


def prediction_to_masks(
    prediction: NDArray[np.integer[typing.Any]],
) -> dict[SEMSegmentationLabels, NDArray[np.bool_]]:
    return {label: prediction == label.value for label in SEMSegmentationLabels}


def clean_prediction(
    prediction: NDArray[np.integer[typing.Any]],
) -> NDArray[np.uint8]:
    _logger.debug(
        "Cleaning prediction with shape: %s",
        str(prediction.shape),
    )

    # Generate bool masks
    masks = prediction_to_masks(prediction=prediction)

    mask_largest_foreground = keep_only_largest_object(
        masks[SEMSegmentationLabels.LAMELLA]
        + masks[SEMSegmentationLabels.GIS]
        + masks[SEMSegmentationLabels.CRACK]
    )

    mask_foreground_lamella = (
        mask_largest_foreground & masks[SEMSegmentationLabels.LAMELLA]
    )
    mask_foreground_gis = mask_largest_foreground & masks[SEMSegmentationLabels.GIS]
    mask_foreground_crack = (
        # Only include the cracks that are connected to some lamella
        filter_connected(mask_foreground_lamella, masks[SEMSegmentationLabels.CRACK])
        & masks[SEMSegmentationLabels.CRACK]
    )

    non_foreground_gis = masks[SEMSegmentationLabels.GIS] & ~mask_foreground_gis
    non_foreground_crack = masks[SEMSegmentationLabels.CRACK] & ~mask_foreground_crack

    return masks_to_labels(
        masks={
            SEMSegmentationLabels.LAMELLA: mask_foreground_lamella,
            SEMSegmentationLabels.GIS: mask_foreground_gis,
            SEMSegmentationLabels.CRACK: mask_foreground_crack,
            SEMSegmentationLabels.VACUUM: masks[SEMSegmentationLabels.VACUUM]
            + non_foreground_crack,
            SEMSegmentationLabels.BACKGROUND: masks[SEMSegmentationLabels.BACKGROUND]
            + non_foreground_gis,
        },
        # Use an unused value so anything that isn't filled in is visible
        default_value=len(SEMSegmentationLabels),
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


def filter_gis_thickness(
    gis_thickness_px: NDArray[np.integer | np.float64 | np.float32],
    xlims_px: tuple[int, int],
    sigma: float,
) -> NDArray[np.float64]:
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
    masks: dict[SEMSegmentationLabels, NDArray[np.bool_]],
    default_value: int | float = np.nan,
) -> NDArray[np.float64]:
    return np.select(
        list(masks.values()),
        [_.value for _ in masks],
        default=default_value,
    )


def get_gis_thickness(
    prediction: NDArray[np.integer[typing.Any]],
    xlims: tuple[int | None, int | None] = (None, None),
    ylims: tuple[int | None, int | None] = (None, None),
    image_shape: tuple[int, int] | None = None,
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

    mask_lamella = prediction_slice == SEMSegmentationLabels.LAMELLA.value
    mask_gis = prediction_slice == SEMSegmentationLabels.GIS.value
    mask_background = prediction_slice == SEMSegmentationLabels.BACKGROUND.value
    mask_bad = np.isin(
        prediction_slice,
        (
            SEMSegmentationLabels.CRACK.value,
            SEMSegmentationLabels.VACUUM.value,
        ),
    )

    mask_gis_background = mask_gis + mask_background

    mask_good = mask_gis + mask_lamella

    # Set everything above the good bottom to False for mask_bad
    for y, x in get_mask_edge(mask_good, axis=0, side="max"):
        mask_bad[: y + 1, x] = False

    # Ignore GIS/background below the top of the lower crack/vacuum area
    for y, x in get_mask_edge(mask_bad, axis=0, side="min"):
        mask_gis_background[y:, x] = False

    # Ignore GIS/background above the bottom of the lamella
    for y, x in get_mask_edge(mask_lamella, axis=0, side="max"):
        mask_gis_background[: y + 1, x] = False

    new_mask_gis: NDArray[np.bool_ | np.float64]
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

    return sliced_gis_thickness, xlims_out  # type: ignore


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


def get_lamella_gis_boundary_peturbations(
    prediction: NDArray[np.integer[typing.Any]],
    sigma: float,
) -> NDArray[np.float64]:
    lamella_mask = prediction == SEMSegmentationLabels.LAMELLA.value
    lower_edge_coords = get_mask_edge(lamella_mask, axis=0, side="max").astype(
        np.float64
    )
    filtered_edge = gaussian_filter1d(
        lower_edge_coords[:, 0],
        sigma=sigma,
        mode="reflect",
        truncate=6,
    )
    lower_edge_coords[:, 0] = filtered_edge - lower_edge_coords[:, 0]
    return lower_edge_coords
