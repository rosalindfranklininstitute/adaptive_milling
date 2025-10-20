from __future__ import annotations
import typing
from functools import partial
from math import ceil, floor

import numpy as np
from scipy.signal import correlate, correlation_lags
from skimage import measure, transform

from fibsem import conversions
from fibsem.structures import Point

from adaptive_polish.exceptions import CentringException

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray
    from collections.abc import Callable, Sequence


def count_objects(
    mask: NDArray[np.integer[typing.Any] | np.bool_],
    connectivity: int = 2,
) -> int:
    _, num = measure.label(mask, return_num=True, connectivity=connectivity)
    return num


def keep_only_largest_object(
    mask: NDArray[np.integer[typing.Any] | np.bool_],
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


def get_mask_edge(
    mask_2d: NDArray[np.bool_],
    axis: int,
    side: typing.Literal["min", "max"] = "min",
    filter_valid: bool = True,
) -> NDArray[np.uint16]:
    """Get the edge along an axis of a 2D mask

    Args:
        mask_2d (NDArray[np.bool_]): boolean array
        axis (int): which axis to check along (NumPy indexing)
        side (typing.Literal["min", "max"], optional): Whether to get the
            minimum or maximum side along an axis. Defaults to "min".
        filter_valid (bool, optional): Whether to filter out values that aren't
            connected to an object. Warning: if False, "min" may return the
            final index of the axis, and "max" may return 0s but gains a small
            performance improvement. Defaults to True.

    Raises:
        ValueError: Invalid argument for side

    Returns:
        NDArray[np.uint16]: Array of (y, x) coordinates
    """
    if side == "max":
        indexes = (
            slice(None, None, -1 if axis == 0 else None),
            slice(None, None, -1 if axis == 1 else None),
        )
        edge = (
            mask_2d.shape[axis]
            - np.argmax(mask_2d[indexes[0], indexes[1]], axis=axis)
            - 1
        )
    elif side == "min":
        edge = np.argmax(mask_2d, axis=axis)
    else:
        raise ValueError(f"Invalid argument for side '{side}'")

    other_axis_range = np.arange(mask_2d.shape[1 - axis], dtype=edge.dtype)

    coordinates = np.stack([edge, other_axis_range], axis=-1)
    if axis != 0:
        coordinates = coordinates[:, ::-1]

    if filter_valid:
        # Only include edge values that are True
        # This filters out any rows/columns that were all False
        valid_idxs = mask_2d[coordinates[:, 0], coordinates[:, 1]]
        coordinates = coordinates[valid_idxs]
    return coordinates.astype(np.uint16)


def get_mask_edges(
    mask: NDArray[np.bool_],
) -> typing.List[typing.List[NDArray[np.uint16]]]:
    return [
        [
            get_mask_edge(mask, axis=0, side="min"),
            get_mask_edge(mask, axis=0, side="max"),
        ],
        [
            get_mask_edge(mask, axis=1, side="min"),
            get_mask_edge(mask, axis=1, side="max"),
        ],
    ]


def get_bounding_box_from_edges(
    edges_array: Sequence[Sequence[NDArray[np.integer[typing.Any]]]],
    edge_finding: typing.Literal[
        "median", "mean", "min", "max", "percentile"
    ] = "median",
    percentile: float | None = None,
) -> tuple[float, float, float, float]:
    edge_fn_max: Callable[..., np.integer[typing.Any] | np.floating[typing.Any]]
    edge_fn_min: Callable[..., np.integer[typing.Any] | np.floating[typing.Any]]
    if edge_finding == "median":
        edge_fn_min = np.median
        edge_fn_max = np.median
    elif edge_finding == "mean":
        edge_fn_min = np.mean
        edge_fn_max = np.mean
    elif edge_finding == "min":
        edge_fn_min = np.max
        edge_fn_max = np.min
    elif edge_finding == "max":
        edge_fn_min = np.min
        edge_fn_max = np.max
    elif edge_finding == "percentile":
        if percentile is None:
            raise ValueError(
                "Edge finding method 'percentile' requires additional argument 'percentile'"
            )
        edge_fn_min = partial(np.percentile, q=100 - percentile)
        edge_fn_max = partial(np.percentile, q=percentile)
    else:
        raise ValueError(f"Invalid option edge_finding='{edge_finding}'")
    xmin = edge_fn_min(edges_array[1][0][:, 1]).item()
    xmax = edge_fn_max(edges_array[1][1][:, 1]).item()
    ymin = edge_fn_min(edges_array[0][0][:, 0]).item()
    ymax = edge_fn_max(edges_array[0][1][:, 0]).item()
    bbox = (ymin, xmin, ymax, xmax)

    if np.any(np.isnan(bbox)):
        raise CentringException("Bounding box coordinates contains a NaN")
    return bbox


def get_mask_bounding_box(
    mask: NDArray[np.bool_],
    edge_finding: typing.Literal[
        "median", "mean", "min", "max", "percentile"
    ] = "median",
    percentile: float | None = None,
) -> tuple[float, float, float, float]:
    return get_bounding_box_from_edges(
        get_mask_edges(mask), edge_finding=edge_finding, percentile=percentile
    )


@typing.overload
def get_centre_from_bounding_box(
    bbox: tuple[int, int, int, int] | tuple[float, float, float, float],
    subpixel_accuracy: typing.Literal[False] = ...,
) -> tuple[int, int]: ...


@typing.overload
def get_centre_from_bounding_box(
    bbox: tuple[int, int, int, int] | tuple[float, float, float, float],
    subpixel_accuracy: typing.Literal[True] = ...,
) -> tuple[float, float]: ...


@typing.overload
def get_centre_from_bounding_box(
    bbox: tuple[int, int, int, int] | tuple[float, float, float, float],
    subpixel_accuracy: bool = ...,
) -> tuple[int, int] | tuple[float, float]: ...


def get_centre_from_bounding_box(
    bbox: tuple[int, int, int, int] | tuple[float, float, float, float],
    subpixel_accuracy: bool = False,
) -> tuple[int, int] | tuple[float, float]:
    cx = (bbox[1] + bbox[3]) / 2
    cy = (bbox[0] + bbox[2]) / 2
    if not subpixel_accuracy:
        cx = int(round(cx))
        cy = int(round(cy))
    return (cy, cx)


def bbox_to_ylims(
    bbox: tuple[float, float, float, float],
    y_bounds: tuple[int, int],
    pad: int = 0,
) -> tuple[int, int]:
    return (
        max(int(floor(bbox[0])) - pad, y_bounds[0]),
        min(int(ceil(bbox[2])) + pad, y_bounds[1]),
    )


def bbox_to_xlims(
    bbox: tuple[float, float, float, float],
    x_bounds: tuple[int, int],
    pad: int = 0,
) -> tuple[int, int]:
    return (
        int(max(floor(bbox[1]) - pad, x_bounds[0])),
        int(min(ceil(bbox[3]) + pad, x_bounds[1])),
    )


def get_bounding_box_scaled_to_image(
    image: NDArray[typing.Any],
    mask: NDArray[np.bool_],
    edge_finding: typing.Literal[
        "median", "mean", "min", "max", "percentile"
    ] = "median",
    percentile: typing.Optional[float] = None,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    # This does assume square pixels
    prediction_to_image_scale_multiplier: float
    if image.shape[1] == mask.shape[1]:
        prediction_to_image_scale_multiplier = 1
    else:
        prediction_to_image_scale_multiplier = image.shape[1] / mask.shape[1]

    bbox_mask = get_mask_bounding_box(
        mask, edge_finding=edge_finding, percentile=percentile
    )

    return (
        bbox_mask[0] * prediction_to_image_scale_multiplier,
        bbox_mask[1] * prediction_to_image_scale_multiplier,
        bbox_mask[2] * prediction_to_image_scale_multiplier,
        bbox_mask[3] * prediction_to_image_scale_multiplier,
    ), bbox_mask


def get_centre_points_from_bounding_box(
    bbox: typing.Tuple[float, float, float, float],
    image: NDArray[typing.Any],
    pixel_size_m: float,
) -> tuple[Point, Point]:
    centre_px = get_centre_from_bounding_box(bbox, subpixel_accuracy=True)

    if np.any(np.isnan(centre_px)):
        raise CentringException("Centre pixel coordinates contain a NaN")

    centre_px_point = Point(x=centre_px[1], y=centre_px[0])

    # Convert to microscope image coordinates (0, 0 at centre of image)
    centre_m = conversions.image_to_microscope_image_coordinates(
        centre_px_point,
        image,
        pixel_size_m,
        subpixel_precision=True,
    )
    return centre_m, centre_px_point


def center_subtract_1d(
    arr1: NDArray[typing.Any], arr2: NDArray[typing.Any]
) -> NDArray[typing.Any]:
    len1 = len(arr1)
    len2 = len(arr2)

    if len1 < len2:
        start = len2 // 2 - len1 // 2
        end = start + len1
        return arr1 - arr2[start:end]
    elif len1 > len2:
        start = len1 // 2 - len2 // 2
        end = start + len2
        return arr1[start:end] - arr2

    # If lengths are equal
    return arr1 - arr2


def correlate_arrays_1d(
    target_array: NDArray[typing.Any], new_array: NDArray[typing.Any]
) -> int:
    target_length = len(target_array)
    target_length_quater = target_length // 4
    # Use the middle half of the target_array to correlate:
    weights = target_array[target_length_quater : 3 * target_length_quater]
    correlation_mode = "valid"
    correlation = correlate(new_array, weights, mode=correlation_mode)
    # # Remove padded correlations
    # correlation = correlation[
    #     (len(new_array) - target_length_quater * 2 - 1) : len(correlation)
    #     - (len(new_array) - target_length_quater * 2 - 1)
    # ]
    # argmax gets the middle of the target array, we want the left edge
    offsets = correlation_lags(len(new_array), len(weights), mode=correlation_mode)
    corr_idx = np.argmax(correlation)
    return offsets[corr_idx] - (2 * target_length_quater)


def align_and_subtract_signals(
    arr1: NDArray[typing.Any], arr2: NDArray[typing.Any]
) -> NDArray[typing.Any]:
    corr_idx = -correlate_arrays_1d(arr1, arr2)

    if corr_idx < 0:  # decides which array sets the minimum limit
        corr_idx = abs(corr_idx)
        arr1_slice = slice(0, min(len(arr1), len(arr2) - corr_idx))
        arr2_slice = slice(corr_idx, min(len(arr2), len(arr1) + corr_idx))
    else:
        arr1_slice = slice(corr_idx, min(len(arr1), len(arr2) + corr_idx))
        arr2_slice = slice(0, min(len(arr2), len(arr1) - corr_idx))

    return np.subtract(arr2[arr2_slice], arr2[arr1_slice])
