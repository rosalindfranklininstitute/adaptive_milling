from __future__ import annotations
import typing
from functools import partial
from math import ceil, floor

import numpy as np
from skimage import measure, transform

from adaptive_milling.exceptions import CentringException

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
        mask (NDArray[np.integer[typing.Any] | np.bool_]): Segmentation mask
        connectivity (int): connectivity when finding objects (1 is edges only, 2 includes corners)

    Returns:
        NDArray[np.bool_]: Boolean mask with all but the largest object set to False
    """
    labels, num = measure.label(mask, return_num=True, connectivity=connectivity)
    if num == 1:
        return labels.astype(np.bool_)
    prop = max(measure.regionprops(labels), key=lambda x: x.area)
    return labels == prop.label


def filter_connected(
    mask1: NDArray[np.bool_],
    mask2: NDArray[np.bool_],
    connectivity: int = 2,
) -> NDArray[np.bool_]:
    labels = measure.label(mask1 + mask2, connectivity=connectivity)

    connected_labels = np.zeros_like(labels, dtype=np.bool_)
    for prop in measure.regionprops(labels):
        region_label = labels == prop.label
        if np.any(mask1 & region_label) and np.any(mask2 & region_label):
            # If region contains both labels, add it to the connected_labels mask
            connected_labels += region_label
    return connected_labels


def resize_image(
    image: NDArray[typing.Any], new_shape: tuple[int, int]
) -> NDArray[np.float64]:
    if not isinstance(image.dtype, np.floating):
        # Needs to be floating type if we want interpolation
        image = image.astype(np.float64)
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
) -> list[list[NDArray[np.uint16]]]:
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
    percentile: float | None = None,
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


def resize_interp_1d(
    array_1d: Sequence[int | float] | NDArray[typing.Any],
    target_size: int,
) -> NDArray[np.float64]:
    if len(array_1d) == target_size:
        return np.asarray(array_1d, dtype=float)
    return np.interp(
        np.linspace(0, len(array_1d), target_size),
        range(len(array_1d)),
        array_1d,
    )


def rescale_values(
    array: NDArray[typing.Any],
    input_range: tuple[float, float],
    output_range: tuple[float, float],
) -> NDArray[np.float64]:
    return np.interp(array, input_range, output_range)
