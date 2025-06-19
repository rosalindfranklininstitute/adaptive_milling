from __future__ import annotations
import typing

import numpy as np

from fibsem import conversions
from fibsem.structures import Point

from adaptive_polish.exceptions import CentringException

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray
    from collections.abc import Callable, Sequence


def get_mask_edge(
    mask_2d: NDArray[np.bool_], axis: int, side: typing.Literal["min", "max"] = "min"
) -> NDArray[np.uint16]:
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

    # Only include edge values that are True
    # This filters out any rows/columns that were all False
    valid_idxs = mask_2d[coordinates[:, 0], coordinates[:, 1]]
    return coordinates[valid_idxs].astype(np.uint16)


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
    edge_finding: typing.Literal["median", "mean", "min", "max"] = "median",
) -> typing.Tuple[float, float, float, float]:
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
    edge_finding: typing.Literal["median", "mean", "min", "max"] = "median",
) -> typing.Tuple[float, float, float, float]:
    return get_bounding_box_from_edges(get_mask_edges(mask), edge_finding=edge_finding)


def get_centre_from_bounding_box(
    bbox: typing.Union[
        typing.Tuple[int, int, int, int], typing.Tuple[float, float, float, float]
    ],
    subpixel_accuracy: bool = False,
):
    cx = (bbox[1] + bbox[3]) / 2
    cy = (bbox[0] + bbox[2]) / 2
    if not subpixel_accuracy:
        cx = int(round(cx))
        cy = int(round(cy))
    return (cy, cx)


def get_lamella_centre(
    array: NDArray[np.bool_],
    edge_finding: typing.Literal["median", "mean"] = "median",
    subpixel_accuracy: bool = False,
) -> typing.Union[typing.Tuple[int, int], typing.Tuple[float, float]]:
    bbox = get_bounding_box_from_edges(
        get_mask_edges(mask=array), edge_finding=edge_finding
    )
    return get_centre_from_bounding_box(bbox, subpixel_accuracy=subpixel_accuracy)


def get_bounding_box_scaled_to_image(
    image: NDArray[typing.Any],
    mask: NDArray[np.bool_],
    edge_finding: typing.Literal["median", "mean", "min", "max"] = "median",
) -> typing.Tuple[float, float, float, float]:
    # This does assume square pixels
    if image.shape[1] == mask.shape[1]:
        prediction_to_image_scale_multiplier = 1
    else:
        prediction_to_image_scale_multiplier = image.shape[1] / mask.shape[1]

    bbox_mask = get_mask_bounding_box(mask, edge_finding=edge_finding)

    return (
        bbox_mask[0] * prediction_to_image_scale_multiplier,
        bbox_mask[1] * prediction_to_image_scale_multiplier,
        bbox_mask[2] * prediction_to_image_scale_multiplier,
        bbox_mask[3] * prediction_to_image_scale_multiplier,
    )


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
