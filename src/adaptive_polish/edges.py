from __future__ import annotations
import typing

import numpy as np

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


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
