from __future__ import annotations
import typing

import numpy as np

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


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
