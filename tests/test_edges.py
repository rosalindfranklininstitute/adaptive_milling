from __future__ import annotations
from numpy.testing import assert_array_equal

import numpy as np

from adaptive_polish.centring import (
    get_mask_edges,
)

from .setup import SimpleRectangleLamellaMask


def test_get_mask_edges() -> None:
    test_lamella = SimpleRectangleLamellaMask((100, 200), 20)
    edges = get_mask_edges(test_lamella.array)
    y_range = np.arange(test_lamella.bounding_box[0], test_lamella.bounding_box[2] + 1)
    x_range = np.arange(test_lamella.bounding_box[1], test_lamella.bounding_box[3] + 1)
    expected_edges = [
        [
            np.stack(
                ([test_lamella.bounding_box[0]] * len(x_range), x_range), axis=-1
            ).astype(np.uint16),
            np.stack(
                ([test_lamella.bounding_box[2]] * len(x_range), x_range), axis=-1
            ).astype(np.uint16),
        ],
        [
            np.stack(
                (y_range, [test_lamella.bounding_box[1]] * len(y_range)), axis=-1
            ).astype(np.uint16),
            np.stack(
                (y_range, [test_lamella.bounding_box[3]] * len(y_range)), axis=-1
            ).astype(np.uint16),
        ],
    ]
    for i, axis in enumerate(edges):
        for j, edge in enumerate(axis):
            assert_array_equal(
                edge,
                expected_edges[i][j],
                err_msg=f"Edge {['X', 'Y'][i]} {['min', 'max'][j]} do not match",
            )
