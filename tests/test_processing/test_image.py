from __future__ import annotations
import pytest
from numpy.testing import assert_array_equal

from typing import Literal

import numpy as np

from adaptive_milling.processing.image import (
    get_mask_bounding_box,
    get_bounding_box_from_edges,
    get_centre_from_bounding_box,
    get_mask_edges,
)

from ..setup import SimpleRectangleLamellaMask


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


@pytest.mark.parametrize("edge_finding", ["median", "mean", "min", "max"])
def test_get_bounding_box_from_edges(
    edge_finding: Literal["median", "mean", "min", "max"],
) -> None:
    expected_bounding_box = np.asarray(
        (
            50,
            20,
            70,
            60,
        )
    )
    y_range = np.arange(expected_bounding_box[0], expected_bounding_box[2] + 1)
    x_range = np.arange(expected_bounding_box[1], expected_bounding_box[3] + 1)
    edges = [
        [
            np.stack(
                ([expected_bounding_box[0]] * len(x_range), x_range), axis=-1
            ).astype(np.uint16),
            np.stack(
                ([expected_bounding_box[2]] * len(x_range), x_range), axis=-1
            ).astype(np.uint16),
        ],
        [
            np.stack(
                (y_range, [expected_bounding_box[1]] * len(y_range)), axis=-1
            ).astype(np.uint16),
            np.stack(
                (y_range, [expected_bounding_box[3]] * len(y_range)), axis=-1
            ).astype(np.uint16),
        ],
    ]
    bounding_box = get_bounding_box_from_edges(edges, edge_finding=edge_finding)
    assert_array_equal(
        bounding_box, expected_bounding_box, err_msg="Bounding box is not as expected"
    )


@pytest.mark.parametrize("subpixel_accuracy", [True, False], ids=["subpixel", "pixel"])
def test_get_centre_from_bounding_box(subpixel_accuracy: bool):
    expected_centre = np.asarray((20.5, 50.5))
    diffs = np.asarray((5.5, 12.5))
    if not subpixel_accuracy:
        expected_centre = expected_centre.astype(np.uint8)
        diffs = diffs.astype(np.uint8)

    bounding_box = np.concatenate((expected_centre - diffs, expected_centre + diffs))
    assert_array_equal(
        get_centre_from_bounding_box(
            (bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]),
            subpixel_accuracy=subpixel_accuracy,
        ),
        expected_centre,
        err_msg="Failed to find correct centre",
    )


@pytest.mark.parametrize("edge_finding", ["median", "mean", "min", "max"])
def test_get_mask_bounding_box_simple(
    edge_finding: Literal["median", "mean", "min", "max"],
) -> None:
    test_lamella = SimpleRectangleLamellaMask((100, 200), 20)
    # All the edge finding methods should be the same for this case
    found_bounding_box = get_mask_bounding_box(
        test_lamella.array, edge_finding=edge_finding
    )
    np.testing.assert_array_equal(
        found_bounding_box,
        test_lamella.bounding_box,
        err_msg="The found bounding box does not match the actual bounding box",
    )


def test_get_lamella_bounding_box_median() -> None:
    test_lamella = SimpleRectangleLamellaMask((100, 200), 20)

    # Add to under half of one edge
    test_lamella.array[
        test_lamella.bounding_box[0] : test_lamella.bounding_box[0]
        - (test_lamella.bounding_box[2] - test_lamella.bounding_box[0]) // 2
        - 1,
        test_lamella.bounding_box[1] - 1,
    ] = True

    # Cut away just under half of the edge
    test_lamella.array[
        test_lamella.bounding_box[0]
        - (test_lamella.bounding_box[2] - test_lamella.bounding_box[0])
        // 2 : test_lamella.bounding_box[2] + 1,
        test_lamella.bounding_box[1] + 1,
    ] = False

    # All the edge finding methods should be the same for this case
    found_bounding_box = get_mask_bounding_box(
        test_lamella.array, edge_finding="median"
    )
    np.testing.assert_array_equal(
        found_bounding_box,
        test_lamella.bounding_box,
        err_msg="The found bounding box does not match the expected bounding box",
    )


def test_get_lamella_bounding_box_mean() -> None:
    shape = np.asarray((100, 200))
    triangle_array = np.triu(np.ones(shape, dtype=np.bool_), k=0)
    triangle_array = np.roll(triangle_array, 1, axis=1)  # roll so the edges are filled
    expected_bounding_box = [
        0,
        0,
        # expected mean:
        (((shape[0] - 1) / 2) * shape[0] + (shape[0] - 1) * (shape[1] - shape[0]))
        / shape[1],
        shape[1] - 1,
    ]

    found_bounding_box = get_mask_bounding_box(triangle_array, edge_finding="mean")

    np.testing.assert_array_equal(
        found_bounding_box,
        expected_bounding_box,
        err_msg="The found bounding box does not match the expected bounding box",
    )
