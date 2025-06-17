import pytest

import numpy as np

from fibsem.detection.detection import AdaptiveLamellaCentre
from adaptive_polish.centring import (
    get_mask_bounding_box,
    get_lamella_centre,
)

from .setup import SimpleRectangleLamellaMask


def test_methods_equivalent_for_simple_rectangle() -> None:
    # Required to be fairly big due to `detect_centre_point` threshold defaulting to 500
    test_lamella = SimpleRectangleLamellaMask((1000, 2000))
    centre_point = AdaptiveLamellaCentre().detect(
        test_lamella.array, mask=test_lamella.array.astype(np.uint8) * 2
    )
    centre_1 = (centre_point.y, centre_point.x)
    # get_lamella_centre returns (y, x), whereas Point is (x, y)
    centre_2 = get_lamella_centre(array=test_lamella.array, edge_finding="median")
    assert centre_1 == centre_2, "Centres do not match"


@pytest.mark.parametrize("edge_finding", ["median", "mean", "min", "max"])
def test_get_mask_bounding_box_simple(edge_finding: str) -> None:
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


def test_get_lamella_centre() -> None:
    test_lamella = SimpleRectangleLamellaMask((100, 200), 20)
    found_centre = get_lamella_centre(test_lamella.array)
    np.testing.assert_array_equal(
        found_centre,
        test_lamella.centre,
        err_msg="The found centre does not match the actual centre",
    )
