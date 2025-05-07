import pytest

import typing
from dataclasses import dataclass, field, InitVar
import numpy as np

from adaptive_polish.centring import (
    AdaptiveLamellaCentre,
    AdaptivePolishLamellaCentre,
    get_mask_bounding_box,
    get_lamella_centre,
)

from setup import SimpleRectangleLamellaMask


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


def test_methods_equivalent_for_simple_rectangle() -> None:
    # Required to be fairly big due to `detect_centre_point` threshold defaulting to 500
    test_lamella = SimpleRectangleLamellaMask((1000, 2000))
    centre_1 = AdaptiveLamellaCentre().detect(
        test_lamella.array, mask=test_lamella.array
    )
    centre_2 = AdaptivePolishLamellaCentre().detect(
        test_lamella.array, mask=test_lamella.array
    )
    assert centre_1 == centre_2, "Centres do not match"


@pytest.mark.skip("No need to run this unless speed is being checked")
def test_relative_speed() -> None:
    from timeit import timeit

    repeats = 100

    test_lamella = SimpleRectangleLamellaMask((2048, 3072))  # Typical dims

    # AdaptiveLamellaCentre2 is ~2.1x slower for this size array (gets worse
    # with size). However, it should be more accurate.
    centre_feature_1 = AdaptiveLamellaCentre()
    centre_feature_2 = AdaptivePolishLamellaCentre()

    centre_1_time = (
        timeit(
            lambda: centre_feature_1.detect(
                test_lamella.array, mask=test_lamella.array
            ),
            number=repeats,
        )
        / repeats
    )
    centre_2_time = (
        timeit(
            lambda: centre_feature_2.detect(
                test_lamella.array, mask=test_lamella.array
            ),
            number=repeats,
        )
        / repeats
    )
    print(f"AdaptiveLamellaCentre: {centre_1_time} s")
    print(f"AdaptiveLamellaCentre2: {centre_2_time} s")
