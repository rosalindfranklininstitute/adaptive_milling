from __future__ import annotations
import pytest

import numpy as np
from numpy.testing import assert_array_equal

from adaptive_milling.processing import bitmap as bitmap_proc


@pytest.mark.parametrize(
    "max_output_range,use_blanking",
    [((0, 255), True), ((0, 1), True), ((0, 1), False)],
    ids=["bitmap range", "points range", "no blanking"],
)
def test_create_dwell_and_blanking_arrays(
    max_output_range: tuple[int, int], use_blanking: bool
) -> None:
    expected_dwell_array = np.asarray(
        [
            [
                0.0,
                0.0,
                (max_output_range[1] - max_output_range[0]) / 2 + max_output_range[0],
                max_output_range[1],
                max_output_range[1],
            ]
        ],
        dtype=float,
    )
    expected_blanking_array = np.asarray(
        [[False, use_blanking, False, False, False]], dtype=bool
    )

    blanking_threshold = 0.5 if use_blanking else None
    min_threshold = 1.0
    max_threshold = 3.0
    input_array = np.asarray(
        [
            min_threshold,
            0.0,
            (max_threshold - min_threshold) / 2 + min_threshold,
            max_threshold,
            16.0,
        ]
    )

    dwell_array, blanking_array = bitmap_proc.create_dwell_and_blanking_arrays(
        input_signal=input_array,
        min_dwell_threshold=min_threshold,
        max_dwell_threshold=max_threshold,
        blanking_threshold=blanking_threshold,
        max_output_range=max_output_range,
    )

    assert_array_equal(dwell_array, expected_dwell_array)
    if use_blanking:
        assert blanking_array is not None
        assert_array_equal(blanking_array, expected_blanking_array)
    else:
        assert blanking_array is None


def test_bitmap_to_points() -> None:
    bitmap_image = np.zeros((1, 256, 3), dtype=np.uint8)
    bitmap_image[:, :, 2] = np.arange(0, 256)
    bitmap_image[0, 0, 1] = 1

    assert bitmap_image.dtype == np.uint8

    output = bitmap_proc.bitmap_to_points(bitmap_image)
    del bitmap_image

    assert output[0, 0, 0] == 0, "The first value should stay 0"
    assert output[0, 1, 0] == 1 / 255, "The second value should be 1/255"
    assert output[0, 255, 0] == 1, "The final value should be 1"
    assert_array_equal(
        output[0, :, 0],
        np.linspace(0, 1, num=256, endpoint=True, dtype=object),
        err_msg="Dwell channel is not as expected",
    )
    assert output[0, 0, 1] == 0, (
        "The first blanking value should be 0 (blanked)"
    )  # blanking is all 1s
    assert all(output[0, 1:, 1] == 1), (
        "The rest of blanking should be all 1s (not blanked)"
    )
