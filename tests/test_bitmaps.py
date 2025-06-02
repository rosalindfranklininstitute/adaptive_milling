import numpy as np
from numpy.testing import assert_array_equal

from adaptive_polish.bitmaps import bitmap_to_points


def test_bitmap_to_points():
    bitmap_image = np.zeros((1, 256, 3), dtype=np.uint8)
    bitmap_image[:, :, 2] = np.arange(0, 256)
    bitmap_image[0, 0, 1] = 1

    assert bitmap_image.dtype == np.uint8

    output = bitmap_to_points(bitmap_image)
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
