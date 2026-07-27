from __future__ import annotations
import typing

import numpy as np

from fibsem import conversions
from fibsem.structures import Point

from adaptive_milling.exceptions import CentringException
from adaptive_milling.processing.image import get_centre_from_bounding_box

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


def bounding_box_to_centre_points(
    bbox: tuple[float, float, float, float],
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
