from __future__ import annotations
from dataclasses import dataclass
import typing

import numpy as np

from fibsem import conversions
from fibsem.structures import Point
from fibsem.detection.detection import AdaptiveLamellaCentre

from adaptive_polish.exceptions import CentringException

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class AdaptivePolishLamellaCentre(AdaptiveLamellaCentre):
    name: str = "AdaptivePolishLamellaCentre"

    def detect(
        self, img: np.ndarray, mask: np.ndarray = None, point: Point = None
    ) -> Point:
        if not np.issubdtype(mask.dtype, np.bool_):
            # Only accept boolean arrays as this requires the mask to have been cleaned up beforehand
            raise TypeError(f"{self.__class__.__name__} only accepts boolean arrays")
        mask_lamella_centre = np.asarray(
            get_lamella_centre(mask, subpixel_accuracy=True)
        )

        img_shape = np.asarray(img.shape)
        mask_shape = np.asarray(mask.shape)
        if np.any(img_shape != mask_shape):
            # Scale mask to image (in case mask is binned)
            lamella_centre = mask_lamella_centre * (img_shape / mask_shape)
        else:
            lamella_centre = mask_lamella_centre

        self.px = Point(
            x=lamella_centre[1],
            y=lamella_centre[0],
        )
        return self.px


def get_mask_bounding_box(
    mask: NDArray[np.bool_],
    edge_finding: typing.Literal["median", "mean", "min", "max"] = "median",
) -> typing.Tuple[float, float, float, float]:
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

    x_mins = np.argmax(mask, axis=1)
    x_maxs = mask.shape[1] - np.argmax(mask[:, ::-1], axis=1) - 1
    y_mins = np.argmax(mask, axis=0)
    y_maxs = mask.shape[0] - np.argmax(mask[::-1, :], axis=0) - 1
    x_range = np.arange(mask.shape[1])
    y_range = np.arange(mask.shape[0])
    # Only include edge values that are True
    # This filters out any rows/columns that were all False
    valid_xmins = x_mins[mask[y_range, x_mins]]
    valid_xmaxs = x_maxs[mask[y_range, x_maxs]]
    valid_ymins = y_mins[mask[y_mins, x_range]]
    valid_ymaxs = y_maxs[mask[y_maxs, x_range]]
    xmin = edge_fn_min(valid_xmins)
    xmax = edge_fn_max(valid_xmaxs)
    ymin = edge_fn_min(valid_ymins)
    ymax = edge_fn_max(valid_ymaxs)
    bbox = (ymin, xmin, ymax, xmax)

    if np.any(np.isnan(bbox)):
        raise CentringException("Bounding box coordinates contains a NaN")
    return bbox


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
    bbox = get_mask_bounding_box(array, edge_finding=edge_finding)
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
