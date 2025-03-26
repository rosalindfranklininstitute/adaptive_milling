import typing

import numpy as np

from fibsem.detection.detection import AdaptiveLamellaCentre
from fibsem.structures import Point


class AdaptiveLamellaCentre2(AdaptiveLamellaCentre):
    def detect(
        self, img: np.ndarray, mask: np.ndarray = None, point: Point = None
    ) -> Point:
        if not np.issubdtype(mask.dtype, np.bool_):
            # Only accept boolean arrays as this requires the mask to have been cleaned up beforehand
            raise TypeError(f"{self.__class__.__name__} only accepts boolean arrays")
        mask_lamella_centre = get_lamella_centre(mask, subpixel_accuracy=True)
        if any(img.shape != mask.shape):
            multiplier = (
                img_dim / mask_dim for img_dim, mask_dim in zip(img.shape, mask.shape)
            )
            self.px = Point(
                x=mask_lamella_centre.x * multiplier[1],
                y=mask_lamella_centre.y * multiplier[0],
            )
        else:
            self.px = mask_lamella_centre
        return self.px


def get_lamella_bounding_box(
    array: np.typing.NDArray[np.bool_],
    edge_finding: typing.Literal["median", "mean", "max", "min"] = "median",
) -> tuple[int, int, int, int]:
    if edge_finding == "median":
        edge_fn = np.median
    elif edge_finding == "mean":
        edge_fn = np.mean
    elif edge_finding == "max":
        edge_fn = np.max
    elif edge_finding == "min":
        edge_fn == np.min
    else:
        raise ValueError(f"Invalid option edge_finding='{edge_finding}'")

    x_mins = np.argmax(array, axis=1)
    x_maxs = array.shape[0] - np.argmax(array[:, ::-1], axis=1) - 1
    y_mins = np.argmax(array, axis=0)
    y_maxs = array.shape[1] - np.argmax(array[::-1, :], axis=0) - 1
    x_range = np.arange(array.shape[1])
    y_range = np.arange(array.shape[0])
    valid_xmins = x_mins[array[y_range, x_mins]]
    valid_xmaxs = x_maxs[array[y_range, x_maxs]]
    valid_ymins = y_mins[array[y_mins, x_range]]
    valid_ymaxs = y_maxs[array[y_maxs, x_range]]
    xmin = edge_fn(valid_xmins)
    xmax = edge_fn(valid_xmaxs)
    ymin = edge_fn(valid_ymins)
    ymax = edge_fn(valid_ymaxs)
    return (int(ymin), int(xmin), int(ymax), int(xmax))


def get_lamella_centre(
    array: np.typing.NDArray[np.bool_],
    edge_finding: typing.Literal["median", "mean", "max", "min"] = "median",
    subpixel_accuracy: bool = False,
) -> Point:
    bbox = get_lamella_bounding_box(array, edge_finding=edge_finding)
    cx = (bbox[1] + bbox[3]) / 2
    cy = (bbox[0] + bbox[2]) / 2
    if not subpixel_accuracy:
        cx = int(round(cx))
        cy = int(round(cy))
    return Point(x=cx, y=cy)
