import typing
from dataclasses import dataclass

import numpy as np

from fibsem.detection.detection import AdaptiveLamellaCentre
from fibsem.structures import Point

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
            lamella_centre *= img_shape / mask_shape
        else:
            lamella_centre = mask_lamella_centre

        self.px = Point(
            x=lamella_centre[1],
            y=lamella_centre[0],
        )
        return self.px


def get_lamella_bounding_box(
    array: np.typing.NDArray[np.bool_],
    edge_finding: typing.Literal["median", "mean"] = "median",
) -> typing.Union[tuple[int, int, int, int], tuple[float, float, float, float]]:
    if edge_finding == "median":
        edge_fn = np.median
    elif edge_finding == "mean":
        edge_fn = np.mean
    else:
        raise ValueError(f"Invalid option edge_finding='{edge_finding}'")

    x_mins = np.argmax(array, axis=1)
    x_maxs = array.shape[1] - np.argmax(array[:, ::-1], axis=1) - 1
    y_mins = np.argmax(array, axis=0)
    y_maxs = array.shape[0] - np.argmax(array[::-1, :], axis=0) - 1
    x_range = np.arange(array.shape[1])
    y_range = np.arange(array.shape[0])
    # Only include edge values that are True
    # This filters out any rows/columns that were all False
    valid_xmins = x_mins[array[y_range, x_mins]]
    valid_xmaxs = x_maxs[array[y_range, x_maxs]]
    valid_ymins = y_mins[array[y_mins, x_range]]
    valid_ymaxs = y_maxs[array[y_maxs, x_range]]
    xmin = edge_fn(valid_xmins)
    xmax = edge_fn(valid_xmaxs)
    ymin = edge_fn(valid_ymins)
    ymax = edge_fn(valid_ymaxs)
    return (ymin, xmin, ymax, xmax)

def get_centre_from_bounding_box(
    bbox: typing.Union[tuple[int, int, int, int], tuple[float, float, float, float]],
    subpixel_accuracy: bool = False,
):
    cx = (bbox[1] + bbox[3]) / 2
    cy = (bbox[0] + bbox[2]) / 2
    if not subpixel_accuracy:
        cx = int(round(cx))
        cy = int(round(cy))
    return (cy, cx)

def get_lamella_centre(
    array: np.typing.NDArray[np.bool_],
    edge_finding: typing.Literal["median", "mean"] = "median",
    subpixel_accuracy: bool = False,
) -> typing.Union[typing.Tuple[int, int], typing.Tuple[float, float]]:
    bbox = get_lamella_bounding_box(array, edge_finding=edge_finding)
    return get_centre_from_bounding_box(bbox, subpixel_accuracy=subpixel_accuracy)
