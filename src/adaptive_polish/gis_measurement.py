# Copyright 2024 Rosalind Franklin Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import annotations
import logging
import typing
import xml.etree.ElementTree as ET  # to handle metadata as xml

import skimage
import numpy as np
from scipy.signal.windows import gaussian

from adaptive_polish.dl_segmentation import sem_lamella_segmentor as sgm

if typing.TYPE_CHECKING:
    from collections.abc import Sequence
    from numpy.typing import NDArray


_logger = logging.getLogger(__name__)

DEFAULT_SEM_MODEL_GENERATION = sgm._get_newest_generation_key()

load_sem_model = sgm.load_model

def get_pixel_width(img) -> float | None:
    """Gets the pixel width of an AdornedImage, or if not possible, returns None

    Args:
        img (AdornedImage): TFS AutoScript AdornedImage
    """
    try:
        xml_sett = img.metadata.metadata_as_xml
        xml_parse = ET.fromstring(xml_sett)
        pixel_size_x = xml_parse.find("BinaryResult/PixelSize/X")
        if pixel_size_x is None:
            raise ValueError("No BinaryResult/PixelSize/X found")
        return float(pixel_size_x.text)  # type: ignore
    except Exception:
        # if there is no metadata with 'PixelWidth' defaults to the value above
        return None


def keep_only_largest_object(mask: NDArray[np.integer]) -> NDArray[np.bool_]:
    """Find the largest object in a mask and sets everything in that object to
    `fill_value`, background is 0.

    Args:
        mask (np.array): Segmentation mask

    Returns:
        np.array: Boolean mask with all but the largest object set to False
    """
    instances = skimage.measure.label(mask)
    largest_area = max([_.area for _ in skimage.measure.regionprops(instances)])
    mask_largest_only = skimage.morphology.remove_small_objects(
        instances, min_size=largest_area - 1
    )
    return mask_largest_only > 0


def check_minimum_area(
    mask: NDArray[np.bool_], pixel_size: float, minimum_size: float
) -> bool:
    # Ensure lamella and GIS object is bigger than minimum size
    return bool(np.sum(mask) * pixel_size <= minimum_size)


def clean_prediction(
    prediction: NDArray[np.integer],
    additional_labels: typing.Optional[
        Sequence[
            typing.Literal[
                sgm.SegmentationLabels.GIS,
                sgm.SegmentationLabels.CRACK,
            ]
        ]
    ] = None,
) -> typing.Tuple[
    NDArray[np.bool_],
    typing.Optional[NDArray[np.bool_]],
    typing.Optional[NDArray[np.bool_]],
]:
    labels = [sgm.SegmentationLabels.LAMELLA]
    if additional_labels is not None:
        labels.extend(additional_labels)

    _logger.debug(
        "Cleaning prediction with shape: %s, finding: %s",
        str(prediction.shape),
        ", ".join(_.name for _ in labels),
    )

    # Generate bool masks for GIS and lamella
    masks = {label: prediction == label.value for label in labels}

    # Doing these all together could be an issue if the model fails too hard
    # (e.g. layers of gis and crack would mess up the GIS reading) but this
    # seems unlikely.
    # TODO: check whether the crack need to be touching lamella to be a crack
    mask_largest_foreground = keep_only_largest_object(
        np.sum(list(masks.values()), axis=0, dtype=np.bool_)
    )

    # Gets the label that's connected to the other foreground elements
    connected_masks = {
        label: np.logical_and(mask_largest_foreground, masks[label]) for label in labels
    }

    mask_connected_lamella = connected_masks.pop(sgm.SegmentationLabels.LAMELLA)

    for key, mask in tuple(connected_masks.items()):
        if not np.sum(mask):
            # No need to keep an array of 0s
            del connected_masks[key]

    return (
        mask_connected_lamella,
        connected_masks.get(sgm.SegmentationLabels.GIS, None),
        connected_masks.get(sgm.SegmentationLabels.CRACK, None),
    )


def apply_binary_opening(
    array: NDArray[np.bool_], window_size_m: float, pixel_size_m: float
):
    # Get window size in px
    if window_size_m > 0:
        window_size_px = int(round(window_size_m / pixel_size_m))
        return skimage.morphology.binary_opening(
            array,
            footprint=[
                (np.ones((window_size_px, 1)), 1),
                (np.ones((1, window_size_px)), 1),
            ],
            mode="ignore",
        )


def get_xlims(
    mask_1d: NDArray[typing.Union[np.integer, np.floating]],
) -> NDArray[np.uint32]:
    mask_1d = mask_1d.squeeze()
    return np.asarray(
        (np.argmax(mask_1d), mask_1d.size - 1 - np.argmax(mask_1d[::-1])),
        dtype=np.uint32,
    )


def filter_gis_thickness(
    gis_thickness_px: NDArray[typing.Union[np.integer, np.floating]],
    window_size_m: float,
    pixel_size_m: float,
) -> NDArray[np.float64]:
    # TODO: Change window_size_m to beam FWHM
    # FWHM is 2 * sqrt(2 * np.log(2)) * sigma, which is approx 2.355 * sigma
    # The number of points in the gaussian curve should be approx 6 * std for convolution
    window_size_px = int(window_size_m / pixel_size_m)
    if window_size_px % 2 == 0:
        # An even window size means we won't get central point of the curve
        window_size_px += 1
    sigma = window_size_px / 6
    gaussian_curve = gaussian(window_size_px, std=sigma)
    gaussian_curve /= gaussian_curve.sum()
    return np.convolve(
        np.pad(gis_thickness_px, int(gaussian_curve.size / 2)),
        gaussian_curve,
        mode="valid",
    )


def filter_gis_thickness_fast(
    gis_thickness_px: NDArray[typing.Union[np.integer, np.floating]],
    window_size_m: int,
    pixel_size_m: float,
) -> NDArray[np.float64]:
    # Get window size in px
    window_size_px = int(window_size_m / pixel_size_m)
    _logger.info(f"window_size_px: {window_size_px}")
    cumsum_vec = np.cumsum(np.pad(gis_thickness_px, int(window_size_px / 2)))
    return (cumsum_vec[window_size_px:] - cumsum_vec[:-window_size_px]) / window_size_px


def resize_image(
    image: NDArray[typing.Any], new_shape: typing.Tuple[int, int]
) -> NDArray[np.float_]:
    if not isinstance(image.dtype, np.floating):
        # Needs to be floating type if we want interpolation
        image = image.astype(np.float_)
    return skimage.transform.resize(
        image,
        output_shape=new_shape,
        preserve_range=True,
    )


def measure_gis(
    gis_mask: NDArray[typing.Any],
    window_size_m: float,
    pixel_size_m: float,
) -> NDArray[np.floating]:
    # Sum to get thickness along x in pixels
    gis_thickness_px = np.sum(gis_mask, axis=0)

    return filter_gis_thickness(
        gis_thickness_px=gis_thickness_px,
        window_size_m=window_size_m,
        pixel_size_m=pixel_size_m,
    )


def cleanup_crack_segmentation(
    prediction: NDArray[np.integer[typing.Any]],
) -> NDArray[np.bool_]:
    """Measure the area in the prediction for cracks in um2.

    Cracks are only considered if they are within the largest combined lamella+
    GIS+crack object in the prediction.

    Args:
        prediction (np.array): Segmentation mask

    Returns:
        NDArray[np.bool_]: Segmentation of cracks that are connected to the lamella and/or GIS
    """
    # Restrict crack search area to the largest object which is not background
    mask_gis_lamella = np.isin(
        prediction,
        (
            sgm.SegmentationLabels.LAMELLA.value,
            sgm.SegmentationLabels.GIS.value,
        ),
    )
    mask_crack = prediction == 3

    mask_foreground_largest_only = keep_only_largest_object(
        mask_gis_lamella + mask_crack
    )

    return np.logical_and(mask_crack, mask_foreground_largest_only)


def get_mask_area_um2(mask: NDArray[np.bool_], pixel_size_um: float) -> float:
    return float(np.sum(mask) * (pixel_size_um**2))


def masks_to_labels(
    lamella_mask: typing.Optional[NDArray[np.bool_]] = None,
    gis_mask: typing.Optional[NDArray[np.bool_]] = None,
    crack_mask: typing.Optional[NDArray[np.bool_]] = None,
    background_mask: typing.Optional[NDArray[np.bool_]] = None,
    vacuum_mask: typing.Optional[NDArray[np.bool_]] = None,
    default_value: typing.Union[int, float] = np.nan,
) -> NDArray[typing.Any]:
    mask_label_pairs = [
        (lamella_mask, sgm.SegmentationLabels.LAMELLA),
        (gis_mask, sgm.SegmentationLabels.GIS),
        (crack_mask, sgm.SegmentationLabels.CRACK),
        (background_mask, sgm.SegmentationLabels.BACKGROUND),
        (vacuum_mask, sgm.SegmentationLabels.VACUUM),
    ]

    masks = []
    label_values = []
    for mask, label in mask_label_pairs:
        if mask is not None:
            masks.append(mask)
            label_values.append(label.value)
    return np.select(masks, label_values, default=default_value)
