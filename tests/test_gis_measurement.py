from __future__ import annotations
import pytest
from numpy.testing import assert_array_equal

import typing
import numpy as np

from adaptive_polish.dl_segmentation.sem_lamella_segmentor import SegmentationLabels

import adaptive_polish.gis_measurement as gm

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray

@pytest.mark.parametrize("with_crack", [True, False], ids=["crack", "no crack"])
def test_get_gis_thickness(with_crack: bool) -> None:
    background_lamella_overlap = 20
    image_shape: tuple[int, int] = (200, 250)
    lamella_bbox: tuple[int, int, int, int] = (10, 50, 70, 150)
    vacuum_bottom_pixels = 3
    segmented_gis_thickness: NDArray[np.uint8] = np.asarray(
        (
            ([0] * lamella_bbox[1])
            + [_ // 2 for _ in range(lamella_bbox[3] - lamella_bbox[1])]
            + ([0] * (image_shape[1] - lamella_bbox[3]))
        ),
        dtype=np.uint8,
    )

    # Add on overlapping background to segmented GIS area
    expected_gis_thickness = segmented_gis_thickness.copy()
    gis_background_thickness = image_shape[0] - (lamella_bbox[2] + 1)
    expected_gis_thickness[
        lamella_bbox[1] : lamella_bbox[1] + background_lamella_overlap
    ] = gis_background_thickness
    expected_gis_thickness[
        lamella_bbox[3] + 1 - background_lamella_overlap : lamella_bbox[3] + 1
    ] = gis_background_thickness

    prediction = np.full(image_shape, SegmentationLabels.VACUUM.value, dtype=np.uint8)

    # Draw background (edges)
    prediction[:, : lamella_bbox[1] + background_lamella_overlap] = (
        SegmentationLabels.BACKGROUND.value
    )
    prediction[:, lamella_bbox[3] + 1 - background_lamella_overlap :] = (
        SegmentationLabels.BACKGROUND.value
    )

    # Add background to bottom of vacuum (should be ignored)
    prediction[image_shape[0] - vacuum_bottom_pixels :, :] = (
        SegmentationLabels.BACKGROUND.value
    )

    # Draw lamella
    prediction[
        lamella_bbox[0] : lamella_bbox[2] + 1, lamella_bbox[1] : lamella_bbox[3] + 1
    ] = SegmentationLabels.LAMELLA.value

    # Draw GIS
    gis_top = lamella_bbox[2] + 1
    for i, thickness in enumerate(segmented_gis_thickness):
        prediction[gis_top : gis_top + thickness, i] = SegmentationLabels.GIS.value

    # Add hole of vacuum to GIS layer
    hole_coords = (lamella_bbox[2] + 1, lamella_bbox[3] - lamella_bbox[1])
    prediction[hole_coords[0], hole_coords[1]] = SegmentationLabels.VACUUM.value
    expected_gis_thickness[hole_coords[1]] -= 1

    if with_crack:
        # Add a crack to GIS layer
        remaining_gis_thickness = 1
        crack_coords = (
            lamella_bbox[2] + 1 + remaining_gis_thickness,
            lamella_bbox[3] - lamella_bbox[1] + 1,
        )
        prediction[
            crack_coords[0] : crack_coords[0]
            + expected_gis_thickness[crack_coords[1]]
            - remaining_gis_thickness,
            crack_coords[1],
        ] = SegmentationLabels.VACUUM.value
        expected_gis_thickness[crack_coords[1]] = remaining_gis_thickness
        mask_crack = prediction == SegmentationLabels.CRACK.value
    else:
        mask_crack = None

    gis_thickness = gm.get_gis_thickness(
        mask_gis=prediction == SegmentationLabels.GIS.value,
        mask_lamella=prediction == SegmentationLabels.LAMELLA.value,
        mask_background=prediction == SegmentationLabels.BACKGROUND.value,
        mask_vacuum=prediction == SegmentationLabels.VACUUM.value,
        mask_crack=mask_crack,
        lamella_mask_bbox=lamella_bbox,
        image_shape=None,
    )

    assert_array_equal(
        gis_thickness,
        np.asarray(expected_gis_thickness, dtype=np.float32),
        err_msg="Incorrect GIS thicknesses found",
    )


@pytest.mark.parametrize("with_crack", [True, False], ids=["crack", "no crack"])
def test_get_gis_thickness_2(with_crack: bool) -> None:
    background_lamella_overlap = 20
    image_shape: tuple[int, int] = (200, 250)
    lamella_bbox: tuple[int, int, int, int] = (10, 50, 70, 150)
    vacuum_bottom_pixels = 3
    segmented_gis_thickness: NDArray[np.uint8] = np.asarray(
        (
            ([0] * lamella_bbox[1])
            + [_ // 2 for _ in range(lamella_bbox[3] - lamella_bbox[1])]
            + ([0] * (image_shape[1] - lamella_bbox[3]))
        ),
        dtype=np.uint8,
    )

    # Add on overlapping background to segmented GIS area
    expected_gis_thickness = segmented_gis_thickness.copy()
    gis_background_thickness = image_shape[0] - (lamella_bbox[2] + 1)
    expected_gis_thickness[
        lamella_bbox[1] : lamella_bbox[1] + background_lamella_overlap
    ] = gis_background_thickness
    expected_gis_thickness[
        lamella_bbox[3] + 1 - background_lamella_overlap : lamella_bbox[3] + 1
    ] = gis_background_thickness

    prediction = np.full(image_shape, SegmentationLabels.VACUUM.value, dtype=np.uint8)

    # Draw background (edges)
    prediction[:, : lamella_bbox[1] + background_lamella_overlap] = (
        SegmentationLabels.BACKGROUND.value
    )
    prediction[:, lamella_bbox[3] + 1 - background_lamella_overlap :] = (
        SegmentationLabels.BACKGROUND.value
    )

    # Add background to bottom of vacuum (should be ignored)
    prediction[image_shape[0] - vacuum_bottom_pixels :, :] = (
        SegmentationLabels.BACKGROUND.value
    )

    # Draw lamella
    prediction[
        lamella_bbox[0] : lamella_bbox[2] + 1, lamella_bbox[1] : lamella_bbox[3] + 1
    ] = SegmentationLabels.LAMELLA.value

    # Draw GIS
    gis_top = lamella_bbox[2] + 1
    for i, thickness in enumerate(segmented_gis_thickness):
        prediction[gis_top : gis_top + thickness, i] = SegmentationLabels.GIS.value

    # Add hole of vacuum to GIS layer
    hole_coords = (lamella_bbox[2] + 1, lamella_bbox[3] - lamella_bbox[1])
    prediction[hole_coords[0], hole_coords[1]] = SegmentationLabels.VACUUM.value
    expected_gis_thickness[hole_coords[1]] -= 1

    if with_crack:
        # Add a crack to GIS layer
        remaining_gis_thickness = 1
        crack_coords = (
            lamella_bbox[2] + 1 + remaining_gis_thickness,
            lamella_bbox[3] - lamella_bbox[1] + 1,
        )
        prediction[
            crack_coords[0] : crack_coords[0]
            + expected_gis_thickness[crack_coords[1]]
            - remaining_gis_thickness,
            crack_coords[1],
        ] = SegmentationLabels.VACUUM.value
        expected_gis_thickness[crack_coords[1]] = remaining_gis_thickness
        mask_crack = prediction == SegmentationLabels.CRACK.value
    else:
        mask_crack = None

    gis_thickness = gm.get_gis_thickness_2(
        mask_gis=prediction == SegmentationLabels.GIS.value,
        mask_lamella=prediction == SegmentationLabels.LAMELLA.value,
        mask_vacuum=prediction == SegmentationLabels.VACUUM.value,
        mask_crack=mask_crack,
        lamella_mask_bbox=lamella_bbox,
        image_shape=None,
    )

    assert_array_equal(
        gis_thickness,
        np.asarray(expected_gis_thickness, dtype=np.float32),
        err_msg="Incorrect GIS thicknesses found",
    )


@pytest.mark.parametrize("with_crack", [True, False], ids=["crack", "no crack"])
def test_get_gis_thickness_functions_equivalent(with_crack: bool) -> None:
    background_lamella_overlap = 20
    image_shape: tuple[int, int] = (200, 250)
    lamella_bbox: tuple[int, int, int, int] = (10, 50, 70, 150)
    vacuum_bottom_pixels = 3
    segmented_gis_thickness: NDArray[np.uint8] = np.asarray(
        (
            ([0] * lamella_bbox[1])
            + [_ // 2 for _ in range(lamella_bbox[3] - lamella_bbox[1])]
            + ([0] * (image_shape[1] - lamella_bbox[3]))
        ),
        dtype=np.uint8,
    )

    # Add on overlapping background to segmented GIS area
    expected_gis_thickness = segmented_gis_thickness.copy()
    gis_background_thickness = image_shape[0] - (lamella_bbox[2] + 1)
    expected_gis_thickness[
        lamella_bbox[1] : lamella_bbox[1] + background_lamella_overlap
    ] = gis_background_thickness
    expected_gis_thickness[
        lamella_bbox[3] + 1 - background_lamella_overlap : lamella_bbox[3] + 1
    ] = gis_background_thickness

    prediction = np.full(image_shape, SegmentationLabels.VACUUM.value, dtype=np.uint8)

    # Draw background (edges)
    prediction[:, : lamella_bbox[1] + background_lamella_overlap] = (
        SegmentationLabels.BACKGROUND.value
    )
    prediction[:, lamella_bbox[3] + 1 - background_lamella_overlap :] = (
        SegmentationLabels.BACKGROUND.value
    )

    # Add background to bottom of vacuum (should be ignored)
    prediction[image_shape[0] - vacuum_bottom_pixels :, :] = (
        SegmentationLabels.BACKGROUND.value
    )

    # Draw lamella
    prediction[
        lamella_bbox[0] : lamella_bbox[2] + 1, lamella_bbox[1] : lamella_bbox[3] + 1
    ] = SegmentationLabels.LAMELLA.value

    # Draw GIS
    gis_top = lamella_bbox[2] + 1
    for i, thickness in enumerate(segmented_gis_thickness):
        prediction[gis_top : gis_top + thickness, i] = SegmentationLabels.GIS.value

    # Add hole of vacuum to GIS layer
    hole_coords = (lamella_bbox[2] + 1, lamella_bbox[3] - lamella_bbox[1])
    prediction[hole_coords[0], hole_coords[1]] = SegmentationLabels.VACUUM.value
    expected_gis_thickness[hole_coords[1]] -= 1

    if with_crack:
        # Add a crack to GIS layer
        remaining_gis_thickness = 1
        crack_coords = (
            lamella_bbox[2] + 1 + remaining_gis_thickness,
            lamella_bbox[3] - lamella_bbox[1] + 1,
        )
        prediction[
            crack_coords[0] : crack_coords[0]
            + expected_gis_thickness[crack_coords[1]]
            - remaining_gis_thickness,
            crack_coords[1],
        ] = SegmentationLabels.VACUUM.value
        expected_gis_thickness[crack_coords[1]] = remaining_gis_thickness
        mask_crack = prediction == SegmentationLabels.CRACK.value
    else:
        mask_crack = None

    gis_thickness_1 = gm.get_gis_thickness(
        mask_gis=prediction == SegmentationLabels.GIS.value,
        mask_lamella=prediction == SegmentationLabels.LAMELLA.value,
        mask_background=prediction == SegmentationLabels.BACKGROUND.value,
        mask_vacuum=prediction == SegmentationLabels.VACUUM.value,
        mask_crack=mask_crack,
        lamella_mask_bbox=lamella_bbox,
        image_shape=None,
    )

    gis_thickness_2 = gm.get_gis_thickness_2(
        mask_gis=prediction == SegmentationLabels.GIS.value,
        mask_lamella=prediction == SegmentationLabels.LAMELLA.value,
        mask_vacuum=prediction == SegmentationLabels.VACUUM.value,
        mask_crack=mask_crack,
        lamella_mask_bbox=lamella_bbox,
        image_shape=None,
    )

    assert_array_equal(
        gis_thickness_1,
        gis_thickness_2,
        err_msg="GIS thicknesses don't match",
    )

    ## 1 is quicker but handles undefined sections differently
    # from timeit import timeit

    # repeats = 10000
    # time_1 = (
    #     timeit(
    #         lambda: gm.get_gis_thickness(
    #             mask_gis=prediction == SegmentationLabels.GIS.value,
    #             mask_lamella=prediction == SegmentationLabels.LAMELLA.value,
    #             mask_background=prediction == SegmentationLabels.BACKGROUND.value,
    #             mask_vacuum=prediction == SegmentationLabels.VACUUM.value,
    #             mask_crack=mask_crack,
    #             lamella_mask_bbox=lamella_bbox,
    #             image_shape=None,
    #         ),
    #         number=repeats,
    #     )
    #     / repeats
    # )
    # time_2 = (
    #     timeit(
    #         lambda: gm.get_gis_thickness_2(
    #             mask_gis=prediction == SegmentationLabels.GIS.value,
    #             mask_lamella=prediction == SegmentationLabels.LAMELLA.value,
    #             mask_vacuum=prediction == SegmentationLabels.VACUUM.value,
    #             mask_crack=mask_crack,
    #             lamella_mask_bbox=lamella_bbox,
    #             image_shape=None,
    #         ),
    #         number=repeats,
    #     )
    #     / repeats
    # )
    # print(time_1, time_2)
