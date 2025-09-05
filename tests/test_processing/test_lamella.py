from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch, ANY
from numpy.testing import assert_array_equal

import typing
import itertools
import numpy as np

from adaptive_polish.processing import lamella as lamella_proc
from adaptive_polish.processing import image as image_proc
from adaptive_polish.processing.sem_segmentation import (
    SegmentationLabels as SemLabels,
)

from fibsem.detection.detection import AdaptiveLamellaCentre
from ..setup import SimpleRectangleLamellaMask

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


def _create_mock_prediction_gis_thickness(
    image_shape: tuple[int, int],
    lamella_bbox: tuple[int, int, int, int],
    background_lamella_overlap: int,
    vacuum_bottom_pixels: int,
    add_crack: bool = True,
    top_background_size: int = 3,
    top_vacuum_size: int = 3,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
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

    prediction = np.full(image_shape, SemLabels.VACUUM.value, dtype=np.uint8)

    # Draw background (edges)
    prediction[:, : lamella_bbox[1] + background_lamella_overlap] = (
        SemLabels.BACKGROUND.value
    )
    prediction[:, lamella_bbox[3] + 1 - background_lamella_overlap :] = (
        SemLabels.BACKGROUND.value
    )

    # Add background to bottom of vacuum (should be ignored)
    prediction[image_shape[0] - vacuum_bottom_pixels :, :] = SemLabels.BACKGROUND.value

    # Draw lamella
    prediction[
        lamella_bbox[0] : lamella_bbox[2] + 1, lamella_bbox[1] : lamella_bbox[3] + 1
    ] = SemLabels.LAMELLA.value

    # Draw GIS
    gis_top = lamella_bbox[2] + 1
    for i, thickness in enumerate(segmented_gis_thickness):
        prediction[gis_top : gis_top + thickness, i] = SemLabels.GIS.value

    # Mislable some GIS as background
    # This should not affect the GIS thickness
    hole_coords = (lamella_bbox[2] + 1, lamella_bbox[3])
    prediction[hole_coords[0], hole_coords[1]] = SemLabels.BACKGROUND.value

    # Add hole of vacuum to GIS layer
    hole_coords = (lamella_bbox[2] + 1, lamella_bbox[3] - lamella_bbox[1])
    prediction[hole_coords[0], hole_coords[1]] = SemLabels.VACUUM.value
    expected_gis_thickness[hole_coords[1]] -= 1

    # Add some background to top of lamella
    # This should not affect the GIS thickness
    slicer = (
        slice(None, lamella_bbox[0] + top_background_size),
        slice(lamella_bbox[1], lamella_bbox[1] + top_background_size),
    )
    prediction[slicer] = SemLabels.BACKGROUND.value

    # Add some vacuum to top of lamella
    # This should not affect the GIS thickness
    slicer = (
        slice(None, lamella_bbox[0] + top_vacuum_size),
        slice(lamella_bbox[3], lamella_bbox[3] - top_vacuum_size - 1, -1),
    )
    prediction[slicer] = SemLabels.VACUUM.value

    if add_crack:
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
        ] = SemLabels.CRACK.value
        expected_gis_thickness[crack_coords[1]] = remaining_gis_thickness

    return prediction, segmented_gis_thickness, expected_gis_thickness


@patch.object(lamella_proc, "resize_image")
@pytest.mark.parametrize("prediction_binning", [None, 2], ids=["unscaled", "scaled"])
@pytest.mark.parametrize("pad_limits", [True, False], ids=["padded", "unpadded"])
@pytest.mark.parametrize("with_crack", [True, False], ids=["crack", "no crack"])
def test_get_gis_thickness(
    mock_resize_image: MagicMock,
    with_crack: bool,
    pad_limits: bool,
    prediction_binning: int | None,
) -> None:
    background_lamella_overlap = 20
    prediction_shape: tuple[int, int] = (200, 250)
    lamella_bbox: tuple[int, int, int, int] = (10, 50, 70, 150)
    padding: tuple[int, int] = (20, 10)
    if prediction_binning is None:
        image_shape = None
    else:
        image_shape = (
            prediction_shape[0] * prediction_binning,
            prediction_shape[1] * prediction_binning,
        )

    def passthrough_resize(image, new_shape):
        return image

    mock_resize_image.side_effect = passthrough_resize

    xlims = image_proc.bbox_to_xlims(
        lamella_bbox,
        pad=padding[1] if pad_limits else 0,
        x_bounds=(0, prediction_shape[1] - 1),
    )
    ylims = image_proc.bbox_to_ylims(
        lamella_bbox,
        pad=padding[0] if pad_limits else 0,
        y_bounds=(0, prediction_shape[0] - 1),
    )
    vacuum_bottom_pixels = 3

    expected_gis_thickness: NDArray[typing.Any]
    prediction, _, expected_gis_thickness = _create_mock_prediction_gis_thickness(
        prediction_shape,
        lamella_bbox,
        background_lamella_overlap,
        vacuum_bottom_pixels,
        add_crack=with_crack,
    )

    gis_thickness, xlims_out = lamella_proc.get_gis_thickness(
        prediction=prediction,
        xlims=xlims,
        ylims=(ylims[0], None),
        image_shape=image_shape,
    )

    if prediction_binning is None:
        assert xlims_out == xlims, "xlims should not be changed if image_shape is None"
    else:
        assert xlims_out == (
            xlims[0] * prediction_binning,
            xlims[1] * prediction_binning,
        ), "xlims should be scaled by the binning amount"

    expected_gis_thickness = np.asarray(expected_gis_thickness, dtype=np.float32)

    # Add the thickness due to background at the edges
    expected_gis_thickness[: lamella_bbox[1]] = prediction_shape[0] - ylims[0]
    expected_gis_thickness[lamella_bbox[3] + 1 :] = prediction_shape[0] - ylims[0]

    expected_gis_thickness[: xlims[0] - 1] = 0
    expected_gis_thickness[xlims[1] + 2 :] = 0

    if image_shape is None:
        mock_resize_image.assert_not_called()
    else:
        mock_resize_image.assert_called_once_with(ANY, new_shape=image_shape)

    assert_array_equal(
        gis_thickness,
        expected_gis_thickness,
        err_msg="Incorrect GIS thicknesses found",
    )


@pytest.mark.parametrize("scaling", np.arange(1, 2.01, 0.01).tolist())
def test_get_gis_thickness_edge_issues(scaling: float) -> None:
    background_lamella_overlap = 20
    prediction_shape: tuple[int, int] = (200, 250)
    lamella_bbox: tuple[int, int, int, int] = (10, 50, 70, 150)

    image_shape = (
        int(np.ceil(prediction_shape[0] * scaling)),
        int(np.ceil(prediction_shape[1] * scaling)),
    )

    xlims = image_proc.bbox_to_xlims(
        lamella_bbox,
        pad=0,
        x_bounds=(0, prediction_shape[1] - 1),
    )
    ylims = image_proc.bbox_to_ylims(
        lamella_bbox,
        pad=0,
        y_bounds=(0, prediction_shape[0] - 1),
    )
    vacuum_bottom_pixels = 3

    prediction, _, _ = _create_mock_prediction_gis_thickness(
        prediction_shape,
        lamella_bbox,
        background_lamella_overlap,
        vacuum_bottom_pixels,
    )

    gis_thickness, xlims_out = lamella_proc.get_gis_thickness(
        prediction=prediction,
        xlims=xlims,
        ylims=(ylims[0], None),
        image_shape=image_shape,
    )

    assert np.all(gis_thickness[xlims_out[0] : xlims_out[1] + 1] >= 1), (
        "If a 0 shows up, it's due to rounding issues at the boundaries of the analysed area"
    )


@pytest.mark.parametrize(
    "scaling", (None, 1, 2, 5), ids=["unscaled", "scaling 1", "scaling 2", "scaling 5"]
)
def test_get_gis_thickness_equivalent_to_old_method(scaling: int | None) -> None:
    background_lamella_overlap = 20
    prediction_shape: tuple[int, int] = (200, 250)
    lamella_bbox: tuple[int, int, int, int] = (10, 50, 70, 150)

    xlims = image_proc.bbox_to_xlims(
        lamella_bbox,
        # Undo the padding to make equivalent (should be only difference)
        pad=-1,
        x_bounds=(0, prediction_shape[1] - 1),
    )

    ylims = image_proc.bbox_to_ylims(
        lamella_bbox,
        pad=0,
        y_bounds=(0, prediction_shape[0] - 1),
    )
    vacuum_bottom_pixels = 3

    image_shape: tuple[int, int] | None
    if scaling is None:
        image_shape = None
    else:
        image_shape = (prediction_shape[0] * scaling, prediction_shape[1] * scaling)

    prediction, _, _ = _create_mock_prediction_gis_thickness(
        prediction_shape,
        lamella_bbox,
        background_lamella_overlap,
        vacuum_bottom_pixels,
        add_crack=True,
    )

    gis_thickness_old = lamella_proc.get_gis_thickness_old(
        prediction=prediction,
        lamella_mask_bbox=lamella_bbox,
        image_shape=image_shape,
    )

    gis_thickness, xlims_out = lamella_proc.get_gis_thickness(
        prediction=prediction,
        xlims=xlims,
        ylims=(ylims[0], None),
        image_shape=image_shape,
        clean_edges=False,
    )

    assert_array_equal(
        gis_thickness,
        gis_thickness_old,
        err_msg="Thicknesses don't match if clean_edges=False",
    )

    gis_thickness_clean_edges, xlims_out = lamella_proc.get_gis_thickness(
        prediction=prediction,
        xlims=xlims,
        ylims=(ylims[0], None),
        image_shape=image_shape,
        clean_edges=True,
    )

    if scaling is None or scaling == 1:
        assert_array_equal(
            gis_thickness_clean_edges,
            gis_thickness_old,
            err_msg="Thicknesses don't match if clean_edges=True",
        )
    else:
        assert_array_equal(
            gis_thickness_clean_edges[xlims_out[0] : xlims_out[1] + 1],
            gis_thickness_old[xlims_out[0] : xlims_out[1] + 1],
            err_msg="Thicknesses don't match inside xlims if clean_edges=True",
        )


@pytest.mark.parametrize("dtype,connectivity", itertools.product([bool, int], [1, 2]))
def test_keep_only_largest_object(dtype: type, connectivity: int) -> None:
    input_array = np.zeros((50, 50), dtype=np.uint8)

    # Draw largest object
    input_array[5:20, 5:20] = 5

    expected_output = input_array.copy().astype(np.bool_)

    # Add corner pixel to test connectivity
    input_array[20, 20] = 5
    if connectivity > 1:
        expected_output[20, 20] = True

    # Add other objects
    input_array[:4, :4] = 3

    input_array[22:25, :4] = 8

    input_array[40:45, 40:45] = 2

    # Same label as largest object but not touching
    input_array[:4, 22:25] = 5

    if dtype is bool:
        input_array = input_array.astype(np.bool_)
    elif dtype is int:
        # Attached to largest object but with different value
        # Check it does recognise them as different
        input_array[40:45, 20:30] = 2
    else:
        raise TypeError(f"Unsupported dtype '{dtype}'")

    output = lamella_proc.keep_only_largest_object(
        mask=input_array, connectivity=connectivity
    )

    assert_array_equal(output, expected_output)


def test_clean_prediction() -> None:
    background_lamella_overlap = 20
    image_shape: tuple[int, int] = (200, 250)
    lamella_bbox: tuple[int, int, int, int] = (10, 50, 70, 150)
    vacuum_bottom_pixels = 3
    prediction, segmented_gis_thickness, _ = _create_mock_prediction_gis_thickness(
        image_shape,
        lamella_bbox,
        background_lamella_overlap,
        vacuum_bottom_pixels,
        add_crack=True,
    )
    max_gis_thickness = segmented_gis_thickness.max()

    messy_prediction = prediction.copy()

    num_random_crack = 20
    for _ in range(num_random_crack):
        idx = (
            # Beneath the lamella and GIS
            np.random.randint(
                lamella_bbox[2] + max_gis_thickness + 1,
                image_shape[0] - vacuum_bottom_pixels,
            ),
            # Between the edges of the background
            np.random.randint(
                lamella_bbox[1] + background_lamella_overlap,
                lamella_bbox[3] - background_lamella_overlap,
            ),
        )
        messy_prediction[idx] = SemLabels.CRACK.value

    num_random_gis = 20
    for _ in range(num_random_gis):
        idx = (
            # Anywhere in Y
            np.random.randint(0, image_shape[0]),
            # Before the lamella (without touching it)
            np.random.randint(0, lamella_bbox[1] - 1),
        )
        messy_prediction[idx] = SemLabels.GIS.value

    clean_prediction = lamella_proc.clean_prediction(prediction=messy_prediction)

    assert_array_equal(
        clean_prediction,
        prediction,
        err_msg="Prediction cleaning has not worked as expected",
    )


def test_get_lamella_centre() -> None:
    test_lamella = SimpleRectangleLamellaMask((100, 200), 20)
    found_centre = lamella_proc.get_lamella_centre(test_lamella.array)
    np.testing.assert_array_equal(
        found_centre,
        test_lamella.centre,
        err_msg="The found centre does not match the actual centre",
    )


def test_methods_equivalent_for_simple_rectangle() -> None:
    # Required to be fairly big due to `detect_centre_point` threshold defaulting to 500
    test_lamella = SimpleRectangleLamellaMask((1000, 2000))
    centre_point = AdaptiveLamellaCentre().detect(
        test_lamella.array, mask=test_lamella.array.astype(np.uint8) * 2
    )
    centre_1 = (centre_point.y, centre_point.x)  # type: ignore
    # get_lamella_centre returns (y, x), whereas Point is (x, y)
    centre_2 = lamella_proc.get_lamella_centre(
        array=test_lamella.array, edge_finding="median"
    )
    assert centre_1 == centre_2, "Centres do not match"
