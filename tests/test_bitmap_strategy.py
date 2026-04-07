from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock
from numpy.testing import assert_array_equal, assert_array_almost_equal

import typing
from pathlib import Path

import numpy as np

from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol

from adaptive_milling.strategy import bitmap as bitmap_strategy
from adaptive_milling._dataclasses import (
    LamellaStatistics,
)

from . import setup

if typing.TYPE_CHECKING:
    from fibsem.milling.base import FibsemMillingStage

TIMESTAMP = "timestamp"


def get_polishing_stages(
    config_dict: dict[str, typing.Any],
    protocol_template_path: Path,
    temporary_path: Path,
) -> list[FibsemMillingStage]:
    protocol_path = setup.setup_protocol_path(
        protocol_template_path,
        temporary_path,
        config_dict,
        ap_only=True,
        ap_type="bitmap",
    )

    protocol = AutoLamellaTaskProtocol.load(str(protocol_path))

    return protocol.task_config["Polishing"].milling["mill_polishing"].stages


def test_default_config(tmp_path: Path) -> None:
    """Tests that default config works with no errors"""
    # Create AdaptivePolish obj with default input parameters from millingstrategy
    model_path = tmp_path / "model.file"
    model_path.touch()
    config = bitmap_strategy.BitmapAdaptivePolishMillingConfig(
        model_path=str(model_path)
    )
    strategy = bitmap_strategy.BitmapAdaptivePolishMillingStrategy.from_dict(
        {"config": {"model_path": str(model_path)}}
    )

    assert config == strategy.config, "Configs do not match"


def test_milling_stage_loads_defaults(
    protocol_template_path: Path, tmp_path: Path
) -> None:
    """Tests that default the strategy loads the defaults when setup via stages"""
    model_path = tmp_path / "model.file"
    model_path.touch()
    config_dict = {"model_path": str(model_path)}
    config = bitmap_strategy.BitmapAdaptivePolishMillingConfig.from_dict(config_dict)
    polishing_stages = get_polishing_stages(
        config_dict, protocol_template_path, tmp_path
    )
    strategy = polishing_stages[0].strategy
    assert isinstance(strategy, bitmap_strategy.BitmapAdaptivePolishMillingStrategy), (
        "Milling stage strategy is not AdaptivePolishMillingStrategy"
    )
    assert config == strategy.config, "Configs do not match"


@pytest.mark.parametrize("mask_cracks", [True, False])
@patch.object(bitmap_strategy, "create_bitmap_array")
@patch.object(bitmap_strategy, "resize_interp_1d")
def test_create_bitmap_array_calls(
    mock_resize_interp_1d, mock_create_bitmap_array, mask_cracks: bool
) -> None:
    pattern_width = 1e-5
    lamella_width_px = 1000
    sigma = 19
    erosion_px = 13
    gis_width = 10
    crack_location = int(gis_width / 2)
    bitmap_signal = np.random.rand(gis_width)
    refined_xlims = [5, 10]

    lamella_stats = MagicMock()
    lamella_stats.gis_thickness_filtered_um = bitmap_signal.copy()
    lamella_stats.image_pixel_size_m = (
        pattern_width / lamella_width_px,
        pattern_width / lamella_width_px,
    )

    with patch("os.path.isfile") as mock_isfile:
        mock_isfile.return_value = True
        bitmap_config = bitmap_strategy.BitmapAdaptivePolishMillingConfig(
            model_path="model_path",
            bitmap_gaussian_sigma=sigma,
            bitmap_erosion_px=erosion_px,
            mask_cracks=mask_cracks,
        )

    strategy = bitmap_strategy.BitmapAdaptivePolishMillingStrategy(config=bitmap_config)

    mock_resize_interp_1d.return_value = np.asarray([0] * gis_width, dtype=float)
    mock_resize_interp_1d.return_value[crack_location] = 1

    with (
        patch.object(strategy, "_refine_xlims") as mock_refine_xlims,
        patch.object(strategy, "_filter_bitmap_signal") as mock_filter_bitmap_signal,
    ):
        mock_refine_xlims.return_value = refined_xlims
        mock_filter_bitmap_signal.return_value = np.arange(
            refined_xlims[0], refined_xlims[1] + 1
        )

        bitmap_array = strategy.create_bitmap_array(
            pattern_width_m=1e-5, stats=lamella_stats
        )

    mock_refine_xlims.assert_called_once_with(
        lamella_width=lamella_width_px,
        gis_thickness=lamella_stats.gis_thickness_filtered_um,
        xlims=lamella_stats.xlims_image_px,
    )

    if mask_cracks:
        mock_resize_interp_1d.assert_called_once_with(
            lamella_stats.crack_thickness_prediction_px, target_size=gis_width
        )
        bitmap_signal[crack_location] = 0

    else:
        mock_resize_interp_1d.assert_not_called()

    mock_filter_bitmap_signal.assert_called_once()
    assert_array_equal(
        mock_filter_bitmap_signal.call_args[0][0],
        bitmap_signal[refined_xlims[0] : refined_xlims[1] + 1],
        "argument to _filter_bitmap_signal is not as expected",
    )

    mock_create_bitmap_array.assert_called_once_with(
        input_signal=mock_filter_bitmap_signal.return_value,
        xlims=(0, len(mock_filter_bitmap_signal.return_value) - 1),
        min_dwell_threshold=bitmap_config.gis_min * 1e6,
        max_dwell_threshold=bitmap_config.gis_max * 1e6,
        as_image=False,
    )

    assert bitmap_array is mock_create_bitmap_array.return_value


@pytest.mark.parametrize("mask_cracks", [True, False])
def test_create_bitmap_array_simple_values(mask_cracks: bool) -> None:
    pattern_width_m = 1e-5
    lamella_width_px = 1000
    image_width = 100
    pixel_size_m = pattern_width_m / lamella_width_px
    pixel_size_m_tuple = (pixel_size_m, pixel_size_m)
    xlims = (20, 50)
    bitmap_width = xlims[1] - xlims[0] + 1
    crack_x_loc = 30

    gis_thickness_filtered_image_px = [0.0] * image_width
    gis_thickness_filtered_image_px[xlims[0] : xlims[1] + 1] = np.linspace(
        5e-6 / pixel_size_m, 10e-6 / pixel_size_m, bitmap_width
    )

    crack_thickness_prediction_px = [0.0] * image_width
    crack_thickness_prediction_px[crack_x_loc] = 5.0

    lamella_stats = LamellaStatistics(
        image_pixel_size_m=pixel_size_m_tuple,
        prediction_pixel_size_m=pixel_size_m_tuple,
        crack_count=0,
        lamella_thickness_prediction_px=list(range(image_width)),
        crack_thickness_prediction_px=crack_thickness_prediction_px,
        gis_thickness_filtered_image_px=gis_thickness_filtered_image_px,
        xlims_prediction_px=xlims,
        xlims_image_px=xlims,
    )
    lamella_stats.calculate_statistics()

    with patch("os.path.isfile") as mock_isfile:
        mock_isfile.return_value = True
        bitmap_config = bitmap_strategy.BitmapAdaptivePolishMillingConfig(
            model_path="model_path",
            bitmap_gaussian_sigma=0,
            bitmap_erosion_px=0,
            apply_boundary_smoothing=False,
            mask_cracks=mask_cracks,
            gis_min=5e-6,
            gis_max=10e-6,
            incident_angle_sputter_ratio=1,  # Disables incident angle scaling
        )
    strategy = bitmap_strategy.BitmapAdaptivePolishMillingStrategy(config=bitmap_config)

    expected_bitmap_array = np.zeros((1, bitmap_width, 2), dtype=int).astype(object)
    expected_bitmap_array[0, :, 0] = np.linspace(0, 1, bitmap_width, dtype=np.float32)
    if mask_cracks:
        expected_bitmap_array[0, crack_x_loc - xlims[0], 0] = 0

    with patch.object(strategy, "_refine_xlims") as mock_refine_xlims:
        mock_refine_xlims.return_value = xlims
        bitmap_array = strategy.create_bitmap_array(
            pattern_width_m=pattern_width_m, stats=lamella_stats
        )

    assert_array_almost_equal(
        bitmap_array, expected_bitmap_array, err_msg="bitmap arrays aren't as expected"
    )
