from __future__ import annotations

import typing
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol
from fibsem.milling.patterning.patterns2 import RectanglePattern, TrenchPattern
from fibsem.structures import Point
from numpy.testing import assert_array_almost_equal, assert_array_equal

from adaptive_milling._dataclasses import LamellaInformation, LamellaStatistics
from adaptive_milling.strategy import bitmap as bitmap_strategy

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


@pytest.mark.parametrize(
    "pattern_time", [10.0, 0.0], ids=["pattern time", "pattern depth"]
)
def test_optimise_pattern_parameters(pattern_time: float) -> None:
    pattern_depth = 20
    bitmap_array = np.zeros(
        (
            1,  # This could be expanded if some falloff is wanted
            6,
            2,
        ),
        dtype=object,
    )
    bitmap_array[:, :, 0] = np.arange(0.0, 0.6, step=0.1, dtype=np.float64)

    new_pattern_depth, new_pattern_time = (
        bitmap_strategy.BitmapAdaptivePolishMillingStrategy._optimise_pattern_parameters(
            bitmap_array, pattern_depth=pattern_depth, pattern_time=pattern_time
        )
    )
    assert np.max(bitmap_array[0]) == 1, (
        "Dwell time multiplier has not been properly scaled"
    )
    assert np.min(bitmap_array[0]) == 0, (
        "Dwell time multiplier has not been properly scaled"
    )

    assert new_pattern_depth == pattern_depth / 2, (
        "Pattern depth has not been properly scaled"
    )

    if pattern_time == 0:
        assert new_pattern_time == 0, "Pattern time should be 0"
    else:
        assert new_pattern_time == pattern_time / 2, (
            "Pattern time has not been properly scaled"
        )


@pytest.mark.parametrize("pattern_time", [0, 100])
def test_set_bitmap_pattern_statistics(pattern_time: float) -> None:
    old_pattern = MagicMock()
    old_pattern.depth = 100
    old_pattern.time = pattern_time

    dwell_multiplier_array = np.linspace(0, 1, num=100)
    blanking_array = np.zeros_like(dwell_multiplier_array, dtype=np.int64)

    bitmap_array = np.zeros(
        (
            1,  # This could be expanded if some falloff is wanted
            dwell_multiplier_array.size,
            2,
        ),
        dtype=object,
    )

    bitmap_array[:, :, 0] = dwell_multiplier_array[np.newaxis, :]
    bitmap_array[:, :, 1] = blanking_array[np.newaxis, :]

    new_pattern = MagicMock()
    new_pattern.depth = 25
    new_pattern.time = pattern_time / 2
    new_pattern.array = bitmap_array

    stats = LamellaStatistics(
        image_pixel_size_m=(0, 0),
        prediction_pixel_size_m=(0, 0),
        crack_count=0,
        lamella_thickness_prediction_px=[0],
        crack_thickness_prediction_px=[0],
    )

    bitmap_strategy.BitmapAdaptivePolishMillingStrategy._set_bitmap_pattern_statistics(
        stats=stats, new_pattern=new_pattern, old_pattern=old_pattern
    )

    assert_array_equal(stats.pattern_dwell_multiplier, dwell_multiplier_array.tolist())
    assert_array_equal(stats.pattern_blanking, blanking_array.astype(bool).tolist())
    assert stats.pattern_depth_multiplier == 0.25
    if pattern_time:
        assert stats.pattern_time_multiplier == 0.5
    else:
        assert stats.pattern_time_multiplier is None


@pytest.mark.parametrize("input_pattern_type", [RectanglePattern, TrenchPattern])
@pytest.mark.parametrize(
    "incident_angle_sputter_ratio",
    [1, 4.3],
    ids=["no angle sputter ratio", "angle sputter ratio"],
)
@pytest.mark.parametrize(
    "boundary_smoothing_sigma",
    [0, 5],
    ids=["no boundary smoothing", "boundary smoothing"],
)
@patch.object(bitmap_strategy, "get_mask_edge")
@patch.object(bitmap_strategy, "get_lamella_gis_boundary_peturbations")
@patch.object(
    bitmap_strategy.BitmapAdaptivePolishMillingStrategy,
    "_set_bitmap_pattern_statistics",
)
def test_update_milling_stage(
    mock_set_bitmap_pattern_statistics,
    mock_get_lamella_gis_boundary_peturbations,
    mock_get_mask_edge,
    input_pattern_type: type[RectanglePattern | TrenchPattern],
    incident_angle_sputter_ratio: float,
    boundary_smoothing_sigma: float,
) -> None:

    input_pattern = input_pattern_type(Point())
    lamella_stats = LamellaStatistics(
        image_pixel_size_m=(1, 1),
        prediction_pixel_size_m=(1, 1),
        crack_count=1,
        lamella_thickness_prediction_px=[0] * 10,
        crack_thickness_prediction_px=[0] * 10,
    )
    lamella_info = LamellaInformation(
        identifier=f"test_{input_pattern_type}",
        sem_image=MagicMock(),  # pyright: ignore[reportArgumentType]
        fib_image=MagicMock(),  # pyright: ignore[reportArgumentType]
        prediction=None,  # pyright: ignore[reportArgumentType]
        clean_prediction=MagicMock(),  # pyright: ignore[reportArgumentType]
        statistics=lamella_stats,
    )

    with patch("os.path.isfile") as mock_isfile:
        mock_isfile.return_value = True
        bitmap_config = bitmap_strategy.BitmapAdaptivePolishMillingConfig(
            model_path="model_path",
            bitmap_gaussian_sigma=0,
            bitmap_erosion_px=0,
            apply_boundary_smoothing=True,
            mask_cracks=False,
            gis_min=5e-6,
            gis_max=10e-6,
            incident_angle_sputter_ratio=incident_angle_sputter_ratio,
            boundary_smoothing_sigma=boundary_smoothing_sigma,
        )
    strategy = bitmap_strategy.BitmapAdaptivePolishMillingStrategy(config=bitmap_config)

    mock_stage = MagicMock()
    mock_stage.pattern = input_pattern

    new_pattern = MagicMock()

    with (
        patch.object(
            strategy, "_convert_trench_pattern"
        ) as mock_convert_trench_pattern,
        patch.object(
            strategy, "_convert_rectangle_pattern"
        ) as mock_convert_rectangle_pattern,
    ):
        mock_convert_trench_pattern.return_value = new_pattern
        mock_convert_rectangle_pattern.return_value = new_pattern

        updated_stage = strategy._update_milling_stage(
            stage=mock_stage, lamella_info=lamella_info
        )

        if boundary_smoothing_sigma > 0:
            lamella_gis_boundary_peturbations = (
                mock_get_lamella_gis_boundary_peturbations.return_value
            )
            mock_get_lamella_gis_boundary_peturbations.assert_called_once_with(
                lamella_info.clean_prediction,
                sigma=bitmap_config.boundary_smoothing_sigma,
            )
        else:
            lamella_gis_boundary_peturbations = None
            mock_get_lamella_gis_boundary_peturbations.assert_not_called()

        if incident_angle_sputter_ratio > 1:
            gis_lower_edge_coordinates = mock_get_mask_edge.return_value
            mock_get_mask_edge.assert_called_once_with(
                lamella_info.clean_prediction
                == bitmap_strategy.SEMSegmentationLabels.GIS.value,
                axis=0,
                side="max",
            )
        else:
            gis_lower_edge_coordinates = None
            mock_get_mask_edge.assert_not_called()

        if input_pattern_type is RectanglePattern:
            mock_convert_rectangle_pattern.assert_called_once_with(
                input_pattern,
                stats=lamella_stats,
                lamella_gis_boundary_peturbations=lamella_gis_boundary_peturbations,
                gis_lower_edge_coordinates=gis_lower_edge_coordinates,
            )
            mock_convert_trench_pattern.assert_not_called()
        elif input_pattern_type is TrenchPattern:
            mock_convert_trench_pattern.assert_called_once_with(
                input_pattern,
                stats=lamella_stats,
                lamella_gis_boundary_peturbations=lamella_gis_boundary_peturbations,
                gis_lower_edge_coordinates=gis_lower_edge_coordinates,
            )
            mock_convert_rectangle_pattern.assert_not_called()
        else:
            raise TypeError("Unexpected input pattern type")

        mock_set_bitmap_pattern_statistics.assert_called_once_with(
            lamella_stats, new_pattern=new_pattern, old_pattern=input_pattern
        )

        assert updated_stage.pattern == new_pattern
