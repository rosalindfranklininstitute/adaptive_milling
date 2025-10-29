from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock, PropertyMock, call, ANY
from numpy.testing import assert_array_equal

import json
import typing
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import numpy as np

from fibsem import utils as fibsem_utils, acquire
from fibsem.structures import BeamType, Point, FibsemImage
from fibsem.milling.base import get_milling_stages
from fibsem.applications.autolamella.protocol.validation import validate_protocol

from adaptive_polish.strategy import adaptive_polish as ap_strategy
from adaptive_polish._dataclasses import CycleInformation
from adaptive_polish.processing.sem_segmentation import SegmentationLabels as SemLabels
from adaptive_polish.exceptions import SegmentationException
from adaptive_polish.enums import StopReasons

from . import setup, utils

if typing.TYPE_CHECKING:
    from fibsem.milling.base import FibsemMillingStage

_AP_PASS_CHECKS_CONFIG = {
    "gis_stop_min_um": 0,
    "max_crack_area_um2": np.inf,
    "minimum_lamella_area_um2": 0,
    "maximum_drift_um": np.inf,
}

TIMESTAMP = "timestamp"


def setup_protocol_and_milling_stages(
    config_dict: dict[str, typing.Any],
    protocol_template_path: Path,
    temporary_path: Path,
) -> tuple[dict[str, typing.Any], list[FibsemMillingStage]]:
    protocol_path = setup.setup_protocol_path(
        protocol_template_path, temporary_path, config_dict, ap_only=True
    )

    protocol = validate_protocol(
        fibsem_utils.load_protocol(protocol_path=protocol_path)
    )

    milling_stages = get_milling_stages("mill_polishing", protocol["milling"])

    return protocol, milling_stages


def assert_results_dicts_equal(
    results: dict[str, typing.Any],
    expected_results: dict[str, typing.Any],
    allow_missing_expected: bool = True,
) -> None:
    """Used to check whether results dicts match, including nested dicts and tuples/lists and allowing for ANY"""

    def compare(value: typing.Any, expected: typing.Any) -> None:
        if expected is ANY:
            return
        elif type(value) is not type(expected):
            raise TypeError(
                f"types do not match: expected {type(expected)} got {type(value)}"
            )
        elif isinstance(expected, dict):
            return assert_results_dicts_equal(
                value, expected, allow_missing_expected=allow_missing_expected
            )
        elif isinstance(expected, (list, tuple)):
            expected_num = len(expected)
            num = len(value)
            assert num == expected_num, (
                f"different lengths: expected {expected_num}, got {num}"
            )
            for i in range(expected_num):
                try:
                    compare(value=value[i], expected=expected[i])
                except Exception as e:
                    raise type(e)(f"index {i} {e}") from e
        else:
            assert value == expected, (
                f"does not match: expected {expected}, got {value}"
            )

    all_keys = set(results.keys()) | set(expected_results.keys())
    for k in all_keys:
        if k not in results:
            raise KeyError(f"{k} is missing from results")
        elif k not in expected_results:
            if allow_missing_expected:
                continue
            raise KeyError(f"{k} is missing from expected results")

        value = results[k]
        expected = expected_results[k]
        try:
            compare(value=value, expected=expected)
        except Exception as e:
            raise AssertionError(f"Mismatch: {k} {e}") from e


@patch("os.path.isfile")
def test_default_config(mock_isfile) -> None:
    """Tests that default config works with no errors"""
    # Create AdaptivePolish obj with default input parameters from millingstrategy
    mock_isfile.return_value = True
    model_path = "path/to/model.file"
    config = ap_strategy.AdaptivePolishMillingConfig(model_path=str(model_path))
    strategy = ap_strategy.AdaptivePolishMillingStrategy.from_dict(
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
    config = ap_strategy.AdaptivePolishMillingConfig.from_dict(config_dict)
    _, milling_stages = setup_protocol_and_milling_stages(
        config_dict, protocol_template_path, tmp_path
    )
    strategy = milling_stages[0].strategy
    assert isinstance(strategy, ap_strategy.AdaptivePolishMillingStrategy), (
        "Milling stage strategy is not AdaptivePolishMillingStrategy"
    )
    assert config == strategy.config, "Configs do not match"


@patch.object(
    ap_strategy.fs_utils, "current_timestamp", new=MagicMock(return_value=TIMESTAMP)
)
def test_ap_folders_created(
    protocol_template_path: Path,
    microscope_config_path,
    tmp_path: Path,
) -> None:
    """Tests that the lamella folders are created in the correct place"""
    model_path = tmp_path / "model.file"
    model_path.touch()
    ap_config = ap_strategy.AdaptivePolishMillingConfig(model_path=str(model_path))
    _, stages = setup_protocol_and_milling_stages(
        ap_config.to_dict(), protocol_template_path, tmp_path
    )
    stage = stages[0]

    microscope, _ = fibsem_utils.setup_session(config_path=microscope_config_path)

    strategy = ap_strategy.AdaptivePolishMillingStrategy(config=ap_config)

    lamella_directory = tmp_path / "lamella"
    lamella_directory.mkdir()

    stage.imaging.path = lamella_directory

    with pytest.raises(SegmentationException):
        # This will raise an error but should make directories first
        strategy.run(microscope, stage)

    adaptive_polish_dir = lamella_directory / f"adaptive_polish_{TIMESTAMP}"
    # Check directories were created
    assert adaptive_polish_dir.is_dir(), "adaptive_polish directory wasn't created"
    for name in ("plots", "sem", "fib"):
        assert (adaptive_polish_dir / name).is_dir(), (
            f"{name} subdirectory wasn't created"
        )


@patch.object(ap_strategy, "_restore_beam_shifts")
def test_loads_sem_model_on_first_run(
    mock_restore_beam_shifts,
    protocol_template_path: Path,
    microscope_config_path: Path,
    tmp_path: Path,
) -> None:
    """Tests that the sem model loading is called correctly"""
    model_path = tmp_path / "model.file"
    model_path.touch()
    ap_config = ap_strategy.AdaptivePolishMillingConfig(
        model_path=str(model_path), align_sem=False
    )
    _, stages = setup_protocol_and_milling_stages(
        ap_config.to_dict(), protocol_template_path, tmp_path
    )
    stage = stages[0]

    microscope, _ = fibsem_utils.setup_session(config_path=microscope_config_path)

    strategy = ap_strategy.AdaptivePolishMillingStrategy(config=ap_config)

    stage.imaging.path = tmp_path

    with patch.object(strategy, "_load_model") as mock_load_model:
        mock_restore_beam_shifts.side_effect = utils.ExceptionForMocking
        with pytest.raises(utils.ExceptionForMocking):
            strategy.run(microscope, stage)

        mock_load_model.assert_called_once()
        assert mock_restore_beam_shifts.call_count == 1, "Should have been called once"

        strategy.model = "model"  # type: ignore

        with pytest.raises(utils.ExceptionForMocking):
            strategy.run(microscope, stage)

        # Check it wasn't called again after model is set
        mock_load_model.assert_called_once()
        assert mock_restore_beam_shifts.call_count == 2, "Should have been called twice"


@patch.object(
    ap_strategy.fs_utils, "current_timestamp", new=MagicMock(return_value=TIMESTAMP)
)
def test_reference_images_saved_correctly(
    protocol_template_path: Path,
    microscope_config_path: Path,
    tmp_path: Path,
) -> None:
    """Test that reference images are saved in the correct file naming
    convention
    """
    model_path = tmp_path / "model.file"
    model_path.touch()
    ap_config = ap_strategy.AdaptivePolishMillingConfig(
        model_path=str(model_path), align_sem=False
    )
    _, stages = setup_protocol_and_milling_stages(
        ap_config.to_dict(), protocol_template_path, tmp_path
    )

    microscope, _ = fibsem_utils.setup_session(config_path=microscope_config_path)
    stage = stages[0]

    strategy = ap_strategy.AdaptivePolishMillingStrategy(config=ap_config)

    lamella_directory = tmp_path / "lamella"
    lamella_directory.mkdir()

    stage.imaging.path = lamella_directory

    with patch.object(strategy, "model") as mock_model:
        # Exit test at prediction step
        mock_model.predict.side_effect = utils.ExceptionForMocking("Expected exception")

        with pytest.raises(utils.ExceptionForMocking):
            strategy.run(microscope, stage)

        mock_model.predict.assert_called_once()

    adaptive_polish_dir = lamella_directory / f"adaptive_polish_{TIMESTAMP}"
    for image_type in ("sem", "fib"):
        image_path = adaptive_polish_dir / image_type
        image_paths = tuple(image_path.glob("*.tif"))
        assert len(image_paths) == 1, f"Only one {image_type} image expected"
        expected_plot_path = image_path / f"lamella_AP_img_000_{image_type.upper()}.tif"
        assert image_paths[0] == expected_plot_path, (
            f"Expected {image_type} image paths do not match found .tif paths"
        )


@patch.object(ap_strategy, "create_milling_cycle_plot")
@patch.object(
    ap_strategy.fs_utils, "current_timestamp", new=MagicMock(return_value=TIMESTAMP)
)
def test_max_milling_cycles_not_exceeded(
    mock_create_milling_cycle_plot: MagicMock,
    protocol_template_path: Path,
    microscope_config_path: Path,
    tmp_path: Path,
) -> None:
    """Tests that milling cycles cannot exceed the max"""
    max_milling_cycles = 2

    model_path = tmp_path / "model.file"
    model_path.touch()
    pass_checks_kwargs = _AP_PASS_CHECKS_CONFIG.copy()
    pass_checks_kwargs["max_milling_cycles"] = max_milling_cycles

    ap_config = ap_strategy.AdaptivePolishMillingConfig(
        model_path=str(model_path),
        align_sem=False,
        **pass_checks_kwargs,  # type: ignore[arg-type]
    )
    _, stages = setup_protocol_and_milling_stages(
        ap_config.to_dict(), protocol_template_path, tmp_path
    )
    stage = stages[0]

    microscope, _ = fibsem_utils.setup_session(config_path=microscope_config_path)

    strategy = ap_strategy.AdaptivePolishMillingStrategy(config=ap_config)

    lamella_directory = tmp_path / "lamella"
    lamella_ap_folder = lamella_directory / f"adaptive_polish_{TIMESTAMP}"
    plots_directory = lamella_ap_folder / "plots"

    lamella_directory.mkdir()

    stage.imaging.path = lamella_directory

    # Stop it trying to load a model
    with (
        patch.object(strategy, "_load_model") as mock_load_model,
        patch.object(
            strategy, "_get_lamella_info", autospec=True
        ) as mock_get_lamella_info,
        patch.object(strategy, "_check_lamella", autospec=True) as mock_check_lamella,
        patch.object(strategy, "_update_milling_stage") as mock_update_milling_stage,
        patch.object(strategy, "_save_predictions") as mock_save_predictions,
        patch.object(strategy, "_mill") as mock_mill,
    ):
        mock_stats = MagicMock()
        # Mock properties
        mock_stats.gis_thickness_filtered_um = PropertyMock()
        mock_stats.gis_thickness_min_um = PropertyMock()
        mock_stats.crack_area_um2 = PropertyMock()
        # Create a new dict each time to_dict is called
        mock_stats.to_dict.side_effect = dict
        # Create LamellaInformation mocks with statistics set
        lamella_infos = [
            MagicMock(statistics=mock_stats) for _ in range(max_milling_cycles + 1)
        ]
        mock_get_lamella_info.side_effect = lamella_infos
        mock_update_milling_stage.return_value = "stage"

        strategy.run(microscope, stage)

        mock_load_model.assert_called_once()

        assert mock_get_lamella_info.call_count == max_milling_cycles + 1, (
            "An incorrect number of cycles were run"
        )
        for i in range(max_milling_cycles + 1):
            call = mock_get_lamella_info.call_args_list[i]
            assert not call.args, "No positional args expected"
            cycle_info = call.kwargs.get("cycle_info")
            assert cycle_info, "cycle_info argument was not given"
            assert cycle_info.milling_cycle == i
            assert cycle_info.identifier == f"{lamella_directory.stem}_AP_img_{i:03}"

            lamella_pad_x = call.kwargs.get("lamella_pad_x")
            assert lamella_pad_x == strategy.config.lamella_pad_x

        # One extra round of checks should be run
        mock_check_lamella.assert_has_calls(
            [
                call(
                    lamella_info=lamella_infos[i],
                    expected_lamella_centre_m=None,  # Due to align_sem=False
                    plots_directory=plots_directory,
                )
                for i in range(max_milling_cycles + 1)
            ]
        )

        mock_update_milling_stage.assert_has_calls(
            [
                call(stage=stage, lamella_info=lamella_infos[i])
                for i in range(max_milling_cycles + 1)
            ]
        )

        mock_save_predictions.assert_has_calls(
            [
                call(
                    directory=lamella_ap_folder / "predictions",
                    lamella_info=lamella_infos[i],
                )
                for i in range(max_milling_cycles + 1)
            ]
        )

        mock_create_milling_cycle_plot.assert_has_calls(
            [
                call(
                    save_path=plots_directory
                    / f"{lamella_infos[i].identifier}_plot.png",
                    sem_image=lamella_infos[i].sem_image,
                    fib_image=lamella_infos[i].fib_image,
                    first_prediction=lamella_infos[i].prediction,
                    clean_prediction=lamella_infos[i].clean_prediction,
                    gis_thickness_um=lamella_infos[
                        i
                    ].statistics.gis_thickness_filtered_um,
                    gis_thickness_min_um=lamella_infos[
                        i
                    ].statistics.gis_thickness_min_um,
                    crack_area_um2=lamella_infos[i].statistics.crack_area_um2,
                    gis_stop_threshold_um=strategy.config.gis_stop_min_um,
                    milling_stage=mock_update_milling_stage.return_value,
                    image_xlims=lamella_infos[i].statistics.xlims_image_px,
                    max_crack_area_um2=strategy.config.max_crack_area_um2,
                    img_name=lamella_infos[i].identifier,
                )
                for i in range(max_milling_cycles + 1)
            ],
        )

        # Ensure no extra rounds of milling are run
        mock_mill.assert_has_calls(
            [
                call(
                    i,
                    microscope=microscope,
                    stage=mock_update_milling_stage.return_value,
                    asynch=False,
                    parent_ui=None,
                )
                for i in range(max_milling_cycles)
            ]
        )


@patch.object(
    ap_strategy.fs_utils, "current_timestamp", new=MagicMock(return_value=TIMESTAMP)
)
@patch.object(ap_strategy.lamella_proc, "clean_prediction")
@patch.object(ap_strategy.lamella_proc, "filter_gis_thickness")
def test_results_saved(
    mock_filter_gis_thickness,
    mock_clean_prediction,
    protocol_template_path: Path,
    microscope_config_path: Path,
    tmp_path: Path,
) -> None:
    """Tests that results are saved in correct file naming convention"""
    max_milling_cycles = 3

    model_path = tmp_path / "model.file"
    model_path.touch()
    pass_checks_kwargs = _AP_PASS_CHECKS_CONFIG.copy()
    pass_checks_kwargs["max_milling_cycles"] = max_milling_cycles
    max_checks = max_milling_cycles + 1

    sem_res: tuple[int, int] = (1536, 1024)

    ap_config = ap_strategy.AdaptivePolishMillingConfig(
        model_path=str(model_path),
        align_sem=False,
        **pass_checks_kwargs,  # type: ignore[arg-type]
    )
    _, stages = setup_protocol_and_milling_stages(
        ap_config.to_dict(), protocol_template_path, tmp_path
    )
    stage = stages[0]

    lamella_name = f"lamella_{uuid4()}"
    lamella_directory = tmp_path / lamella_name
    lamella_directory.mkdir()
    adaptive_polish_dir = lamella_directory / f"adaptive_polish_{TIMESTAMP}"

    stage.imaging.path = lamella_directory

    microscope, _ = fibsem_utils.setup_session(config_path=microscope_config_path)

    strategy = ap_strategy.AdaptivePolishMillingStrategy(config=ap_config)

    _, sem_imaging_settings = strategy._get_imaging_settings(stage)

    sem_image, _ = acquire.take_reference_images(microscope, sem_imaging_settings)

    # Use this resolution to avoid any scaling so know the exact output
    mask_shape = sem_res[::-1]
    lamella_bottom = mask_shape[0] // 3
    gis_bottom = mask_shape[0] * 2 // 3

    lamella_mask = np.zeros(mask_shape, dtype=np.bool_)
    gis_mask = lamella_mask.copy()

    lamella_mask[:lamella_bottom, :] = True
    gis_mask[lamella_bottom:gis_bottom, :] = True

    prediction = np.full(mask_shape, SemLabels.VACUUM.value, dtype=np.uint8)
    prediction[lamella_mask] = SemLabels.LAMELLA.value
    prediction[gis_mask] = SemLabels.GIS.value

    mock_clean_prediction.return_value = prediction

    expected_gis_thickness = np.asarray(
        [gis_bottom - lamella_bottom] * mask_shape[1], dtype=np.float32
    )
    expected_min_gis_thicknesses = [5 * i for i in range(1, max_checks + 1)]
    expected_filtered_gis_thicknesses = [
        np.arange(
            i,
            mask_shape[1] + i,
            dtype=float,
        )
        for i in expected_min_gis_thicknesses
    ]

    mock_filter_gis_thickness.side_effect = expected_filtered_gis_thicknesses
    # Stop it trying to load a model
    with (
        patch.object(strategy, "model") as mock_model,
        patch.object(strategy, "_update_milling_stage") as mock_update_milling_stage,
        patch.object(strategy, "_mill") as mock_mill,
    ):
        mock_model.predict.return_value = prediction
        mock_update_milling_stage.return_value = deepcopy(stage)

        strategy.run(microscope, stage)

        # # The calls for cracks will be skipped as mask_crack_clean is None
        # mock_get_mask_area_um2.assert_has_calls(
        #     [call(lamella_mask, pixel_size_um=ANY) for _ in range(max_checks)]
        # )
        assert mock_filter_gis_thickness.call_count == max_checks, (
            f"Checks didn't run the full {max_checks} times"
        )
        mock_mill.assert_has_calls(
            [
                call(
                    _,
                    microscope=microscope,
                    stage=mock_update_milling_stage.return_value,
                    asynch=False,
                    parent_ui=None,
                )
                for _ in range(max_milling_cycles)
            ]
        )
    # Results (csv)
    results_path = adaptive_polish_dir / "AP_metadata.json"
    assert results_path.is_file(), "Results file does not exist"

    expected_results = {
        "strategy_name": strategy.name,
        "stage_name": mock_update_milling_stage.return_value.name,
        "lamella_name": lamella_name,
        "timestamps": ANY,
        "strategy_end_reason": StopReasons.MAX_CYCLES.value,
        "cycle_information": [
            {
                "milling_cycle": i,
                "identifier": f"{lamella_directory.stem}_AP_img_{i:03}",
                "timestamps": ANY,
                "lamella_statistics": {
                    "image_pixel_size_m": [
                        sem_image.metadata.pixel_size.x,
                        sem_image.metadata.pixel_size.y,
                    ],
                    "prediction_pixel_size_m": ANY,
                    "crack_count": 0,
                    "lamella_thickness_prediction_px": ANY,
                    "crack_thickness_prediction_px": ANY,
                    "estimated_milling_time_s": ANY,
                    "lamella_bounding_box_prediction_px": ANY,
                    "xlims_prediction_px": ANY,
                    "lamella_bounding_box_image_px": ANY,
                    "xlims_image_px": ANY,
                    "gis_thickness_image_px": expected_gis_thickness.tolist(),
                    "gis_thickness_filtered_image_px": expected_filtered_gis_thicknesses[
                        i
                    ]
                    .astype(np.float_)
                    .tolist(),
                    "gis_thickness_min_image_px": float(
                        expected_min_gis_thicknesses[i]
                    ),
                    "gis_thickness_median_image_px": float(
                        np.nanmedian(expected_filtered_gis_thicknesses[i])
                    ),
                    "gis_thickness_mean_image_px": float(
                        np.nanmean(expected_filtered_gis_thicknesses[i])
                    ),
                    "pattern_xlims_px": None,
                    "pattern_dwell_multiplier": None,
                    "pattern_blanking": None,
                },
            }
            for i in range(max_checks)
        ],
    }

    with results_path.open("r") as f:
        results_dict = json.loads(f.read())

    assert_results_dicts_equal(
        results_dict, expected_results, allow_missing_expected=False
    )


@pytest.mark.parametrize("hits_limits", [False, True], ids=["normal", "hits_limits"])
@patch.object(ap_strategy.image_proc, "get_bounding_box_scaled_to_image")
def test_align_beam(
    mock_get_bounding_box_scaled_to_image,
    hits_limits,
    microscope_config_path: Path,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.file"
    model_path.touch()

    plot_path = tmp_path / "plot.png"
    ap_config = ap_strategy.AdaptivePolishMillingConfig(model_path=str(model_path))

    # connect to microscope
    microscope, settings = fibsem_utils.setup_session(
        config_path=microscope_config_path
    )
    settings.image.path = tmp_path

    # Necessary to set imaging settings path
    sem_image, _ = acquire.take_reference_images(microscope, settings.image)

    sem_imaging_settings = microscope.get_imaging_settings(BeamType.ELECTRON)
    microscope.electron_system.beam.shift = Point(0, 0)

    # Y,X
    lamella_centre_px = (60, 40)
    lamella_centre_m = (
        lamella_centre_px[0] * sem_image.metadata.pixel_size.y,
        lamella_centre_px[1] * sem_image.metadata.pixel_size.x,
    )
    test_lamella = setup.SimpleRectangleLamellaMask(
        (sem_image.data.shape[0], sem_image.data.shape[1]),
        20,
        centre_px=(
            sem_image.data.shape[0] // 2 - lamella_centre_px[0],
            sem_image.data.shape[1] // 2 + lamella_centre_px[1],
        ),
    )

    if hits_limits:
        # Points are X, Y
        new_beam_shift = Point(-30e-6, -20e-6)
        expected_new_lamella_centre_m = Point(
            lamella_centre_m[1] + new_beam_shift.x,
            lamella_centre_m[0] + new_beam_shift.y,
        )
    else:
        new_beam_shift = Point(-lamella_centre_m[1], -lamella_centre_m[0])
        expected_new_lamella_centre_m = Point(0, 0)

    strategy = ap_strategy.AdaptivePolishMillingStrategy(config=ap_config)

    mock_get_bounding_box_scaled_to_image.return_value = (
        test_lamella.bounding_box,
        test_lamella.bounding_box,
    )

    with (
        patch.object(strategy, "model") as mock_model,
        patch.object(microscope, "beam_shift") as mock_beam_shift,
    ):
        mock_model.predict.return_value = (
            test_lamella.array.astype(np.uint8) * SemLabels.LAMELLA
        )

        def mock_beam_shift_fn(dx, dy, beam_type) -> None:
            microscope.electron_system.beam.shift = new_beam_shift
            return

        mock_beam_shift.side_effect = mock_beam_shift_fn

        new_lamella_centre_m = strategy._align_beam(
            microscope, sem_imaging_settings=sem_imaging_settings, plot_path=plot_path
        )

        mock_get_bounding_box_scaled_to_image.assert_called_once()

        assert_array_equal(
            mock_get_bounding_box_scaled_to_image.call_args_list[0].kwargs["mask"],
            test_lamella.array,
        )

        mock_beam_shift.assert_called_once_with(
            -lamella_centre_m[1], -lamella_centre_m[0], BeamType.ELECTRON
        )

    assert plot_path.is_file(), "Plot was not created"

    assert new_lamella_centre_m == expected_new_lamella_centre_m


@pytest.mark.usefixtures("skip_if_no_models")
@pytest.mark.parametrize(
    "failure_reason",
    ["min_gis", "mean_gis", "crack", "lamella area", "centring", "none"],
)
def test_check_lamella(
    failure_reason: str,
    latest_sem_segmentation_model: tuple[str, Path],
    fib_image_dir: Path,
    sem_image_dir: Path,
    tmp_path: Path,
) -> None:
    """Tests that milling stops when either GIS is too thin or crack
    is found"""
    milling_cycle = 0
    image_name = "test_image"
    lamella_ap_folder = tmp_path / "lamella"
    lamella_ap_plots_folder = lamella_ap_folder / "plots"
    lamella_ap_folder.mkdir()
    lamella_ap_plots_folder.mkdir()

    saves_results: bool
    pass_checks_kwargs = _AP_PASS_CHECKS_CONFIG.copy()
    check_exception: type[Exception] | None = None
    info_exception: type[Exception] | None = None
    if failure_reason == "min_gis":
        saves_results = True
        check_exception = ap_strategy.StopMillingException
        pass_checks_kwargs["gis_stop_min_um"] = 500
    elif failure_reason == "mean_gis":
        saves_results = True
        check_exception = ap_strategy.StopMillingException
        pass_checks_kwargs["gis_stop_median_um"] = 500
    elif failure_reason == "crack":
        saves_results = True
        check_exception = ap_strategy.StopMillingException
        pass_checks_kwargs["max_crack_area_um2"] = 0
    elif failure_reason == "lamella area":
        saves_results = False
        check_exception = ap_strategy.StopEarlyError
        pass_checks_kwargs["minimum_lamella_area_um2"] = 1e5
    elif failure_reason == "centring":
        saves_results = True
        check_exception = ap_strategy.StopEarlyError
        pass_checks_kwargs["maximum_drift_um"] = 0
    elif failure_reason == "none":
        saves_results = True
    else:
        raise NotImplementedError(f"Invalid failure reason {failure_reason}")

    ap_config = ap_strategy.AdaptivePolishMillingConfig(
        model_generation=latest_sem_segmentation_model[0],
        model_path=str(latest_sem_segmentation_model[1]),
        **pass_checks_kwargs,  # type: ignore[arg-type]
    )

    strategy = ap_strategy.AdaptivePolishMillingStrategy(config=ap_config)
    strategy._load_model()

    # open images
    fib_image_path = next(fib_image_dir.glob("*.tif"))
    sem_image_path = next(sem_image_dir.glob("*.tif"))
    fib_image = FibsemImage.load(str(fib_image_path))
    sem_image = FibsemImage.load(str(sem_image_path))

    stage = MagicMock()
    stage.pattern.time = 20

    with utils.assert_raises(info_exception):
        lamella_info = strategy._get_lamella_info(
            cycle_info=CycleInformation(
                milling_cycle=milling_cycle, identifier=image_name
            ),
            sem_image=sem_image,
            fib_image=fib_image,
            lamella_pad_x=0,
        )

    with utils.assert_raises(check_exception):
        strategy._check_lamella(
            lamella_info=lamella_info,
            plots_directory=lamella_ap_plots_folder,
            expected_lamella_centre_m=Point(0, 0),
        )

    for key, value in lamella_info.statistics.to_dict().items():
        if key in (
            "estimated_milling_time_s",
            "pattern_xlims_px",
            "pattern_dwell_multiplier",
            "pattern_blanking",
        ):
            # Not set in this test, as that is done by _mill
            assert value is None, f"{key} should be None"
        elif not saves_results and key in (
            "lamella_bounding_box_prediction_px",
            "lamella_bounding_box_image_px",
            "gis_thickness_image_px",
            "xlims_prediction_px",
            "xlims_image_px",
            "gis_thickness_image_px",
            "gis_thickness_filtered_image_px",
            "gis_thickness_min_image_px",
            "gis_thickness_median_image_px",
            "gis_thickness_mean_image_px",
        ):
            assert value is None, f"{key} should be None"
        else:
            assert value is not None, f"{key} should not be None"


@pytest.mark.parametrize("file_exists", [True, False], ids=["file", "no file"])
@patch("pathlib.Path.is_file")
@patch("os.path.isfile")
@patch("adaptive_polish.strategy.adaptive_polish.sem_seg_proc.load_model")
def test_load_model(
    mock_load_sem_model, mock_os_isfile, mock_pathlib_is_file, file_exists: bool
) -> None:
    model_generation = "model_generation"
    model_path = "model_path"
    mock_os_isfile.return_value = True

    mock_pathlib_is_file.return_value = file_exists

    ap_config = ap_strategy.AdaptivePolishMillingConfig(
        model_generation=model_generation,
        model_path=model_path,
    )

    strategy = ap_strategy.AdaptivePolishMillingStrategy(config=ap_config)

    with utils.assert_raises(None if file_exists else FileNotFoundError):
        strategy._load_model()

    if file_exists:
        mock_load_sem_model.assert_called_once_with(
            model_path=Path(model_path), generation=model_generation
        )
        assert strategy.model == mock_load_sem_model.return_value
    else:
        mock_load_sem_model.assert_not_called()
        assert strategy.model is None


def test_restore_beam_shifts(microscope_config_path: Path) -> None:
    microscope, _ = fibsem_utils.setup_session(config_path=microscope_config_path)
    initial_electron_shift = Point(-11, 12)
    initial_ion_shift = Point(50, -20)

    new_electron_shift = Point(20, -80)
    new_ion_shift = Point(-2, 10)

    microscope.set("shift", initial_electron_shift, BeamType.ELECTRON)
    microscope.set("shift", initial_ion_shift, BeamType.ION)

    assert microscope.get("shift", BeamType.ELECTRON) == initial_electron_shift, (
        "Failed to set electron beam shift"
    )
    assert microscope.get("shift", BeamType.ION) == initial_ion_shift, (
        "Failed to set ion beam shift"
    )

    with ap_strategy._restore_beam_shifts(microscope):
        microscope.set("shift", new_electron_shift, BeamType.ELECTRON)
        microscope.set("shift", new_ion_shift, BeamType.ION)

        assert microscope.get("shift", BeamType.ELECTRON) == new_electron_shift, (
            "Failed to set electron beam shift"
        )
        assert microscope.get("shift", BeamType.ION) == new_ion_shift, (
            "Failed to set ion beam shift"
        )

    assert microscope.get("shift", BeamType.ELECTRON) == initial_electron_shift, (
        "Failed to restore electron beam shift"
    )
    assert microscope.get("shift", BeamType.ION) == initial_ion_shift, (
        "Failed to restore ion beam shift"
    )
