from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock, call, ANY
from numpy.testing import assert_array_equal

import typing
from pathlib import Path
import numpy as np
import pandas as pd

from fibsem import utils as fibsem_utils, acquire
from fibsem.structures import BeamType, Point
from fibsem.milling.base import get_milling_stages
from autolamella.protocol.validation import validate_protocol

from adaptive_polish import strategy as ap_strategy
from adaptive_polish.dl_segmentation.sem_lamella_segmentor import SegmentationLabels

from . import setup, utils

if typing.TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import MicroscopeSettings
    from fibsem.milling.base import FibsemMillingStage

_AP_PASS_CHECKS_CONFIG = {
    "gis_stop_um": 0,
    "max_crack_area_um2": np.inf,
    "minimum_lamella_area_um2": 0,
    "maximum_drift_um": np.inf,
}

TIMESTAMP = "timestamp"


def setup_microscope_and_settings(
    microscope_config_path: Path,
    temporary_path: Path,
    fib_image_dir: typing.Optional[str] = None,
    sem_image_dir: typing.Optional[str] = None,
) -> tuple[FibsemMicroscope, MicroscopeSettings]:
    microscope_config_path = setup.setup_test_microscope_config(
        microscope_template_path=microscope_config_path,
        temporary_directory=temporary_path,
        fib_image_dir=fib_image_dir,
        sem_image_dir=sem_image_dir,
    )
    return fibsem_utils.setup_session(config_path=microscope_config_path)


def setup_protocol_and_milling_stages(
    config_dict: dict[str, typing.Any],
    protocol_template_path: Path,
    temporary_path: Path,
) -> tuple[dict[str, typing.Any], FibsemMillingStage]:
    protocol_path = setup.setup_protocol_path(
        protocol_template_path, temporary_path, config_dict, ap_only=True
    )

    protocol = validate_protocol(
        fibsem_utils.load_protocol(protocol_path=protocol_path)
    )

    milling_stages = get_milling_stages("mill_polishing", protocol["milling"])

    return protocol, milling_stages


def test_default_config() -> None:
    """Tests that default config works with no errors"""
    # Create AdaptivePolish obj with default input parameters from millingstrategy
    model_path = "path/to/model.file"
    config = ap_strategy.AdaptivePolishMillingConfig(model_path=model_path)
    strategy = ap_strategy.AdaptivePolishMillingStrategy.from_dict(
        {"config": {"model_path": model_path}}
    )
    assert config == strategy.config, "Configs do not match"


def test_milling_stage_loads_defaults(
    protocol_template_path: Path, tmp_path: Path
) -> None:
    """Tests that default the strategy loads the defaults when setup via stages"""
    model_path = "path/to/model.file"
    config_dict = {"model_path": model_path}
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

    ap_config = ap_strategy.AdaptivePolishMillingConfig(model_path="path/to/model.file")
    _, stages = setup_protocol_and_milling_stages(
        ap_config.to_dict(), protocol_template_path, tmp_path
    )
    stage = stages[0]

    microscope, settings = setup_microscope_and_settings(
        microscope_config_path, tmp_path
    )

    strategy = ap_strategy.AdaptivePolishMillingStrategy(config=ap_config)

    lamella_directory = tmp_path / "lamella"
    lamella_directory.mkdir()
    settings.image.path = lamella_directory

    # Necessary to set imaging settings path
    acquire.take_reference_images(microscope, settings.image)

    with pytest.raises(FileNotFoundError):
        # This will raise an error but should make directories first
        strategy.run(microscope, stage)

    adaptive_polish_dir = lamella_directory / f"adaptive_polish_{TIMESTAMP}"
    # Check directories were created
    assert adaptive_polish_dir.is_dir(), "adaptive_polish directory wasn't created"
    for name in ("plots", "sem", "fib"):
        assert (adaptive_polish_dir / name).is_dir(), (
            f"{name} subdirectory wasn't created"
        )


@patch.object(ap_strategy.ap_utils, "setup_results_df")
def test_loads_sem_model_on_first_run(
    mock_setup_results_df,
    protocol_template_path: Path,
    microscope_config_path: Path,
    tmp_path: Path,
) -> None:
    """Tests that the sem model loading is called correctly"""
    ap_config = ap_strategy.AdaptivePolishMillingConfig(align_sem=False)
    _, stages = setup_protocol_and_milling_stages(
        ap_config.to_dict(), protocol_template_path, tmp_path
    )
    stage = stages[0]

    microscope, settings = setup_microscope_and_settings(
        microscope_config_path, tmp_path
    )

    strategy = ap_strategy.AdaptivePolishMillingStrategy(config=ap_config)

    lamella_directory = tmp_path / "lamella"
    lamella_directory.mkdir()
    settings.image.path = lamella_directory

    # Necessary to set imaging settings path
    acquire.take_reference_images(microscope, settings.image)

    with patch.object(strategy, "_load_model") as mock_load_model:
        mock_setup_results_df.side_effect = utils.ExceptionForMocking
        with pytest.raises(utils.ExceptionForMocking):
            strategy.run(microscope, stage)

        mock_load_model.assert_called_once()
        assert mock_setup_results_df.call_count == 1, "Should have been called once"

        strategy.model = "model"

        with pytest.raises(utils.ExceptionForMocking):
            strategy.run(microscope, stage)

        # Check it wasn't called again after model is set
        mock_load_model.assert_called_once()
        assert mock_setup_results_df.call_count == 2, "Should have been called twice"


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

    model_path = "path/to/model.file"
    ap_config = ap_strategy.AdaptivePolishMillingConfig(
        model_path=model_path, align_sem=False
    )
    _, stages = setup_protocol_and_milling_stages(
        ap_config.to_dict(), protocol_template_path, tmp_path
    )

    microscope, settings = setup_microscope_and_settings(
        microscope_config_path, tmp_path
    )
    stage = stages[0]

    strategy = ap_strategy.AdaptivePolishMillingStrategy(config=ap_config)

    lamella_directory = tmp_path / "lamella"
    lamella_directory.mkdir()
    settings.image.path = lamella_directory

    # Necessary to set imaging settings path
    acquire.take_reference_images(microscope, settings.image)

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


@patch.object(
    ap_strategy.fs_utils, "current_timestamp", new=MagicMock(return_value=TIMESTAMP)
)
def test_max_milling_cycles_not_exceeded(
    protocol_template_path: Path,
    microscope_config_path: Path,
    fib_image_dir: Path,
    sem_image_dir: Path,
    tmp_path: Path,
) -> None:
    """Tests that milling cycles cannot exceed the max"""
    max_milling_cycles = 2

    model_path = "path/to/model.file"
    pass_checks_kwargs = _AP_PASS_CHECKS_CONFIG.copy()
    pass_checks_kwargs["max_milling_cycles"] = max_milling_cycles

    ap_config = ap_strategy.AdaptivePolishMillingConfig(
        model_path=model_path,
        align_sem=False,
        **pass_checks_kwargs,
    )
    _, stages = setup_protocol_and_milling_stages(
        ap_config.to_dict(), protocol_template_path, tmp_path
    )
    stage = stages[0]

    microscope, settings = setup_microscope_and_settings(
        microscope_config_path,
        tmp_path,
        fib_image_dir=str(fib_image_dir),
        sem_image_dir=str(sem_image_dir),
    )

    strategy = ap_strategy.AdaptivePolishMillingStrategy(config=ap_config)

    lamella_directory = tmp_path / "lamella"
    lamella_directory.mkdir()
    settings.image.path = lamella_directory

    lamella_ap_folder = lamella_directory / f"adaptive_polish_{TIMESTAMP}"

    # Necessary to set imaging settings path
    acquire.take_reference_images(microscope, settings.image)

    # Stop it trying to load a model
    with (
        patch.object(strategy, "_load_model") as mock_load_model,
        patch.object(strategy, "_check_lamella") as mock_check_lamella,
        patch.object(strategy, "_mill") as mock_mill,
    ):
        strategy.run(microscope, stage)

        mock_load_model.assert_called_once()

        # One extra round of checks should be run
        mock_check_lamella.assert_has_calls(
            [
                call(
                    i,
                    image_name=f"{lamella_directory.stem}_AP_img_{i:03}",
                    fib_image=ANY,
                    sem_image=ANY,
                    plots_folder=lamella_ap_folder / "plots",
                    results_dict={
                        "image": f"{lamella_directory.stem}_AP_img_{i:03}",
                        "milling_time_s": ap_config.milling_interval_s * i,
                    },
                    expected_lamella_centre_m=None,  # Due to align_sem=False
                )
                for i in range(max_milling_cycles + 1)
            ]
        )

        # Ensure no extra rounds of milling are run
        mock_mill.assert_has_calls(
            [
                call(microscope=microscope, stage=stage)
                for _ in range(max_milling_cycles)
            ]
        )


@patch.object(
    ap_strategy.fs_utils, "current_timestamp", new=MagicMock(return_value=TIMESTAMP)
)
@patch.object(ap_strategy.gm, "clean_prediction")
@patch.object(ap_strategy.gm, "filter_gis_thickness")
def test_results_saved(
    mock_filter_gis_thickness,
    mock_clean_prediction,
    protocol_template_path: Path,
    microscope_config_path: Path,
    fib_image_dir: Path,
    sem_image_dir: Path,
    tmp_path: Path,
) -> None:
    """Tests that results are saved in correct file naming convention"""
    max_milling_cycles = 3

    model_path = "path/to/model.file"
    pass_checks_kwargs = _AP_PASS_CHECKS_CONFIG.copy()
    pass_checks_kwargs["max_milling_cycles"] = max_milling_cycles
    max_checks = max_milling_cycles + 1

    sem_res = (1536, 1024)

    ap_config = ap_strategy.AdaptivePolishMillingConfig(
        model_path=model_path,
        align_sem=False,
        sem_res_x=sem_res[0],
        sem_res_y=sem_res[1],
        **pass_checks_kwargs,
    )
    _, stages = setup_protocol_and_milling_stages(
        ap_config.to_dict(), protocol_template_path, tmp_path
    )
    stage = stages[0]

    microscope, _ = setup_microscope_and_settings(
        microscope_config_path,
        tmp_path,
        fib_image_dir=str(fib_image_dir),
        sem_image_dir=str(sem_image_dir),
    )

    strategy = ap_strategy.AdaptivePolishMillingStrategy(config=ap_config)

    _, sem_imaging_settings = strategy._update_imaging_settings(microscope)

    lamella_directory = tmp_path / "lamella"
    lamella_directory.mkdir()
    sem_imaging_settings.path = lamella_directory
    adaptive_polish_dir = lamella_directory / f"adaptive_polish_{TIMESTAMP}"

    # Necessary to set imaging settings path
    sem_image, _ = acquire.take_reference_images(microscope, sem_imaging_settings)

    # Use this resolution to avoid any scaling so know the exact output
    mask_shape = sem_res[::-1]
    lamella_bottom = mask_shape[0] // 3
    gis_bottom = mask_shape[0] * 2 // 3

    lamella_mask = np.zeros(mask_shape, dtype=np.bool_)
    gis_mask = lamella_mask.copy()

    lamella_mask[:lamella_bottom, :] = True
    gis_mask[lamella_bottom:gis_bottom, :] = True

    mock_clean_prediction.return_value = (lamella_mask, gis_mask, None)

    expected_gis_thickness = np.asarray(
        [gis_bottom - lamella_bottom] * mask_shape[1], dtype=np.float32
    )
    expected_min_gis_thicknesses = [5 * i for i in range(1, max_checks + 1)]
    expected_filtered_gis_thicknesses = [
        np.arange(
            i,
            mask_shape[1] + i,
            dtype=np.uint16,
        )
        for i in expected_min_gis_thicknesses
    ]

    mock_filter_gis_thickness.side_effect = expected_filtered_gis_thicknesses
    # Stop it trying to load a model
    with patch.object(strategy, "model"), patch.object(strategy, "_mill") as mock_mill:
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
                call(microscope=microscope, stage=stage)
                for _ in range(max_milling_cycles)
            ]
        )
    # Results (csv)
    results_path = adaptive_polish_dir / "GIS_thickness.json"
    detailed_results_path = adaptive_polish_dir / "GIS_thickness_detailed.json"
    assert results_path.is_file(), "Results file does not exist"
    assert detailed_results_path.is_file(), "Detailed results file does not exist"

    images_names = [
        f"{lamella_directory.stem}_AP_img_{i:03}" for i in range(max_checks)
    ]
    milling_times = [ap_config.milling_interval_s * i for i in range(max_checks)]

    expected_results_df = pd.DataFrame(
        {
            "image": images_names,
            "milling_time_s": milling_times,
            "min_GIS_um": expected_min_gis_thicknesses,
            "crack_area_um2": [0] * max_checks,
        }
    )

    # results_df = pd.read_csv(results_path, index_col=0)
    results_df = pd.read_json(results_path)
    pd.testing.assert_frame_equal(
        results_df, expected_results_df, check_dtype=False, obj="Results DataFrame"
    )

    expected_detailed_results_df = pd.DataFrame(
        {
            "image": images_names,
            "milling_time_s": milling_times,
            "gis_thickness_um": pd.Series(
                [
                    [
                        _ * sem_image.metadata.pixel_size.x * 1e6
                        for _ in expected_gis_thickness
                    ]
                ]
                * max_checks,
                dtype=object,
            ),
            "gis_thickness_filtered_um": pd.Series(
                [_.tolist() for _ in expected_filtered_gis_thicknesses], dtype=object
            ),
            "xlims_px": pd.Series(
                [
                    (0, len(expected_gis_thickness) - 1)
                    for _ in range(len(expected_filtered_gis_thicknesses))
                ],
                dtype=object,
            ),
        }
    )

    detailed_results_df = pd.read_csv(detailed_results_path, index_col=0)
    detailed_results_df = pd.read_json(detailed_results_path)
    first_gis_thicknesses = detailed_results_df.loc[0, "gis_thickness_um"]
    assert len(first_gis_thicknesses) == sem_res[0], (
        "Unexpected length of gis_thickness_um"
    )
    pd.testing.assert_frame_equal(
        detailed_results_df,
        expected_detailed_results_df,
        check_dtype=False,
        obj="Detailed results DataFrame",
    )


# def test_milling_time_adjustment() -> None:
#     """Tests that milling cycle time cannot be < 10s"""
#     pass


@pytest.mark.parametrize("hits_limits", [False, True], ids=["normal", "hits_limits"])
@patch.object(ap_strategy, "get_bounding_box_scaled_to_image")
def test_align_beam(
    mock_get_bounding_box_scaled_to_image,
    hits_limits,
    microscope_config_demo2_path: Path,
    sem_image_dir: Path,
    fib_image_dir: Path,
    tmp_path: Path,
) -> None:
    plot_path = tmp_path / "plot.png"
    ap_config = ap_strategy.AdaptivePolishMillingConfig(model_path="path/to/model.file")

    microscope_config_path = setup.setup_test_microscope_config(
        microscope_config_demo2_path,
        tmp_path,
        fib_image_dir=str(fib_image_dir),
        sem_image_dir=str(sem_image_dir),
        cycle_images=True,
    )

    # connect to microscope
    microscope, settings = fibsem_utils.setup_session(
        config_path=microscope_config_path
    )

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
        sem_image.data.shape,
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

    mock_get_bounding_box_scaled_to_image.return_value = test_lamella.bounding_box

    with (
        patch.object(strategy, "model") as mock_model,
        patch.object(microscope, "beam_shift") as mock_beam_shift,
    ):
        mock_model.predict.return_value = (
            test_lamella.array.astype(np.uint8) * SegmentationLabels.LAMELLA
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


@pytest.mark.parametrize(
    "failure_reason", ["gis", "crack", "lamella area", "centring", "none"]
)
def test_check_lamella(
    failure_reason: str,
    microscope_config_demo2_path: Path,
    sem_segmentation_model: tuple[str, Path],
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
    results_dict = {}
    ap_config = ap_strategy.AdaptivePolishMillingConfig(model_path="path/to/model.file")

    pass_checks_kwargs = _AP_PASS_CHECKS_CONFIG.copy()
    if failure_reason == "gis":
        saves_results = True
        exception = ap_strategy.StopMillingException
        pass_checks_kwargs["gis_stop_um"] = 500
    elif failure_reason == "crack":
        saves_results = True
        exception = ap_strategy.StopMillingException
        pass_checks_kwargs["max_crack_area_um2"] = 0
    elif failure_reason == "lamella area":
        saves_results = False
        exception = ap_strategy.StopEarlyError
        pass_checks_kwargs["minimum_lamella_area_um2"] = 1e5
    elif failure_reason == "centring":
        saves_results = True
        exception = ap_strategy.StopEarlyError
        pass_checks_kwargs["maximum_drift_um"] = 0
    elif failure_reason == "none":
        saves_results = True
        exception = None

    ap_config = ap_strategy.AdaptivePolishMillingConfig(
        model_generation=sem_segmentation_model[0],
        model_path=sem_segmentation_model[1],
        **pass_checks_kwargs,
    )

    microscope, settings = setup_microscope_and_settings(
        microscope_config_demo2_path,
        tmp_path,
        fib_image_dir=str(fib_image_dir),
        sem_image_dir=str(sem_image_dir),
    )

    strategy = ap_strategy.AdaptivePolishMillingStrategy(config=ap_config)
    strategy._load_model()

    settings.image.path = lamella_ap_folder

    # Necessary to set imaging settings path
    sem_image, fib_image = acquire.take_reference_images(microscope, settings.image)

    with utils.assert_raises(exception):
        strategy._check_lamella(
            milling_cycle=milling_cycle,
            image_name=image_name,
            fib_image=fib_image,
            sem_image=sem_image,
            plots_folder=lamella_ap_plots_folder,
            results_dict=results_dict,
            expected_lamella_centre_m=Point(0, 0),
        )

        assert bool(results_dict) is saves_results, "Results should%s be empty" % (
            "n't" if saves_results else ""
        )


@pytest.mark.parametrize("file_exists", [True, False], ids=["file", "no file"])
@patch("pathlib.Path.is_file")
@patch("adaptive_polish.strategy.gm.load_sem_model")
def test_load_model(mock_load_sem_model, mock_is_file, file_exists: bool) -> None:
    model_generation = "model_generation"
    model_path = "model_path"

    mock_is_file.return_value = file_exists

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
