from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock, ANY

import typing
from pathlib import Path

from fibsem import utils as fibsem_utils, acquire
from fibsem.milling.base import get_milling_stages

# TODO: look into why this is required
# Necessary to ensure AP strategy is registered:
from fibsem.milling.strategy import register_strategy
from autolamella.protocol.validation import validate_protocol


from adaptive_polish import strategy as ap_strategy

from . import setup

if typing.TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import MicroscopeSettings
    from fibsem.milling.base import FibsemMillingStage

_AP_PASS_CHECKS_CONFIG = {
    "gis_stop_um": -1,
    "max_crack_area_um2": 1e4,
    "minimum_lamella_area_um2": 0,
    "maximum_drift_um": 1e4,
}


class ExceptionForMocking(Exception):
    pass


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

    adaptive_polish_dir = lamella_directory / "adaptive_polish"
    # Check directories were created
    assert adaptive_polish_dir.is_dir(), "adaptive_polish directory wasn't created"
    for name in ("plots", "sem", "fib"):
        assert (adaptive_polish_dir / name).is_dir(), (
            f"{name} subdirectory wasn't created"
        )


@patch("pathlib.Path.is_file", mew=MagicMock(return_value=True))
@patch("adaptive_polish.strategy.gm.load_sem_model")
def test_loads_sem_model(
    mock_load_sem_model: MagicMock,
    mock_is_file: MagicMock,
    protocol_template_path: Path,
    microscope_config_path: Path,
    tmp_path: Path,
) -> None:
    """Tests that the sem model loading is called correctly"""

    model_path = "path/to/model.file"
    ap_config = ap_strategy.AdaptivePolishMillingConfig(model_path=model_path)
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

    mock_load_sem_model.side_effect = ExceptionForMocking("Expected exception")
    with pytest.raises(ExceptionForMocking):
        strategy.run(microscope, stage)

    mock_load_sem_model.assert_called_once_with(
        model_path=Path(model_path), generation=None
    )
    mock_is_file.assert_called_once()


def test_imaging_settings_applied() -> None:
    """Test that the config imaging settings are applied rather than
    existing settings if they are different"""
    # TODO: turns out I don't understand what is needed here
    pass


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
        mock_model.predict.side_effect = ExceptionForMocking("Expected exception")

        with pytest.raises(ExceptionForMocking):
            strategy.run(microscope, stage)

        mock_model.predict.assert_called_once()

    adaptive_polish_dir = lamella_directory / "adaptive_polish"
    for image_type in ("sem", "fib"):
        image_path = adaptive_polish_dir / image_type
        image_paths = tuple(image_path.glob("*.tif"))
        assert len(image_paths) == 1, f"Only one {image_type} image expected"
        expected_plot_path = image_path / f"lamella_AP_img_000_{image_type.upper()}.tif"
        assert image_paths[0] == expected_plot_path, (
            f"Expected {image_type} image paths do not match found .tif paths"
        )


@pytest.mark.parametrize("failure_reason", ["gis", "crack", "lamella area", "centring"])
def test_milling_stops_when_check_fails(
    failure_reason: str,
    protocol_template_path: Path,
    microscope_config_demo2_path: Path,
    sem_segmentation_model: tuple[str, Path],
    fib_image_dir: Path,
    sem_image_dir: Path,
    tmp_path: Path,
) -> None:
    """Tests that milling stops when either GIS is too thin or crack
    is found"""

    pass_checks_kwargs = _AP_PASS_CHECKS_CONFIG.copy()
    if failure_reason == "gis":
        pass_checks_kwargs["gis_stop_um"] = 10
    elif failure_reason == "crack":
        pass_checks_kwargs["max_crack_area_um2"] = 0
    elif failure_reason == "lamella area":
        pass_checks_kwargs["minimum_lamella_area_um2"] = 1e5
    elif failure_reason == "centring":
        pass_checks_kwargs["maximum_drift_um"] = 0

    ap_config = ap_strategy.AdaptivePolishMillingConfig(
        model_generation=sem_segmentation_model[0],
        model_path=sem_segmentation_model[1],
        align_sem=False,
        **pass_checks_kwargs,
    )
    _, stages = setup_protocol_and_milling_stages(
        ap_config.to_dict(), protocol_template_path, tmp_path
    )
    stage = stages[0]

    microscope, settings = setup_microscope_and_settings(
        microscope_config_demo2_path,
        tmp_path,
        fib_image_dir=str(fib_image_dir),
        sem_image_dir=str(sem_image_dir),
    )

    strategy = ap_strategy.AdaptivePolishMillingStrategy(config=ap_config)

    lamella_directory = tmp_path / "lamella"
    lamella_directory.mkdir()
    settings.image.path = lamella_directory

    # Necessary to set imaging settings path
    acquire.take_reference_images(microscope, settings.image)

    with patch.object(
        ap_strategy,
        "draw_patterns",
        MagicMock(
            # Ensure test doesn't keep looping:
            side_effect=ExceptionForMocking("Unexpected exception")
        ),
    ) as mock_draw_patterns:
        strategy.run(microscope, stage)
        mock_draw_patterns.assert_not_called()


# def test_max_milling_cycles_not_exceeded() -> None:
#     """Tests that milling cycles cannot exceed the max"""
#     pass


# def test_results_saved() -> None:
#     """Tests that results are saved in correct file naming convention"""
#     pass


# def test_milling_time_adjustment() -> None:
#     """Tests that milling cycle time cannot be < 10s"""
#     pass
