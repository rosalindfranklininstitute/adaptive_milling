from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock

import typing
from pathlib import Path

from fibsem import utils as fibsem_utils, acquire
from fibsem.milling.base import get_milling_stages

# TODO: look into why this is required
# Necessary to ensure AP strategy is registered:
from fibsem.milling.strategy import register_strategy
from autolamella.protocol.validation import validate_protocol


from adaptive_polish.strategy import (
    AdaptivePolishMillingConfig,
    AdaptivePolishMillingStrategy,
)

from . import setup

if typing.TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import MicroscopeSettings
    from fibsem.milling.base import FibsemMillingStage


class TestException(Exception):
    pass


@pytest.fixture
def microscope_and_settings(
    microscope_config_path: Path, tmp_path: Path
) -> tuple[FibsemMicroscope, MicroscopeSettings]:
    microscope_config_path = setup.setup_test_microscope_config(
        microscope_template_path=microscope_config_path,
        temporary_directory=tmp_path,
    )
    return fibsem_utils.setup_session(config_path=microscope_config_path)


@pytest.fixture
def setup_protocol_and_milling_stages_function(
    protocol_template_path: Path, tmp_path: Path
) -> typing.Callable[..., tuple[dict[str, typing.Any], FibsemMillingStage]]:
    def setup_protocol_and_milling_stages(
        config_dict: dict[str, typing.Any],
    ) -> tuple[dict[str, typing.Any], FibsemMillingStage]:
        protocol_path = setup.setup_protocol_path(
            protocol_template_path, tmp_path, config_dict, ap_only=True
        )

        protocol = validate_protocol(
            fibsem_utils.load_protocol(protocol_path=protocol_path)
        )

        milling_stages = get_milling_stages("mill_polishing", protocol["milling"])

        return protocol, milling_stages

    return setup_protocol_and_milling_stages


def test_default_config() -> None:
    """Tests that default config works with no errors"""
    # Create AdaptivePolish obj with default input parameters from millingstrategy
    model_path = "path/to/model.file"
    ap_config = AdaptivePolishMillingConfig(model_path=model_path)
    ap_strategy = AdaptivePolishMillingStrategy.from_dict(
        {"config": {"model_path": model_path}}
    )
    assert ap_config == ap_strategy.config, "Configs do not match"


def test_milling_stage_loads_defaults(
    setup_protocol_and_milling_stages_function: typing.Callable[
        ..., tuple[dict[str, typing.Any], FibsemMillingStage]
    ],
) -> None:
    """Tests that default the strategy loads the defaults when setup via stages"""
    model_path = "path/to/model.file"
    config_dict = {"model_path": model_path}
    ap_config = AdaptivePolishMillingConfig.from_dict(config_dict)
    _, milling_stages = setup_protocol_and_milling_stages_function(config_dict)
    ap_strategy = milling_stages[0].strategy
    assert isinstance(ap_strategy, AdaptivePolishMillingStrategy), (
        "Milling stage strategy is not AdaptivePolishMillingStrategy"
    )
    assert ap_config == ap_strategy.config, "Configs do not match"


def test_ap_folders_created(
    microscope_and_settings: tuple[FibsemMicroscope, MicroscopeSettings],
    setup_protocol_and_milling_stages_function: typing.Callable[
        ..., tuple[dict[str, typing.Any], FibsemMillingStage]
    ],
    tmp_path: Path,
) -> None:
    """Tests that the lamella folders are created in the correct place"""

    ap_config = AdaptivePolishMillingConfig(model_path="path/to/model.file")
    _, stages = setup_protocol_and_milling_stages_function(ap_config.to_dict())

    microscope, settings = microscope_and_settings

    strategy = AdaptivePolishMillingStrategy(config=ap_config)

    lamella_directory = tmp_path / "lamella"
    lamella_directory.mkdir()
    settings.image.path = lamella_directory

    # Necessary to set imaging settings path
    acquire.take_reference_images(microscope, settings.image)

    with pytest.raises(FileNotFoundError):
        # This will raise an error but should make directories first
        strategy.run(microscope, stages[0])

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
    mock_load_sem_model,
    mock_is_file,
    microscope_and_settings: tuple[FibsemMicroscope, MicroscopeSettings],
    setup_protocol_and_milling_stages_function: typing.Callable[
        ..., tuple[dict[str, typing.Any], FibsemMillingStage]
    ],
    tmp_path: Path,
) -> None:
    """Tests that the sem model loading is called correctly"""

    model_path = "path/to/model.file"
    ap_config = AdaptivePolishMillingConfig(model_path=model_path)
    _, stages = setup_protocol_and_milling_stages_function(ap_config.to_dict())

    microscope, settings = microscope_and_settings

    strategy = AdaptivePolishMillingStrategy(config=ap_config)

    lamella_directory = tmp_path / "lamella"
    lamella_directory.mkdir()
    settings.image.path = lamella_directory

    # Necessary to set imaging settings path
    acquire.take_reference_images(microscope, settings.image)

    mock_load_sem_model.side_effect = TestException("Expected exception")
    with pytest.raises(TestException):
        # This will raise an error but should make directories first
        strategy.run(microscope, stages[0])

    mock_load_sem_model.assert_called_once_with(
        model_path=Path(model_path), generation=None
    )
    mock_is_file.assert_called_once()


def test_imaging_settings_applied() -> None:
    """Test that the config imaging settings are applied rather than
    existing settings if they are different"""
    pass


def test_reference_images_saved_correctly() -> None:
    """Test that reference images are saved in the correct file naming
    convention
    """
    pass


def test_milling_stops_when_stop_criterion_reached() -> None:
    """Tests that milling stops when either GIS is too thin or crack
    is found"""
    pass


def test_max_milling_cycles_not_exceeded() -> None:
    """Tests that milling cycles cannot exceed the max"""
    pass


def test_results_saved() -> None:
    """Tests that results are saved in correct file naming convention"""
    pass


def test_milling_time_adjustment() -> None:
    """Tests that milling cycle time cannot be < 10s"""
    pass
