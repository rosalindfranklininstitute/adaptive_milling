from __future__ import annotations
import pytest
from unittest.mock import patch

import functools
import typing
from pathlib import Path

from fibsem import utils, acquire
from fibsem.milling import get_milling_stages, mill_stages
from fibsem.milling.strategy import register_strategy
from fibsem.structures import BeamType
from autolamella.protocol.validation import validate_protocol
from autolamella.structures import Experiment, AutoLamellaProtocol
from adaptive_polish.strategy import AdaptivePolishMillingConfig

from . import setup

if typing.TYPE_CHECKING:
    from pathlib import Path

_AP_MILLING_CONFIG_SETTINGS = {
    "align_sem": True,
    "milling_interval_s": 3,
    "gis_stop_um": -1,  # Ensures check passes
    "max_crack_area_um2": 1e4,  # Ensures check passes
    "max_milling_cycles": 2,
    "window_size_px": 8,
    "minimum_lamella_area_um2": 0,  # Ensures check passes
    "maximum_drift_um": 1e4,  # Ensures check passes
}


@pytest.fixture
def microscope_config_path(
    microscope_config_demo2_path: Path,
    fib_image_dir: Path,
    sem_image_dir: Path,
    tmp_path: Path,
) -> Path:
    return setup.setup_test_microscope_config(
        microscope_config_demo2_path,
        tmp_path,
        fib_image_dir=fib_image_dir,
        sem_image_dir=sem_image_dir,
        cycle_images=True,
    )


@pytest.fixture
def protocol_path(
    protocol_template_path: Path,
    tmp_path: Path,
    sem_segmentation_model,
) -> Path:
    ap_config = AdaptivePolishMillingConfig(
        model_generation=sem_segmentation_model[0],
        model_path=sem_segmentation_model[1],
        **_AP_MILLING_CONFIG_SETTINGS,
    )

    return setup.setup_protocol_path(
        protocol_template_path, tmp_path, ap_config, ap_only=True
    )


@pytest.fixture
def experiment_config_path(experiment_template_path, tmp_path: Path):
    return setup.setup_test_experiment(experiment_template_path, tmp_path)


def test_runs(
    microscope_config_path: Path,
    experiment_config_path: Path,
    protocol_path: Path,
    tmp_path: Path,
) -> None:
    calls_before_exception = 1
    expected_loops = calls_before_exception + 1

    # connect to microscope
    microscope, settings = utils.setup_session(config_path=microscope_config_path)

    original_stop_milling = microscope.stop_milling

    def raise_error_after_num_calls(
        fn: typing.Callable,
        calls_before_exception: int,
    ) -> typing.Any:
        _calls = 0

        @functools.wraps(fn)
        def raise_error_after_wrapper(
            *args: typing.Any, **kwargs: typing.Any
        ) -> typing.Any:
            nonlocal _calls
            if _calls >= calls_before_exception:
                _calls = 0  # Reset for other loops
                raise Exception("Raising error to exit test")
            _calls += 1
            return fn(*args, **kwargs)

        return raise_error_after_wrapper

    with patch.object(microscope, "stop_milling") as mock_stop_milling:
        mock_stop_milling.side_effect = raise_error_after_num_calls(
            original_stop_milling, calls_before_exception=calls_before_exception
        )

        protocol = validate_protocol(utils.load_protocol(protocol_path=protocol_path))

        lamella_directory = tmp_path / "lamella"
        lamella_directory.mkdir()
        fib_adaptive_polish_dir = lamella_directory / "adaptive_polish"

        settings.image.path = lamella_directory

        acquire.take_reference_images(microscope, settings.image)

        milling_stages = get_milling_stages("mill_polishing", protocol["milling"])
        strategy_config: AdaptivePolishMillingConfig = milling_stages[0].strategy.config
        print("Adaptive Polishing Config:")
        print(f"Milling Interval: {strategy_config.milling_interval_s}")
        print(f"Maximum Cycles: {strategy_config.max_milling_cycles}")
        print(f"Model: {strategy_config.model_path}")

        # run milling stages
        mill_stages(microscope=microscope, stages=milling_stages)

        assert mock_stop_milling.call_count == 2, (
            "stop_milling was called an unexpected number of times"
        )

        # Check directories were created
        assert fib_adaptive_polish_dir.is_dir(), (
            "adaptive_polish directory wasn't created"
        )
        for name in ("plots", "sem", "fib"):
            assert (fib_adaptive_polish_dir / name).is_dir(), (
                f"{name} subdirectory wasn't created"
            )

        exp = Experiment.load(experiment_config_path)
        pos = exp.positions[0]
        protocol = AutoLamellaProtocol.load(protocol_path)
        print(protocol.method.workflow)

        print(f"Last Completed: {pos.last_completed}")
        print(f"Next: {protocol.method.get_next(pos.workflow)}")
        print(f"Previous: {protocol.method.get_previous(pos.workflow)}")
        print(f"Workflow: {protocol.method.workflow}")

        acquire.take_reference_images(microscope, settings.image)

        dirs = tuple(fib_adaptive_polish_dir.rglob("*"))

        milling_stages = protocol.milling["mill_polishing"]
        strategy_config: AdaptivePolishMillingConfig = milling_stages[0].strategy.config
        for k, v in _AP_MILLING_CONFIG_SETTINGS.items():
            value = getattr(strategy_config, k)
            assert getattr(strategy_config, k) == v, (
                f"Strategy config does not match expected value: {k} is {value}, expected {v}"
            )

        # run milling stages
        mill_stages(microscope=microscope, stages=milling_stages)

        assert mock_stop_milling.call_count == expected_loops * 2, (
            "stop_milling was called an unexpected number of times"
        )

    dirs = tuple(fib_adaptive_polish_dir.rglob("*"))

    expected_stems = [f"lamella_AP_img_{_:>03}_" for _ in range(expected_loops)]

    # Check plots exist
    centring_plot_path = fib_adaptive_polish_dir / "centring.png"
    assert centring_plot_path.is_file(), f"{centring_plot_path.name} was not created"
    gis_thickness_plot_path = fib_adaptive_polish_dir / "lamella_GIS_thickness.png"
    assert gis_thickness_plot_path.is_file(), (
        f"{gis_thickness_plot_path.name} plot was not created"
    )
    plots_dir = fib_adaptive_polish_dir / "plots"
    plot_paths = set(plots_dir.glob("*.png"))
    assert len(plot_paths), "No plots have been created"
    expected_plot_file_names = (f"{_}plot.png" for _ in expected_stems)
    expected_plot_paths = set((plots_dir / _ for _ in expected_plot_file_names))
    assert plot_paths == expected_plot_paths, (
        "Expected plot file paths do not match found plot paths"
    )

    results_paths = set(fib_adaptive_polish_dir.glob("*.csv"))
    assert len(results_paths), "No results have been created"
    expected_results_file_names = ("GIS_thickness.csv", "GIS_thickness_detailed.csv")
    expected_results_paths = set(
        (fib_adaptive_polish_dir / _ for _ in expected_results_file_names)
    )
    assert results_paths == expected_results_paths, (
        "Expected results file paths do not match found .csv paths"
    )

    for image_type in ("sem", "fib"):
        image_path = fib_adaptive_polish_dir / image_type
        image_paths = set(image_path.glob("*.tif"))
        assert len(image_paths), f"No {image_type} images have been created"
        expected_plot_file_names = (
            f"{_}{image_type.upper()}.tif" for _ in expected_stems
        )
        expected_plot_paths = set((image_path / _ for _ in expected_plot_file_names))
        assert image_paths == expected_plot_paths, (
            f"Expected {image_type} image paths do not match found .tif paths"
        )
