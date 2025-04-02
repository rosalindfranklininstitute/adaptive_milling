from __future__ import annotations
import pytest

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
    "gis_stop_um": 0.1,
    "max_crack_area_um2": 5,
    "max_milling_cycles": 2,
    "window_size_px": 8,
    "minimum_lamella_area_um2": 50.0,
    "maximum_drift_um": 0.01,
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
    # connect to microscope
    microscope, settings = utils.setup_session(config_path=microscope_config_path)

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

    dirs = tuple(fib_adaptive_polish_dir.rglob("*"))

    # Check directories were created
    assert fib_adaptive_polish_dir.is_dir(), "adaptive_polish directory wasn't created"
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

    fib_images = (fib_adaptive_polish_dir / "fib").glob("*.tif")
    sem_images = (fib_adaptive_polish_dir / "sem").glob("*.tif")

    dirs = tuple(fib_adaptive_polish_dir.rglob("*"))
    # Check plots exist
    centring_plot_path = fib_adaptive_polish_dir / "centring.png"
    assert centring_plot_path.is_file(), f"{centring_plot_path.name} was not created"
    gis_thickness_plot_path = fib_adaptive_polish_dir / "lamella_GIS_thickness.png"
    assert gis_thickness_plot_path.is_file(), (
        f"{gis_thickness_plot_path.name} plot was not created"
    )
    plots_dir = fib_adaptive_polish_dir / "plots"
    plot_paths = tuple(plots_dir.glob("*.png"))
    assert len(plot_paths), "No plots have been created"

    results_paths = tuple(fib_adaptive_polish_dir.glob("*.csv"))
    assert len(results_paths), "No results have been created"
