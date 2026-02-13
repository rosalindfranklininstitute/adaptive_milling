from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock

import functools
import itertools
import typing
from pathlib import Path

from fibsem import utils
from fibsem.structures import FibsemImage, BeamType
from fibsem.milling import mill_stages
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol

from adaptive_polish.config import (
    AdaptivePolishMillingConfig,
    BitmapAdaptivePolishMillingConfig,
)

from . import setup

if typing.TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from fibsem.structures import ImageSettings

_AP_MILLING_CONFIG_SETTINGS = {
    "align_sem": True,
    "gis_stop_min": 0,  # Ensures check passes
    "gis_stop_median": 0,  # Ensures check passes
    "max_crack_area": 1e-6,  # Ensures check passes
    "max_milling_cycles": 2,
    "minimum_lamella_area": 0,  # Ensures check passes
    "maximum_drift": 1e-2,  # Ensures check passes
}


TIMESTAMP = "timestamp"


@pytest.fixture
def protocol_path(
    request: pytest.FixtureRequest,
    protocol_template_path: Path,
    tmp_path: Path,
    sem_segmentation_model: tuple[str, Path],
) -> Path:
    strategy_type = request.param
    if strategy_type == "bitmap":
        config_class = BitmapAdaptivePolishMillingConfig
    else:
        config_class = AdaptivePolishMillingConfig

    ap_config = config_class(
        model_generation=sem_segmentation_model[0],
        model_path=str(sem_segmentation_model[1]),
        **_AP_MILLING_CONFIG_SETTINGS,
    )

    return setup.setup_protocol_path(
        protocol_template_path,
        tmp_path,
        ap_config.to_dict(),
        ap_only=True,
        ap_type=strategy_type,
    )


def create_dummy_acquire_image_function(
    fib_image_dir: Path, sem_image_dir: Path
) -> Callable[[ImageSettings], FibsemImage]:
    fib_image_paths = itertools.cycle(fib_image_dir.glob("*.tif"))
    sem_image_paths = itertools.cycle(sem_image_dir.glob("*.tif"))

    def dummy_acquire_image(image_settings: ImageSettings) -> FibsemImage:
        if image_settings.beam_type == BeamType.ELECTRON:
            return FibsemImage.load(str(next(sem_image_paths)))
        elif image_settings.beam_type == BeamType.ION:
            return FibsemImage.load(str(next(fib_image_paths)))
        raise ValueError(f"Invalid beam time {image_settings.beam_type}")

    return dummy_acquire_image


def raise_error_after_num_calls(
    fn: Callable,
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


@pytest.mark.usefixtures("skip_if_no_models")
@pytest.mark.parametrize("protocol_path", ["normal", "bitmap"], indirect=True)
@patch(
    "adaptive_polish.strategy.adaptive_polish.fs_utils.current_timestamp",
    new=MagicMock(return_value=TIMESTAMP),
)
def test_runs(
    microscope_config_path: Path,
    protocol_path: Path,
    fib_image_dir: Path,
    sem_image_dir: Path,
    tmp_path: Path,
) -> None:
    calls_before_exception = 1
    expected_loops = calls_before_exception + 1

    # connect to microscope
    microscope, _ = utils.setup_session(config_path=microscope_config_path)

    # Check Autolamella loads protocol correctly
    protocol = AutoLamellaTaskProtocol.load(str(protocol_path))
    milling_stages = protocol.task_config["Polishing"].milling["mill_polishing"].stages
    protocol_strategy_config = milling_stages[0].strategy.config

    assert isinstance(
        protocol_strategy_config,
        (AdaptivePolishMillingConfig, BitmapAdaptivePolishMillingConfig),
    ), f"Strategy config is the wrong type: {type(protocol_strategy_config)}"

    for k, v in _AP_MILLING_CONFIG_SETTINGS.items():
        value = getattr(protocol_strategy_config, k)
        assert getattr(protocol_strategy_config, k) == v, (
            f"Strategy config does not match expected value: {k} is {value}, expected {v}"
        )

    # Check FibsemMillingTaskConfig loads strategy correctly
    ddict = utils.load_yaml(protocol_path)
    task_ddict = ddict["tasks"]["Polishing"]["milling"]["mill_polishing"]
    milling_task_config = FibsemMillingTaskConfig.from_dict(task_ddict)

    assert milling_task_config.stages[0].strategy.config == protocol_strategy_config, (
        "The strategy and protocol configs do not match"
    )

    lamella_directory = tmp_path / "lamella"
    lamella_directory.mkdir()
    adaptive_polish_dir = lamella_directory / f"adaptive_polish_{TIMESTAMP}"

    # Set stage imaging settings
    milling_stages[0].imaging.resolution = (3072, 2048)
    milling_stages[0].imaging.hfw = 4e-5
    milling_stages[0].imaging.dwell_time = 2e-7
    milling_stages[0].imaging.frame_integration = 2
    milling_stages[0].imaging.path = lamella_directory

    # Check milling loop runs but exits at the end of loop calls_before_exception + 1
    with (
        patch.object(
            microscope,
            "acquire_image",
            new=create_dummy_acquire_image_function(
                fib_image_dir=fib_image_dir, sem_image_dir=sem_image_dir
            ),
        ),
        patch.object(
            microscope,
            "stop_milling",
            MagicMock(
                side_effect=raise_error_after_num_calls(
                    microscope.stop_milling,
                    calls_before_exception=calls_before_exception,
                )
            ),
        ) as mock_stop_milling,
    ):
        # run milling stages
        mill_stages(microscope=microscope, stages=milling_stages)

        assert mock_stop_milling.call_count == expected_loops, (
            "stop_milling was called an unexpected number of times"
        )

    # Check directories were created
    assert adaptive_polish_dir.is_dir(), "adaptive_polish directory wasn't created"
    for name in ("plots", "sem", "fib"):
        assert (adaptive_polish_dir / name).is_dir(), (
            f"{name} subdirectory wasn't created"
        )

    # Check all expected output files have been created:
    expected_fn_stems = [f"lamella_AP_img_{_:>03}_" for _ in range(expected_loops)]

    # Plots (png)
    centring_plot_path = adaptive_polish_dir / "centring.png"
    assert centring_plot_path.is_file(), f"{centring_plot_path.name} was not created"
    gis_thickness_plot_path = adaptive_polish_dir / "AP_summary_plots.png"
    assert gis_thickness_plot_path.is_file(), (
        f"{gis_thickness_plot_path.name} plot was not created"
    )
    plots_dir = adaptive_polish_dir / "plots"
    plot_paths = set(plots_dir.glob("*.png"))
    assert len(plot_paths), "No plots have been created"
    expected_plot_file_names = (f"{_}plot.png" for _ in expected_fn_stems)
    expected_plot_paths = set((plots_dir / _ for _ in expected_plot_file_names))
    assert plot_paths == expected_plot_paths, (
        "Expected plot file paths do not match found plot paths"
    )

    # Results (csv)
    expected_results_file_name = adaptive_polish_dir / "AP_metadata.json"
    assert expected_results_file_name.is_file(), (
        f"{expected_results_file_name.name} plot was not created"
    )

    # FIB and SEM images (tif)
    for image_type in ("sem", "fib"):
        image_path = adaptive_polish_dir / image_type
        image_paths = set(image_path.glob("*.tif"))
        assert len(image_paths), f"No {image_type} images have been created"
        expected_plot_file_names = (
            f"{_}{image_type.upper()}.tif" for _ in expected_fn_stems
        )
        expected_plot_paths = set((image_path / _ for _ in expected_plot_file_names))
        assert image_paths == expected_plot_paths, (
            f"Expected {image_type} image paths do not match found .tif paths"
        )
