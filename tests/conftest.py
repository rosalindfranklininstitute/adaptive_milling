from __future__ import annotations

import typing
from importlib import resources
from pathlib import Path

import pytest

_MODELS_DIR: Path = (
    Path.home()
    / "OneDrive - The Rosalind Franklin Institute"
    / "Documents"
    / "test data"
    / "adaptive milling"
    / "sem_models"
)

# Note: Gen 1 models below v4 apply padding before normalisation, so will not be accurate
_MODEL_PATHS: dict[str, list[Path]] = {
    "1.0": [
        _MODELS_DIR / "Gen1" / "gen01_V8_FPN_RGB" / "cryo_sem_epoch_49.pth",
        _MODELS_DIR / "Gen1" / "gen01_quality_1536_v9_FPN" / "cryo_sem_epoch_73.pth",
    ],
}


def models_directory_exists() -> bool:
    return _MODELS_DIR.is_dir()


@pytest.fixture(scope="session")
def skip_if_no_models():
    if not models_directory_exists():
        pytest.skip("Unable to find models")


@pytest.fixture(scope="session")
def latest_sem_segmentation_model() -> tuple[str, Path]:
    model_name, model_paths = tuple(_MODEL_PATHS.items())[-1]
    return model_name, model_paths[-1]


@pytest.fixture(
    params=[(k, p) for k, v in _MODEL_PATHS.items() for p in v],
    ids=[
        f"Model {k} {i}" for k, v in _MODEL_PATHS.items() for i in range(1, len(v) + 1)
    ],
    scope="session",
)
def sem_segmentation_model(
    request: pytest.FixtureRequest,
) -> typing.Generator[tuple[str, Path]]:
    yield request.param


@pytest.fixture(scope="session")
def experiment_template_path() -> Path:
    return (Path(__file__).parent / "data" / "experiment.yaml").resolve()


@pytest.fixture(scope="session")
def protocol_template_path() -> Path:
    return (Path(__file__).parent / "data" / "protocol.yaml").resolve()


@pytest.fixture(scope="session")
def microscope_config_path() -> Path:
    return (
        Path(str(resources.files("fibsem")))
        / "config"
        / "microscope-configuration.yaml"
    ).resolve()


@pytest.fixture(scope="session")
def fib_image_dir() -> Path:
    return (Path(__file__).parent / "data" / "images" / "fib").resolve()


@pytest.fixture(scope="session")
def sem_image_dir() -> Path:
    return (Path(__file__).parent / "data" / "images" / "sem").resolve()
