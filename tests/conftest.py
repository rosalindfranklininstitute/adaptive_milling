from __future__ import annotations
from importlib import resources
from pathlib import Path
import pytest

import typing

_MODELS_PATH = (
    Path.home()
    / "OneDrive - The Rosalind Franklin Institute"
    / "Documents"
    / "test data"
    / "adaptive milling"
    / "sem_models"
)

# Note: Gen 1 models below v4 apply padding before normalisation, so will not be accurate
_MODEL_PATHS = {
    "0": [
        _MODELS_PATH / "Gen0" / "2024-02-24_0013_gis_lamela_crack_pytorch_AUnet.ptchkp"
    ],
    "1.0p": [
        _MODELS_PATH
        / "Gen1"
        / "gen01_performance_768_v3"
        / "gen01_performance_768_v3.pth"
    ],
    "1.0q": [
        _MODELS_PATH / "Gen1" / "gen01_quality_1536_v3" / "gen01_quality_1536_v3.pth"
    ],
    "1.1q": [
        _MODELS_PATH / "Gen1" / "gen01_quality_1536_v4" / "gen01_quality_1536_v4.pth"
    ],
    "1.2q": [
        _MODELS_PATH
        / "Gen1"
        / "gen01_quality_1536_v5_grayscale"
        / "cryo_sem_epoch_56.pth"
    ],
    "1.3fpn": [
        _MODELS_PATH
        / "Gen1"
        / "gen01_quality_1536_v7_FPN"
        / "gen01_quality_1536_v7_FPN.pth"
    ],
}

assert _MODELS_PATH.is_dir(), "SEM models path does not exist"

@pytest.fixture(scope="session")
def latest_sem_segmentation_model() -> tuple[str, Path]:
    model_name, model_paths = tuple(_MODEL_PATHS.items())[-1]
    return model_name, model_paths[-1]


@pytest.fixture(
    params=list(_MODEL_PATHS.keys()),
    ids=[f"Model {_}" for _ in _MODEL_PATHS.keys()],
    scope="session",
)
def sem_segmentation_model(
    request: pytest.FixtureRequest,
) -> typing.Generator[tuple[str, Path]]:
    for path in _MODEL_PATHS[request.param]:
        yield request.param, path


@pytest.fixture(scope="session")
def experiment_template_path() -> Path:
    return (Path(__file__).parent / "data" / "experiment.yaml").resolve()


@pytest.fixture(scope="session")
def protocol_template_path() -> Path:
    return (Path(__file__).parent / "data" / "protocol.yaml").resolve()


@pytest.fixture(scope="session")
def microscope_config_path() -> Path:
    return (
        Path(resources.files("fibsem")) / "config" / "microscope-configuration.yaml"
    ).resolve()


@pytest.fixture(scope="session")
def microscope_config_demo2_path() -> Path:
    return (
        Path(resources.files("fibsem"))
        / "config"
        / "microscope-configuration-demo2.yaml"
    ).resolve()


@pytest.fixture(scope="session")
def fib_image_dir() -> Path:
    return (Path(__file__).parent / "data" / "images" / "fib").resolve()


@pytest.fixture(scope="session")
def sem_image_dir() -> Path:
    return (Path(__file__).parent / "data" / "images" / "sem").resolve()
