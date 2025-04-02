from __future__ import annotations
import pkg_resources
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

_MODEL_PATHS = {
    "0": [
        _MODELS_PATH / "Gen0" / "2024-02-24_0013_gis_lamela_crack_pytorch_AUnet.ptchkp"
    ],
    "1": [
        _MODELS_PATH
        / "Gen1"
        / "gen01_quality_1536_v2"
        / "20250228_gen01_quality_1536_v2.pth"
    ],
}

assert _MODELS_PATH.is_dir(), "SEM models path does not exist"


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
    return Path(__file__).parent / "data" / "experiment.yaml"


@pytest.fixture(scope="session")
def protocol_template_path() -> Path:
    return Path(__file__).parent / "data" / "protocol.yaml"


@pytest.fixture(scope="session")
def microscope_config_demo2_path() -> Path:
    return Path(
        pkg_resources.resource_filename(
            "fibsem", "config/microscope-configuration-demo2.yaml"
        )
    )


@pytest.fixture(scope="session")
def fib_image_dir() -> Path:
    return Path(__file__).parent / "data" / "images" / "fib"


@pytest.fixture(scope="session")
def sem_image_dir() -> Path:
    return Path(__file__).parent / "data" / "images" / "sem"
