from __future__ import annotations
from pathlib import Path
import pytest

import typing

# from setup import setup_experiment, setup_microscope


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
