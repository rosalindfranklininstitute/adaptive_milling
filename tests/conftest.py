from __future__ import annotations

import typing
from importlib import resources
from pathlib import Path

import pytest
import requests

_MODELS_DIR: Path = Path(__file__).parent / "data" / "models"
_MODELS_DIR.mkdir(exist_ok=True, parents=False)

_MODELS: dict[str, list[tuple[str, Path | None]]] = {
    "1.0": [
        ("https://zenodo.org/records/21804785/files/AM_SEM_All_V01.pth", None),
    ],
}


def _get_model_path(generation: str, url: str) -> Path:
    return _MODELS_DIR / generation / url.rsplit("/")[-1]


def _download_from_url(url: str, local_path: Path) -> None:
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with local_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):  # 8 KiB
                f.write(chunk)


def _ensure_model_downloaded(generation: str, url: str) -> Path:
    local_path = _get_model_path(generation, url)
    if not local_path.is_file():
        local_path.parent.mkdir(exist_ok=True, parents=False)
        _download_from_url(url, local_path)
    return local_path


def _ensure_models_downloaded() -> None:
    for generation, model_tuples in _MODELS.items():
        for i, model_tuple in enumerate(tuple(model_tuples)):
            if model_tuple[1] is None:
                model_tuples[i] = (
                    model_tuple[0],
                    _ensure_model_downloaded(generation, model_tuple[0]),
                )


_ensure_models_downloaded()


@pytest.fixture(scope="session")
def latest_sem_segmentation_model() -> tuple[str, Path]:
    model_name, model_tuples = tuple(_MODELS.items())[-1]
    model_path = model_tuples[-1][1]
    if model_path is None:
        raise ValueError("Failed to get model path")
    return model_name, model_path


@pytest.fixture(
    params=[(k, p[1]) for k, v in _MODELS.items() for p in v],
    ids=[f"Model {k} {i}" for k, v in _MODELS.items() for i in range(1, len(v) + 1)],
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
