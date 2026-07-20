from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from adaptive_milling.models.abstract import AbstractAdaptivePolishingModel
from adaptive_milling.models.gen1 import Gen1Model

if TYPE_CHECKING:
    from os import PathLike
    from typing import Union

    DeviceLikeType = Union[str, torch.device, int]

_logger = logging.getLogger(__name__)


# Using str keys allows for semantic versioning
MODEL_GENERATIONS_DICT: dict[str, type[AbstractAdaptivePolishingModel]] = {
    "1.0": Gen1Model,
}


def get_latest_generation_key() -> str:
    # If no generation is specified, get the last one specified
    return str(tuple(MODEL_GENERATIONS_DICT.keys())[-1])


def load_model(
    model_path: str | PathLike[str],
    generation: int | str | None = None,
    device: DeviceLikeType | None = None,
) -> AbstractAdaptivePolishingModel:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if generation is None:
        # If no generation is specified, get the latest generation one
        generation = get_latest_generation_key()

    generation = str(generation)
    if generation not in MODEL_GENERATIONS_DICT:
        raise ValueError(
            f"Invalid model generation '{generation}' specified, available generations are: {', '.join(MODEL_GENERATIONS_DICT.keys())}"
        )

    _logger.info("Loading %s generation model", generation)
    model_class = MODEL_GENERATIONS_DICT[generation]
    if model_class is None:
        raise ValueError("%s")
    try:
        return model_class(model_path, device=device)
    except Exception:
        _logger.exception("Failed to load model with '%s'", model_class.__name__)
        raise
