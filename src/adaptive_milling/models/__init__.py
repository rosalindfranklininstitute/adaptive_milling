from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import torch

from adaptive_milling.models.abstract import AbstractAdaptivePolishingModel
from adaptive_milling.models.gen0 import Gen0Model
from adaptive_milling.models.gen1 import (
    Gen1PerformanceModel,
    Gen1QualityModel,
    Gen1ImprovedPerformanceModel,
    Gen1ImprovedQualityModel,
    Gen1GreyscalePerformanceModel,
    Gen1GreyscaleQualityModel,
    Gen1ImprovedPreprocessingPerformanceModel,
    Gen1ImprovedPreprocessingQualityModel,
    Gen1ImprovedPreprocessingFPNModel,
    Gen1RGBImprovedPreprocessingFPNModel,
)

if TYPE_CHECKING:
    from typing import Union
    from os import PathLike

    DeviceLikeType = Union[str, torch.device, int]

_logger = logging.getLogger(__name__)


# Using str keys allows for semantic versioning
MODEL_GENERATIONS_DICT: dict[str, type[AbstractAdaptivePolishingModel]] = {
    "0": Gen0Model,
    "1.0p": Gen1PerformanceModel,  # v<=3
    "1.0q": Gen1QualityModel,  # v<=3
    "1.1p": Gen1ImprovedPerformanceModel,  # v4
    "1.1q": Gen1ImprovedQualityModel,  # v4
    "1.2p": Gen1GreyscalePerformanceModel,  # v5
    "1.2q": Gen1GreyscaleQualityModel,  # v5
    # Updated preprocessing:
    "1.3p": Gen1ImprovedPreprocessingPerformanceModel,
    "1.3q": Gen1ImprovedPreprocessingQualityModel,
    "1.3fpn": Gen1ImprovedPreprocessingFPNModel,
    "1.4fpn": Gen1RGBImprovedPreprocessingFPNModel,
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
