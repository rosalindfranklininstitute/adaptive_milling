from __future__ import annotations
from typing import Literal
from pydantic import (
    ConfigDict,
    NonNegativeInt,
    PositiveFloat,
    NonNegativeFloat,
    Field,
)
from pydantic.dataclasses import dataclass

from adaptive_polish.config import AdaptivePolishMillingConfig


@dataclass(config=ConfigDict(validate_assignment=True))
class BitmapAdaptivePolishMillingConfig(AdaptivePolishMillingConfig):
    gis_max_um: PositiveFloat = 2
    gis_min_um: NonNegativeFloat = 0.15
    stop_mill_fraction_mean: float = Field(default=0.03, ge=0, le=1)
    stop_mill_fraction_max: float = Field(default=0.05, ge=0, le=1)
    bitmap_erosion_px: NonNegativeInt = 20
    bitmap_gaussian_sigma: NonNegativeFloat = 5
    mask_cracks: bool = True
    pattern_alignment: Literal["centre", "convolve"] = "centre"
