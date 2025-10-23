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
    dwell_multiplier_mean_stop: float = Field(
        default=0.03,
        ge=0,
        le=1,
        title="Mean dwell multiplier threshold",
        description="Milling will be stopped if the mean dwell time multiplier drops below this value",
    )
    dwell_multiplier_max_stop: float = Field(
        default=0.05,
        ge=0,
        le=1,
        title="Max. dwell multiplier threshold",
        description="Milling will be stopped if the maximum dwell time multiplier drops below this value",
    )
    bitmap_erosion_px: NonNegativeInt = 20
    bitmap_gaussian_sigma: NonNegativeFloat = 5
    mask_cracks: bool = True
    pattern_alignment: Literal["centre", "convolve"] = "centre"
