from __future__ import annotations
from typing import Literal, ClassVar
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
    bitmap_gaussian_sigma: NonNegativeFloat = 20
    boundary_smoothing_sigma: NonNegativeFloat = 5.0
    apply_boundary_smoothing: bool = False
    incident_angle_sputter_ratio: PositiveFloat = Field(
        default=4.3,
        ge=1,
        title="Incident angle sputter ratio",
        description="The ratio between the maximum sputter rate and the sputter rate at an incident angle of 0 degrees. This is disabled if a ratio of 1 is given.",
    )
    incident_angle_scale_with_gis_min: bool = False
    mask_cracks: bool = True
    pattern_alignment: Literal["centre", "convolve"] = "centre"

    _advanced_attributes: ClassVar[tuple[str, ...]] = (
        *AdaptivePolishMillingConfig._advanced_attributes,
        "save_predictions",
        "pattern_alignment",
    )
