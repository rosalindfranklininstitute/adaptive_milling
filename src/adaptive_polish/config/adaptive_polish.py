from __future__ import annotations
import os
import typing
from pydantic import ConfigDict, PositiveInt, NonNegativeFloat, Field, field_validator
from pydantic.dataclasses import dataclass

from fibsem.milling.base import MillingStrategyConfig

from adaptive_polish.processing.sem_segmentation import get_latest_generation_key

DEFAULT_MODEL_GENERATION = get_latest_generation_key()


@dataclass(config=ConfigDict(validate_assignment=True))
class AdaptivePolishMillingConfig(MillingStrategyConfig):
    model_path: str = ""
    align_sem: bool = True
    gis_filter_sigma: NonNegativeFloat = 10 / 6
    gis_stop_um: NonNegativeFloat = 0.2
    max_crack_area_um2: NonNegativeFloat = 2
    max_milling_cycles: PositiveInt = 30
    minimum_lamella_area_um2: NonNegativeFloat = 30.0  # 30μm²
    maximum_drift_um: NonNegativeFloat = 70
    lamella_pad_x: float = Field(  # fraction of lamella width to pad by on each side
        default=0.0, ge=0, le=1
    )
    model_generation: str = DEFAULT_MODEL_GENERATION
    save_predictions: bool = True

    @field_validator("model_path", mode="after")
    @classmethod
    def ensure_model_path(cls, v: str) -> str:
        if not os.path.isfile(v):
            raise FileNotFoundError(f"'{v}' is not a valid file path")
        return v

    def get_model_generation(self) -> typing.Optional[str]:
        model_generation = self.model_generation.strip()
        if not model_generation:
            # Interpret empty strings as None
            return None
        return model_generation
