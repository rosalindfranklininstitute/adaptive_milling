from __future__ import annotations
import typing
from os import PathLike
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from fibsem.milling.base import MillingStrategyConfig

from adaptive_polish.gis_measurement import DEFAULT_SEM_MODEL_GENERATION


@dataclass(config=ConfigDict(validate_assignment=True))
class AdaptivePolishMillingConfig(MillingStrategyConfig):
    model_path: typing.Union[str, PathLike] = "Undefined"
    align_sem: bool = True
    gis_stop_um: float = 0.2
    max_crack_area_um2: float = 2
    max_milling_cycles: int = 30
    window_size_px: int = 10
    minimum_lamella_area_um2: float = 30.0  # 30μm²
    maximum_drift_um: float = 0.1
    model_generation: str = DEFAULT_SEM_MODEL_GENERATION
    maximum_side_difference_um: float = 0.05

    _advanced_attributes = []

    @staticmethod
    def from_dict(d: dict[str, typing.Any]) -> "AdaptivePolishMillingConfig":
        return AdaptivePolishMillingConfig(**d)

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "model_path": str(self.model_path),
            "align_sem": self.align_sem,
            "gis_stop_um": self.gis_stop_um,
            "max_milling_cycles": self.max_milling_cycles,
            "window_size_px": self.window_size_px,
            "max_crack_area_um2": self.max_crack_area_um2,
            "minimum_lamella_area_um2": self.minimum_lamella_area_um2,
            "maximum_drift_um": self.maximum_drift_um,
            "model_generation": self.model_generation,
            "maximum_side_difference_um": self.maximum_side_difference_um,
        }

    def get_model_generation(self) -> typing.Optional[str]:
        model_generation = self.model_generation.strip()
        if not model_generation:
            # Interpret empty strings as None
            return None
        return model_generation
