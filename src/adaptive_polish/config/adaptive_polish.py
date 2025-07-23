from __future__ import annotations
import typing
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from fibsem.milling.base import MillingStrategyConfig

from adaptive_polish.gis_measurement import DEFAULT_SEM_MODEL_GENERATION


@dataclass(config=ConfigDict(validate_assignment=True))
class AdaptivePolishMillingConfig(MillingStrategyConfig):
    model_path: str = ""
    align_sem: bool = True
    gis_stop_um: float = 0.2
    max_crack_area_um2: float = 2
    max_milling_cycles: int = 30
    window_size_px: int = 10
    minimum_lamella_area_um2: float = 30.0  # 30μm²
    maximum_drift_um: float = 0.1
    lamella_pad_x: float = 0.0  # fraction of lamella width to pad by on each side
    model_generation: str = DEFAULT_SEM_MODEL_GENERATION

    def get_model_generation(self) -> typing.Optional[str]:
        model_generation = self.model_generation.strip()
        if not model_generation:
            # Interpret empty strings as None
            return None
        return model_generation
