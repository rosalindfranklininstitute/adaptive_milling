from __future__ import annotations
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


from adaptive_polish.config import AdaptivePolishMillingConfig

@dataclass(config=ConfigDict(validate_assignment=True))
class BitmapAdaptivePolishMillingConfig(AdaptivePolishMillingConfig):
    gis_max_um: float = 2
    gis_min_um: float | None = None
