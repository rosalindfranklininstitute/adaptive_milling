from __future__ import annotations
from dataclasses import dataclass
import typing

from fibsem.milling.base import MillingStrategyConfig

if typing.TYPE_CHECKING:
    from os import PathLike


@dataclass
class AdaptivePolishMillingConfig(MillingStrategyConfig):
    model_path: typing.Union[str, PathLike]
    align_sem: bool = True
    milling_interval_s: int = 10
    gis_stop_um: float = 0.2
    max_crack_area_um2: float = 2
    max_milling_cycles: int = 30
    window_size_px: int = 10
    minimum_lamella_area_um2: float = 30.0  # 30μm²
    maximum_drift_um: float = 0.05
    model_generation: typing.Optional[str] = None  # Uses latest generation

    _advanced_attributes = []

    @staticmethod
    def from_dict(d: dict[str, typing.Any]) -> typing.Self:
        return AdaptivePolishMillingConfig(**d)

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "model_path": str(self.model_path),
            "align_sem": self.align_sem,
            "milling_interval_s": self.milling_interval_s,
            "gis_stop_um": self.gis_stop_um,
            "max_milling_cycles": self.max_milling_cycles,
            "window_size_px": self.window_size_px,
            "max_crack_area_um2": self.max_crack_area_um2,
            "minimum_lamella_area_um2": self.minimum_lamella_area_um2,
            "maximum_drift_um": self.maximum_drift_um,
            "model_generation": self.model_generation,
        }
