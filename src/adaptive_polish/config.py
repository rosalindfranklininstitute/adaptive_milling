from __future__ import annotations
import typing
from os import PathLike
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


from fibsem.milling.base import MillingStrategyConfig


@dataclass(config=ConfigDict(validate_assignment=True))
class AdaptivePolishMillingConfig(MillingStrategyConfig):
    model_path: typing.Union[str, PathLike] = "none"
    align_sem: bool = True
    milling_interval_s: int = 10
    gis_stop_um: float = 0.2
    max_crack_area_um2: float = 2
    max_milling_cycles: int = 30
    window_size_px: int = 10
    minimum_lamella_area_um2: float = 30.0  # 30μm²
    maximum_drift_um: float = 0.1
    model_generation: typing.Optional[str] = None  # Uses latest generation
    maximum_side_difference_um: float = 0.05
    fib_resolution: list = [3072, 2048]
    fib_dwell_time_us: float = 2
    fib_hfw_um: float = 50
    fib_autocontrast: bool = False
    fib_autogamma: bool = False
    sem_resolution: list = [3072, 2048]
    sem_dwell_time_us: float = 2
    sem_hfw_um: float = 50
    sem_autocontrast: bool = False
    sem_autogamma: bool = False

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
            "maximum_side_difference_um": self.maximum_side_difference_um,
            "fib_resolution": self.fib_resolution,
            "fib_dwell_time_us": self.fib_dwell_time_us,
            "fib_autocontrast": self.fib_autocontrast,
            "fib_autogamma": self.fib_autogamma,
            "sem_resolution": self.sem_resolution,
            "sem_dwell_time_us": self.sem_dwell_time_us,
            "sem_hfw_um": self.sem_hfw_um,
            "sem_autocontrast": self.sem_autocontrast,
            "sem_autogamma": self.sem_autogamma,
        }
