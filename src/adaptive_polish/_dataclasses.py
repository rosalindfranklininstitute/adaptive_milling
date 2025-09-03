from __future__ import annotations
import time
from functools import cached_property
from dataclasses import dataclass, asdict, field
from types import TracebackType
from typing import TYPE_CHECKING

import numpy as np

from adaptive_polish.enums import StopReasons

if TYPE_CHECKING:
    from typing import Any
    import numpy as np
    from numpy.typing import NDArray
    from fibsem.structures import FibsemImage


@dataclass
class ProcessTimestamps:
    start: float | None = None
    end: float | None = None
    exception: str | None = None

    def __enter__(self) -> None:
        self.start = time.time()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.end = time.time()
        if exc_val is not None:
            self.exception = f"{exc_val.__class__.__name__}({exc_val})"


@dataclass
class CycleTimestamps:
    cycle: ProcessTimestamps = field(default_factory=ProcessTimestamps)
    image: ProcessTimestamps = field(default_factory=ProcessTimestamps)
    process: ProcessTimestamps = field(default_factory=ProcessTimestamps)
    update_stage: ProcessTimestamps = field(default_factory=ProcessTimestamps)
    predict: ProcessTimestamps = field(default_factory=ProcessTimestamps)
    check_lamella: ProcessTimestamps = field(default_factory=ProcessTimestamps)
    mill: ProcessTimestamps = field(default_factory=ProcessTimestamps)


@dataclass
class StrategyTimestamps:
    strategy: ProcessTimestamps = field(default_factory=ProcessTimestamps)
    setup: ProcessTimestamps = field(default_factory=ProcessTimestamps)
    centre_lamella: ProcessTimestamps = field(default_factory=ProcessTimestamps)


@dataclass
class CycleInformation:
    milling_cycle: int
    identifier: str
    timestamps: CycleTimestamps = field(default_factory=CycleTimestamps)
    lamella_statistics: LamellaStatistics | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyRunInformation:
    strategy_name: str
    stage_name: str
    lamella_name: str | None = None
    timestamps: StrategyTimestamps = field(default_factory=StrategyTimestamps)
    strategy_end_reason: str | None = None
    cycle_information: list[CycleInformation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def set_end_reason(self, reason: StopReasons | str | None) -> None:
        if isinstance(reason, StopReasons):
            reason = reason.value
        self.strategy_end_reason = reason


@dataclass
class LamellaStatistics:
    image_pixel_size_m: tuple[float, float]
    prediction_pixel_size_m: tuple[float, float]

    crack_count: int
    lamella_thickness_prediction_px: list[float | int]
    crack_thickness_prediction_px: list[float | int]

    estimated_milling_time_s: float | None = None

    lamella_bounding_box_prediction_px: tuple[float, float, float, float] | None = (
        None  # ymin, xmin, ymax, xmax
    )
    xlims_prediction_px: tuple[int, int] | None = None  # min, max (prediction xlims)
    lamella_bounding_box_image_px: tuple[float, float, float, float] | None = (
        None  # ymin, xmin, ymax, xmax
    )
    xlims_image_px: tuple[int, int] | None = None  # min, max (image xlims)

    gis_thickness_image_px: list[float | int] | None = None
    gis_thickness_filtered_image_px: list[float | int] | None = None

    # These can be calculated by calculate_gis_statistics
    gis_min_image_px: float | None = None
    gis_median_image_px: float | None = None
    gis_mean_image_px: float | None = None

    def calculate_gis_statistics(self) -> None:
        if (
            self.gis_thickness_filtered_image_px is not None
            and self.xlims_image_px is not None
        ):
            gis_thickness_slice_image_px = self.gis_thickness_filtered_image_px[
                self.xlims_image_px[0] : self.xlims_image_px[1] + 1
            ]
            self.gis_min_image_px = float(np.nanmin(gis_thickness_slice_image_px))
            self.gis_mean_image_px = float(np.nanmean(gis_thickness_slice_image_px))
            self.gis_median_image_px = float(np.nanmedian(gis_thickness_slice_image_px))

    @cached_property
    def gis_thickness_um(self) -> NDArray[np.float_] | None:
        if self.gis_thickness_image_px is None:
            return None
        return np.asarray(self.gis_thickness_image_px) / (
            self.image_pixel_size_m[1] * 1e6
        )

    @cached_property
    def gis_filtered_thickness_um(self) -> NDArray[np.float_] | None:
        if self.gis_thickness_filtered_image_px is None:
            return None
        return np.asarray(self.gis_thickness_filtered_image_px, dtype=float) * (
            self.image_pixel_size_m[1] * 1e6
        )

    @property
    def gis_min_um(self) -> float | None:
        if self.gis_min_image_px is None:
            self.calculate_gis_statistics()
            if self.gis_min_image_px is None:
                return None
        return float(self.gis_min_image_px * (self.image_pixel_size_m[1] * 1e6))

    @property
    def gis_mean_um(self) -> float | None:
        if self.gis_mean_image_px is None:
            self.calculate_gis_statistics()
            if self.gis_mean_image_px is None:
                return None
        return float(self.gis_mean_image_px * (self.image_pixel_size_m[1] * 1e6))

    @property
    def gis_median_um(self) -> float | None:
        if self.gis_median_image_px is None:
            self.calculate_gis_statistics()
            if self.gis_median_image_px is None:
                return None
        return float(self.gis_median_image_px * (self.image_pixel_size_m[1] * 1e6))

    @cached_property
    def crack_area_um2(self) -> float:
        return float(
            np.sum(self.crack_thickness_prediction_px)
            * (self.prediction_pixel_size_m[0] * self.prediction_pixel_size_m[1] * 1e12)
        )

    @cached_property
    def lamella_thickness_um(self) -> float:
        return float(
            np.sum(self.lamella_thickness_prediction_px)
            * (self.prediction_pixel_size_m[0] * self.prediction_pixel_size_m[1] * 1e12)
        )

    @cached_property
    def lamella_area_um2(self) -> float:
        return float(
            np.sum(self.lamella_thickness_prediction_px)
            * (self.prediction_pixel_size_m[0] * self.prediction_pixel_size_m[1] * 1e12)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LamellaInformation:
    identifier: str
    sem_image: FibsemImage
    fib_image: FibsemImage
    prediction: NDArray[np.uint8]
    clean_prediction: NDArray[np.uint8]
    statistics: LamellaStatistics

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
