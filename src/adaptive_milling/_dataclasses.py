from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from adaptive_milling.enums import StopReasons

if TYPE_CHECKING:
    from typing import Any

    from fibsem.structures import FibsemImage
    from numpy.typing import NDArray


@dataclass
class ProcessTimestamps:
    start: float | None = None
    end: float | None = None
    exception: str | None = None

    @property
    def duration(self) -> float | None:
        if self.start is None or self.end is None:
            return None
        return self.end - self.start

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

    def to_dict(self) -> dict[str, Any]:
        ddict = asdict(self)
        ddict["duration"] = self.duration
        return ddict


@dataclass
class CycleTimestamps:
    cycle: ProcessTimestamps = field(default_factory=ProcessTimestamps)
    image: ProcessTimestamps = field(default_factory=ProcessTimestamps)
    process: ProcessTimestamps = field(default_factory=ProcessTimestamps)
    update_stage: ProcessTimestamps = field(default_factory=ProcessTimestamps)
    predict: ProcessTimestamps = field(default_factory=ProcessTimestamps)
    check_lamella: ProcessTimestamps = field(default_factory=ProcessTimestamps)
    check_stage: ProcessTimestamps = field(default_factory=ProcessTimestamps)
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

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CycleInformation:
        timestamps = CycleTimestamps(**d.pop("timestamps"))
        lamella_stats = LamellaStatistics(**d.pop("lamella_statistics"))
        lamella_stats.calculate_statistics()
        return cls(
            timestamps=timestamps,
            lamella_statistics=lamella_stats,
            **d,
        )

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

    def set_end_reason(self, reason: StopReasons | str | None) -> None:
        if isinstance(reason, StopReasons):
            reason = reason.value
        self.strategy_end_reason = reason

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StrategyRunInformation:
        timestamps = StrategyTimestamps(**d.pop("timestamps"))
        cycle_information = [
            CycleInformation.from_dict(_) for _ in d.pop("cycle_information")
        ]
        return cls(timestamps=timestamps, cycle_information=cycle_information, **d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert strategy run information into a pandas DataFrame."""

        ddict = asdict(self)
        cycle_info = ddict.pop("cycle_information")
        df_cycles = pd.json_normalize(cycle_info)

        df = df_cycles.join(pd.json_normalize(ddict), how="outer").ffill()

        # compute duration columns for all .start/.end pairs
        for c in df.columns[df.columns.str.contains(r"timestamps\.[\w_]+\.start$")]:
            # find the matching .end column
            col_base = c.rsplit(".", maxsplit=1)[0]
            end_col = col_base + ".end"
            if end_col not in df.columns:
                continue
            duration_col = col_base + ".duration"
            # compute the duration in seconds
            df[duration_col] = df[end_col] - df[c]

        return df

    def to_final_dataframe(self) -> pd.DataFrame:
        """Return only the final row of the summary dataframe."""

        df = self.to_dataframe()
        if df.empty:
            return df
        return df.tail(1).reset_index(drop=True)


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

    # These can be calculated by calculate_statistics
    gis_thickness_min_image_px: float | None = None
    gis_thickness_median_image_px: float | None = None
    gis_thickness_mean_image_px: float | None = None

    # Calculated physical values
    gis_thickness_um: list[float] | None = None
    gis_thickness_filtered_um: list[float] | None = None
    gis_thickness_min_um: float | None = None
    gis_thickness_mean_um: float | None = None
    gis_thickness_median_um: float | None = None
    crack_area_um2: float | None = None
    lamella_thickness_um: list[float] | None = None
    lamella_area_um2: float | None = None

    # Bitmap specific values
    pattern_xlims_px: tuple[int, int] | None = None
    pattern_dwell_multiplier: list[float] | None = None
    pattern_blanking: list[bool] | None = None

    def calculate_statistics(self) -> None:
        if self.gis_thickness_image_px is not None:
            self.gis_thickness_um = (
                np.asarray(self.gis_thickness_image_px)
                * (self.image_pixel_size_m[1] * 1e6)
            ).tolist()

        if (
            self.gis_thickness_filtered_image_px is not None
            and self.xlims_image_px is not None
        ):
            gis_thickness_slice_image_px = self.gis_thickness_filtered_image_px[
                self.xlims_image_px[0] : self.xlims_image_px[1] + 1
            ]
            self.gis_thickness_min_image_px = float(
                np.nanmin(gis_thickness_slice_image_px)
            )
            self.gis_thickness_mean_image_px = float(
                np.nanmean(gis_thickness_slice_image_px)
            )
            self.gis_thickness_median_image_px = float(
                np.nanmedian(gis_thickness_slice_image_px)
            )

            # Physical units
            self.gis_thickness_filtered_um = (
                np.asarray(self.gis_thickness_filtered_image_px)
                * (self.image_pixel_size_m[1] * 1e6)
            ).tolist()
            self.gis_thickness_min_um = self.gis_thickness_min_image_px * (
                self.image_pixel_size_m[1] * 1e6
            )
            self.gis_thickness_mean_um = self.gis_thickness_mean_image_px * (
                self.image_pixel_size_m[1] * 1e6
            )
            self.gis_thickness_median_um = self.gis_thickness_median_image_px * (
                self.image_pixel_size_m[1] * 1e6
            )
        if self.crack_thickness_prediction_px is not None:
            self.crack_area_um2 = float(
                np.sum(self.crack_thickness_prediction_px)
                * (
                    self.prediction_pixel_size_m[0]
                    * self.prediction_pixel_size_m[1]
                    * 1e12
                )
            )

        self.lamella_thickness_um = (
            np.asarray(self.lamella_thickness_prediction_px, dtype=float)
            * (self.prediction_pixel_size_m[0] * 1e6)
        ).tolist()
        self.lamella_area_um2 = float(
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
