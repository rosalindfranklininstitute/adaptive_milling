from __future__ import annotations
import time
from dataclasses import asdict, dataclass, field, fields
from functools import cached_property
from types import TracebackType
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from adaptive_polish.enums import StopReasons

if TYPE_CHECKING:
    from typing import Any

    import numpy as np
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

    def to_cycle_information_dataframe(self):
        return pd.json_normalize([asdict(_) for _ in self.cycle_information])

    def to_dataframe2(self) -> pd.DataFrame:
        return pd.json_normalize(asdict(self))

    def to_full_dataframe(self) -> pd.DataFrame:
        df = self.to_dataframe2()

        # drop cycle_information column
        df = df.drop(columns=["cycle_information"])

        # # join df to each row of df_cycles on index
        df_cycles = self.to_cycle_information_dataframe()
        df_full = df_cycles.join(df, how="outer").ffill()

        # compute duration columns for all .start/.end pairs
        columns = df_full.columns.tolist()
        for c in columns:
            if ".start" not in c:
                continue
            # find the matching .end column
            end_col = c.replace(".start", ".end")
            if end_col not in columns:
                continue
            # compute the duration in seconds
            df_full[f"{c.replace('.start', '')}_duration"] = (df_full[end_col] - df_full[c])

        return df_full

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def set_end_reason(self, reason: StopReasons | str | None) -> None:
        if isinstance(reason, StopReasons):
            reason = reason.value
        self.strategy_end_reason = reason

    def to_dataframe(self) -> pd.DataFrame:
        """Convert strategy run information into a pandas DataFrame."""

        base_data: dict[str, Any] = {
            "strategy_name": self.strategy_name,
            "stage_name": self.stage_name,
            "lamella_name": self.lamella_name,
            "strategy_end_reason": self.strategy_end_reason,
        }

        # Add strategy timestamps
        for field_info in fields(self.timestamps):
            process_timestamps: ProcessTimestamps = getattr(self.timestamps, field_info.name)
            for key, val in process_timestamps.to_dict().items():
                base_data[f"strategy_{field_info.name}_{key}"] = val
        # return df
        records = []
        for cycle_info in self.cycle_information:
            record = base_data.copy()
            record["milling_cycle"] = cycle_info.milling_cycle
            record["identifier"] = cycle_info.identifier
            # Add cycle timestamps
            for field_info in fields(cycle_info.timestamps):
                process_timestamps: ProcessTimestamps = getattr(cycle_info.timestamps, field_info.name)
                for key, val in process_timestamps.to_dict().items():
                    record[f"cycle_{field_info.name}_{key}"] = val
            # Add lamella statistics if available
            if cycle_info.lamella_statistics:
                for key, val in cycle_info.lamella_statistics.to_summary_dict().items():
                    record[f"lamella_stat_{key}"] = val
            records.append(record)

        return pd.DataFrame(records)

    def to_summary_dataframe(self) -> pd.DataFrame:
        """Return a subset of the dataframe containing duration and summary metrics."""

        df = self.to_dataframe()

        core_columns = [
            "strategy_name",
            "stage_name",
            "lamella_name",
            "strategy_end_reason",
        ]

        strategy_duration_columns = [
            f"strategy_{field_info.name}_duration" for field_info in fields(self.timestamps)
        ]
        cycle_duration_columns = [
            f"cycle_{field_info.name}_duration" for field_info in fields(CycleTimestamps)
        ]
        summary_columns = [
            "lamella_stat_lamella_area_um2",
            "lamella_stat_gis_thickness_min_um",
            "lamella_stat_gis_thickness_mean_um",
            "lamella_stat_crack_count",
        ]

        desired_columns = (
            core_columns
            + strategy_duration_columns
            + cycle_duration_columns
            + summary_columns
        )

        rename_map: dict[str, str] = {
            "strategy_name": "Strategy Name",
            "stage_name": "Stage Name",
            "lamella_name": "Lamella Name",
            "strategy_end_reason": "Strategy End Reason",
            "lamella_stat_lamella_area_um2": "Lamella Area (um2)",
            "lamella_stat_crack_count": "Crack Count",
            "lamella_stat_gis_thickness_min_um": "GIS Thickness Min (um)",
            "lamella_stat_gis_thickness_mean_um": "GIS Thickness Mean (um)",
        }

        for field_info in fields(self.timestamps):
            col_name = f"strategy_{field_info.name}_duration"
            label = field_info.name.replace("_", " ").title()
            rename_map[col_name] = f"Strategy {label} Duration"

        for field_info in fields(CycleTimestamps):
            col_name = f"cycle_{field_info.name}_duration"
            label = field_info.name.replace("_", " ").title()
            rename_map[col_name] = f"Cycle {label} Duration"

        available_columns = [col for col in desired_columns if col in df.columns]

        if not available_columns:
            renamed_columns = [rename_map.get(col, col) for col in desired_columns]
            return pd.DataFrame(columns=renamed_columns)

        df = df.loc[:, available_columns]

        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        return df

    def to_final_dataframe(self) -> pd.DataFrame:
        """Return only the final row of the summary dataframe."""

        summary_df = self.to_summary_dataframe()
        if summary_df.empty:
            return summary_df
        return summary_df.tail(1).reset_index(drop=True)


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
    gis_thickness_min_image_px: float | None = None
    gis_thickness_median_image_px: float | None = None
    gis_thickness_mean_image_px: float | None = None

    def to_summary_dict(self) -> dict[str, Any]:
        data = {}

        lamella_thickness_um = self.lamella_thickness_um
        if lamella_thickness_um is None or lamella_thickness_um.size == 0:
            data["lamella_thickness_um"] = None
        else:
            data["lamella_thickness_um"] = float(np.nanmean(lamella_thickness_um))

        gis_thickness_um = self.gis_thickness_um
        if gis_thickness_um is None or gis_thickness_um.size == 0:
            data["gis_thickness_um"] = None
        else:
            data["gis_thickness_um"] = float(np.nanmean(gis_thickness_um))

        gis_thickness_filtered_um = self.gis_thickness_filtered_um
        if gis_thickness_filtered_um is None or gis_thickness_filtered_um.size == 0:
            data["gis_thickness_filtered_um"] = None
        else:
            data["gis_thickness_filtered_um"] = float(
                np.nanmean(gis_thickness_filtered_um)
            )

        data["crack_area_um2"] = self.crack_area_um2
        data["lamella_area_um2"] = self.lamella_area_um2
        data["gis_thickness_min_um"] = self.gis_thickness_min_um
        data["gis_thickness_mean_um"] = self.gis_thickness_mean_um
        data["gis_thickness_median_um"] = self.gis_thickness_median_um
        data["crack_count"] = self.crack_count

        return data

    def calculate_gis_statistics(self) -> None:
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

    @cached_property
    def gis_thickness_um(self) -> NDArray[np.float_] | None:
        if self.gis_thickness_image_px is None:
            return None
        return np.asarray(self.gis_thickness_image_px) / (
            self.image_pixel_size_m[1] * 1e6
        )

    @cached_property
    def gis_thickness_filtered_um(self) -> NDArray[np.float_] | None:
        if self.gis_thickness_filtered_image_px is None:
            return None
        return np.asarray(self.gis_thickness_filtered_image_px, dtype=float) * (
            self.image_pixel_size_m[1] * 1e6
        )

    @property
    def gis_thickness_min_um(self) -> float | None:
        if self.gis_thickness_min_image_px is None:
            self.calculate_gis_statistics()
            if self.gis_thickness_min_image_px is None:
                return None
        return float(
            self.gis_thickness_min_image_px * (self.image_pixel_size_m[1] * 1e6)
        )

    @property
    def gis_thickness_mean_um(self) -> float | None:
        if self.gis_thickness_mean_image_px is None:
            self.calculate_gis_statistics()
            if self.gis_thickness_mean_image_px is None:
                return None
        return float(
            self.gis_thickness_mean_image_px * (self.image_pixel_size_m[1] * 1e6)
        )

    @property
    def gis_thickness_median_um(self) -> float | None:
        if self.gis_thickness_median_image_px is None:
            self.calculate_gis_statistics()
            if self.gis_thickness_median_image_px is None:
                return None
        return float(
            self.gis_thickness_median_image_px * (self.image_pixel_size_m[1] * 1e6)
        )

    @cached_property
    def crack_area_um2(self) -> float:
        return float(
            np.sum(self.crack_thickness_prediction_px)
            * (self.prediction_pixel_size_m[0] * self.prediction_pixel_size_m[1] * 1e12)
        )
    @cached_property
    def lamella_thickness_um(self) -> NDArray[np.float_] | None:
        return np.asarray(self.lamella_thickness_prediction_px, dtype=float) * (
            self.prediction_pixel_size_m[0] * self.prediction_pixel_size_m[1] * 1e12
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
