from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from fibsem.milling.patterning import (
    TrenchPattern,
    RectanglePattern,
    TrenchBitmapPattern,
    BitmapPattern,
)

from adaptive_polish.strategy import AdaptivePolishMillingStrategy
from adaptive_polish.config import BitmapAdaptivePolishMillingConfig
from adaptive_polish.bitmaps import create_bitmap_array
from adaptive_polish.gis_measurement import find_milling_edges
from adaptive_polish.plot import create_milling_cycle_plot
from adaptive_polish.exceptions import StopMillingException

if TYPE_CHECKING:
    from typing import ClassVar
    from pathlib import Path
    from numpy.typing import NDArray
    from fibsem.milling import FibsemMillingStage
    from fibsem.structures import Point
    from adaptive_polish._dataclasses import LamellaInformation, LamellaStatistics

_logger = logging.getLogger(__name__)


class BitmapAdaptivePolishMillingStrategy(
    AdaptivePolishMillingStrategy[BitmapAdaptivePolishMillingConfig]
):
    name: str = "BitmapAdaptivePolishing"
    fullname: str = "Adaptive polishing using bitmap milling"
    config_class: ClassVar[type[BitmapAdaptivePolishMillingConfig]] = (
        BitmapAdaptivePolishMillingConfig
    )

    def _update_milling_stage(
        self, stage: FibsemMillingStage, lamella_info: LamellaInformation
    ) -> FibsemMillingStage:
        fib_image = lamella_info.fib_image
        sem_image = lamella_info.sem_image
        stats = lamella_info.statistics

        if fib_image.metadata is None:
            raise ValueError("FIB image has no metadata")
        elif sem_image.metadata is None:
            raise ValueError("SEM image has no metadata")

        fib_pixel_size = fib_image.metadata.pixel_size
        sem_pixel_size = sem_image.metadata.pixel_size

        pattern = stage.pattern

        if isinstance(pattern, TrenchPattern):
            new_pattern = self._convert_trench_pattern(
                pattern,
                fib_pixel_size=fib_pixel_size,
                sem_pixel_size=sem_pixel_size,
                stats=stats,
            )
        elif isinstance(pattern, RectanglePattern):
            new_pattern = self._convert_rectangle_pattern(
                pattern,
                fib_pixel_size=fib_pixel_size,
                sem_pixel_size=sem_pixel_size,
                stats=stats,
            )
        else:
            raise TypeError(
                f"Invalid pattern type {pattern.name}, only {TrenchPattern.name} and {RectanglePattern.name} are supported"
            )

        stage = super()._update_milling_stage(stage=stage, lamella_info=lamella_info)
        stage.pattern = new_pattern

        _logger.info(
            "Created new %s for %s",
            new_pattern.name,
            stage.name,
        )

        return stage

    def _convert_trench_pattern(
        self,
        pattern: TrenchPattern,
        fib_pixel_size: Point,
        sem_pixel_size: Point,
        stats: LamellaStatistics,
    ) -> TrenchBitmapPattern:
        upper_dimensions_px = (
            BitmapAdaptivePolishMillingStrategy._get_milling_pixel_dimensions(
                (pattern.upper_trench_height, pattern.width),
                (fib_pixel_size.y, fib_pixel_size.x),
            )
        )
        lower_dimensions_px = (
            BitmapAdaptivePolishMillingStrategy._get_milling_pixel_dimensions(
                (pattern.lower_trench_height, pattern.width),
                (fib_pixel_size.y, fib_pixel_size.x),
            )
        )

        bitmap_array = self.create_bitmap_array(
            pattern=pattern,
            fib_pixel_size_x=fib_pixel_size.x,
            sem_pixel_size_x=sem_pixel_size.x,
            stats=stats,
        )
        if pattern.time != 0:
            _logger.warning(
                "Bitmap adaptive polishing won't work as expected because pattern time has been set"
            )

        return TrenchBitmapPattern(
            width=pattern.width,
            spacing=pattern.spacing,
            depth=pattern.depth,
            upper_trench_height=pattern.upper_trench_height,
            lower_trench_height=pattern.lower_trench_height,
            time=pattern.time,
            array=np.tile(bitmap_array, (upper_dimensions_px[0], 1, 1)),
            array_lower=np.tile(bitmap_array, (lower_dimensions_px[0], 1, 1)),
        )

    def _convert_rectangle_pattern(
        self,
        pattern: RectanglePattern,
        fib_pixel_size: Point,
        sem_pixel_size: Point,
        stats: LamellaStatistics,
    ) -> BitmapPattern:
        dimensions_px = (
            BitmapAdaptivePolishMillingStrategy._get_milling_pixel_dimensions(
                (pattern.height, pattern.width),
                (fib_pixel_size.y, fib_pixel_size.x),
            )
        )

        bitmap_array = self.create_bitmap_array(
            pattern=pattern,
            fib_pixel_size_x=fib_pixel_size.x,
            sem_pixel_size_x=sem_pixel_size.x,
            stats=stats,
        )

        if pattern.time != 0:
            _logger.warning(
                "Bitmap adaptive polishing won't work as expected because pattern time has been set"
            )

        return BitmapPattern(
            width=pattern.width,
            height=pattern.height,
            depth=pattern.depth,
            rotation=pattern.rotation,
            time=pattern.time,
            passes=pattern.passes,
            scan_direction=pattern.scan_direction,
            array=np.tile(bitmap_array, (dimensions_px[0], 1, 1)),
        )


    @staticmethod
    def _get_milling_pixel_dimensions(
        dimensions: tuple[float, float], fib_pixel_size_m: tuple[float, float]
    ) -> tuple[int, int]:
        return (
            int(round(dimensions[0] / fib_pixel_size_m[0])),
            int(round(dimensions[1] / fib_pixel_size_m[1])),
        )

    def _create_milling_cycle_plot(
        self,
        plots_directory: Path,
        lamella_info: LamellaInformation,
        stage: FibsemMillingStage | None,
    ) -> None:
        stats = lamella_info.statistics
        # Create plots
        create_milling_cycle_plot(
            save_path=plots_directory / f"{lamella_info.identifier}_plot.png",
            sem_image=lamella_info.sem_image,
            fib_image=lamella_info.fib_image,
            first_prediction=lamella_info.prediction,
            clean_prediction=lamella_info.clean_prediction,
            gis_thickness_um=stats.gis_thickness_filtered_um,
            crack_area_um2=stats.crack_area_um2,
            min_gis_um=stats.min_GIS_um,
            gis_stop_threshold_um=self.config.gis_stop_um,
            gis_min_threshold_um=self.config.gis_min_um,
            gis_max_threshold_um=self.config.gis_max_um,
            milling_stage=stage,
            xlims=stats.xlims_px,
            total_milling_time=stats.milling_time_s,
            max_crack_area_um2=self.config.max_crack_area_um2,
            img_name=lamella_info.identifier,
        )

    def _refine_xlims(
        self,
        lamella_width,
        gis_thickness: NDArray[np.float32 | np.float64],
        lamella_thickness: NDArray[np.float32 | np.float64],
        xlims: tuple[int, int],
    ) -> tuple[int, int]:
        crop_amount = (1 + xlims[1] - xlims[0] - lamella_width) / 2
        return (
            xlims[0] + math.floor(crop_amount),
            xlims[1] - math.ceil(crop_amount),
        )

    def create_bitmap_array(
        self,
        pattern: RectanglePattern | TrenchPattern,
        fib_pixel_size_x: float,
        sem_pixel_size_x: float,
        stats: LamellaStatistics,
    ) -> NDArray:
        if stats.gis_thickness_um is None:
            raise ValueError('"gis_thickness_um" is not defined')
        elif stats.lamella_thickness_um is None:
            raise ValueError('"lamella_thickness_um" is not defined')
        elif stats.xlims_px is None:
            raise ValueError('"xlims_px" is not defined')

        min_dwell_thickness_um = (
            self.config.gis_stop_um
            if self.config.gis_min_um == 0
            else self.config.gis_min_um
        )

        lamella_width_px = int(round(pattern.width / fib_pixel_size_x))

        gis_thickness_um = np.asarray(stats.gis_thickness_filtered_um, dtype=np.float32)
        lamella_thickness_um = np.asarray(stats.lamella_thickness_um, dtype=np.float32)

        new_x_lims = self._refine_xlims(
            lamella_width=lamella_width_px,
            gis_thickness=gis_thickness_um,
            lamella_thickness=lamella_thickness_um,
            xlims=stats.xlims_px,
        )

        bitmap_array = create_bitmap_array(
            gis_thickness_m=gis_thickness_um * 1e-6,
            xlims=new_x_lims,
            gis_resolution_m=sem_pixel_size_x,
            bitmap_resolution_m=fib_pixel_size_x,
            min_dwell_thickness_m=min_dwell_thickness_um * 1e-6,
            max_dwell_thickness_m=self.config.gis_max_um * 1e-6,
            as_image=False,
        )

        if (
            dwell_multiplier_max := bitmap_array[:, :, 0].max()
        ) < self.config.stop_mill_fraction_max:
            raise StopMillingException(
                f"The maximum dwell time multiplier is {dwell_multiplier_max:.3f}, below the threshold of {self.config.stop_mill_fraction_max:.3f}"
            )
        elif (
            dwell_multiplier_max := bitmap_array[:, :, 0].mean()
        ) <= self.config.stop_mill_fraction_mean:
            raise StopMillingException(
                f"The mean dwell time multiplier is {dwell_multiplier_max:.3f}, below the threshold of {self.config.stop_mill_fraction_max:.3f}"
            )

        return bitmap_array
