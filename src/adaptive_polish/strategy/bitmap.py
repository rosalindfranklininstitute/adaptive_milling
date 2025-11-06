from __future__ import annotations
import logging
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
from adaptive_polish.processing.bitmap import (
    filter_bitmap_signal,
    create_bitmap_array,
    get_angle_dwell_multiplier,
)
from adaptive_polish.processing.lamella import (
    crop_xlims_centre,
    crop_xlims_convolve_filtered,
    get_lamella_gis_boundary_peturbations,
)
from adaptive_polish.processing.image import resize_interp_1d, get_mask_edge
from adaptive_polish.processing.sem_segmentation import (
    SegmentationLabels as SemSegmentationLabels,
)
from adaptive_polish.plot import create_milling_cycle_plot
from adaptive_polish.exceptions import StopMillingException
from adaptive_polish.enums import StopReasons

if TYPE_CHECKING:
    from typing import ClassVar, Any
    from pathlib import Path
    from numpy.typing import NDArray
    from fibsem.milling import FibsemMillingStage
    from adaptive_polish._dataclasses import (
        LamellaInformation,
        LamellaStatistics,
    )

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
        self,
        stage: FibsemMillingStage,
        lamella_info: LamellaInformation,
    ) -> FibsemMillingStage:
        fib_image = lamella_info.fib_image
        sem_image = lamella_info.sem_image
        stats = lamella_info.statistics

        if fib_image.metadata is None:
            raise ValueError("FIB image has no metadata")
        elif sem_image.metadata is None:
            raise ValueError("SEM image has no metadata")

        stage = super()._update_milling_stage(stage=stage, lamella_info=lamella_info)

        pattern = stage.pattern

        if (
            self.config.apply_boundary_smoothing
            and self.config.boundary_smoothing_sigma > 0
        ):
            boundary_peturbations = get_lamella_gis_boundary_peturbations(
                lamella_info.clean_prediction,
                sigma=self.config.boundary_smoothing_sigma,
            )
        else:
            boundary_peturbations = None

        if self.config.incident_angle_scaling:
            gis_mask = lamella_info.clean_prediction == SemSegmentationLabels.GIS.value
            gis_lower_edge_coordinates = get_mask_edge(gis_mask, axis=0, side="max")
        else:
            gis_lower_edge_coordinates = None

        if isinstance(pattern, TrenchPattern):
            new_pattern = self._convert_trench_pattern(
                pattern,
                stats=stats,
                lamella_gis_boundary_peturbations=boundary_peturbations,
                gis_lower_edge_coordinates=gis_lower_edge_coordinates,
            )
        elif isinstance(pattern, RectanglePattern):
            new_pattern = self._convert_rectangle_pattern(
                pattern,
                stats=stats,
                lamella_gis_boundary_peturbations=boundary_peturbations,
                gis_lower_edge_coordinates=gis_lower_edge_coordinates,
            )
        else:
            raise TypeError(
                f"Invalid pattern type {pattern.name}, only {TrenchPattern.name} and {RectanglePattern.name} are supported"
            )

        stage.pattern = new_pattern

        _logger.info(
            "Created new %s for %s",
            new_pattern.name,
            stage.name,
        )
        return stage



    def _check_milling_stage(self, stage: FibsemMillingStage) -> None:
        super()._check_milling_stage(stage=stage)
        if not isinstance(stage.pattern, (BitmapPattern, TrenchBitmapPattern)):
            raise TypeError(f"Stage pattern type is {type(stage.pattern)}, expected pattern types are BitmapPattern, TrenchBitmapPattern.")
        elif stage.pattern.array is None:
            raise AttributeError(
                f"{stage.pattern.name} attribute 'array' has not been set"
            )
        self._check_bitmap(stage.pattern.array)

    def _convert_trench_pattern(
        self,
        pattern: TrenchPattern,
        stats: LamellaStatistics,
        lamella_gis_boundary_peturbations: NDArray[
            np.integer[Any] | np.float32 | np.float64
        ]
        | None = None,
        gis_lower_edge_coordinates: NDArray[np.float64 | np.float32 | np.integer[Any]]
        | None = None,
    ) -> TrenchBitmapPattern:
        bitmap_array = self.create_bitmap_array(
            pattern_width_m=pattern.width,
            stats=stats,
            lamella_gis_boundary_peturbations=lamella_gis_boundary_peturbations,
            gis_lower_edge_coordinates=gis_lower_edge_coordinates,
        )

        if pattern.time != 0:
            _logger.warning(
                "Bitmap adaptive polishing won't work as expected because pattern time has been set"
            )

        return TrenchBitmapPattern(
            point=pattern.point,
            width=pattern.width,
            spacing=pattern.spacing,
            depth=pattern.depth,
            upper_trench_height=pattern.upper_trench_height,
            lower_trench_height=pattern.lower_trench_height,
            time=pattern.time,
            array=bitmap_array,
        )

    def _convert_rectangle_pattern(
        self,
        pattern: RectanglePattern,
        stats: LamellaStatistics,
        lamella_gis_boundary_peturbations: NDArray[
            np.integer[Any] | np.float32 | np.float64
        ]
        | None = None,
        gis_lower_edge_coordinates: NDArray[np.float64 | np.float32 | np.integer[Any]]
        | None = None,
    ) -> BitmapPattern:
        bitmap_array = self.create_bitmap_array(
            pattern_width_m=pattern.width,
            stats=stats,
            lamella_gis_boundary_peturbations=lamella_gis_boundary_peturbations,
            gis_lower_edge_coordinates=gis_lower_edge_coordinates,
        )

        if pattern.time != 0:
            _logger.warning(
                "Bitmap adaptive polishing won't work as expected because pattern time has been set"
            )

        return BitmapPattern(
            point=pattern.point,
            width=pattern.width,
            height=pattern.height,
            depth=pattern.depth,
            rotation=pattern.rotation,
            time=pattern.time,
            passes=pattern.passes,
            scan_direction=pattern.scan_direction,
            array=bitmap_array,
        )

    @staticmethod
    def _get_milling_pixel_dimensions(
        dimensions: tuple[float, float],
        fib_pixel_size_m: tuple[float, float],
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
            gis_thickness_min_um=stats.gis_thickness_min_um,
            gis_thickness_median_um=stats.gis_thickness_median_um,
            crack_area_um2=stats.crack_area_um2,
            gis_min_stop_threshold_um=self.config.gis_stop_min_um,
            gis_median_stop_threshold_um=self.config.gis_stop_median_um,
            gis_min_threshold_um=self.config.gis_min_um,
            gis_max_threshold_um=self.config.gis_max_um,
            milling_stage=stage,
            image_xlims=stats.xlims_image_px,
            max_crack_area_um2=self.config.max_crack_area_um2,
            img_name=lamella_info.identifier,
            pattern_dwell_multiplier=stats.pattern_dwell_multiplier,
            pattern_xlims=stats.pattern_xlims_px,
            dwell_multiplier_mean_threshold=self.config.dwell_multiplier_mean_stop,
            dwell_multiplier_max_threshold=self.config.dwell_multiplier_max_stop,
        )

    def _refine_xlims(
        self,
        lamella_width,
        gis_thickness: NDArray[np.float64 | np.float32],
        xlims: tuple[int, int],
    ) -> tuple[int, int]:
        if self.config.pattern_alignment == "centre":
            return crop_xlims_centre(
                xlims=xlims,
                lamella_width=lamella_width,
            )
        elif self.config.pattern_alignment == "convolve":
            return crop_xlims_convolve_filtered(
                xlims=xlims,
                lamella_width=lamella_width,
                gis_thickness=gis_thickness,
            )

        return crop_xlims_centre(lamella_width=lamella_width, xlims=xlims)

    def create_bitmap_array(
        self,
        pattern_width_m: float,
        stats: LamellaStatistics,
        lamella_gis_boundary_peturbations: NDArray[
            np.integer[Any] | np.float32 | np.float64
        ]
        | None = None,
        gis_lower_edge_coordinates: NDArray[np.float64 | np.float32 | np.integer[Any]]
        | None = None,
    ) -> NDArray:
        if stats.gis_thickness_filtered_um is None:
            raise ValueError('"gis_thickness_filtered_um" is not available')
        elif stats.crack_thickness_prediction_px is None:
            raise ValueError('"crack_thickness_prediction_px" is not defined')
        elif stats.xlims_image_px is None:
            raise ValueError('"xlims_image_px" is not defined')

        lamella_width_px = int(round(pattern_width_m / stats.image_pixel_size_m[0]))

        pattern_xlims = self._refine_xlims(
            lamella_width=lamella_width_px,
            gis_thickness=stats.gis_thickness_filtered_um,
            xlims=stats.xlims_image_px,
        )
        stats.pattern_xlims_px = pattern_xlims

        bitmap_signal = stats.gis_thickness_filtered_um.copy()

        if gis_lower_edge_coordinates is not None:
            # TODO: is there a neater way to do this?
            angle_dwell_multiplier = get_angle_dwell_multiplier(
                gis_lower_edge_coordinates=gis_lower_edge_coordinates,
                pixel_size=stats.image_pixel_size_m,
            )

            angle_dwell_multiplier = np.interp(
                np.linspace(
                    0,
                    len(stats.lamella_thickness_prediction_px),
                    len(bitmap_signal),
                    endpoint=False,
                ),
                gis_lower_edge_coordinates[:, 1],
                angle_dwell_multiplier,
            )

            bitmap_signal[pattern_xlims[0] : pattern_xlims[1] + 1] = (
                (
                    bitmap_signal[pattern_xlims[0] : pattern_xlims[1] + 1]
                    - self.config.gis_min_um
                )
                * angle_dwell_multiplier[pattern_xlims[0] : pattern_xlims[1] + 1]
            ) + self.config.gis_min_um

        if lamella_gis_boundary_peturbations is not None:
            # Interpolate boundary peturbations to have image_px width
            interpolated_boundary_peturbations_um = np.interp(
                np.linspace(
                    0,
                    len(stats.lamella_thickness_prediction_px),
                    len(bitmap_signal),
                    endpoint=False,
                ),
                lamella_gis_boundary_peturbations[:, 1],
                lamella_gis_boundary_peturbations[:, 0]
                * stats.prediction_pixel_size_m[0]
                * 1e6,
            )
            # We don't want to aim for below gis_min_um, ensure we aren't
            # increasing any values in bitmap_signal
            interpolated_boundary_peturbations_um = np.clip(
                interpolated_boundary_peturbations_um, 0, None
            )
            # Remove the dips in to aim for a smoother GIS layer
            bitmap_signal -= interpolated_boundary_peturbations_um

        if self.config.mask_cracks:
            # Interpolate cracks to have image_px width
            crack_thickness_image_px = resize_interp_1d(
                stats.crack_thickness_prediction_px, target_size=len(bitmap_signal)
            )

            # Set any region with cracks a thickness of 0 for the purposes of the bitmap
            bitmap_signal[crack_thickness_image_px > 0] = 0

        filtered_trimmed_bitmap_signal = self._filter_bitmap_signal(
            bitmap_signal[pattern_xlims[0] : pattern_xlims[1] + 1]
        )

        bitmap_array = create_bitmap_array(
            input_signal=filtered_trimmed_bitmap_signal,
            xlims=(0, len(filtered_trimmed_bitmap_signal) - 1),
            min_dwell_threshold=self.config.gis_min_um,
            max_dwell_threshold=self.config.gis_max_um,
            as_image=False,
        )

        stats.pattern_dwell_multiplier = bitmap_array[0, :, 0].tolist()
        stats.pattern_blanking = bitmap_array[0, :, 1].astype(bool).tolist()

        return bitmap_array

    def _filter_bitmap_signal(
        self,
        array_1d: NDArray[np.float32 | np.float64],
    ) -> NDArray[np.float32 | np.float64]:
        return filter_bitmap_signal(
            array_1d=array_1d,
            erosion_px=self.config.bitmap_erosion_px,
            gaussian_sigma=self.config.bitmap_gaussian_sigma,
            minimum_value=self.config.gis_min_um,
            maximum_value=self.config.gis_max_um,
        )

    def _check_bitmap(self, bitmap_array: NDArray[Any]) -> None:
        if (
            dwell_multiplier_max := bitmap_array[:, :, 0].max()
        ) < self.config.dwell_multiplier_max_stop:
            raise StopMillingException(
                f"The maximum dwell time multiplier is {dwell_multiplier_max:.3f}, below the threshold of {self.config.dwell_multiplier_max_stop:.3f}",
                reason=StopReasons.MAX_DWELL_MULTIPLIER,
            )
        elif (
            dwell_multiplier_mean := bitmap_array[:, :, 0].mean()
        ) <= self.config.dwell_multiplier_mean_stop:
            raise StopMillingException(
                f"The mean dwell time multiplier is {dwell_multiplier_mean:.3f}, below the threshold of {self.config.dwell_multiplier_mean_stop:.3f}",
                reason=StopReasons.MEAN_DWELL_MULTIPLIER,
            )
