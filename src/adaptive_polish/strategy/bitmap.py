from __future__ import annotations
import logging
import typing
import numpy as np

from fibsem.milling.patterning import (
    TrenchPattern,
    RectanglePattern,
    TrenchBitmapPattern,
    BitmapPattern,
)

from adaptive_polish.strategy import AdaptivePolishMillingStrategy
from adaptive_polish.config import BitmapAdaptivePolishMillingConfig
from adaptive_polish.processing.bitmap import create_bitmap_array
from adaptive_polish.processing.lamella import find_milling_edges

if typing.TYPE_CHECKING:
    from fibsem.milling import FibsemMillingStage
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import FibsemImage, ImageSettings, Point

    from adaptive_polish._dataclasses import LamellaInformation


_logger = logging.getLogger(__name__)


class BitmapAdaptivePolishMillingStrategy(
    AdaptivePolishMillingStrategy[BitmapAdaptivePolishMillingConfig]
):
    name: str = "BitmapAdaptivePolishing"
    fullname: str = "Adaptive polishing using bitmap milling"
    config_class: typing.ClassVar[typing.Type[BitmapAdaptivePolishMillingConfig]] = (
        BitmapAdaptivePolishMillingConfig
    )

    def _run_milling_cycle(
        self,
        milling_cycle: int,
        identifier: str,
        fib_imaging_settings: ImageSettings,
        sem_imaging_settings: ImageSettings,
        plots_directory: Path,
        results_dict: dict[str, typing.Any],
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
        expected_lamella_centre_m: Point | None,
        mill: bool = True,
        asynch: bool = False,
        parent_ui=None,
    ) -> None:
        # Acquire images
        _logger.info(
            "Acquiring images for milling cycle %i/%i",
            milling_cycle,
            self.config.max_milling_cycles,
        )

        sem_imaging_settings.filename = f"{identifier}_SEM.tif"
        sem_image = self._acquire_image(microscope, sem_imaging_settings)
        fib_imaging_settings.filename = f"{identifier}_FIB.tif"
        fib_image = self._acquire_image(microscope, fib_imaging_settings)

        lamella_info = self._get_lamella_info(
            milling_cycle=milling_cycle,
            identifier=identifier,
            sem_image=sem_image,
            fib_image=fib_image,
            milling_stage=stage,
        )
        self._check_lamella(
            lamella_info=lamella_info,
            plots_directory=plots_directory,
            expected_lamella_centre_m=expected_lamella_centre_m,
        )

        results_dict["image"] = identifier
        results_dict.update(lamella_info.statistics.to_dict())

        if mill:
            patterns = self._create_bitmap_patterns(
                lamella_info=lamella_info, milling_stage=stage
            )
            self._mill(
                milling_cycle,
                microscope=microscope,
                stage=stage,
                patterns=patterns,
                asynch=asynch,
                parent_ui=parent_ui,
            )

    def _create_bitmap_patterns(
        self,
        lamella_info: LamellaInformation,
        milling_stage: FibsemMillingStage,
    ) -> list[FibsemBitmapSettings]:
        fib_image = lamella_info.fib_image
        sem_image = lamella_info.sem_image
        stats = lamella_info.statistics

        if fib_image.metadata is None:
            raise ValueError("FIB image has no metadata")
        elif sem_image.metadata is None:
            raise ValueError("SEM image has no metadata")
        elif stats.gis_thickness_um is None:
            raise ValueError('"gis_thickness_um" is not defined')
        elif stats.lamella_thickness_um is None:
            raise ValueError('"lamella_thickness_um" is not defined')
        elif stats.xlims_px is None:
            raise ValueError('"xlims_px" is not defined')

        fib_pixel_size = fib_image.metadata.pixel_size
        sem_pixel_size = sem_image.metadata.pixel_size
        gis_thickness_um = np.asarray(stats.gis_thickness_um, dtype=np.float32)
        lamella_thickness_um = np.asarray(stats.lamella_thickness_um, dtype=np.float32)

        pattern = milling_stage.pattern
        min_dwell_thickness_um = (
            self.config.gis_stop_um
            if self.config.gis_min_um is None
            else self.config.gis_min_um
        )

        if isinstance(pattern, TrenchPattern):
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

            lamella_width_px = int(round(pattern.width / fib_pixel_size.x))

            new_x_lims = find_milling_edges(
                lamella_thickness=lamella_thickness_um,
                gis_thickness=gis_thickness_um,
                lamella_width=lamella_width_px,
            )

            bitmap_array = create_bitmap_array(
                gis_thickness_m=gis_thickness_um * 1e-6,
                xlims=new_x_lims,
                gis_resolution_m=sem_pixel_size.x,
                bitmap_resolution_m=fib_pixel_size.x,
                min_dwell_thickness_m=min_dwell_thickness_um * 1e-6,
                max_dwell_thickness_m=self.config.gis_max_um * 1e-6,
                as_image=False,
            )

            new_pattern = TrenchBitmapPattern(
                width=pattern.width,
                spacing=pattern.spacing,
                depth=pattern.depth,
                upper_trench_height=pattern.upper_trench_height,
                lower_trench_height=pattern.lower_trench_height,
                time=pattern.time,
                array=np.tile(bitmap_array, (upper_dimensions_px[0], 0)),
                array_lower=np.tile(bitmap_array, (lower_dimensions_px[0], 0)),
            )

        elif isinstance(pattern, RectanglePattern):
            dimensions_px = (
                BitmapAdaptivePolishMillingStrategy._get_milling_pixel_dimensions(
                    (pattern.height, pattern.width),
                    (fib_pixel_size.y, fib_pixel_size.x),
                )
            )

            lamella_width_px = int(round(pattern.width / fib_pixel_size.x))

            new_x_lims = find_milling_edges(
                lamella_thickness=lamella_thickness_um,
                gis_thickness=gis_thickness_um,
                lamella_width=lamella_width_px,
            )

            bitmap_array = create_bitmap_array(
                gis_thickness_m=gis_thickness_um * 1e-6,
                xlims=new_x_lims,
                gis_resolution_m=sem_pixel_size.x,
                bitmap_resolution_m=fib_pixel_size.x,
                min_dwell_thickness_m=min_dwell_thickness_um * 1e-6,
                max_dwell_thickness_m=self.config.gis_max_um * 1e-6,
                as_image=False,
            )

            new_pattern = BitmapPattern(
                width=pattern.width,
                height=pattern.height,
                depth=pattern.depth,
                rotation=pattern.rotation,
                time=pattern.time,
                passes=pattern.passes,
                scan_direction=pattern.scan_direction,
                array=np.tile(bitmap_array, (dimensions_px[0], 0)),
            )

        else:
            raise TypeError(
                f"Invalid pattern type {pattern.name}, only {TrenchPattern.name} and {RectanglePattern.name} are supported"
            )

        return new_pattern.define()

    @staticmethod
    def _get_milling_pixel_dimensions(
        dimensions: tuple[float, float], fib_pixel_size_m: tuple[float, float]
    ) -> tuple[int, int]:
        return (
            int(round(dimensions[0] / fib_pixel_size_m[0])),
            int(round(dimensions[1] / fib_pixel_size_m[1])),
        )
