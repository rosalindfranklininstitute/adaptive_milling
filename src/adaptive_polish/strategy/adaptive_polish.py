from __future__ import annotations
import logging
import math
import time
import typing
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

# fibsem
from fibsem import acquire, constants, utils as fs_utils
from fibsem.milling import MillingStrategy
from fibsem.milling import (
    setup_milling,
    run_milling,
)
from fibsem.structures import BeamType

# Adaptive polish
import adaptive_polish.utils as ap_utils
from adaptive_polish.exceptions import (
    StopEarlyError,
    StopMillingException,
    SegmentationException,
    CentringException,
)
from adaptive_polish.enums import StopReasons
from adaptive_polish.processing import sem_segmentation as sem_seg_proc
from adaptive_polish.processing import image as image_proc
from adaptive_polish.processing import lamella as lamella_proc
from adaptive_polish.plot import (
    create_centring_plot,
    create_milling_cycle_plot,
    create_summary_gis_plot,
)
from adaptive_polish.config import (
    AdaptivePolishMillingConfig,
    TAdaptivePolishMillingConfig,
)
from adaptive_polish._dataclasses import (
    LamellaInformation,
    LamellaStatistics,
    StrategyRunInformation,
    CycleInformation,
)

if typing.TYPE_CHECKING:
    from os import PathLike
    from collections.abc import Generator
    from numpy.typing import NDArray
    from fibsem.milling import FibsemMillingStage
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import (
        FibsemImage,
        ImageSettings,
        Point,
    )
    from fibsem.ui import FibsemMillingWidget
    from adaptive_polish.processing.sem_segmentation import (
        AbstractAdaptivePolishingModel,
    )


_logger = logging.getLogger(__name__)


@contextmanager
def _restore_beam_shifts(
    microscope: FibsemMicroscope,
) -> Generator[None, None, None]:
    sem_shift = microscope.get_beam_shift(BeamType.ELECTRON)
    fib_shift = microscope.get_beam_shift(BeamType.ION)
    try:
        yield None
    finally:
        microscope.set_beam_shift(sem_shift, BeamType.ELECTRON)
        microscope.set_beam_shift(fib_shift, BeamType.ION)


class AdaptivePolishMillingStrategy(MillingStrategy[TAdaptivePolishMillingConfig]):
    name: str = "AdaptivePolishing"
    fullname: str = "Adaptive polishing according to GIS thickness"
    config_class: typing.ClassVar[type[AdaptivePolishMillingConfig]] = (
        AdaptivePolishMillingConfig
    )

    def __init__(self, config: TAdaptivePolishMillingConfig | None = None) -> None:
        super().__init__(config=config)
        self.model: AbstractAdaptivePolishingModel | None = None

    def run(
        self,
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
        asynch: bool = False,  # what does this do
        parent_ui: FibsemMillingWidget | None = None,  # what does this do
    ) -> None:
        """Run adaptive polishing

        Args:
            microscope (FibsemMicroscope): See `fibsem.microscope.FibsemMicroscope`
            stage (FibsemMillingStage): See `fibsem.milling.base.FibsemMillingStage`
            asynch (bool, optional): Run asynchronously? Defaults to False.
            parent_ui (FibsemMillingWidget, optional): Defaults to None.
        """
        logging.info("Running %s for %s", self.fullname, stage.name)
        run_info = StrategyRunInformation(
            strategy_name=self.name,
            stage_name=stage.name,
        )
        with run_info.timestamps.strategy:
            with run_info.timestamps.setup:
                # setup milling
                setup_milling(microscope=microscope, milling_stage=stage)

                fib_imaging_settings, sem_imaging_settings = self._get_imaging_settings(
                    stage
                )

                if fib_imaging_settings.path is None:
                    raise ValueError(f"Imaging path has not been set for {stage.name}")

                lamella_directory = Path(fib_imaging_settings.path)
                lamella_name = lamella_directory.stem
                run_info.lamella_name = lamella_name
                lamella_ap_directory = (
                    lamella_directory
                    / f"adaptive_polish_{fs_utils.current_timestamp()}"
                )
                if lamella_ap_directory.is_dir():
                    logging.info(
                        "Lamella directory %s already exists, some data may be overwritten",
                        lamella_ap_directory,
                    )
                else:
                    lamella_ap_directory.mkdir()

                (
                    lamella_ap_plots_directory,
                    lamella_ap_sem_directory,
                    lamella_ap_fib_directory,
                ) = ap_utils.ensure_subdirectories(
                    lamella_ap_directory, "plots", "sem", "fib"
                )

                if self.config.save_predictions:
                    lamella_ap_predictions_directory = ap_utils.ensure_subdirectories(
                        lamella_ap_directory, "predictions"
                    )[0]
                else:
                    lamella_ap_predictions_directory = None

                # load model
                if self.model is None:
                    self._load_model()

            with _restore_beam_shifts(microscope):
                # align SEM
                lamella_centre_m = None
                if self.config.align_sem:
                    with run_info.timestamps.centre_lamella:
                        try:
                            alignment_sem_imaging_settings = deepcopy(
                                sem_imaging_settings
                            )
                            # Set path and name in case save is set to True
                            alignment_sem_imaging_settings.path = lamella_ap_directory
                            alignment_sem_imaging_settings.filename = (
                                f"{lamella_name}_centring_SEM.tif"
                            )
                            lamella_centre_m = self._align_beam(
                                microscope=microscope,
                                sem_imaging_settings=alignment_sem_imaging_settings,
                                plot_path=lamella_ap_directory / "centring.png",
                            )
                        except SegmentationException:
                            _logger.error(
                                "Exception occurred during segmentation for SEM beam alignment",
                                exc_info=True,
                            )
                        except CentringException:
                            _logger.error(
                                "Error occurred calculating the lamella centre",
                                exc_info=True,
                            )
                        except Exception:
                            _logger.error(
                                "Unexpected error occurred aligning SEM.",
                                exc_info=True,
                            )
                        finally:
                            if lamella_centre_m is None:
                                _logger.info(
                                    "Failed to align SEM. Attempting to continue..."
                                )

                # Set lamella directorys for saving images
                fib_imaging_settings.save = True
                sem_imaging_settings.save = True
                fib_imaging_settings.path = lamella_ap_fib_directory
                sem_imaging_settings.path = lamella_ap_sem_directory

                # run adaptive polishing
                try:
                    # Do one extra cycle without milling to run checks and get stats
                    for milling_cycle in range(self.config.max_milling_cycles + 1):
                        cycle_info = CycleInformation(
                            milling_cycle=milling_cycle,
                            identifier=f"{lamella_name}_AP_img_{milling_cycle:03}",
                        )
                        run_info.cycle_information.append(cycle_info)
                        with cycle_info.timestamps.cycle:
                            self._run_milling_cycle(
                                cycle_info=cycle_info,
                                fib_imaging_settings=fib_imaging_settings,
                                sem_imaging_settings=sem_imaging_settings,
                                plots_directory=lamella_ap_plots_directory,
                                predictions_directory=lamella_ap_predictions_directory,
                                microscope=microscope,
                                stage=stage,
                                expected_lamella_centre_m=lamella_centre_m,
                                # Don't mill on the final cycle, just run checks
                                mill=milling_cycle < self.config.max_milling_cycles,
                                asynch=asynch,
                                parent_ui=parent_ui,
                            )
                    _logger.info(
                        "%s complete (ended due to maximum milling cycles)", self.name
                    )
                    run_info.set_end_reason(StopReasons.MAX_CYCLES)
                except StopMillingException as e:
                    run_info.set_end_reason(e.reason)
                    _logger.info("Stopping %s due to: %s", self.name, str(e))
                except StopEarlyError as e:
                    # Likely due to something not working correctly (e.g.
                    # segmentation issues)
                    run_info.set_end_reason(e.reason)
                    _logger.warning("Stopping %s early due to: %s", self.name, str(e))
                except Exception as e:
                    run_info.set_end_reason(f"{e.__class__.__name__}({e})")
                    _logger.error(
                        "Stopping %s due to unexpected exception",
                        self.name,
                        exc_info=True,
                    )
                    raise
                finally:
                    # Save results again now to include the end reason
                    self._save_results(
                        run_info=run_info,
                        save_directory=lamella_ap_directory,
                    )
                    # Always try to create a summary plot(s) and finish milling
                    try:
                        self._create_summary_plots(
                            run_info,
                            save_directory=lamella_ap_directory,
                        )
                    except Exception:
                        _logger.error("Failed to create summary plot(s)", exc_info=True)

    def _run_milling_cycle(
        self,
        cycle_info: CycleInformation,
        fib_imaging_settings: ImageSettings,
        sem_imaging_settings: ImageSettings,
        plots_directory: Path,
        predictions_directory: Path | None,
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
        expected_lamella_centre_m: Point | None,
        mill: bool = True,
        asynch: bool = False,
        parent_ui: FibsemMillingWidget | None = None,
    ) -> None:
        # Acquire images
        _logger.info(
            "Acquiring images for %s milling cycle %i/%i",
            self.name,
            cycle_info.milling_cycle,
            self.config.max_milling_cycles,
        )
        with cycle_info.timestamps.image:
            sem_imaging_settings.filename = f"{cycle_info.identifier}_SEM.tif"
            sem_image = self._acquire_image(microscope, sem_imaging_settings)
            fib_imaging_settings.filename = f"{cycle_info.identifier}_FIB.tif"
            fib_image = self._acquire_image(microscope, fib_imaging_settings)

        with cycle_info.timestamps.process:
            lamella_info = self._get_lamella_info(
                cycle_info=cycle_info,
                sem_image=sem_image,
                fib_image=fib_image,
                lamella_pad_x=self.config.lamella_pad_x,
            )
        try:
            with cycle_info.timestamps.update_stage:
                stage = self._update_milling_stage(
                    stage=stage, lamella_info=lamella_info
                )

            with cycle_info.timestamps.check_lamella:
                self._check_lamella(
                    lamella_info=lamella_info,
                    expected_lamella_centre_m=expected_lamella_centre_m,
                    plots_directory=plots_directory,
                )
            with cycle_info.timestamps.check_stage:
                self._check_milling_stage(stage=stage)

            if mill:
                with cycle_info.timestamps.mill:
                    estimated_milling_time_s = self._mill(
                        cycle_info.milling_cycle,
                        microscope=microscope,
                        stage=stage,
                        asynch=asynch,
                        parent_ui=parent_ui,
                    )
                    if cycle_info.lamella_statistics is not None:
                        cycle_info.lamella_statistics.estimated_milling_time_s = (
                            estimated_milling_time_s
                        )
        finally:
            if predictions_directory is not None:
                try:
                    self._save_predictions(
                        directory=predictions_directory,
                        lamella_info=lamella_info,
                    )
                except Exception:
                    _logger.error(
                        "Exception occurred saving the predictions", exc_info=True
                    )

            try:
                self._create_milling_cycle_plot(
                    plots_directory=plots_directory,
                    lamella_info=lamella_info,
                    stage=stage,
                )
            except Exception:
                _logger.error(
                    "Exception occurred creating the milling cycle plot", exc_info=True
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
            milling_stage=stage,
            image_xlims=stats.xlims_image_px,
            max_crack_area_um2=self.config.max_crack_area_um2,
            img_name=lamella_info.identifier,
        )

    def _save_predictions(
        self, directory: Path, lamella_info: LamellaInformation
    ) -> None:
        filename = f"{lamella_info.identifier}_SEM.tif"

        clean_prediction_dir = directory / "clean"
        clean_prediction_dir.mkdir(exist_ok=True)
        Image.fromarray(lamella_info.clean_prediction).save(
            clean_prediction_dir / filename
        )

        prediction_dir = directory / "raw"
        prediction_dir.mkdir(exist_ok=True)
        Image.fromarray(lamella_info.prediction).save(prediction_dir / filename)

    def _update_milling_stage(
        self,
        stage: FibsemMillingStage,
        lamella_info: LamellaInformation,
    ) -> FibsemMillingStage:
        # Make a copy of the milling stage before anything is changed
        return deepcopy(stage)

    def _get_imaging_settings(
        self, stage: FibsemMillingStage
    ) -> tuple[ImageSettings, ImageSettings]:
        fib_imaging_settings = deepcopy(stage.imaging)
        sem_imaging_settings = deepcopy(stage.imaging)

        # FIB
        fib_imaging_settings.beam_type = BeamType.ION
        _logger.debug("%s FIB settings: %s", self.name, str(fib_imaging_settings))

        # SEM
        sem_imaging_settings.beam_type = BeamType.ELECTRON
        _logger.debug("%s SEM settings: %s", self.name, str(sem_imaging_settings))

        return fib_imaging_settings, sem_imaging_settings

    def _load_model(self) -> None:
        model_path = Path(self.config.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Failed to find SEM segmentation model '{model_path}'"
            )
        try:
            self.model = sem_seg_proc.load_model(
                model_path=model_path,
                generation=self.config.get_model_generation(),
            )
        except Exception as e:
            raise SegmentationException("Failed to load SEM segmentation model") from e

    def _get_lamella_info(
        self,
        cycle_info: CycleInformation,
        sem_image: FibsemImage,
        fib_image: FibsemImage,
        lamella_pad_x: float = 0.1,
    ) -> LamellaInformation:
        if sem_image.metadata is None:
            raise ValueError("Unable to get pixel size from SEM image with no metadata")

        with cycle_info.timestamps.predict:
            prediction = self._segment_sem_image(sem_image.data)

        clean_prediction = lamella_proc.clean_prediction(prediction)

        image_pixel_size_m = (
            sem_image.metadata.pixel_size.x,
            sem_image.metadata.pixel_size.y,
        )

        prediction_pixel_size_m = (
            image_pixel_size_m[0] * (sem_image.data.shape[1] / prediction.shape[1]),
            image_pixel_size_m[1] * (sem_image.data.shape[0] / prediction.shape[0]),
        )

        mask_lamella_clean = (
            clean_prediction == sem_seg_proc.SegmentationLabels.LAMELLA.value
        )

        mask_crack_clean = (
            clean_prediction == sem_seg_proc.SegmentationLabels.CRACK.value
        )

        lamella_thickness = np.sum(mask_lamella_clean, axis=0)

        crack_thickness = np.sum(mask_crack_clean, axis=0)

        crack_count = image_proc.count_objects(mask_crack_clean)

        # Save guaranteed values
        statistics = LamellaStatistics(
            image_pixel_size_m=image_pixel_size_m,
            prediction_pixel_size_m=prediction_pixel_size_m,
            crack_count=crack_count,
            lamella_thickness_prediction_px=lamella_thickness.tolist(),
            crack_thickness_prediction_px=crack_thickness.tolist(),
        )
        cycle_info.lamella_statistics = statistics

        try:
            lamella_area_um2 = ap_utils.pixels_to_size(
                np.sum(statistics.lamella_thickness_prediction_px),
                pixel_size=(statistics.prediction_pixel_size_m[0] * 1e6)
                * (statistics.prediction_pixel_size_m[1] * 1e6),
            )
            if self._get_lamella_too_small(lamella_area_um2=lamella_area_um2):
                raise StopEarlyError(
                    f"Lamella found was only {lamella_area_um2:.4e} um2, below the threshold of {self.config.minimum_lamella_area_um2:.4e} um2",
                    reason=StopReasons.LAMELLA_AREA,
                )

            try:
                # Get lamella position
                lamella_image_bbox, lamella_prediction_bbox = (
                    image_proc.get_bounding_box_scaled_to_image(
                        image=sem_image.data,
                        mask=mask_lamella_clean,
                        edge_finding="percentile",
                        percentile=90,
                    )
                )
                statistics.lamella_bounding_box_prediction_px = lamella_prediction_bbox
                statistics.lamella_bounding_box_image_px = lamella_image_bbox
            except CentringException as e:
                _logger.warning(
                    "Failed to get lamella centre, drift check will be skipped: %s",
                    str(e),
                )

            # Apply lamella_pad_x padding to each side in X
            x_pad = int(
                round(
                    (lamella_prediction_bbox[3] - lamella_prediction_bbox[1])
                    * lamella_pad_x
                )
            )

            # Measure GIS
            # Note the returned values are scaled to the image, not the prediction
            ylims_px = image_proc.bbox_to_ylims(
                lamella_prediction_bbox,
                y_bounds=(0, clean_prediction.shape[0] - 1),
                pad=0,
            )
            statistics.xlims_prediction_px = image_proc.bbox_to_xlims(
                lamella_prediction_bbox,
                x_bounds=(0, clean_prediction.shape[1] - 1),
                pad=x_pad,
            )
            gis_thickness_image_px, xlims_image_px = lamella_proc.get_gis_thickness(
                clean_prediction,
                xlims=statistics.xlims_prediction_px,
                ylims=(ylims_px[0], None),
                image_shape=(sem_image.data.shape[0], sem_image.data.shape[1]),
            )

            statistics.xlims_image_px = xlims_image_px
            statistics.gis_thickness_image_px = gis_thickness_image_px.tolist()

            gis_thickness_filtered_image_px = lamella_proc.filter_gis_thickness(
                gis_thickness_px=gis_thickness_image_px,
                xlims_px=xlims_image_px,
                sigma=self.config.gis_filter_sigma,
            )

            statistics.gis_thickness_filtered_image_px = (
                gis_thickness_filtered_image_px.tolist()
            )
            # Calculate GIS min, median, etc.
            statistics.calculate_statistics()

        finally:
            return LamellaInformation(
                identifier=cycle_info.identifier,
                sem_image=sem_image,
                fib_image=fib_image,
                prediction=prediction,
                clean_prediction=clean_prediction,
                statistics=statistics,
            )

    def _check_lamella(
        self,
        lamella_info: LamellaInformation,
        expected_lamella_centre_m: Point | None = None,
        plots_directory: str | PathLike[str] | None = None,
    ) -> None:
        if lamella_info.sem_image.metadata is None:
            raise ValueError("SEM image has no metadata")

        stats = lamella_info.statistics

        if (
            self.config.align_sem
            and expected_lamella_centre_m is not None
            and stats.lamella_bounding_box_image_px is not None
        ):
            centre_m, centre_px = image_proc.get_centre_points_from_bounding_box(
                stats.lamella_bounding_box_image_px,
                image=lamella_info.sem_image.data,
                pixel_size_m=lamella_info.sem_image.metadata.pixel_size.x,
            )
            centre_drift_um = (
                math.sqrt(
                    (centre_m.x - expected_lamella_centre_m.x) ** 2
                    + (centre_m.y - expected_lamella_centre_m.y) ** 2
                )
                * constants.SI_TO_MICRO
            )

            # Only a valid check if sem is aligned
            if self._get_drift_too_large(centre_drift_um):
                # Create centring plot if centring is found to be beyond the threshold
                if plots_directory is not None:
                    create_centring_plot(
                        sem_image=lamella_info.sem_image,
                        mask_lamella_clean=lamella_info.clean_prediction
                        == sem_seg_proc.SegmentationLabels.LAMELLA.value,
                        centre_px=centre_px,
                        centre_m=centre_m,
                        plot_path=Path(plots_directory)
                        / f"{lamella_info.identifier}_centring_problem.png",
                        bounding_box=stats.lamella_bounding_box_image_px,
                    )
                raise StopEarlyError(
                    f"Total drift (um) {centre_drift_um:.4e} > threshold {self.config.maximum_drift_um:.4e} (might be a segmentation problem)",
                    reason=StopReasons.LAMELLA_DRIFT,
                )

        if self._get_min_gis_too_thin(stats.gis_thickness_min_um):
            raise StopMillingException(
                f"Minimum GIS thickness (um) {stats.gis_thickness_min_um:.4e} < threshold {self.config.gis_stop_min_um:.4e} um",
                reason=StopReasons.MIN_GIS_THICKNESS,
            )

        if self._get_median_gis_too_thin(stats.gis_thickness_median_um):
            raise StopMillingException(
                f"Median GIS thickness (um) {stats.gis_thickness_median_um:.4e} < threshold {self.config.gis_stop_median_um:.4e} um",
                reason=StopReasons.MEDIAN_GIS_THICKNESS,
            )

        if self._get_crack_too_large(stats.crack_area_um2):
            raise StopMillingException(
                f"Crack area (um2) {stats.crack_area_um2:.4e} > threshold {self.config.max_crack_area_um2:.4e} um2",
                reason=StopReasons.CRACK_AREA,
            )

    def _check_milling_stage(self, stage: FibsemMillingStage) -> None:
        # The stage isn't updated in this strategy
        return

    def _mill(
        self,
        milling_cycle: int,
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
        asynch: bool = False,
        parent_ui: FibsemMillingWidget | None = None,
    ) -> float | None:
        # Process UI stop after checks to ensure reporting isn't skipped
        if parent_ui is not None and hasattr(parent_ui, "_milling_stop_event"):
            if parent_ui._milling_stop_event.is_set():
                raise StopEarlyError(
                    "Stop milling requested via the UI",
                    reason=StopReasons.USER,
                )

        # ensure milling settings are still correctly set
        microscope.setup_milling(mill_settings=stage.milling)

        # draw patterns
        microscope.draw_patterns(patterns=stage.pattern.define())

        try:
            # Log patterns created
            for pattern_str in _format_microscope_patterns(microscope=microscope):
                logging.debug("Pattern created: %s", pattern_str)
        except Exception:
            logging.debug("Failed to log pattern information", exc_info=True)

        try:
            estimated_time = microscope.estimate_milling_time()
            logging.info(
                f"Estimated time for {stage.name}: {estimated_time:.2f} seconds"
            )
            if parent_ui is not None and hasattr(parent_ui, "milling_progress_signal"):
                parent_ui.milling_progress_signal.emit(
                    {
                        "msg": f"Running {stage.name} cycle {milling_cycle}...",
                        "progress": {
                            "started": True,
                            "start_time": time.time(),
                            "estimated_time": estimated_time,
                            "name": stage.name,
                        },
                    }
                )

            # mill
            run_milling(
                microscope=microscope,
                milling_current=stage.milling.milling_current,
                milling_voltage=stage.milling.milling_voltage,
                asynch=asynch,
            )
            _logger.info("Completed milling cycle %i", milling_cycle)
            return estimated_time
        except Exception:
            _logger.error(
                "An error occurred during milling cycle %i",
                milling_cycle,
                exc_info=True,
            )
            return None
        finally:
            microscope.stop_milling()  # dont use milling.finish_milling as it would clear patterns

    def _align_beam(
        self,
        microscope: FibsemMicroscope,
        sem_imaging_settings: ImageSettings,
        plot_path: Path | None = None,
    ) -> Point:
        _logger.info("Aligning SEM beam in order to centre the lamella")

        # Take reference images
        sem_image = self._acquire_image(microscope, sem_imaging_settings)

        try:
            prediction = self._segment_sem_image(sem_image.data)

            mask_lamella_clean = lamella_proc.clean_lamella(prediction)
        except Exception as e:
            raise SegmentationException(
                "Failed to get clean lamella mask required for SEM alignment"
            ) from e

        centre_m: Point | None = None
        centre_px: Point | None = None
        lamella_bbox: tuple[float, float, float, float] | None = None
        try:
            lamella_bbox, _ = image_proc.get_bounding_box_scaled_to_image(
                sem_image.data,
                mask=mask_lamella_clean,
                edge_finding="percentile",
                percentile=90,
            )
            if sem_image.metadata is None:
                raise ValueError(
                    "Unable to get pixel size from SEM image with no metadata"
                )

            centre_m, centre_px = image_proc.get_centre_points_from_bounding_box(
                lamella_bbox,
                image=sem_image.data,
                pixel_size_m=sem_image.metadata.pixel_size.x,
            )
            initial_beam_shift = microscope.get_beam_shift(BeamType.ELECTRON)
            expected_new_beam_shift = initial_beam_shift - centre_m

            # shift beam
            dx, dy = -centre_m.x, -centre_m.y
            microscope.beam_shift(dx, dy, BeamType.ELECTRON)

            new_beam_shift = microscope.get_beam_shift(BeamType.ELECTRON)

            new_lamella_centre_m = new_beam_shift - expected_new_beam_shift

            _logger.info("Completed SEM beam alignment")

            return new_lamella_centre_m
        finally:
            if plot_path is not None:
                create_centring_plot(
                    sem_image=sem_image,
                    prediction=prediction,
                    mask_lamella_clean=mask_lamella_clean,
                    centre_px=centre_px,
                    centre_m=centre_m,
                    plot_path=plot_path,
                    bounding_box=lamella_bbox,
                )

    def _get_lamella_too_small(self, lamella_area_um2: float) -> bool:
        return lamella_area_um2 < self.config.minimum_lamella_area_um2

    def _get_drift_too_large(self, centre_drift_um: float) -> bool:
        return centre_drift_um > float(self.config.maximum_drift_um)

    def _get_min_gis_too_thin(self, min_gis_um: float | None) -> bool:
        if min_gis_um is None:
            raise StopEarlyError("No minimum GIS measurement")
        # Minumum GIS thickness check
        return min_gis_um < float(self.config.gis_stop_min_um)

    def _get_median_gis_too_thin(self, median_gis_um: float | None) -> bool:
        if median_gis_um is None:
            raise StopEarlyError("No mean GIS measurement")
        # Mean GIS thickness check
        return median_gis_um < float(self.config.gis_stop_median_um)

    def _get_crack_too_large(self, crack_area_um2: float) -> bool:
        # Total crack area check
        return crack_area_um2 > float(self.config.max_crack_area_um2)

    def _save_results(
        self,
        run_info: StrategyRunInformation,
        save_directory: Path,
    ) -> None:
        ap_utils.save_dict_as_json(
            run_info.to_dict(), save_directory / "AP_metadata.json"
        )

    def _create_summary_plots(
        self, run_info: StrategyRunInformation, save_directory: Path
    ) -> None:
        """Creates any summary plots from the strategy run information"""
        create_summary_gis_plot(
            run_info=run_info,
            save_path=save_directory / "AP_summary_plots.png",
        )

    def _segment_sem_image(
        self, sem_image: NDArray[np.number], full_size: bool = False
    ) -> NDArray[typing.Any]:
        if self.model is None:
            raise SegmentationException(
                "Unable to continue as no SEM segmentation model has been loaded"
            )
        _logger.debug("Starting SEM segmentation")
        prediction = self.model.predict(sem_image, full_size=full_size)
        _logger.debug("SEM segmentation complete")
        return prediction

    def _acquire_image(
        self, microscope: FibsemMicroscope, imaging_settings: ImageSettings
    ) -> FibsemImage:
        return acquire.new_image(microscope, imaging_settings)


def _format_microscope_patterns(microscope: FibsemMicroscope) -> list[str]:
    pattern_string_list: list[str] = []
    for pattern in microscope._patterns:
        pattern_name = pattern.__class__.__name__
        attr_list: list[str] = []
        for attr_name in dir(pattern):
            if attr_name.startswith("_"):
                continue
            try:
                value = getattr(pattern, attr_name)
                if isinstance(value, str):
                    value = f'"{value}"'
                attr_list.append(f"{attr_name}={value}")
            except Exception:
                pass
        pattern_string_list.append(f"{pattern_name}({', '.join(attr_list)})")
    return pattern_string_list
