from __future__ import annotations
import logging
import math
import time
import typing
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# fibsem
from fibsem import acquire, constants, utils as fs_utils
from fibsem.milling import MillingStrategy
from fibsem.milling import (
    setup_milling,
    draw_patterns,
    run_milling,
    finish_milling,
)
from fibsem.structures import BeamType


# Adaptive polish
import adaptive_polish.gis_measurement as gm
import adaptive_polish.utils as ap_utils
from adaptive_polish.exceptions import (
    StopEarlyError,
    StopMillingException,
    SegmentationException,
)
from adaptive_polish.dl_segmentation.sem_lamella_segmentor import SegmentationLabels
from adaptive_polish.centring import (
    get_bounding_box_scaled_to_image,
    get_centre_points_from_bounding_box,
    CentringException,
)
from adaptive_polish.plot import (
    create_centring_plot,
    create_milling_cycle_plot,
    create_summary_gis_plot,
)
from adaptive_polish.config import AdaptivePolishMillingConfig


if typing.TYPE_CHECKING:
    from pandas import DataFrame
    from numpy.typing import NDArray
    from fibsem.milling import FibsemMillingStage
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import FibsemImage, ImageSettings, Point
    from adaptive_polish.dl_segmentation.sem_lamella_segmentor import (
        AbstractAdaptivePolishingModel,
    )

_logger = logging.getLogger(__name__)


def _results_entry_helper(
    *dfs: DataFrame, milling_cycle: int, results: typing.Dict[str, typing.Any]
) -> None:
    for df in dfs:
        df.loc[milling_cycle] = {
            key: results.get(key, None) for key in df.columns.values
        }


@contextmanager
def _restore_beam_shifts(
    microscope: FibsemMicroscope,
) -> typing.Generator[None, None, None]:
    sem_shift = microscope.get_beam_shift(BeamType.ELECTRON)
    fib_shift = microscope.get_beam_shift(BeamType.ION)
    try:
        yield None
    finally:
        microscope.set_beam_shift(sem_shift, BeamType.ELECTRON)
        microscope.set_beam_shift(fib_shift, BeamType.ION)


@dataclass
class AdaptivePolishMillingStrategy(MillingStrategy):
    name: str = "AdaptivePolishing"
    fullname: str = "Adaptive polishing according to GIS thickness"

    def __init__(self, config: AdaptivePolishMillingConfig) -> None:
        self.config = config
        self.model: typing.Optional[AbstractAdaptivePolishingModel] = None

    def to_dict(self) -> dict[str, typing.Any]:
        return {"name": self.name, "config": self.config.to_dict()}

    @classmethod
    def from_dict(cls, d: dict[str, typing.Any]) -> typing.Self:
        config = AdaptivePolishMillingConfig.from_dict(d["config"])
        return cls(config=config)

    def run(
        self,
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
        asynch: bool = False,  # what does this do
        parent_ui=None,  # what does this do
    ) -> None:
        """Run adaptive polishing
        #TODO: improve docs

        Args:
            microscope (FibsemMicroscope): See `fibsem.microscope.FibsemMicroscope`
            stage (FibsemMillingStage): See `fibsem.milling.base.FibsemMillingStage`
            asynch (bool, optional): Run asynchronously? Defaults to False.
            parent_ui (_type_, optional): Napari UI. Defaults to None.
        """
        logging.info("Running %s for %s", self.fullname, stage.name)

        # setup milling
        setup_milling(microscope=microscope, milling_stage=stage)

        fib_imaging_settings, sem_imaging_settings = self._get_imaging_settings(stage)

        lamella_folder = Path(fib_imaging_settings.path)
        lamella_name = lamella_folder.stem
        lamella_ap_folder = (
            lamella_folder / f"adaptive_polish_{fs_utils.current_timestamp()}"
        )
        if lamella_ap_folder.is_dir():
            logging.info(
                "Lamella folder %s already exists, some data may be overwritten",
                lamella_ap_folder,
            )
        else:
            lamella_ap_folder.mkdir()

        lamella_ap_plots_folder, lamella_ap_sem_folder, lamella_ap_fib_folder = (
            ap_utils.ensure_subdirectories(lamella_ap_folder, "plots", "sem", "fib")
        )

        # load model
        if self.model is None:
            self._load_model()

        with _restore_beam_shifts(microscope):
            # align SEM
            lamella_centre_m = None
            if self.config.align_sem:
                try:
                    alignment_sem_imaging_settings = deepcopy(sem_imaging_settings)
                    # Set path and name in case save is set to True
                    alignment_sem_imaging_settings.path = lamella_ap_folder
                    alignment_sem_imaging_settings.filename = (
                        f"{lamella_name}_centring_SEM.tif"
                    )
                    lamella_centre_m = self._align_beam(
                        microscope=microscope,
                        sem_imaging_settings=alignment_sem_imaging_settings,
                        plot_path=lamella_ap_folder / "centring.png",
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
                        _logger.info("Failed to align SEM. Attempting to continue...")

            results_dataframes = self._setup_results_dataframes()

            # Set lamella folders for saving images
            fib_imaging_settings.save = True
            sem_imaging_settings.save = True
            fib_imaging_settings.path = lamella_ap_fib_folder
            sem_imaging_settings.path = lamella_ap_sem_folder

            # run adaptive polishing
            try:
                # Do one extra cycle without milling to run checks and get stats
                for milling_cycle in range(self.config.max_milling_cycles + 1):
                    image_name = f"{lamella_name}_AP_img_{milling_cycle:03}"
                    results_dict = {
                        "image": image_name,
                        "milling_time_s": self.config.milling_interval_s
                        * milling_cycle,
                    }
                    try:
                        self._run_milling_cycle(
                            milling_cycle=milling_cycle,
                            image_name=image_name,
                            fib_imaging_settings=fib_imaging_settings,
                            sem_imaging_settings=sem_imaging_settings,
                            plots_folder=lamella_ap_plots_folder,
                            results_dict=results_dict,
                            microscope=microscope,
                            stage=stage,
                            expected_lamella_centre_m=lamella_centre_m,
                            # Don't mill on the final cycle, just run checks
                            mill=milling_cycle < self.config.max_milling_cycles,
                            asynch=asynch,
                            parent_ui=parent_ui,
                        )
                    finally:
                        self._handle_results(
                            milling_cycle,
                            results_dict,
                            *results_dataframes,
                            save_directory=lamella_ap_folder,
                        )
                _logger.info(
                    "Adaptive milling complete (ended due to maximum milling cycles)"
                )
            except StopMillingException as e:
                _logger.info("Stopping milling due to: %s", str(e))
            except StopEarlyError as e:
                # Likely due to something not working correctly (e.g.
                # segmentation issues)
                _logger.warning("Stopping milling early due to: %s", str(e))
            except Exception:
                _logger.error("Stopping due to unexpected exception", exc_info=True)
                raise
            finally:
                # Always try to create a summary plot(s) and finish milling
                try:
                    self._create_summary_plots(
                        *results_dataframes,
                        lamella_name=lamella_name,
                        save_directory=lamella_ap_folder,
                    )
                except Exception:
                    _logger.error("Failed to create summary plot(s)", exc_info=True)
                # finish milling (clear patterns, restore imaging current)
                finish_milling(
                    microscope=microscope,
                    imaging_current=microscope.system.ion.beam.beam_current,
                    imaging_voltage=microscope.system.ion.beam.voltage,
                )

    def _run_milling_cycle(
        self,
        milling_cycle: int,
        image_name: str,
        fib_imaging_settings: ImageSettings,
        sem_imaging_settings: ImageSettings,
        plots_folder: Path,
        results_dict: typing.Dict[str, typing.Any],
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
        expected_lamella_centre_m: Point,
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

        sem_imaging_settings.filename = f"{image_name}_SEM.tif"
        sem_image = acquire.new_image(microscope, sem_imaging_settings)
        fib_imaging_settings.filename = f"{image_name}_FIB.tif"
        fib_image = acquire.new_image(microscope, fib_imaging_settings)

        self._check_lamella(
            milling_cycle,
            image_name=image_name,
            fib_image=fib_image,
            sem_image=sem_image,
            plots_folder=plots_folder,
            results_dict=results_dict,
            expected_lamella_centre_m=expected_lamella_centre_m,
        )
        if mill:
            self._mill(
                milling_cycle,
                microscope=microscope,
                stage=stage,
                asynch=asynch,
                parent_ui=parent_ui,
            )

    def _get_imaging_settings(
        self, stage: FibsemMillingStage
    ) -> typing.Tuple[ImageSettings, ImageSettings]:
        # TODO: Figure out how to get lamella directory without assuming previous image's path was correct
        fib_imaging_settings = deepcopy(stage.imaging)
        sem_imaging_settings = deepcopy(stage.imaging)

        # FIB
        fib_imaging_settings.beam_type = BeamType.ION
        _logger.debug("Adaptive polish FIB settings: %s", str(fib_imaging_settings))

        # SEM
        sem_imaging_settings.beam_type = BeamType.ELECTRON
        _logger.debug("Adaptive polish SEM settings: %s", str(sem_imaging_settings))

        return fib_imaging_settings, sem_imaging_settings

    def _load_model(self):
        model_path = Path(self.config.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Failed to find SEM segmentation model '{model_path}'"
            )
        self.model = gm.load_sem_model(
            model_path=model_path,
            generation=self.config.get_model_generation(),
        )

    def _check_lamella(
        self,
        milling_cycle: int,
        image_name: str,
        fib_image: FibsemImage,
        sem_image: FibsemImage,
        plots_folder: Path,
        results_dict: typing.Dict[str, typing.Any],
        expected_lamella_centre_m: typing.Optional[Point] = None,
    ) -> None:
        prediction = self._segment_sem_image(sem_image.data)

        mask_lamella_clean, mask_gis_clean, mask_crack_clean = gm.clean_prediction(
            prediction,
            additional_labels=(
                SegmentationLabels.GIS,
                SegmentationLabels.CRACK,
            ),
        )

        prediction_pixel_size_um = (
            sem_image.metadata.pixel_size.x
            * constants.SI_TO_MICRO
            * sem_image.data.shape[1]
            / prediction.shape[1]
        )

        lamella_area_um2 = gm.get_mask_area_um2(
            mask_lamella_clean, pixel_size_um=prediction_pixel_size_um
        )
        if lamella_area_um2 < self.config.minimum_lamella_area_um2:
            raise StopEarlyError(
                f"Lamella found was only {lamella_area_um2:.4e} um2, below the threshold of {self.config.minimum_lamella_area_um2:.4e} um2"
            )
        if mask_gis_clean is None:
            raise StopEarlyError("No GIS found beneath the lamella")

        try:
            # Get lamella position
            lamella_bbox = get_bounding_box_scaled_to_image(
                sem_image.data, mask=mask_lamella_clean
            )
        except CentringException:
            raise StopEarlyError("Failed to get lamella bounds from the segmentation")

        # Measure GIS
        gis_thickness_um = (
            np.sum(
                gm.resize_image(mask_gis_clean, new_shape=sem_image.data.shape),
                axis=0,
            )
            * sem_image.metadata.pixel_size.x
            * constants.SI_TO_MICRO
        )
        results_dict["gis_thickness_um"] = gis_thickness_um.tolist()

        lamella_xlims_px = np.round(
            (
                lamella_bbox[1],
                lamella_bbox[3],
            )
        ).astype(np.uint32)

        maximum_side_difference_px = int(
            round(
                self.config.maximum_side_difference_um
                / (sem_image.metadata.pixel_size.x * constants.SI_TO_MICRO)
            )
        )

        gis_thickness_filtered_um = gm.filter_gis_thickness(
            gis_thickness_um,
            window_size_m=self.config.window_size_px * sem_image.metadata.pixel_size.x,
            pixel_size_m=sem_image.metadata.pixel_size.x,
        )
        results_dict["gis_thickness_filtered_um"] = gis_thickness_filtered_um.tolist()

        gis_above_threshold = gis_thickness_filtered_um > self.config.gis_stop_um

        gis_xlims_px = np.asarray(
            (
                np.argmax(gis_above_threshold),
                len(gis_above_threshold) - 1 - np.argmax(gis_above_threshold[::-1]),
            ),
            dtype=np.uint32,
        )

        # Allow maximum of maximum_side_difference_um inward from lamella edge
        xlims_px = (
            min(gis_xlims_px[0], lamella_xlims_px[0] + maximum_side_difference_px),
            max(gis_xlims_px[1], lamella_xlims_px[1] - maximum_side_difference_px),
        )
        results_dict["xlims_px"] = xlims_px

        min_gis_um = np.nanmin(gis_thickness_filtered_um[xlims_px[0] : xlims_px[1] + 1])
        _logger.info(f"Took {len(gis_thickness_filtered_um)} GIS measurements along x")
        _logger.info(
            "Minimum GIS thickness for milling cycle %i = %.4e um",
            milling_cycle,
            min_gis_um,
        )
        results_dict["min_GIS_um"] = min_gis_um

        if mask_crack_clean is None:
            crack_area_um2 = 0
        else:
            crack_area_um2 = gm.get_mask_area_um2(
                mask_crack_clean, pixel_size_um=prediction_pixel_size_um
            )

        _logger.info(
            "Area of cracks found in milling cycle %i = %.4e um2",
            milling_cycle,
            crack_area_um2,
        )
        results_dict["crack_area_um2"] = crack_area_um2

        try:
            # Create plots
            create_milling_cycle_plot(
                sem_image=sem_image.data,
                first_prediction=prediction,
                clean_prediction=gm.masks_to_labels(
                    lamella_mask=mask_lamella_clean,
                    gis_mask=mask_gis_clean,
                    crack_mask=mask_crack_clean,
                ),
                fib_image=fib_image.data,
                gis_thickness_um=gis_thickness_filtered_um,
                gis_stop_um=self.config.gis_stop_um,
                crack_area_um2=crack_area_um2,
                min_gis_um=min_gis_um,
                xlims=xlims_px,
                total_milling_time=self.config.milling_interval_s * milling_cycle,
                max_crack_area_um2=self.config.max_crack_area_um2,
                img_name=image_name,
                fib_screenshot=None,
                save_path=plots_folder / f"{image_name}_plot.png",
            )
        except Exception:
            _logger.error(
                "Exception occurred creating the milling cycle plot", exc_info=True
            )

        if self.config.align_sem and expected_lamella_centre_m is not None:
            centre_m: typing.Optional[Point] = None
            mask_centre_px: typing.Optional[Point] = None
            centre_drift_um: float = 0
            try:
                centre_m, mask_centre_px = get_centre_points_from_bounding_box(
                    bbox=lamella_bbox,
                    image=sem_image.data,
                    pixel_size_m=sem_image.metadata.pixel_size.x,
                )

                centre_drift_um = (
                    math.sqrt(
                        (centre_m.x - expected_lamella_centre_m.x) ** 2
                        + (centre_m.y - expected_lamella_centre_m.y) ** 2
                    )
                    * constants.SI_TO_MICRO
                )
            except CentringException as e:
                _logger.warning(
                    f"Failed to get lamella centre, drift check will be skipped: {e}"
                )

            # Only a valid check if sem is aligned
            if self._get_drift_too_large(centre_drift_um):
                # Create centring plot if centring is found to be beyond the threshold
                create_centring_plot(
                    sem_image=sem_image,
                    mask_lamella_clean=mask_lamella_clean,
                    centre_px=mask_centre_px,
                    centre_m=centre_m,
                    plot_path=plots_folder / f"{image_name}_centring_problem.png",
                    bounding_box=lamella_bbox,
                )
                raise StopEarlyError(
                    f"Total drift (um) {centre_drift_um:.4e} > threshold {self.config.maximum_drift_um:.4e} (might be a segmentation problem)"
                )

        if self._get_gis_too_thin(min_gis_um):
            raise StopMillingException(
                f"Minimum GIS thickness (um) {min_gis_um:.4e} < threshold {self.config.gis_stop_um:.4e} um"
            )

        if self._get_crack_too_large(crack_area_um2):
            raise StopMillingException(
                f"Crack area (um2) {crack_area_um2:.4e} > threshold {self.config.max_crack_area_um2:.4e} um2"
            )

    def _mill(
        self,
        milling_cycle: int,
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
        asynch: bool = False,
        parent_ui=None,
    ) -> None:
        # get pattern - this is where bitmap will come in later
        pattern = stage.pattern.define()

        # adjust milling interval TODO
        next_milling_interval = self.config.milling_interval_s
        pattern[0].time = next_milling_interval

        # ensure milling settings are still correctly set
        microscope.setup_milling(mill_settings=stage.milling)

        # draw patterns
        draw_patterns(microscope=microscope, patterns=pattern)
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
            _logger.info("Completed milling")
        except Exception:
            _logger.error("An error occurred during milling", exc_info=True)
        finally:
            microscope.stop_milling()  # dont use milling.finish_milling as it would clear patterns

    def _align_beam(
        self,
        microscope: FibsemMicroscope,
        sem_imaging_settings: ImageSettings,
        plot_path: typing.Optional[Path] = None,
    ) -> Point:
        _logger.info("Aligning SEM beam in order to centre the lamella")

        # Take reference images
        sem_image = acquire.new_image(microscope, sem_imaging_settings)

        try:
            prediction = self._segment_sem_image(sem_image.data)

            mask_lamella_clean, _, _ = gm.clean_prediction(
                prediction,
                additional_labels=(SegmentationLabels.GIS, SegmentationLabels.CRACK),
            )
        except Exception as e:
            raise SegmentationException(
                f"Failed to get clean lamella mask required for SEM alignment: {e}"
            )

        centre_m: typing.Optional[Point] = None
        centre_px: typing.Optional[Point] = None
        lamella_bbox: typing.Optional[tuple[float, float, float, float]] = None
        try:
            lamella_bbox = get_bounding_box_scaled_to_image(
                sem_image.data,
                mask=mask_lamella_clean,
            )
            centre_m, centre_px = get_centre_points_from_bounding_box(
                lamella_bbox,
                image=sem_image.data,
                pixel_size_m=sem_image.metadata.pixel_size.x,
            )
            initial_beam_shift = microscope.get_beam_shift(BeamType.ELECTRON)
            expected_new_beam_shift = initial_beam_shift - centre_m

            # shift beam
            dx, dy = -centre_m.x, -centre_m.y
            new_beam_shift = microscope.beam_shift(dx, dy, BeamType.ELECTRON)

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

    def _get_drift_too_large(self, centre_drift_um: float) -> bool:
        return centre_drift_um > float(self.config.maximum_drift_um)

    def _get_gis_too_thin(self, min_gis_um: float) -> bool:
        # Minumum GIS thickness check
        return min_gis_um < float(self.config.gis_stop_um)

    def _get_crack_too_large(self, crack_area_um2: float) -> bool:
        # Total crack area check
        return crack_area_um2 > float(self.config.max_crack_area_um2)

    def _setup_results_dataframes(self) -> typing.Tuple[DataFrame, DataFrame]:
        return ap_utils.setup_results_df()

    def _save_results_dataframes(self, *dfs, directory: Path) -> None:
        results, gis_results_detailed = dfs
        results.to_json(directory / "GIS_thickness.json")
        gis_results_detailed.to_json(directory / "GIS_thickness_detailed.json")

    def _handle_results(
        self,
        milling_cycle: int,
        results_dict: typing.Dict[str, typing.Any],
        *results_dataframes: DataFrame,
        save_directory: Path,
    ) -> None:
        # Ensure results are always added and saved
        _results_entry_helper(
            *results_dataframes,
            milling_cycle=milling_cycle,
            results=results_dict,
        )
        self._save_results_dataframes(*results_dataframes, directory=save_directory)

    def _create_summary_plots(
        self, *results_dataframes: DataFrame, lamella_name: str, save_directory: Path
    ) -> None:
        """Creates any summary plots from the results dataframes"""
        create_summary_gis_plot(
            results=results_dataframes[0],
            save_path=save_directory / f"{lamella_name}_GIS_thickness.png",
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
