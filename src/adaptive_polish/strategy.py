from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import typing

# fibsem
from fibsem import acquire, constants, utils as fs_utils
from fibsem.milling import MillingStrategy
from fibsem.milling import (
    setup_milling,
    draw_patterns,
    run_milling,
    finish_milling,
)
from fibsem.structures import BeamType, Point


# Adaptive polish
import adaptive_polish.gis_measurement as gm
import adaptive_polish.utils as ap_utils
from adaptive_polish.dl_segmentation.sem_lamella_segmentor import SegmentationLabels
from adaptive_polish.centring import (
    get_bounding_box_scaled_to_image,
    get_centre_points_from_bounding_box,
)
from adaptive_polish.config import AdaptivePolishMillingConfig


if typing.TYPE_CHECKING:
    from numpy.typing import NDArray
    from pandas import DataFrame
    from fibsem.milling import FibsemMillingStage
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import FibsemImage, ImageSettings
    from adaptive_polish.dl_segmentation.sem_lamella_segmentor import (
        AbstractAdaptivePolishingModel,
    )

_logger = logging.getLogger(__name__)


class _AdaptivePolishMillingException(Exception):
    # Base class to make it easy to catch all
    pass


class StopMillingException(_AdaptivePolishMillingException):
    pass


class StopEarlyError(_AdaptivePolishMillingException):
    pass


@dataclass
class AdaptivePolishMillingStrategy(MillingStrategy):
    name: str = "AdaptivePolishing"
    fullname: str = "Adaptive polishing according to GIS thickness"

    def __init__(self, config: AdaptivePolishMillingConfig) -> None:
        self.config = config
        self.model: typing.Optional[AbstractAdaptivePolishingModel] = None

    def to_dict(self) -> dict[str, typing.Any]:
        return {"name": self.name, "config": self.config.to_dict()}

    @staticmethod
    def from_dict(d: dict[str, typing.Any]) -> typing.Self:
        config = AdaptivePolishMillingConfig.from_dict(d["config"])
        return AdaptivePolishMillingStrategy(config=config)

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
        fib_imaging_settings = microscope.get_imaging_settings(BeamType.ION)
        sem_imaging_settings = microscope.get_imaging_settings(BeamType.ELECTRON)
        lamella_folder = Path(fib_imaging_settings.path)

        # set imaging settings
        fib_imaging_settings.resolution = [self.config.fib_resolution_x, self.config.fib_resolution_y]
        fib_imaging_settings.dwell_time = self.config.fib_dwell_time_us * constants.MICRO_TO_SI
        fib_imaging_settings.hfw = self.config.fib_hfw_um
        fib_imaging_settings.autocontrast = self.config.fib_autocontrast
        fib_imaging_settings.autogamma = self.config.fib_autogamma

        sem_imaging_settings.resolution = [self.config.sem_resolution_x, self.config.sem_resolution_y]
        sem_imaging_settings.dwell_time =  self.config.sem_dwell_time_us * constants.MICRO_TO_SI
        sem_imaging_settings.hfw = self.config.sem_hfw_um
        sem_imaging_settings.autocontrast = self.config.sem_autocontrast
        sem_imaging_settings.autogamma = self.config.sem_autogamma

        logging.info(f"Adaptive polish FIB settings: {fib_imaging_settings}")
        logging.info(f"Adaptive polish SEM settings: {sem_imaging_settings}")

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
            model_path = Path(self.config.model_path)
            if not model_path.is_file():
                raise FileNotFoundError(f"Failed to find '{model_path}'")
            self.model = gm.load_sem_model(
                model_path=model_path, generation=self.config.model_generation
            )

        # align SEM
        if self.config.align_sem:
            self._align_beam(
                microscope=microscope,
                sem_imaging_settings=sem_imaging_settings,
                plot_path=lamella_ap_folder / "centring.png",
            )

        # Set lamella folders for saving images
        fib_imaging_settings.save = True
        sem_imaging_settings.save = True
        fib_imaging_settings.path = lamella_ap_fib_folder
        sem_imaging_settings.path = lamella_ap_sem_folder

        # setup results
        results, gis_results_detailed = ap_utils.setup_results_df()

        # run adaptive polishing
        try:
            for milling_cycle in range(int(self.config.max_milling_cycles)):
                f_basename = f"{lamella_folder.stem}_AP_img_{milling_cycle:03}"

                # Acquire images
                _logger.info(
                    "Acquiring images for milling cycle %i/%i",
                    milling_cycle,
                    self.config.max_milling_cycles,
                )

                sem_imaging_settings.filename = f"{f_basename}_SEM.tif"
                sem_image = acquire.new_image(microscope, sem_imaging_settings)
                fib_imaging_settings.filename = f"{f_basename}_FIB.tif"
                fib_image = acquire.new_image(microscope, fib_imaging_settings)

                AdaptivePolishMillingStrategy._check_lamella(
                    milling_cycle,
                    image_name=f_basename,
                    fib_image=fib_image,
                    sem_image=sem_image,
                    config=self.config,
                    model=self.model,
                    lamella_ap_folder=lamella_ap_folder,
                    lamella_ap_plots_folder=lamella_ap_plots_folder,
                    results=results,
                    gis_results_detailed=gis_results_detailed,
                )
                AdaptivePolishMillingStrategy._mill(
                    microscope=microscope, stage=stage, config=self.config
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
            # Always try to create a summary plot and finish milling
            try:
                gm.summary_gis_plot(
                    results=results,
                    save_path=lamella_ap_folder
                    / f"{lamella_folder.stem}_GIS_thickness.png",
                )
            except Exception:
                _logger.error("Failed to create summary plot", exc_info=True)
            # finish milling (clear patterns, restore imaging current)
            finish_milling(
                microscope=microscope,
                imaging_current=microscope.system.ion.beam.beam_current,
                imaging_voltage=microscope.system.ion.beam.voltage,
            )
            microscope.reset_beam_shifts()

    @staticmethod
    def _check_lamella(
        milling_cycle: int,
        image_name: str,
        fib_image: FibsemImage,
        sem_image: FibsemImage,
        config: AdaptivePolishMillingConfig,
        model: AbstractAdaptivePolishingModel,
        lamella_ap_folder: Path,
        lamella_ap_plots_folder: Path,
        results: DataFrame,
        gis_results_detailed: DataFrame,
    ) -> None:
        # Segmentation
        _logger.info("Starting segmentation")
        prediction = model.predict(sem_image.data, full_size=False)
        _logger.info("Segmentation complete")

        prediction_pixel_size_um = (
            sem_image.metadata.pixel_size.x
            * constants.SI_TO_MICRO
            * sem_image.data.shape[1]
            / prediction.shape[1]
        )

        mask_lamella_clean, mask_gis_clean, mask_crack_clean = gm.clean_prediction(
            prediction,
            additional_labels=(
                SegmentationLabels.GIS,
                SegmentationLabels.CRACK,
            ),
        )
        lamella_area_um2 = gm.get_mask_area_um2(
            mask_lamella_clean, pixel_size_um=prediction_pixel_size_um
        )
        if lamella_area_um2 < config.minimum_lamella_area_um2:
            raise StopEarlyError(
                f"Lamella found was only {lamella_area_um2:.4e} um2, below the threshold of {config.minimum_lamella_area_um2:.4e} um2"
            )
        if mask_gis_clean is None:
            raise StopEarlyError("No GIS found beneath the lamella")

        # Get lamella position
        lamella_bbox = get_bounding_box_scaled_to_image(
            sem_image.data, mask=mask_lamella_clean
        )

        # Measure GIS
        gis_thickness_um = (
            np.sum(
                gm.resize_image(mask_gis_clean, new_shape=sem_image.data.shape),
                axis=0,
            )
            * sem_image.metadata.pixel_size.x
            * constants.SI_TO_MICRO
        )

        lamella_xlims_px = np.round(
            (
                lamella_bbox[1],
                lamella_bbox[3],
            )
        ).astype(np.uint32)

        maximum_side_difference_px = int(
            round(
                config.maximum_side_difference_um
                / (sem_image.metadata.pixel_size.x * constants.SI_TO_MICRO)
            )
        )

        gis_thickness_filtered_um = gm.filter_gis_thickness(
            gis_thickness_um,
            window_size_m=config.window_size_px * sem_image.metadata.pixel_size.x,
            pixel_size_m=sem_image.metadata.pixel_size.x,
        )

        gis_above_threshold = gis_thickness_filtered_um > config.gis_stop_um

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

        min_gis_um = np.nanmin(gis_thickness_filtered_um[xlims_px[0] : xlims_px[1] + 1])
        _logger.info(f"Took {len(gis_thickness_filtered_um)} GIS measurements along x")
        _logger.info(
            "Minimum GIS thickness for milling cycle %i = %.4e um",
            milling_cycle,
            min_gis_um,
        )

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

        # Save results
        total_time = config.milling_interval_s * (milling_cycle + 1)
        results.loc[milling_cycle] = {
            "image": image_name,
            "milling_time_s": total_time,
            "min_GIS_um": min_gis_um,
            "crack_area_um2": crack_area_um2,
        }
        results.to_json(lamella_ap_folder / "GIS_thickness.json")

        # Save GIS thickness
        detailed_results = {
            "image": image_name,
            "milling_time_s": total_time,
            "gis_thickness_um": gis_thickness_um.tolist(),
            "gis_thickness_filtered_um": gis_thickness_filtered_um.tolist(),
            "xlims_px": xlims_px,
        }

        gis_results_detailed.loc[milling_cycle] = detailed_results
        gis_results_detailed.to_json(lamella_ap_folder / "GIS_thickness_detailed.json")

        try:
            # Create plots
            gm.milling_cycle_plot(
                sem_image=sem_image.data,
                first_prediction=prediction,
                clean_prediction=gm.masks_to_labels(
                    lamella_mask=mask_lamella_clean,
                    gis_mask=mask_gis_clean,
                    crack_mask=mask_crack_clean,
                ),
                fib_image=fib_image.data,
                gis_thickness_um=gis_thickness_filtered_um,
                gis_stop_um=config.gis_stop_um,
                crack_area_um2=crack_area_um2,
                min_gis_um=min_gis_um,
                xlims=xlims_px,
                img_name=image_name,
                fib_screenshot=None,
                save_path=lamella_ap_plots_folder / f"{image_name}_plot.png",
            )
        except Exception:
            _logger.error(
                "Exception occurred creating the milling cycle plot", exc_info=True
            )

        centre_m, mask_centre_px = get_centre_points_from_bounding_box(
            bbox=lamella_bbox,
            image=sem_image.data,
            pixel_size_m=sem_image.metadata.pixel_size.x,
        )

        centre_drift_um = (
            math.sqrt(centre_m.x**2 + centre_m.y**2) * constants.SI_TO_MICRO
        )
        # Only a valid check if sem is aligned
        if config.align_sem and AdaptivePolishMillingStrategy._get_drift_too_large(
            centre_drift_um, config=config
        ):
            # Create centring plot if centring is found to be beyond the threshold
            AdaptivePolishMillingStrategy._create_centring_plot(
                sem_image=sem_image,
                mask_lamella_clean=mask_lamella_clean,
                centre_px=mask_centre_px,
                centre_m=centre_m,
                plot_path=lamella_ap_plots_folder
                / f"{image_name}_centring_problem.png",
                bounding_box=lamella_bbox,
            )
            raise StopEarlyError(
                f"Total drift (um) {centre_drift_um:.4e} > threshold {config.maximum_drift_um:.4e} (might be a segmentation problem)"
            )

        if AdaptivePolishMillingStrategy._get_gis_too_thin(min_gis_um, config=config):
            raise StopMillingException(
                f"Minimum GIS thickness (um) {min_gis_um:.4e} < threshold {config.gis_stop_um:.4e} um"
            )

        if AdaptivePolishMillingStrategy._get_crack_too_large(
            crack_area_um2, config=config
        ):
            raise StopMillingException(
                f"Crack area (um2) {crack_area_um2:.4e} > threshold {config.max_crack_area_um2:.4e} um2"
            )

    @staticmethod
    def _mill(
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
        config: AdaptivePolishMillingConfig,
    ) -> None:
        # get pattern - this is where bitmap will come in later
        pattern = stage.pattern.define()

        # adjust milling interval TODO
        next_milling_interval = config.milling_interval_s
        pattern[0].time = next_milling_interval

        # mill
        draw_patterns(microscope=microscope, patterns=pattern)
        try:
            run_milling(
                microscope=microscope,
                milling_current=stage.milling.milling_current,
                milling_voltage=stage.milling.milling_voltage,
                asynch=False,
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
    ) -> None:
        beam_shifts = {
            _: microscope.get("shift", _) for _ in (BeamType.ELECTRON, BeamType.ION)
        }
        _logger.info("Using sem beam shift alignment for adaptive polishing")

        # Take reference images
        sem_image = acquire.new_image(microscope, sem_imaging_settings)

        # Find centre
        _logger.info("Starting segmentation")
        prediction = self.model.predict(sem_image.data, full_size=False)
        _logger.info("Segmentation complete")

        mask_lamella_clean, _, _ = gm.clean_prediction(
            prediction,
            additional_labels=(SegmentationLabels.GIS, SegmentationLabels.CRACK),
        )

        lamella_bbox = get_bounding_box_scaled_to_image(
            sem_image.data,
            mask=mask_lamella_clean,
        )
        centre_m, centre_px = get_centre_points_from_bounding_box(
            lamella_bbox,
            image=sem_image.data,
            pixel_size_m=sem_image.metadata.pixel_size.x,
        )

        # shift beam
        dx, dy = -centre_m.x, -centre_m.y
        microscope.beam_shift(dx, dy, BeamType.ELECTRON)
        _logger.info(
            "Beamshift %s by dx=%.4e, dy=%.4e m", BeamType.ELECTRON.name, dx, dy
        )

        if plot_path is not None:
            AdaptivePolishMillingStrategy._create_centring_plot(
                sem_image=sem_image,
                prediction=prediction,
                mask_lamella_clean=mask_lamella_clean,
                centre_px=centre_px,
                centre_m=centre_m,
                plot_path=plot_path,
                bounding_box=lamella_bbox,
            )
        return beam_shifts

    @staticmethod
    def _create_centring_plot(
        sem_image: FibsemImage,
        mask_lamella_clean: NDArray[np.bool_],
        centre_px: Point,
        centre_m: Point,
        plot_path: Path,
        prediction: typing.Optional[NDArray[np.integer]] = None,
        bounding_box: typing.Optional[tuple[float, float, float, float]] = None,
    ) -> None:
        # Plot centring stuff
        if prediction is not None:
            fig, axs = plt.subplots(1, 2)
            axs = axs.ravel()[::-1]
        else:
            fig, ax = plt.subplots(1, 1)
            axs = [ax]

        _ = axs[0].imshow(sem_image.data, cmap="gray")
        extent = _.get_extent()
        axs[0].imshow(
            # Overlay the cleaned lamella
            mask_lamella_clean,
            cmap=ListedColormap(
                [(0, 0, 0, 0), gm.LABEL_CMAP(SegmentationLabels.LAMELLA.value)]
            ),
            extent=extent,
            alpha=0.5,
        )
        if bounding_box is not None:
            axs[0].add_patch(
                Rectangle(
                    (bounding_box[1], bounding_box[0]),
                    width=bounding_box[3] - bounding_box[1],
                    height=bounding_box[2] - bounding_box[0],
                    edgecolor="red",
                    facecolor="none",
                    alpha=0.5,
                )
            )
        if prediction is not None:
            axs[1].imshow(
                prediction,
                cmap=gm.LABEL_CMAP,
                extent=extent,
                vmin=0,
                vmax=len(gm.LABEL_CMAP.colors),
            )

        for ax in axs:
            # Add centre markers to both
            ax.scatter(
                centre_px.x,
                centre_px.y,
                c="r",
                marker="+",
                label="Lamella Centre",
            )
            ax.scatter(
                sem_image.data.shape[1] // 2,
                sem_image.data.shape[0] // 2,
                c="g",
                marker="+",
                label="Image Centre",
            )
            ax.set_xticks([])
            ax.set_yticks([])

        axs[-1].legend()  # No need to have a duplicate legend

        fig.suptitle(
            rf"Lamella centre (x, y): {centre_m.x * constants.SI_TO_MICRO}, {centre_m.y * constants.SI_TO_MICRO} $\mu m$"
        )
        fig.tight_layout()

        fig.savefig(plot_path)
        plt.close(fig)

    @staticmethod
    def _get_drift_too_large(
        centre_drift_um: float, config: AdaptivePolishMillingConfig
    ) -> bool:
        return centre_drift_um > config.maximum_drift_um

    @staticmethod
    def _get_gis_too_thin(
        min_gis_um: float, config: AdaptivePolishMillingConfig
    ) -> bool:
        # Minumum GIS thickness check
        return min_gis_um < float(config.gis_stop_um)

    @staticmethod
    def _get_crack_too_large(
        crack_area_um2: float, config: AdaptivePolishMillingConfig
    ) -> bool:
        # Total crack area check
        return crack_area_um2 > float(config.max_crack_area_um2)
