from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import typing

# fibsem
from fibsem import (
    acquire,
    conversions,
    constants,
)
from fibsem.milling.base import (
    MillingStrategy,
    MillingStrategyConfig,
)
from fibsem.milling import (
    setup_milling,
    draw_patterns,
    run_milling,
    finish_milling,
)
from fibsem.milling.patterning.patterns2 import (
    TrenchPattern,
    TrenchBitmapPattern,
)
from fibsem.structures import BeamType

# Adaptive polish
import adaptive_polish.gis_measurement as gm
import adaptive_polish.utils as ap_utils
from adaptive_polish.dl_segmentation.sem_lamella_segmentor import SegmentationLabels
from adaptive_polish.centring import (
    get_lamella_bounding_box,
    get_centre_from_bounding_box,
)

if typing.TYPE_CHECKING:
    from os import PathLike
    from numpy.typing import NDArray
    from pandas import DataFrame
    from fibsem.milling.base import FibsemMillingStage
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import FibsemImage, ImageSettings, Point
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
class AdaptivePolishMillingConfig(MillingStrategyConfig):
    model_path: typing.Union[str, PathLike]
    align_sem: bool = True
    milling_interval_s: int = 10
    gis_stop_um: float = 0.2
    max_crack_area_um2: float = 2
    max_milling_cycles: int = 30
    window_size_px: int = 10
    model_generation: typing.Optional[str] = None
    minimum_lamella_area_um2: float = 30.0  # 30μm²
    maximum_drift_um: float = 0.05

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
            "model_generation": self.model_generation,
            "minimum_lamella_area_um2": self.minimum_lamella_area_um2,
            "maximum_drift_um": self.maximum_drift_um,
        }


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
        lamella_ap_folder = lamella_folder / "adaptive_polish"
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
                model_path=model_path,
                generation=self.config.model_generation,
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
            for milling_cycle in range(self.config.max_milling_cycles):
                f_basename = f"{lamella_folder.stem}_AP_img_{milling_cycle:03}"

                # Acquire images
                _logger.info(
                    "Acquiring images for milling cycle %i/%i",
                    milling_cycle,
                    self.config.max_milling_cycles,
                )
                fib_imaging_settings.filename = f"{f_basename}_FIB.tif"
                fib_image = acquire.new_image(microscope, fib_imaging_settings)
                sem_imaging_settings.filename = f"{f_basename}_SEM.tif"
                sem_image = acquire.new_image(microscope, sem_imaging_settings)

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
            _logger.info(f"Stopping milling due to: {e}")
        except StopEarlyError as e:
            # Likely due to something not working correctly (e.g.
            # segmentation issues)
            _logger.warning(f"Stopping milling early due to: {e}")

        except Exception as e:
            _logger.error(f"Stopping due to unexpected exception: {e}")
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
            * (sem_image.data.shape[1] / prediction.shape[1])
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
                f"Lamella found was only {lamella_area_um2:.4e} um2, below the threshold of {config.minimum_lamella_area_um2:.4e} um2 ()"
            )
        if mask_gis_clean is None:
            raise StopEarlyError("No GIS found beneath the lamella")

        # Get lamella position
        centre_m, mask_centre_px, lamella_bbox = (
            AdaptivePolishMillingStrategy._get_lamella_position(
                sem_image, lamella_mask=mask_lamella_clean
            )
        )

        # Measure GIS
        gis_thickness_um = (
            gm.filter_gis_thickness(
                window_size_m=config.window_size_px * sem_image.metadata.pixel_size.x,
                pixel_size_m=sem_image.metadata.pixel_size.x,
            )
            * constants.SI_TO_MICRO
        )
        min_gis_um = np.nanmin(gis_thickness_um)
        _logger.info(f"Took {len(gis_thickness_um)} GIS measurements along x")
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
            "image": f"adapt_mill_img_{milling_cycle:03}",
            "milling_time_s": total_time,
            "min_GIS_um": min_gis_um,
            "crack_area_um2": crack_area_um2,
        }
        results.to_csv(lamella_ap_folder / "GIS_thickness.csv")

        # Save GIS thickness for each window
        for window, gis_thickness in enumerate(gis_thickness_um):
            if gis_thickness > 0:
                gis_results_detailed.loc[len(gis_results_detailed)] = {
                    "image": image_name,
                    "milling_time_s": total_time,
                    "window": window,
                    "gis_windowed_um": gis_thickness,
                }
            else:
                pass

        gis_results_detailed.to_csv(lamella_ap_folder / "GIS_thickness_detailed.csv")

        clean_foreground_prediction = gm.masks_to_labels(
            lamella_mask=mask_lamella_clean,
            gis_mask=mask_gis_clean,
            crack_mask=mask_crack_clean,
        )

        # plots
        gm.milling_cycle_plot(
            sem_image=sem_image.data,
            first_prediction=prediction,
            clean_prediction=clean_foreground_prediction,
            fib_image=fib_image.data,
            gis_thickness_um=gis_thickness_um,
            gis_stop_um=config.gis_stop_um,
            crack_area_um2=crack_area_um2,
            xlims=xlims_px,
            img_name=image_name,
            fib_screenshot=None,
            save_path=lamella_ap_plots_folder / f"{image_name}_plot.png",
            bounding_box=lamella_bbox,
        )

        centre_drift_um = (
            math.sqrt(centre_m.x**2 + centre_m.y**2) * constants.SI_TO_MICRO
        )
        if AdaptivePolishMillingStrategy._get_drift_too_large(
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

        centre_m, centre_px, lamella_bbox = (
            AdaptivePolishMillingStrategy._get_lamella_position(
                sem_image, lamella_mask=mask_lamella_clean
            )
        )

        # shift beam
        dx, dy = centre_m.x, centre_m.y
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
            )

    @staticmethod
    def _create_centring_plot(
        sem_image: FibsemImage,
        mask_lamella_clean: NDArray[np.bool_],
        centre_px: Point,
        centre_m: Point,
        plot_path: Path,
        prediction: typing.Optional[NDArray[np.integer]] = None,
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
    def _get_lamella_position(
        sem_image: NDArray[typing.Any], lamella_mask: NDArray[np.bool_]
    ) -> tuple[
        Point, Point, tuple[int, int, int, int] | tuple[float, float, float, float]
    ]:
        # This does assume square pixels
        if sem_image.data.shape[1] == lamella_mask.shape[1]:
            labels_pixel_size_m = sem_image.metadata.pixel_size.x
        else:
            labels_pixel_size_m = sem_image.metadata.pixel_size.x * (
                sem_image.data.shape[1] / lamella_mask.shape[1]
            )

        bbox = get_lamella_bounding_box(sem_image.data, lamella_mask, None)
        centre_px = get_centre_from_bounding_box(bbox)

        # Convert to microscope image coordinates (0, 0 at centre of image)
        centre_m = conversions.image_to_microscope_image_coordinates(
            centre_px, lamella_mask, labels_pixel_size_m, subpixel_precision=True
        )
        return (
            centre_m,
            centre_px,
            bbox,
        )

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
