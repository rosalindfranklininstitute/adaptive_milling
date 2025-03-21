from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
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
from fibsem.detection.detection import AdaptiveLamellaCentre

# Adaptive polish
import adaptive_polish.gis_measurement as gm
import adaptive_polish.utils as ap_utils

if typing.TYPE_CHECKING:
    from os import PathLike
    from fibsem.milling.base import FibsemMillingStage
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import ImageSettings


@dataclass
class AdaptivePolishMillingConfig(MillingStrategyConfig):
    model_path: str | PathLike[str]
    align_sem: bool = True
    milling_interval_s: int = 10
    gis_stop_um: float = 0.2
    max_crack_area_um2: float = 2
    max_milling_cycles: int = 30
    window_size_px: int = 10
    model_generation: str | None = None

    _advanced_attributes = []

    @staticmethod
    def from_dict(d: dict) -> "AdaptivePolishMillingConfig":
        return AdaptivePolishMillingConfig(**d)

    def to_dict(self):
        return {
            "model_path": self.model_path,
            "align_sem": self.align_sem,
            "milling_interval_s": self.milling_interval_s,
            "gis_stop_um": self.gis_stop_um,
            "max_milling_cycles": self.max_milling_cycles,
            "window_size_px": self.window_size_px,
            "max_crack_area_um2": self.max_crack_area_um2,
            "model_generation": self.model_generation,
        }


@dataclass
class AdaptivePolishMillingStrategy(MillingStrategy):
    name: str = "AdaptivePolishing"
    fullname: str = "Adaptive polishing according to GIS thickness"

    def __init__(self, config: AdaptivePolishMillingConfig = None):
        self.config = config or AdaptivePolishMillingConfig()
        self.model = None

    def to_dict(self):
        return {"name": self.name, "config": self.config.to_dict()}

    @staticmethod
    def from_dict(d: dict) -> "AdaptivePolishMillingStrategy":
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
        logging.info(f"Running {self.fullname} for {stage.name}")

        # setup milling
        setup_milling(microscope=microscope, milling_stage=stage)
        fib_imaging_settings = microscope.get_imaging_settings(BeamType.ION)
        sem_imaging_settings = microscope.get_imaging_settings(BeamType.ELECTRON)
        lamella_folder = Path(fib_imaging_settings.path)
        lamella_ap_folder = lamella_folder / "adaptive_polish"
        if lamella_ap_folder.is_dir() is True:
            logging.info(
                "Lamella folder %s already exists, some data may be overwritten.",
                lamella_ap_folder,
            )
        else:
            lamella_ap_folder.mkdir()

        lamella_ap_plots_folder, lamella_ap_sem_folder, lamella_ap_fib_folder = (
            ap_utils.ensure_subdirectories(lamella_ap_folder, "plots", "sem", "fib")
        )

        # load model
        if self.model is None:
            if not self.config.model_path.is_file():
                raise FileNotFoundError(f"Failed to find '{self.config.model_path}'")
            self.model = gm.load_sem_model(
                model_path=self.config.model_path,
                generation=self.config.model_generation,
            )

        # align SEM
        if self.config.align_sem is True:
            self._align_beam(
                microscope=microscope,
                sem_imaging_settings=sem_imaging_settings,
                plot_path=lamella_ap_folder / "centering.png",
            )

        # Set lamella folders for saving images
        fib_imaging_settings.save = True
        sem_imaging_settings.save = True
        fib_imaging_settings.path = lamella_ap_fib_folder
        sem_imaging_settings.path = lamella_ap_sem_folder

        self._mill(
            microscope=microscope,
            fib_imaging_settings=fib_imaging_settings,
            sem_imaging_settings=sem_imaging_settings,
            stage=stage,
            lamella_folder=lamella_folder,
            lamella_ap_folder=lamella_ap_folder,
            lamella_ap_plots_folder=lamella_ap_plots_folder,
        )

    def _mill(
        self,
        microscope: FibsemMicroscope,
        fib_imaging_settings: ImageSettings,
        sem_imaging_settings: ImageSettings,
        stage: FibsemMillingStage,
        lamella_folder: Path,
        lamella_ap_folder: Path,
        lamella_ap_plots_folder: Path,
    ) -> None:
        # setup results
        results, gis_results_detailed = ap_utils.setup_results_df()

        # Initialise counts
        milling_cycle = 0
        total_time = 0

        # run adaptive polishing
        while milling_cycle <= int(self.config.max_milling_cycles):
            f_basename = f"{lamella_folder.stem}_AP_img_{milling_cycle:03}"

            # Acquire images
            logging.info(
                f"Acquiring images for milling cycle {milling_cycle}/{self.config.max_milling_cycles}"
            )
            fib_imaging_settings.filename = f"{f_basename}_FIB.tif"
            FIB_img = acquire.new_image(microscope, fib_imaging_settings)
            sem_imaging_settings.filename = f"{f_basename}_SEM.tif"
            SEM_img = acquire.new_image(microscope, sem_imaging_settings)

            # Segmentation
            logging.info("Starting segmentation")
            prediction = self.model.predict(SEM_img.data, fullsize=False)
            logging.info("Segmentation complete")

            prediction_pixel_size_m = SEM_img.metadata.pixel_size.x * (
                SEM_img.data.shape[1] / prediction.shape[1]
            )

            try:
                mask_gis_clean, mask_lamella_clean, mask_crack_clean = (
                    gm.clean_prediction(
                        prediction,
                        pixel_size_m=prediction_pixel_size_m,
                        minimum_lamella_size_m2=30e-6,  # 30μm²
                    )
                )
            except ValueError as e:
                logging.warning("Failed to clean GIS prediction: %s", str(e))
                break

            # Measure GIS
            gis_thickness_um = (
                gm.filter_gis_thickness(
                    np.sum(
                        gm.resize_image(mask_gis_clean, new_shape=SEM_img.data.shape),
                        axis=0,
                    ),
                    window_size_m=self.config.window_size_m,
                    pixel_size_m=SEM_img.metadata.pixel_size.x,
                )
                * constants.SI_TO_MICRO
            )
            min_gis_um = np.nanmin(gis_thickness_um)
            logging.info(f"Took {len(gis_thickness_um)} GIS measurements along x")
            logging.info(
                f"Minimum GIS thickness for milling cycle {milling_cycle} = {min_gis_um}"
            )

            if mask_crack_clean is None:
                crack_area_um2 = 0
            else:
                crack_area_um2 = gm.get_mask_area_um2(mask_crack_clean)

            logging.info(
                f"Area of cracks found in milling cycle {milling_cycle} = "
                f"{crack_area_um2} um2"
            )

            # Save results
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
                        "image": f_basename,
                        "milling_time_s": total_time,
                        "window": window,
                        "gis_windowed_um": gis_thickness,
                    }
                else:
                    pass

            gis_results_detailed.to_csv(
                lamella_ap_folder / "GIS_thickness_detailed.csv"
            )

            clean_foreground_prediction = gm.masks_to_labels(
                lamella_mask=mask_lamella_clean,
                gis_mask=mask_gis_clean,
                crack_mask=mask_crack_clean,
            )

            # plots
            gm.milling_cycle_plot(
                sem_image=SEM_img.data,
                first_prediction=prediction,
                clean_prediction=clean_foreground_prediction,
                fib_image=FIB_img.data,
                gis_thickness_um=gis_thickness_um,
                gis_stop_um=self.config.gis_stop_um,
                crack_area_um2=crack_area_um2,
                img_name=f_basename,
                fib_screenshot=None,
                save_path=lamella_ap_plots_folder / f"{f_basename}_plot.png",
            )

            # should we continue?
            if min_gis_um < float(self.config.gis_stop_um):
                logging.info(
                    f"Stopping as minimum GIS (um) {min_gis_um} < threshold "
                    f"{self.config.gis_stop_um} um"
                )
                break

            if crack_area_um2 > float(self.config.max_crack_area_um2):
                logging.info(
                    f"Stopping as crack area (um2) {crack_area_um2} > threshold "
                    f"{self.config.max_crack_area_um2} um2"
                )
                break

            # get pattern - this is where bitmap will come in later
            pattern = stage.pattern.define()

            # adjust milling interval TODO
            next_milling_interval = self.config.milling_interval_s
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
                logging.info("Completed milling.")
            except Exception as e:
                logging.error(f"The following error occurred doing milling. {str(e)}")
            finally:
                microscope.stop_milling()  # dont use milling.finish_milling as it would clear patterns

            # Increment counters
            milling_cycle += 1
            total_time += next_milling_interval

        gm.summary_gis_plot(
            results=results,
            save_path=lamella_ap_folder / f"{lamella_folder.stem}_GIS_thickness.png",
        )

        # finish milling (clear patterns, restore imaging current)
        finish_milling(
            microscope=microscope,
            imaging_current=microscope.system.ion.beam.beam_current,
            imaging_voltage=microscope.system.ion.beam.voltage,
        )

    def _align_beam(
        self,
        microscope: FibsemMicroscope,
        sem_imaging_settings: ImageSettings,
        plot_path: Path,
    ) -> None:
        logging.info("Using sem beam shift alignment for adaptive polishing")

        # Take reference images
        SEM_img = acquire.new_image(microscope, sem_imaging_settings)

        # Find center
        logging.info("Starting segmentation")
        prediction = self.model.predict(SEM_img.data)
        logging.info("Segmentation complete")
        feature = AdaptiveLamellaCentre()
        centre_px = feature.detect(SEM_img.data, prediction, None)

        # Convert to microscope image coordinates (0, 0 at centre of image)
        centre_m = conversions.image_to_microscope_image_coordinates(
            centre_px, SEM_img.data, SEM_img.metadata.pixel_size.x
        )

        # shift beam
        dx, dy = centre_m.x, centre_m.y
        microscope.beam_shift(dx, dy, BeamType.ELECTRON)
        logging.info(f"Beamshift {BeamType.ELECTRON} by dx={dx}, dy={dy}")

        # Plot centering stuff
        plt.figure()
        plt.imshow(prediction, cmap="gray")
        plt.scatter(centre_px.x, centre_px.y, c="r", marker="+", label="lamella_centre")
        plt.scatter(
            SEM_img.data.shape[1] // 2,
            SEM_img.data.shape[0] // 2,
            c="g",
            marker="+",
            label="image_centre",
        )
        plt.legend()
        plt.savefig(plot_path)
        plt.close()
