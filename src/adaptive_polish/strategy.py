import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

# fibsem
from fibsem import (
    acquire,
    conversions,
)
from fibsem.milling.base import (
    MillingStrategy,
    MillingStrategyConfig,
    FibsemMillingStage
)
from fibsem.milling import (
    setup_milling,
    draw_patterns,
    run_milling,
    finish_milling,
)
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.patterning.patterns2 import (
    TrenchPattern,
    TrenchBitmapPattern,
)
from fibsem.structures import (
    BeamType,
    ImageSettings,
)
from fibsem.detection.detection import AdaptiveLamellaCentre

# Adaptive polish
import adaptive_polish.gis_measurement as gm
import adaptive_polish.utils as ap_utils


@dataclass
class AdaptivePolishMillingConfig(MillingStrategyConfig):
    align_sem: bool = True
    milling_interval_s: int = 10
    gis_stop_um: float = 0.2
    max_crack_area_um2: float = 2
    max_milling_cycles: int = 30
    window_size_px: int = 10
    model_path: str = "abc"  #TODO add support in FibsemMillingWidget.set_milling_strategy_ui for path inputs

    _advanced_attributes = []

    @staticmethod
    def from_dict(d: dict) -> "AdaptivePolishMillingConfig":
        return AdaptivePolishMillingConfig(**d)

    def to_dict(self):
        return {
            "align_sem": self.align_sem,
            "milling_interval_s": self.milling_interval_s,
            "gis_stop_um": self.gis_stop_um,
            "max_milling_cycles": self.max_milling_cycles,
            "window_size_px": self.window_size_px,
            "max_crack_area_um2": self.max_crack_area_um2,
            "model_path": self.model_path,
        }


@dataclass
class AdaptivePolishMillingStrategy(MillingStrategy):
    name: str = "AdaptivePolishing"
    fullname: str = "Adaptive polishing according to GIS thickness"

    def __init__(self, config: AdaptivePolishMillingConfig = None):
        self.config = config or AdaptivePolishMillingConfig()

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
        parent_ui = None,  # what does this do
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
        setup_milling(
            microscope=microscope,
            milling_stage=stage
        )
        fib_imaging_settings = microscope.get_imaging_settings(BeamType.ION)
        sem_imaging_settings = microscope.get_imaging_settings(BeamType.ELECTRON)
        lamella_folder = Path(fib_imaging_settings.path)  # more robust way of setting lamella folder?
        if Path(f"{lamella_folder}/adaptive_polish").is_dir() is True:
            logging.info(f"Lamella folder {lamella_folder}/adaptive_polish already exists, some data may be overwritten.")
        ap_utils.setup_lamella_ap_folders(
            lamella_folder=lamella_folder
        )

        # setup results
        results, gis_results_detailed = ap_utils.setup_results_df()

        # load model
        gm.init_model_with_path(model_path=self.config.model_path)

        # align SEM
        if self.config.align_sem is True:
            logging.info("Using sem beam shift alignment for adaptive polishing")

            # Take reference images
            SEM_img = acquire.new_image(microscope, sem_imaging_settings)

            # Find center
            logging.info("Starting segmentation")
            prediction = gm.segment(SEM_img.data)
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
            plt.scatter(SEM_img.data.shape[1]//2, SEM_img.data.shape[0]//2, c="g", marker="+", label="image_centre")
            plt.legend()
            plt.savefig(f"{lamella_folder}/adaptive_polish/centering.png")
            plt.close()

        # Set lamella folders for saving images
        fib_imaging_settings.save = True
        sem_imaging_settings.save = True
        fib_imaging_settings.path = Path(f"{lamella_folder}/adaptive_polish/fib")
        sem_imaging_settings.path = Path(f"{lamella_folder}/adaptive_polish/sem")

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
            prediction = gm.segment(SEM_img.data)
            logging.info("Segmentation complete")

            mask_gis_clean = gm.clean_prediction(
                prediction,
                pixel_size_m=SEM_img.metadata.pixel_size.x,
            )

            # Is there any GIS?
            if not np.any(mask_gis_clean):
                # No GIS detected
                logging.info("No GIS layer detected in mask_gis_clean. Stopping")
                break

            # Measure GIS
            GIS_um, xlims = gm.measure_GIS(
                mask_gis_clean=mask_gis_clean,
                window_size_px=self.config.window_size_px,
                pixel_size_m=SEM_img.metadata.pixel_size.x
            )
            min_GIS_um = np.nanmin(GIS_um)
            logging.info(f"Took {len(GIS_um)} GIS measurements along x")
            logging.info(f"Minimum GIS thickness for milling cycle {milling_cycle} = {min_GIS_um}")

            # Crack TODO
            crack_area_um2 = gm.get_crack_area_um2(
                prediction=prediction,
                pixel_size_m=SEM_img.metadata.pixel_size.x
            )

            logging.info(
                f"Area of cracks found in milling cycle {milling_cycle} = "
                f"{crack_area_um2} um2"
            )

            # Save results
            results.loc[milling_cycle] = {
                "image": f"adapt_mill_img_{milling_cycle:03}",
                "milling_time_s": total_time,
                "min_GIS_um": min_GIS_um,
                "crack_area_um2": crack_area_um2,
            }
            results.to_csv(f"{lamella_folder}/adaptive_polish/GIS_thickness.csv")

            # Save GIS thickness for each window
            for window, gis_thickness in enumerate(GIS_um):
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
                f"{lamella_folder}/adaptive_polish/GIS_thickness_detailed.csv"
            )

            # plots
            gm.milling_cycle_plot(
                sem_image=SEM_img.data,
                first_prediction=prediction,
                clean_prediction=mask_gis_clean,
                fib_image=FIB_img.data,
                gis_thickness_um=GIS_um,
                gis_stop_um=self.config.gis_stop_um,
                crack_area_um2=crack_area_um2,
                img_name=f_basename,
                fib_screenshot=None,
                save_path=f"{lamella_folder}/adaptive_polish/plots/{f_basename}_plot.png",
            )

            # should we continue?
            if min_GIS_um < float(self.config.gis_stop_um):
                logging.info(
                    f"Stopping as minimum GIS (um) {min_GIS_um} < threshold "
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
            draw_patterns(
                microscope=microscope,
                patterns=pattern
            )
            try:
                run_milling(
                    microscope=microscope,
                    milling_current=stage.milling.milling_current,
                    milling_voltage=stage.milling.milling_voltage,
                    asynch=False
                )
                logging.info("Completed milling.")
            except Exception as e:
                logging.error(f"The following error occurred doing milling. {str(e)}")
            finally:
                microscope.stop_milling() # dont use milling.finish_milling as it would clear patterns

            # Increment counters
            milling_cycle += 1
            total_time += next_milling_interval

        gm.summary_gis_plot(results=results, lamella_folder=lamella_folder)

        # finish milling (clear patterns, restore imaging current)
        finish_milling(
            microscope=microscope,
            imaging_current=microscope.system.ion.beam.beam_current,
            imaging_voltage=microscope.system.ion.beam.voltage,
        )
