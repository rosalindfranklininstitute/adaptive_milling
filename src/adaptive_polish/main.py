# Copyright 2025 Rosalind Franklin Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import logging
from importlib.metadata import version
from pathlib import Path
import pandas as pd

# fibsem
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import (
    draw_pattern,
    run_milling,
    finish_milling,
)
from fibsem.milling.patterning.patterns2 import (
    TrenchPattern,
    TrenchBitmapPattern,
)
from fibsem.milling.base import FibsemMillingStage

import gis_measurement as gm

class AdaptivePolish():
    def __init__(self, config: dict = None):
        logging.info(f"Using adaptive_polish version {version('adaptive_polish')}")
        self.config = config
        logging.info("Config parameters: %s\n", self.config)

        # load model
        #TODO load DL model

    def adaptive_polish_run(
        self,
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
    ):
        self.lamella_folder = Path(stage.imaging.filename)
        self._setup_folders(lamella_folder=self.lamella_folder)
        self._setup_results_df()

        # Optional centering
        if self.config["align_sem"]:
            self.SEM_alignment()

        # Initialise counts
        milling_cycle = 0
        total_time = 0

        while milling_cycle <= self.config["max_milling_cycles"]:
            ap_basename = f"{self.lamella_folder.stem}_AP_img_{milling_cycle:03}"
            # TODO:Acquire SEM image
            # Create a temporary SEM_settings ImageSettings from self.config and
            # apply the filename based on ap_basename

            # TODO:Acquire FIB image
            # as above
            # Can we use acquire.take_reference_images??

            # TODO: Prediction
            SEM_img = self.segment_SEM()
            prediction = gm.segment(SEM_img.data)
            cleaned_prediction = gm.clean_prediction(
                prediction=prediction,
                pixel_size_m=SEM_img.metadata.pixel_size.x
            )

            # TODO: GIS measurement

            # TODO: Crack detection

            # TODO: Save results

            # TODO: plotting

            # TODO: adjust milling time

            # TODO check if we should continue
            continue_polishing = self._check_continue_polishing()

            if continue_polishing:
                # TODO: draw pattern
                if milling_cycle == 0:  # first milling cycle get existing pattern
                    pattern = stage.pattern
                else:
                    pattern = self.get_pattern()
                draw_pattern(microscope=microscope, pattern=pattern)

                # TODO: mill
                run_milling(
                    microscope=microscope,
                    milling_current=stage.milling.milling_current,
                    milling_voltage=stage.milling.milling_voltage,
                    asynch=False,
                )

            # finish milling (clear patterns, restore imaging current)
            finish_milling(
                microscope=microscope,
                imaging_current=microscope.system.ion.beam.beam_current,
                imaging_voltage=microscope.system.ion.beam.voltage,
            )


    def _setup_folders(
        self,
        lamella_folder: Path,
    ) -> None:
        """Sets up the folders for adaptive polish in the lamella folder (path)

        Args:
            lamella_folder (Path): Path to the lamella folder
        """
        Path(f"{lamella_folder}/plots").mkdir()
        Path(f"{lamella_folder}/sem").mkdir()
        Path(f"{lamella_folder}/fib").mkdir()
        Path(f"{lamella_folder}/centering").mkdir()

    def _setup_results_df(self):
        """Sets up dataframes to hold results for each milling cycle and the summary
        """
        self.results = pd.DataFrame(
            {
                "image": [],
                "milling_time_s": [],
                "min_GIS_m": [],
                "crack_area_m2": [],
            }
        )

        self.gis_results_detailed = pd.DataFrame(
            {
                "image": [],
                "milling_time_s": [],
                "window": [],
                "gis_windowed_m": [],
            }
        )

    def SEM_alignment(self):
        pass
        # Take reference images
        # SEM_img, FIB_img = acquire.take_reference_images(
        #     microscope=microscope_in,
        #     image_settings=imaging_settings,
        # )

        # # Find center
        # logging.info("Starting segmentation")
        # prediction = gm.segment(SEM_img.data)
        # logging.info("Segmentation complete")
        # feature = AdaptiveLamellaCentre()
        # centre_px = feature.detect(SEM_img.data, prediction, None)

        # # Convert to microscope image coordinates (0, 0 at centre of image)
        # centre_m = conversions.image_to_microscope_image_coordinates(
        #     centre_px, SEM_img.data, SEM_img.metadata.pixel_size.x
        # )

        # # shift beam
        # dx, dy = centre_m.x, centre_m.y
        # microscope.beam_shift(dx, dy, BeamType.ELECTRON)
        # logging.info(f"Beamshift {BeamType.ELECTRON} by dx={dx}, dy={dy}")

        # # Plot centering stuff
        # plt.figure()
        # plt.imshow(prediction, cmap="gray")
        # plt.scatter(centre_px.x, centre_px.y, c="r", marker="+", label="lamella_centre")
        # plt.scatter(SEM_img.data.shape[1]//2, SEM_img.data.shape[0]//2, c="g", marker="+", label="image_centre")
        # plt.legend()
        # plt.savefig(f"{lamella_ap_folder}/centering.png")
        # plt.close()

    def segment_SEM(self):
        pass

    def get_pattern(self):
        """Returns a pattern to mill. If adaptive mode is off in the config
        (TODO add this to config), then a TrenchPattern will be returned with
        optional registration (i.e. moving the centre of the pattern to compensate
        for drift between SEM and FIB). If adaptive mode is on then a TrenchBitmapPattern
        is returned.
        """
        pass

    def _check_continue_polishing(
        self,
        gis_thickness_um,
        crack_area_um2,
        milling_cycle
    ):
        """Checks that the GIS is above threshold and crack below threshold"""
        if gis_thickness_um <= self.config["gis_stop_um"]:
            logging.info(
                f"Milling cycle {milling_cycle}: GIS thickness is {gis_thickness_um} um, stopping as this is"
                f" below stop criterion {self.config['gis_stop_um']} um"
            )
            continue_polishing = False
        if crack_area_um2 >= self.config["max_crack_area_um2"]:
            logging.info(
                f"Milling cycle {milling_cycle}: Crack area detected is {crack_area_um2} um2, stopping as "
                f"this is above stop criterion {self.config['max_crack_area_um2']}"
            )
            continue_polishing = False
        else:
            logging.info(
                f"Milling cycle {milling_cycle}: GIS {gis_thickness_um} um, "
                f"Crack area {crack_area_um2} um2"
            )
            continue_polishing = True
        return continue_polishing
