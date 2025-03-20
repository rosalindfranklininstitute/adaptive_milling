# Copyright 2024 Rosalind Franklin Institute
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

# This version uses deep learning (unet+pytorch) to segment the gis layer or lamela

import logging
import sys
from pathlib import Path
import adaptive_polish.gis_measurement as gm
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import napari
from importlib.metadata import version

from fibsem import acquire, utils, milling, conversions
from fibsem.structures import BeamType
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import MicroscopeSettings
from fibsem.patterning import BasePattern
from fibsem.detection.detection import AdaptiveLamellaCentre


class AdaptiveMilling:
    config_dict = None

    def __init__(self, ap_config_dict: dict = None):
        logging.info(f"Using adaptive_polish version {version('adaptive_polish')}")

        if ap_config_dict is None:
            ap_config_dict = self.get_default_config_dict()
        else:
            assert isinstance(ap_config_dict, dict)
            self.config_dict = ap_config_dict.copy()

        # Check dictionary is ok
        if not "imaging_settings" not in ap_config_dict:
            ValueError(
                "No imaging_settings in ap_config. Please check protocol yaml file"
            )

        logging.info("Config parameters: %s\n", self.config_dict)

        self.model_path = self.config_dict.get(
            "model_path",
            f"{Path(__file__).parent}/dl_segmentation/2024-02-24_0013_gis_lamela_crack_pytorch_AUnet.ptchkp",
        )
        logging.info(f"Initialising DL model, path:{self.model_path}")
        self.model = gm.load_sem_model(self.model_path)

    @staticmethod
    def get_default_config_dict():
        config_dict = {
            "do_plots": True,
            "fallback_pixel_size_m": 1.8e-8,
            # Milling cycle control
            "milling_interval_s": 30,
            "gis_stop_m": 2e-7,
            "max_milling_cycles": 30,
            "window_size_m": 1e-7,
            "max_crack_area_m2": 2e-12,
            "model_path": f"{Path(__file__).parent}/dl_segmentation/2024-02-24_0013_gis_lamela_crack_pytorch_AUnet.ptchkp",
            "imaging_settings": {
                "electron": {
                    "resolution": [3072, 2048],
                    "hfw": 40.0e-6,
                    "dwell_time": 200.0e-9,
                    "frame_integration": 8,
                },
                "ion": {
                    "resolution": [3072, 2048],
                    "hfw": 40.0e-6,
                    "dwell_time": 200.0e-9,
                    "frame_integration": 8,
                },
            },
            "use_sem_beam_shift_alignment_adaptive_polish": True,
        }

        return config_dict

    def adaptive_polish_run(
        self,
        microscope_in: FibsemMicroscope = None,
        settings_in: MicroscopeSettings = None,
        patterns_in: BasePattern = None,
        viewer: napari.Viewer = None,
    ):
        if microscope_in is not None:
            microscope = microscope_in
            settings = settings_in
        else:
            try:
                logging.info("Initializing arctis thermo micrsocope with fibsem")
                microscope, settings = utils.setup_session(
                    session_path="../temp",
                    config_path=os.path.join(os.getcwd(), "arctis-configuration.yaml"),
                )  # Check config path
                # TODO: Change config_path to point to the correct path of "arctis-configuration.yaml" file

                # Note that setup_session configures logging differently
            except Exception:
                logging.info(
                    "Could not initialise Thermo microscope with fibsem. Defaulting to use the Demo"
                )
                microscope, settings = utils.setup_session(
                    session_path="../temp",
                    manufacturer="Demo",
                    ip_address="localhost",
                    setup_logging=False,
                )

        # Initialise results
        results = pd.DataFrame(
            {
                "image": [],
                "milling_time_s": [],
                "min_GIS_m": [],
                "crack_area_m2": [],
            }
        )

        gis_results_detailed = pd.DataFrame(
            {
                "image": [],
                "milling_time_s": [],
                "window": [],
                "gis_windowed_m": [],
            }
        )

        lamella_folder = Path(settings_in.image.path)
        lamella_ap_folder = Path(f"{lamella_folder}/adaptive_polish")
        lamella_ap_plots_folder = lamella_ap_folder / "plots"
        lamella_ap_sem_folder = lamella_ap_folder / "sem"
        lamella_ap_fib_folder = lamella_ap_folder / "fib"
        lamella_ap_centering_folder = lamella_ap_folder / "centering"

        # Ensure folders exist
        lamella_ap_folder.mkdir(exist_ok=True)
        lamella_ap_plots_folder.mkdir(exist_ok=True)
        lamella_ap_sem_folder.mkdir(exist_ok=True)
        lamella_ap_fib_folder.mkdir(exist_ok=True)
        lamella_ap_centering_folder.mkdir(exist_ok=True)

        # Running ---------------------------------------------------------------------
        # Align with beamshift
        try:
            align_at_adaptive_polish = self.config_dict[
                "use_sem_beam_shift_alignment_adaptive_polish"
            ]
        except KeyError:
            align_at_adaptive_polish = False
            logging.warning(
                "Protocol does not specify if beamshift alignment is to be used"
                " in adaptive polishing. Defaulting to no beamshift alignment. "
                "Please specify use_sem_beam_shift_alignment_adaptive_polish in protocol.yaml"
            )

        if align_at_adaptive_polish is True:
            logging.info("Using sem beam shift alignment for adaptive polishing")

            # Take reference images
            SEM_img, FIB_img = acquire.take_reference_images(
                microscope=microscope_in,
                image_settings=settings_in.image,
            )

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
            microscope.beam_shift(dx, dy, settings.image.beam_type)
            logging.info(f"Beamshift {settings.image.beam_type} by dx={dx}, dy={dy}")

            # Plot centering stuff
            plt.figure()
            plt.imshow(prediction, cmap="gray")
            plt.scatter(
                centre_px.x, centre_px.y, c="r", marker="+", label="lamella_centre"
            )
            plt.scatter(
                SEM_img.data.shape[1] // 2,
                SEM_img.data.shape[0] // 2,
                c="g",
                marker="+",
                label="image_centre",
            )
            plt.legend()
            plt.savefig(lamella_ap_folder / "centering.png")
            plt.close()

        # Set imaging settings according to the adaptive_polish part of the protocol.yaml
        SEM_settings = microscope.get_imaging_settings(BeamType.ELECTRON)
        FIB_settings = microscope.get_imaging_settings(BeamType.ION)
        for key, value in self.config_dict["imaging_settings"]["electron"].items():
            setattr(SEM_settings, key, value)
        for key, value in self.config_dict["imaging_settings"]["ion"].items():
            setattr(FIB_settings, key, value)
        SEM_settings.save = True
        FIB_settings.save = True
        SEM_settings.path = lamella_ap_sem_folder
        FIB_settings.path = lamella_ap_fib_folder

        # Initialise counts
        scan_count = 0
        total_time = 0

        while scan_count <= int(self.config_dict["max_milling_cycles"]):
            f_basename = f"{lamella_folder.stem}_AP_img_{scan_count:03}"

            # Acquire SEM image
            logging.info(
                "Acquiring SEM image for milling cycle "
                f"{scan_count}/{int(self.config_dict['max_milling_cycles'])}"
            )
            SEM_settings.filename = f"{f_basename}_SEM.tif"
            SEM_img = acquire.new_image(microscope, SEM_settings)

            # Acquire FIB image
            logging.info(
                "Acquiring FIB image for milling cycle "
                f"{scan_count}/{int(self.config_dict['max_milling_cycles'])}"
            )
            FIB_settings.filename = f"{f_basename}_FIB.tif"
            FIB_img = acquire.new_image(microscope, FIB_settings)

            if SEM_img is None:
                # If there are no SEM images to look at, exit while loop
                # Shouldn't happen in a real session, this is only for when there are no
                # more example images to look at

                logging.info("No SEM images acquired, exiting loop.")
                break

            # Measure GIS thickness on SEM image ---------------------------------------

            pixel_size_m = SEM_img.metadata.pixel_size.x

            if pixel_size_m:
                logging.info(f"Using pixel size {pixel_size_m} m")
            else:
                pixel_size_m = float(self.config_dict["fallback_pixel_size_m"])
                logging.info(f"Using fallback pixel size of {pixel_size_m} m")

            # Segmentation
            logging.info("Starting segmentation")
            prediction = gm.segment(SEM_img.data)
            logging.info("Segmentation complete")

            mask_gis_clean = gm.clean_prediction(
                prediction,
                pixel_size_m,
            )
            logging.info("Finished cleaning mask")

            if mask_gis_clean is None:
                logging.info("mask_gis_clean is None. Stopping")
            # Is there any GIS?
            if not np.any(mask_gis_clean):
                # No GIS detected
                logging.info("No GIS layer detected in mask_gis_clean. Stopping")
                break  # Shouold we stop or continue?

            # GIS thickness measurement
            GIS_m, xlims = gm.measure_GIS(
                mask_gis_clean,
                float(self.config_dict["window_size_m"]),
                float(pixel_size_m),
            )

            min_GIS_m = np.nanmin(GIS_m)
            logging.info(f"Took {len(GIS_m)} GIS measurements along x")
            logging.info(
                f"Minimum GIS thickness for milling cycle {scan_count} = {min_GIS_m}"
            )

            # Crack detection
            crack_area_m2 = gm.get_crack_area_m2(prediction, pixel_size_m)

            logging.info(
                f"Area of cracks found in milling cycle {scan_count} = {crack_area_m2} m2"
            )

            # Save results
            results.loc[scan_count] = {
                "image": f"adapt_mill_img_{scan_count:03}",
                "milling_time_s": total_time,
                "min_GIS_m": min_GIS_m,
                "crack_area_m2": crack_area_m2,
            }
            results.to_csv(lamella_ap_folder / "GIS_thickness.csv")

            # Save GIS thickness for each window
            for window, gis_thickness in enumerate(GIS_m):
                if gis_thickness > 0:
                    gis_results_detailed.loc[len(gis_results_detailed)] = {
                        "image": f_basename,
                        "milling_time_s": total_time,
                        "window": window,
                        "gis_windowed_m": gis_thickness,
                    }
                else:
                    pass

            gis_results_detailed.to_csv(
                lamella_ap_folder / "GIS_thickness_detailed.csv"
            )

            # Generate plots
            if viewer is not None:
                fib_screenshot = viewer.screenshot()
            else:
                fib_screenshot = None
            gm.milling_cycle_plot(
                sem_image=SEM_img.data,
                first_prediction=prediction,
                clean_prediction=mask_gis_clean,
                fib_image=FIB_img.data,
                gis_thickness_m=GIS_m,
                gis_stop_m=float(self.config_dict["gis_stop_m"]),
                crack_area_m2=crack_area_m2,
                img_name=f_basename,
                fib_screenshot=fib_screenshot,
                save_path=lamella_ap_plots_folder / f"{f_basename}_plot.png",
            )

            # Should we continue?
            if min_GIS_m < float(self.config_dict["gis_stop_m"]):
                logging.info(
                    f"Stopping as minimum GIS (m) {min_GIS_m} < threshold {self.config_dict['gis_stop_m']}"
                )
                break

            if crack_area_m2 > float(self.config_dict["max_crack_area_m2"]):
                logging.info(
                    f"Stopping as crack area (m2) {crack_area_m2} > threshold {self.config_dict['max_crack_area_m2']}"
                )
                break

            # Adjust milling interval
            if scan_count > 0:
                if min_GIS_m <= 1.2 * float(self.config_dict["gis_stop_m"]):
                    new_milling_interval = (
                        int(self.config_dict["milling_interval_s"]) / 2
                    )
                    self.config_dict["milling_interval_s"] = max(
                        new_milling_interval, 10
                    )
                    logging.info(
                        f"Minimum GIS distance is {10e-6 * (min_GIS_m - self.config_dict['gis_stop_m'])} um from target, "
                        f"Reducing the milling interval from {self.config_dict['milling_interval_s']} s to "
                        f"{new_milling_interval} s or 10 s, whichever is bigger."
                    )

            # Mill for a predetermined amount of time
            millmilling_interval_s = int(self.config_dict["milling_interval_s"])
            logging.info(f"Sleeping {millmilling_interval_s} seconds to mill")
            try:
                microscope.run_milling(
                    settings.milling.milling_current,
                    settings.milling.milling_voltage,
                    asynch=True,
                )
                time.sleep(millmilling_interval_s)
                logging.info("Completed milling.")
            except Exception as e:
                logging.error(f"The following error occurred doing milling. {str(e)}")
            finally:
                microscope.stop_milling()  # dont use milling.finish_milling as it would clear patterns

            scan_count += 1
            total_time += int(self.config_dict["milling_interval_s"])

        if self.config_dict["do_plots"] is True:
            plt.figure()
            plt.plot(
                results.milling_time_s,
                results.min_GIS_m,
                label="Minimum GIS thickness (m)",
            )
            plt.xlabel("Milling Time (s)")
            plt.ylabel("GIS Thickness (m)")
            plt.title(f_basename)
            plt.savefig(f"{f_basename}_GIS_thickness.png")
            plt.close()
