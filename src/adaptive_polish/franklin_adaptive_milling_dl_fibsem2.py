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


#from adaptive_polish.franklin_setup import AdaptiveMillingConfig
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

from fibsem import acquire, utils
from fibsem.structures import BeamType
from fibsem import milling
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import MicroscopeSettings
from fibsem.patterning import BasePattern

#adaptive polish configuration setup
#Is it better to run in a class using __init__ and others?

class AdaptiveMilling():
    config_dict=None

    def __init__(self, ap_config_dict: dict = None):

        if ap_config_dict is None:
            ap_config_dict=self.get_default_config_dict()
        else:
            assert type(ap_config_dict)==dict
            self.config_dict=ap_config_dict.copy()

        # Check dictionary is ok
        if not ("imaging_settings" in ap_config_dict):
            ValueError("No imaging_settings in ap_config. Please check protocol yaml file")
        else:
            self._imaging_settings = self.config_dict["imaging_settings"] 

        logging.info("Config parameters: %s\n", self.config_dict)

        self.model_path = self.config_dict.get(
            "model_path",
            f"{Path(__file__).parent}/dl_segmentation/2024-02-24_0013_gis_lamela_crack_pytorch_AUnet.ptchkp"
        )
        logging.info(f"Initialising DL model, path:{self.model_path}")
        gm.init_model_with_path(self.model_path)

    @staticmethod
    def get_default_config_dict():
        config_dict={
            "do_plots": True,
            "fallback_pixel_size_m": 1.8e-8,
            # Milling cycle control
            "milling_interval_s": 30,
            "gis_stop_m": 2e-7,
            "max_milling_cycles": 30,
            "reject_GIS_distance_m": 1e-5,
            "lam_height_min_m": 2e-6,
            "window_size_m": 1e-7,
            "max_crack_area_m2": 2e-12,
            "model_path": f"{Path(__file__).parent}/dl_segmentation/2024-02-24_0013_gis_lamela_crack_pytorch_AUnet.ptchkp",
            "imaging_settings": {
                "electron":{
                    "resolution":[3072,2048],
                    "hfw": 40.0e-6 ,
                    "dwell_time":200.0e-9,
                    "frame_integration":  8
                },
                "ion":{
                    "resolution":[3072,2048],
                    "hfw":40.0e-6,
                    "dwell_time":200.0e-9,
                    "frame_integration":8
                }
            }
        }

        return config_dict

    def adaptive_polish_run(
            self, 
            microscope_in: FibsemMicroscope = None,
            settings_in : MicroscopeSettings = None,
            patterns_in : BasePattern = None,
            viewer: napari.Viewer = None):

        if microscope_in is not None:
            microscope = microscope_in
            settings=settings_in
        else:
            try:
                logging.info("Initializing arctis thermo micrsocope with fibsem")
                microscope, settings = utils.setup_session(session_path="../temp", config_path= os.path.join(os.getcwd(), "arctis-configuration.yaml" )) #Check config path
                #TODO: Change config_path to point to the correct path of "arctis-configuration.yaml" file

                #Note that setup_session configures logging differently
            except:
                logging.info("Could not initialise Thermo microscope with fibsem. Defaulting to use the Demo")
                microscope, settings = utils.setup_session(session_path="../temp", manufacturer="Demo", ip_address="localhost", setup_logging=False)

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
        if lamella_ap_folder.exists() is False:
            lamella_ap_folder.mkdir()
            Path(f"{lamella_ap_folder}/plots").mkdir()
            Path(f"{lamella_ap_folder}/sem").mkdir()
            Path(f"{lamella_ap_folder}/fib").mkdir()        

        # Running ---------------------------------------------------------------------

        # Initialise counts
        scan_count = 0
        total_time = 0

        while scan_count <= int(self.config_dict["max_milling_cycles"]):
            # Set detector
            #microscope.detector.type.value = "ETD"
            # fibsem detector is set in configuration file "detector_type". No need to set here

            #logging.info(f"Changing to {microscope.detector.type.value} detector")
            # place pattern (temporary because run_milling clears them!)
            #milling.draw_patterns(microscope, patterns_in)
            # Acquire images
            imgs = {}
            SEM_img=None
            FIB_img=None

            img_sett0 = None
            f_basename= f"adapt_mill_img_{scan_count:03}"

            for img_set_name in self._imaging_settings:
                logging.info(
                    f"Acquiring image with {img_set_name} settings for milling cycle "
                    f"{scan_count}/{int(self.config_dict['max_milling_cycles'])}"
                )
                #img_name = f"{self.config_dict.folders['lamella_folder'].stem}_img_{scan_count:03}"
                #logging.info(f"Image is {img_name}")

                if 'SEM' in img_set_name or "electron" in img_set_name:
                    logging.info(f"SEM (electron) image to be acquired")
                    #settings.image.beam_type = BeamType.ELECTRON
                    img_settings = microscope.get_imaging_settings(BeamType.ELECTRON)
                    img_sett0 = self._imaging_settings[img_set_name]
                    # note that by new_image() can save images
                    # but the img_settings.save must be True, and will save to f"{settings.filename}_eb" .tif
                    # TODO: Save to a AP folder?
                    img_settings.path = Path(f"{lamella_ap_folder}/sem")

                elif "FIB" in img_set_name or "ion" in img_set_name:
                    logging.info(f"FIB (ion) image to be acquired")
                    #settings.image.beam_type = BeamType.ION
                    img_settings = microscope.get_imaging_settings(BeamType.ION)
                    img_sett0 = self._imaging_settings[img_set_name]
                    img_settings.path = Path(f"{lamella_ap_folder}/fib")
                
                else:
                    raise ValueError(f"img_set0: {img_set_name} is invalid")

                # TODO: Save to a AP folder, probably doing automatically
                
                img_settings.resolution= img_sett0.get("resolution", img_settings.resolution)
                img_settings.hfw= img_sett0.get("hfw", img_settings.hfw)
                img_settings.dwell_time= img_sett0.get("dwell_time", img_settings.dwell_time)
                img_settings.frame_integration= img_sett0.get("frame_integration", img_settings.frame_integration)
                # img_settings.path = settings_in.image.path
                img_settings.filename = f"{f_basename}_{img_set_name}.tif"

                #TODO. Check the settings_in and microscope_in paramaters top figure out where and how to generate a filename
                img_settings.save=True

                logging.info("Acquiring new image, and saving")
                img = acquire.new_image(microscope, img_settings)

                img_path = img.get_save_path()
                
                imgs[img_set_name]= img

                if 'SEM' in img_set_name or "electron" in img_set_name:
                    SEM_img=img
                elif "FIB" in img_set_name or "ion" in img_set_name:
                    FIB_img=img

                #please note that the imgs will not contain AdornedImage but FibsemImage object
                #Depending on the settings.save, this image is automatically saved


            #microscope.detector.type.value = "ETD"
            #logging.info(f"Changing to {microscope.detector.type.value} detector")

            if SEM_img is None:
                # If there are no SEM images to look at, exit while loop
                # Shouldn't happen in a real session, this is only for when there are no
                # more example images to look at

                logging.info("No SEM images acquired, exiting loop.")
                break

            # Measure GIS thickness on SEM image ---------------------------------------

            #SEM_adornimg = imgs["SEM"]  # find the SEM images in RAM to measure GIS
            #pixel_size_m = gm.get_pixel_width(SEM_adornimg)
            #Try to get pixel size:
            
            pixel_size_m = SEM_img.metadata.pixel_size.x #untested

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
                float(self.config_dict["lam_height_min_m"]),
                float(self.config_dict["reject_GIS_distance_m"])
            )
            logging.info(
                f"Cleaned mask to only include GIS "
                f"{self.config_dict['reject_GIS_distance_m']} m below "
                f"GIS-lamella transition"
            )

            if mask_gis_clean is None:
                logging.info("mask_gis_clean is None. Stopping")
            # Is there any GIS?
            if not np.any(mask_gis_clean):
                #No GIS detected
                logging.info("No GIS layer detected in mask_gis_clean. Stopping")
                break #Shouold we stop or continue?

            # GIS thickness measurement
            #GIS_m = gm.measure_GIS(mask_gis_clean, self.config_dict["window_size_m"], pixel_size_m)
            GIS_m, xlims = gm.measure_GIS(
                mask_gis_clean,
                float(self.config_dict["window_size_m"]),
                float(pixel_size_m)
            )

            min_GIS_m = np.nanmin(GIS_m)
            logging.info(f"Took {len(GIS_m)} GIS measurements along x")
            logging.info(f"Minimum GIS thickness for milling cycle {scan_count} = {min_GIS_m}")

            # Crack detection
            #crack_area_m2 = gm.find_cracks(prediction, pixel_size_m)
            crack_area_m2 = gm.get_crack_area_m2(prediction, pixel_size_m, *xlims)

            logging.info(
                f"Area of cracks found in milling cycle {scan_count} = {crack_area_m2} m2"
            )

            # Save results
            results.loc[scan_count] = {
                "image": img_path,
                "milling_time_s": total_time,
                "min_GIS_m": min_GIS_m,
                "crack_area_m2": crack_area_m2,
            }
            results.to_csv( f"{lamella_ap_folder}/GIS_thickness.csv" )

            # Save GIS thickness for each window
            for window, gis_thickness in enumerate(GIS_m):
                if gis_thickness > 0:
                    gis_results_detailed.loc[len(gis_results_detailed)] = {
                        "image": img_path,
                        "milling_time_s": total_time,
                        "window": window,
                        "gis_windowed_m": gis_thickness,
                    }
                else:
                    pass
            # gis_results_detailed.to_csv(
            #     f"{config.folders['lamella_folder']}/GIS_thickness_detailed.csv"
            # )

            gis_results_detailed.to_csv( f"{lamella_ap_folder}/GIS_thickness_detailed.csv" )

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
                img_name=img_path,
                fib_screenshot=fib_screenshot,
                save_path=f"{lamella_ap_folder}/plots/{Path(img_path).stem}_plot.png",
            )

            # Should we continue?
            if min_GIS_m < float(self.config_dict["gis_stop_m"]):
                logging.info(
                    f"Stopping as minimum GIS (m) {min_GIS_m} < threshold {self.config_dict['gis_stop_m']}"
                )
                break

            if crack_area_m2 > float(self.config_dict["max_crack_area_m2"]):
                logging.info(
                    f"Stopping as crack area (m2) {crack_area_m2} > threshold { self.config_dict['max_crack_area_m2']}"
                )
                break

            # Continue milling
            #milling.run_milling(microscope, settings.milling.milling_current, settings.milling.milling_voltage)
            
            #Mill for a predetermined amoutn of time
            millmilling_interval_s = int(self.config_dict['milling_interval_s'])
            logging.info(f"Sleeping {millmilling_interval_s} seconds to mill")
            #milling.run_milling(microscope, settings.milling.milling_current, settings.milling.milling_voltage, asynch=True)
            try:
                microscope.run_milling(settings.milling.milling_current, settings.milling.milling_voltage, asynch=True)
                time.sleep(millmilling_interval_s)
                logging.info("Completed milling.")
            except Exception as e:
                logging.error(f"The following error occurred doing milling. {str(e)}")
            finally:
                microscope.stop_milling() # dont use milling.finish_milling as it would clear patterns

            # microscope.patterning.start()
            # logging.info(f"Sleeping {_ap_config['milling_interval_s']} seconds to mill")
            # time.sleep(_ap_config["milling_interval_s"])
            # microscope.patterning.stop()

            # run milling (fibsem)
            # start = time.time()
            # pause_milling = start + _ap_config["milling_interval_s"]
            # while True:
            #     if time.time() > pause_milling:
            #         break
            #     milling.run_milling(microscope, settings.milling.milling_current, milling_voltage=settings.milling.milling_voltage)
            #     test_mill()

            # t = threading.Thread(target=test_mill) #, args=(microscope, settings.milling.milling_current, settings.milling.milling_voltage))
            # print(_ap_config['milling_interval_s'])
            # t.start()
            # time.sleep(_ap_config['milling_interval_s'])
            # t.join()

            # p = multiprocessing.Process(target=test_mill)
            # p.start()
            # p.join(30)
            # print("Trying to stop milling")
            # if p.is_alive():
            #     p.terminate()
            #     p.join()
            # print("no more milling")

            # logging.info(f"Sleeping {_ap_config['milling_interval_s']} seconds to mill")
            # time.sleep(_ap_config["milling_interval_s"])

            # Adjust milling interval
            if min_GIS_m <= 1.2 * float(self.config_dict["gis_stop_m"]):
                self.config_dict["milling_interval_s"] = int(self.config_dict["milling_interval_s"]) / 2

            scan_count += 1
            total_time += int(self.config_dict["milling_interval_s"])

        # If we reached the stopping condition of min lamella thickness or max number of cycles
        # microscope.imaging.set_active_view(2)
        # microscope.patterning.stop()

        # finish milling (fibsem)
        # LMAP: Not sure but this line below may be needed
        # milling.finish_milling(microscope, microscope.system.ion.beam.beam_current)



        if self.config_dict["do_plots"] is True:
            plt.figure()
            plt.plot(
                results.milling_time_s, results.min_GIS_m, label="Minimum GIS thickness (m)"
            )
            plt.xlabel("Milling Time (s)")
            plt.ylabel("GIS Thickness (m)")
            #plt.title(f"Lamella {config.folders['lamella_folder'].stem}")
            plt.title(f_basename)
            #plt.savefig(f"{config.folders['lamella_folder']}/GIS_thickness.png")
            plt.savefig(f"{f_basename}_GIS_thickness.png")
            plt.close()