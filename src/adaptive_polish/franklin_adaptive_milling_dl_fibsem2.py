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

_ap_config=None

# def test_mill():
#     print("I AM MILLING!!!!!!")
#     start = time.time()
#     for i in range(100):
#         print(time.time() - start)
#         time.sleep(1)

#adaptive polish configuration setup
#Is it better to run in a class using __init__ and others?
def set_config(ap_config_dict):
    global _ap_config

    _ap_config=ap_config_dict.copy()

    logging.info("Config parameters: %s\n", ap_config_dict)


    model_path = ap_config_dict.get(
        "model_path",
        f"{Path(__file__).parent}/dl_segmentation/2024-02-24_0013_gis_lamela_crack_pytorch_AUnet.ptchkp"
    )
    logging.info(f"Initialising DL model, path:{model_path}")
    gm.init_model_with_path(model_path)
    
def adaptive_polish_run(microscope_in=None, settings_in=None, patterns_in=None, image_settings_sem=None, ):
    global _ap_config

    from fibsem import acquire, utils
    from fibsem.structures import BeamType
    from fibsem import milling

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

    # Running ---------------------------------------------------------------------

    # Initialise counts
    scan_count = 0
    total_time = 0

    # set pattern
    microscope.connection.imaging.set_active_view(BeamType.ION.value)  # the ion beam view
    microscope.connection.imaging.set_active_device(BeamType.ION.value)
    # milling.draw_patterns(microscope, patterns_in)

    while scan_count <= _ap_config["max_milling_cycles"]:
        # Set detector
        #microscope.detector.type.value = "ETD"
        # fibsem detector is set in configuration file "detector_type". No need to set here

        #logging.info(f"Changing to {microscope.detector.type.value} detector")
        # place pattern (temporary because run_milling clears them!)
        milling.draw_patterns(microscope, patterns_in)
        # Acquire images
        imgs = {}
        for gfs_set, gfs in _ap_config.all_gfs.items():
            logging.info(
                f"Acquiring image with {gfs_set} settings for milling cycle "
                f"{scan_count}/{_ap_config['max_milling_cycles']}"
            )
            img_name = f"{config.folders['lamella_folder'].stem}_img_{scan_count:03}"
            logging.info(f"Image is {img_name}")


            # microscope.imaging.set_active_view(gfs["active_view"])
            # try:
            #     adornimg = microscope.imaging.grab_frame(
            #         GrabFrameSettings(
            #             resolution=gfs["resolution"],
            #             dwell_time=gfs["dwell_time"],
            #             frame_integration=gfs["frame_integration"],
            #             bit_depth=gfs["bit_depth"],
            #             drift_correction=gfs["drift_correction"],
            #         )
            #     )
            #     adornimg.save(f"{config.folders[gfs_set]}/{img_name}.tif")  # save to disk
            #     imgs[gfs_set] = adornimg  # store in RAM
            # except StopIteration:
            #     # reached end of example files to look at, this shouldn't happen
            #     # in a real microscopy session
            #     break

            #settings.image.hfw = 400e-6
            # in m, need to define here.
            # There doesn't seem to exist a method to get the field of view from the microscope
            # Can try get_imaging_settings from microscope api.


            if 'SEM' in gfs_set:
                #settings.image.beam_type = BeamType.ELECTRON
                img_settings = microscope.get_imaging_settings(BeamType.ELECTRON)
                img = acquire.new_image(microscope, image_settings_sem)
                # note that by new_image() can save images
                # but the img_settings.save must be True, and will save to f"{settings.filename}_eb" .tif
                #img.save(f"{config.folders[gfs_set]}/{img_name}.tif")  # save to disk
                # TODO: Save to a AP folder

                imgs[gfs_set] = img
            elif "FIB" in gfs_set:
                #settings.image.beam_type = BeamType.ION
                img_settings = microscope.get_imaging_settings(BeamType.ION)
                img = acquire.new_image(microscope, img_settings)
                # img.save(f"{config.folders[gfs_set]}/{img_name}.tif")  # save to disk
                # imgs[gfs_set] = img
                # TODO: Save to a AP folder

            #please note that the imgs will not contain AdornedImage but FibsemImage object
            #Depending on the settings.save, this image is automatically saved


        #microscope.detector.type.value = "ETD"
        #logging.info(f"Changing to {microscope.detector.type.value} detector")

        if len(imgs) < 1:
            # If there are no images to look at, exit while loop
            # Shouldn't happen in a real session, this is only for when there are no
            # more example images to look at

            logging.info("No images acquired, exiting loop")
            break

        # Measure GIS thickness on SEM image ---------------------------------------

        #SEM_adornimg = imgs["SEM"]  # find the SEM images in RAM to measure GIS
        #pixel_size_m = gm.get_pixel_width(SEM_adornimg)
        #Try to get pixel size:
        SEM_img = imgs["SEM"]
        pixel_size_m = SEM_img.metadata.pixel_size.x #untested

        if pixel_size_m:
            logging.info(f"Using pixel size {pixel_size_m} m")
        else:
            pixel_size_m = _ap_config["fallback_pixel_size_m"]
            logging.info(f"Using fallback pixel size of {pixel_size_m} m")

        # Segmentation
        prediction = gm.segment(SEM_img.data)
        logging.info("Segmentation complete")
        mask_gis_clean = gm.clean_prediction(
            prediction, pixel_size_m,
            _ap_config["lam_height_min_m"],
            _ap_config["reject_GIS_distance_m"]
        )
        logging.info(
            f"Cleaned mask to only include GIS "
            f"{_ap_config['reject_GIS_distance_m']} m below "
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
        #GIS_m = gm.measure_GIS(mask_gis_clean, _ap_config["window_size_m"], pixel_size_m)
        GIS_m, xlims = gm.measure_GIS(mask_gis_clean, _ap_config["window_size_m"], pixel_size_m)

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
            "image": img_name,
            "milling_time_s": total_time,
            "min_GIS_m": min_GIS_m,
            "crack_area_m2": crack_area_m2,
        }
        results.to_csv(f"{config.folders['lamella_folder']}/GIS_thickness.csv")

        # Save GIS thickness for each window
        for window, gis_thickness in enumerate(GIS_m):
            if gis_thickness > 0:
                gis_results_detailed.loc[len(gis_results_detailed)] = {
                    "image": img_name,
                    "milling_time_s": total_time,
                    "window": window,
                    "gis_windowed_m": gis_thickness,
                }
            else:
                pass
        gis_results_detailed.to_csv(
            f"{config.folders['lamella_folder']}/GIS_thickness_detailed.csv"
        )

        # Generate plots
        gm.milling_cycle_plot(
            sem_image=SEM_img.data,
            first_prediction=prediction,
            clean_prediction=mask_gis_clean,
            fib_image=imgs["FIB"].data,
            gis_thickness_m=GIS_m,
            gis_stop_m=_ap_config["gis_stop_m"],
            crack_area_m2=crack_area_m2,
            img_name=img_name,
            save_path=f"{config.folders['plots_folder']}/{img_name}_plot.png",
        )

        # Should we continue?
        if min_GIS_m < _ap_config["gis_stop_m"]:
            logging.info(
                f"Stopping as minimum GIS (m) {min_GIS_m} < threshold {_ap_config['gis_stop_m']}"
            )
            break

        if crack_area_m2 > _ap_config["max_crack_area_m2"]:
            logging.info(
                f"Stopping as crack area (m2) {crack_area_m2} > threshold {_ap_config['max_crack_area_m2']}"
            )
            break

        # Continue milling
        #milling.run_milling(microscope, settings.milling.milling_current, settings.milling.milling_voltage)
        
        #Mill for a predetermined amoutn of time
        millmilling_interval_s = _ap_config['milling_interval_s']
        logging.info(f"Sleeping {millmilling_interval_s} seconds to mill")
        #milling.run_milling(microscope, settings.milling.milling_current, settings.milling.milling_voltage, asynch=True)
        microscope.run_milling(microscope, settings.milling.milling_current, settings.milling.milling_voltage, asynch=True)
        time.sleep(millmilling_interval_s)
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
        if min_GIS_m <= 1.2 * _ap_config["gis_stop_m"]:
            _ap_config["milling_interval_s"] = _ap_config["milling_interval_s"] / 2

        scan_count += 1
        total_time += _ap_config["milling_interval_s"]

    # If we reached the stopping condition of min lamella thickness or max number of cycles
    # microscope.imaging.set_active_view(2)
    # microscope.patterning.stop()

    # finish milling (fibsem)
    # LMAP: Not sure but this line below may be needed
    # milling.finish_milling(microscope, microscope.system.ion.beam.beam_current)



    if _ap_config["do_plots"] is True:
        plt.figure()
        plt.plot(
            results.milling_time_s, results.min_GIS_m, label="Minimum GIS thickness (m)"
        )
        plt.xlabel("Milling Time (s)")
        plt.ylabel("GIS Thickness (m)")
        plt.title(f"Lamella {config.folders['lamella_folder'].stem}")
        plt.savefig(f"{config.folders['lamella_folder']}/GIS_thickness.png")
        plt.close()
