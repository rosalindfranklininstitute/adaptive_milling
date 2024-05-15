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


import xml.etree.ElementTree as ET  # to handle metadata as xml
from adaptive_polish.dl_segmentation import sem_lamella_segmentor as sgm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging

def get_pixel_width(img):
    """Gets the pixel width of an AdornedImage, or if not possible, returns None

    Args:
        img (AdornedImage): TFS AutoScript AdornedImage
    """
    try:
        xml_sett = img.metadata.metadata_as_xml
        xml_parse = ET.fromstring(xml_sett)
        pixel_size_x = xml_parse.find("BinaryResult/PixelSize/X")
        pixel_width_m = float(pixel_size_x.text)
        return pixel_width_m
    except:
        # if there is no metadata with 'PixelWidth' defaults to the value above
        return None

_segmentor=None
def init_model_with_path(model_path):
    #Need to call this before doing segmentation
    global _segmentor
    # segmentor = sgm.cSEMLamellaSegmentor(
    #     model_path=f"{Path(__file__).parent}/dl_segmentation/2024-02-24_0013_gis_lamela_crack_pytorch_AUnet.ptchkp"
    # )
    if model_path:
        _segmentor = sgm.cSEMLamellaSegmentor(model_path=model_path)

def segment(img: np.array) -> np.array:
    """
    Segments img using _segmentor instance of cSEMLamellaSegmentor
    
    Returns: annotated image, array of ints with the same shape as img. Returns None if _segmentor is None
    """

    global _segmentor
    prediction=None
    if _segmentor:
        prediction = _segmentor.get_prediction(img)
    return prediction


def clean_prediction(
    prediction: np.array, pixel_size_m: float, lam_height_min_m: float = 1e-7, reject_GIS_distance_m: float = 1e-6
) -> np.array:
    # Generate bool masks for GIS and lamella
    mask_gis = prediction == 1
    mask_lamella = prediction == 2

    lamella_area_px = np.sum(mask_lamella)
    gis_area_px = np.sum(mask_gis)

    # Check if there is any lamella. If not return nothing
    if lamella_area_px == 0:
        logging.info("No lamella detected. Returning NaN")
        return None
    if gis_area_px == 0:
        logging.info("No GIS layer detected. Returning NaN")
        return None
    
    # Find the transition between lamella and GIS
    transition_gis_lamella = np.logical_and(mask_lamella[:-1, :], mask_gis[1:, :])

    # Only take GIS below reject_GIS_distance_um above the transition line

    # The code below can be a bit confusing.
    # It is for selecting all GIS layer below the transition only,
    # with a slight padding of trans_shiftup.
    # We are using the transition line a starting y coordinate for the valid GIS layer
    # This transition line could be one pixel above the GIS layer,
    # hence we create a mask where we shift the y starting coordinates up a bit
    transition_cumsum = np.cumsum(transition_gis_lamella, axis=0)
    trans_shiftup=2
    transition_cumsum = (
        np.pad(
            transition_cumsum[trans_shiftup:, :],
            pad_width=((0, trans_shiftup + 1), (0, 0)),
            mode="edge",
        )
        > 0
    )
    # plt.imshow(transition_cumsum)
    # plt.show()

    #This code below will mask GIS layer segmentation beyond the heigth established by reject_GIS_height_mask
    # Note that it starts measuring from the transition minus shiftup height coordinate.
    mask0 = np.cumsum(transition_cumsum, axis=0)
    height_px = int(reject_GIS_distance_m/pixel_size_m)
    reject_GIS_height_mask = mask0<=height_px
    mask_gis_under_transition =  np.logical_and(mask_gis, reject_GIS_height_mask)

    #mask_gis_under_transition = np.logical_and(mask_gis, transition_cumsum)
    # plt.imshow(mask_gis_under_transition)
    # plt.show()

    # Filter GIS layer by the amount of lamella height  above
    lam_abov_gis_x = np.any(transition_gis_lamella, axis=0)  

    # Consider only GIS which have lamella areas with more than N pixels vertical
    lam_height_threshold_px = int(np.ceil(lam_height_min_m / pixel_size_m))  # pixels
    lam_height_along_x = np.sum(mask_lamella, axis=0) >= lam_height_threshold_px

    large_lam_above_gis_mask_along_x = np.logical_and(lam_abov_gis_x, lam_height_along_x)
    logging.info(f"np.sum(large_lam_above_gis_mask_along_x): {np.sum(large_lam_above_gis_mask_along_x)}")

    #mask along x
    mask = np.tile(large_lam_above_gis_mask_along_x, (mask_gis_under_transition.shape[0],1) )
    mask_gis_clean = np.logical_and(mask_gis_under_transition, mask)

    # debug
    #fig, axs = plt.subplots(1,1, sharex=True, sharey=True)
    # plt.imshow(mask_gis_clean)
    # plt.show()

    return mask_gis_clean


def measure_GIS(
    mask_gis_clean: np.array, window_size_m: int, pixel_size_m: float
) -> np.array:
    # Get window size in px
    window_size_px = int(window_size_m / pixel_size_m)
    logging.info(f"window_size_px: {window_size_px}")

    # Calculate average GIS thickness in windows

    #Sum to get thickness along x in pixels
    GIS_pxbypx = np.sum(mask_gis_clean, axis=0)

    # Pad with zeros to fulfil window size criteria
    GIS_pxbypx_pad = np.pad(
        GIS_pxbypx,
        (0, window_size_px - len(GIS_pxbypx) % window_size_px),
        constant_values=0,
    )

    #Get left and right limits from where the mean should be calculated from
    GIS_pxbypx_where_above_zero = np.where(GIS_pxbypx_pad)
    xlim_min = GIS_pxbypx_where_above_zero[0][0]  # first occurrence along x
    xlim_max = GIS_pxbypx_where_above_zero[0][-1]  # last occurrence along x

    GIS_pxbypx_NaNed = np.copy(GIS_pxbypx_pad).astype(np.float32)
    GIS_pxbypx_NaNed[:xlim_min]=np.nan
    GIS_pxbypx_NaNed[xlim_max:]=np.nan

    # This mean will discard nan areas
    GIS_windowed = np.nanmean(GIS_pxbypx_NaNed.reshape(-1, window_size_px), axis=1)

    # Convert to m
    GIS_m = GIS_windowed * pixel_size_m

    # Make any value with 0's, i.e. no GIS measured, NaNs
    GIS_m[np.where(GIS_m == 0)] = np.nan

    return GIS_m, (xlim_min, xlim_max)

# This will always return zero!!
# def find_cracks(prediction: np.array, pixel_size_m: float) -> float:
#     mask_crack = prediction == 3
#     mask_lamella = prediction == 2
#     mask_crack_within_lamella = np.logical_and(mask_crack, mask_lamella)  
#     crack_area_px = np.sum(mask_crack_within_lamella)
#     crack_area_m2 = crack_area_px * pixel_size_m * pixel_size_m

#     return crack_area_m2

def get_crack_area_m2(prediction: np.array, pixel_size_m: float, xlim_min_px, xlim_max_px) -> float:
    """
    Gets the crack area from a DL prediction where crack is annotated as integer 3

    xlim_min_px, xlim_max_px set the windows where to consider the crack area
    This helps reject annotated cracks outside the lamella area.
    These limits can be determined in measure_GIS()
    
    Params:
        prediction
        pixel_size_m: pixel to meter
        xlim_min_px, xlim_max_px: window limits in pixel units

    Returns:
        crack_area_m2: float value with the crack area in m2 units
    
    """
    
    crack_area_px = prediction == 3
    crack_area_lammellamask = crack_area_px[:, xlim_min_px:xlim_max_px]

    crack_area_px2 = np.sum(crack_area_lammellamask)

    crack_area_m2 = crack_area_px2 *pixel_size_m*pixel_size_m

    return crack_area_m2


def milling_cycle_plot(
    sem_image: np.array,
    first_prediction: np.array,
    clean_prediction: np.array,
    fib_image: np.array,
    gis_thickness_m: np.array,
    gis_stop_m: float,
    crack_area_m2: float,
    img_name: str = None,
    save_path: Path = None,
):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), tight_layout=True)
    fig.suptitle(img_name)

    # SEM
    axs[0, 0].imshow(sem_image, cmap="Greys_r")
    axs[0, 0].axis("off")
    axs[0, 0].set_title("SEM")

    # SEM + 1st pass prediction
    axs[0, 1].imshow(sem_image, cmap="Greys_r")
    axs[0, 1].imshow(first_prediction, alpha=0.4, cmap="tab10", vmin=0, vmax=10)
    axs[0, 1].axis("off")
    axs[0, 1].set_title("SEM, 1st prediction")

    # SEM + clean prediction
    axs[0, 2].imshow(sem_image, cmap="Greys_r")
    axs[0, 2].imshow(clean_prediction, alpha=0.4, cmap="tab10", vmin=0, vmax=10)
    axs[0, 2].axis("off")
    axs[0, 2].set_title(f"SEM, clean, crack area $\mu$m2 = {crack_area_m2*1e12:5}")

    # FIB image
    axs[1, 0].imshow(fib_image, cmap="Greys_r")
    axs[1, 0].axis("off")
    axs[1, 0].set_title("FIB")

    # FIB + milling box
    # TODO

    # GIS thickness
    axs[1, 2].plot(gis_thickness_m * 1e6)
    axs[1, 2].set_xlabel("Distance along x (px)")
    axs[1, 2].set_ylabel("GIS thickness ($\mu$m)")
    axs[1, 2].set_xlim(0, len(gis_thickness_m))
    axs[1, 2].set_ylim(
        0,
    )
    axs[1, 2].hlines(
        y=gis_stop_m * 1e6,
        xmin=0,
        xmax=len(gis_thickness_m),
        label="Target GIS",
        linestyles="dashed",
        colors="C1",
    )
    axs[1, 2].set_title(
        f"GIS thickness, min={np.nanmin(gis_thickness_m)*1e6:.2f} $\mu$m"
    )
    axs[1, 2].legend()

    fig.savefig(save_path)

    plt.close(fig)
