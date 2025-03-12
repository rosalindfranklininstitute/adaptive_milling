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
import pandas as pd
import skimage

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


def keep_only_largest_object(mask: np.array, fill_value: int = 1) -> np.array:
    """Find the largest object in a mask and sets everything in that object to
    `fill_value`, background is 0.

    Args:
        mask (np.array): Segmentation mask
        fill_value (int, optional): Everything inside the largest object
            is set to this value. Defaults to 1.

    Returns:
        np.array: Mask with the largest object only
    """
    instances = skimage.measure.label(mask)
    instance_properties = skimage.measure.regionprops(instances)
    areas = [i.area for i in instance_properties]
    largest_area = max(areas)
    mask_largest_only = skimage.morphology.remove_small_objects(
        instances,
        min_size = 0.99 * largest_area
    )
    mask_largest_only[mask_largest_only > 0] = fill_value
    return mask_largest_only


def clean_prediction(
    prediction: np.array, pixel_size_m: float,
) -> np.array:
    logging.info(f"clean_prediction with prediction.shape:{prediction.shape}, pixel_size_m:{pixel_size_m}")
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

    # Only keep the largest object in the GIS segmentation
    mask_gis_largest_only = keep_only_largest_object(mask_gis)

    return mask_gis_largest_only


def measure_GIS(
    mask_gis_clean: np.array, window_size_px: int, pixel_size_m: float
) -> np.array:

    # Calculate average GIS thickness in windows

    #Sum to get thickness along x in pixels
    GIS_pxbypx = np.sum(mask_gis_clean, axis=0).astype(np.float32)

    # Pad with zeros to fulfil window size criteria
    GIS_pxbypx_pad = np.pad(
        GIS_pxbypx,
        (0, window_size_px - len(GIS_pxbypx) % window_size_px),
        constant_values=np.nan,
    )

    #Get left and right limits from where the mean should be calculated from
    GIS_pxbypx_where_above_zero = np.where(GIS_pxbypx_pad > 0)
    xlim_min = GIS_pxbypx_where_above_zero[0][0]  # first occurrence along x
    xlim_max = GIS_pxbypx_where_above_zero[0][-1]  # last occurrence along x

    # make the L and R limits of the lamella x% smaller
    lamella_width_px = xlim_max - xlim_min
    lamella_width_to_cut = int(0.05 * lamella_width_px)  # cut 5% from each end
    xlim_min = lamella_width_to_cut + xlim_min
    xlim_max = xlim_max - lamella_width_to_cut

    GIS_pxbypx_NaNed = np.copy(GIS_pxbypx_pad).astype(np.float32)
    GIS_pxbypx_NaNed[:xlim_min]=np.nan
    GIS_pxbypx_NaNed[xlim_max:]=np.nan

    # This mean will discard nan areas
    GIS_windowed = np.nanmedian(GIS_pxbypx_NaNed.reshape(-1, window_size_px), axis=1)

    # Convert to m
    GIS_m = GIS_windowed * pixel_size_m

    # Make any value with 0's, i.e. no GIS measured, NaNs
    GIS_m[np.where(GIS_m == 0)] = np.nan

    return GIS_m, (xlim_min, xlim_max)


def get_crack_area_m2(prediction: np.array, pixel_size_m: float) -> float:
    """Measure the area in the prediction for cracks in m2.

    Cracks are only considered if they are within the largest combined lamella+
    GIS+crack object in the prediction.

    Args:
        prediction (np.array): Segmentation mask
        pixel_size_m (float): Pixel size in m

    Returns:
        float: Crack area in m2
    """
    # Restrict crack search area to the largest object which is not background
    mask_anything = prediction > 0
    mask_anything_largest_only = keep_only_largest_object(mask_anything)
    mask_crack = prediction == 3

    # Since crack pixel value is 1 and anything outside largest object is 0
    mask_crack_inside_largest_object = np.multiply(
        mask_crack,
        mask_anything_largest_only
    )
    crack_area_px2 = np.sum(mask_crack_inside_largest_object)
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
    xlims = None,
    fib_screenshot: np.array = None,
    img_name: str = None,
    save_path: Path = None,
):
    logging.info("milling_cycle_plot()")
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), tight_layout=True)
    fig.suptitle(img_name)

    # SEM
    axs[0, 0].imshow(sem_image, cmap="Greys_r")
    axs[0, 0].axis("off")
    axs[0, 0].set_title("SEM")

    # SEM + 1st pass prediction
    axs[0, 1].imshow(sem_image, cmap="Greys_r")
    axs[0, 1].imshow(
        first_prediction,
        alpha=0.4,
        cmap="tab10",
        vmin=0,
        vmax=10,
        interpolation="nearest"
    )
    axs[0, 1].axis("off")
    axs[0, 1].set_title("SEM, 1st prediction")

    # SEM + clean prediction
    axs[0, 2].imshow(sem_image, cmap="Greys_r")
    axs[0, 2].imshow(
        clean_prediction,
        alpha=0.4,
        cmap="tab10",
        vmin=0,
        vmax=10,
        interpolation="nearest"
    )
    if xlims is not None:
        axs[0, 2].axvline(x=xlims[0])
        axs[0, 2].axvline(x=xlims[1])
    axs[0, 2].axis("off")
    axs[0, 2].set_title(f"SEM, clean, crack area $\mu$m2 = {crack_area_m2*1e12:.2f}")

    # FIB image
    axs[1, 0].imshow(fib_image, cmap="Greys_r")
    axs[1, 0].axis("off")
    axs[1, 0].set_title("FIB")

    # FIB + milling box
    if fib_screenshot is not None:
        # Doesn't work -> for some reason I can't open a new napari.Viewer()
        # pattern_viewer = napari.Viewer()
        # pattern_viewer.add_image(fib_image, name="fib_image")
        # _draw_patterns_in_napari(
        #     viewer=pattern_viewer,
        #     ib_image=FibsemImage(data=fib_image),
        #     eb_image=None,
        #     milling_stages=list(adaptive_polish_stage)
        # )
        # screenshot = pattern_viewer.screenshot()
        # axs[1, 1].imshow(screenshot)
        # pattern_viewer.close()
        axs[1, 1].imshow(fib_screenshot[:, int(fib_screenshot.shape[1]/2):, :])
    axs[1, 1].axis("off")

    # GIS thickness
    axs[1, 2].plot(gis_thickness_m * 1e6, ".-")
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


def summary_gis_plot(results: pd.DataFrame, lamella_folder: Path):
    plt.figure()
    plt.plot(
        results.milling_time_s, results.min_GIS_m, label="Minimum GIS thickness (m)"
    )
    plt.xlabel("Milling Time (s)")
    plt.ylabel("GIS Thickness (m)")
    plt.title(lamella_folder.stem)
    plt.savefig(f"{str(lamella_folder)}/adaptive_polish/{lamella_folder.stem}_GIS_thickness.png")
    plt.close()
