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

# Imports
from pathlib import Path
from datetime import datetime
import adaptive_polish.gis_measurement as gm
import tifffile
import numpy as np

# Set parameters
model_path = Path(r"C:\Users\dmv31621\OneDrive - The Rosalind Franklin Institute\2024\00_Adaptive-milling\adaptive_polish\src\adaptive_polish\dl_segmentation\2024-02-24_0013_gis_lamela_crack_pytorch_AUnet.ptchkp")
reject_GIS_distance_m = 1e-5
lam_height_min_m = 2e-6
window_size_m = 1e-7
gis_stop_m = 2e-7
pixel_size_m = 1.3020833333333335e-08
sem_image = Path(r"C:\Users\dmv31621\OneDrive - The Rosalind Franklin Institute\2024\00_Adaptive-milling\test_images\04-easy-orca\adapt_mill_img_003_electron_003.tif")
fib_image = Path(r"C:\Users\dmv31621\OneDrive - The Rosalind Franklin Institute\2024\00_Adaptive-milling\test_images\04-easy-orca\adapt_mill_img_003_ion_003.tif")
test_output_path = Path(f"C:/Users/dmv31621/OneDrive - The Rosalind Franklin Institute/2024/00_Adaptive-milling/temp/{datetime.today().strftime('%Y%m%d')}")

# Create test output file
if not test_output_path.exists():
    test_output_path.mkdir()
out_folder = Path(f"{test_output_path}/{datetime.today().strftime('%H%M%S')}")
out_folder.mkdir()

# Load model
gm.init_model_with_path(model_path)

# Run segmentation
sem_image_data = tifffile.imread(sem_image)
prediction = gm.segment(sem_image_data)
tifffile.imwrite(f"{out_folder}/prediction.tiff", prediction)
print("Created segmentation prediction")

gis_mask_only = prediction == 1
tifffile.imwrite(f"{out_folder}/gis.tiff", gis_mask_only)

# Clean segmentation
mask_gis_clean = gm.clean_prediction(
    prediction,
    pixel_size_m,
    lam_height_min_m,
    reject_GIS_distance_m
)
tifffile.imwrite(f"{out_folder}/cleaned.tiff", mask_gis_clean)
print("Cleaned segmentation")

# Measure GIS
GIS_m, xlims = gm.measure_GIS(
    mask_gis_clean,
    window_size_m,
    pixel_size_m
)
print("Calculated GIS thickness")

# Measure cracks
crack_area_m2 = gm.get_crack_area_m2(prediction, pixel_size_m, *xlims)
print(f"Crack area = {crack_area_m2} m2")

# Generate plots
gm.milling_cycle_plot(
    sem_image_data,
    prediction,
    mask_gis_clean,
    tifffile.imread(fib_image),
    GIS_m,
    gis_stop_m,
    crack_area_m2,
    sem_image.stem,
    f"{out_folder}/plot.png"
)
