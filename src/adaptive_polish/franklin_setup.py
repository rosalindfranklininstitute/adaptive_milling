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


import yaml
import datetime
from pathlib import Path
from glob import glob
import os


class AdaptiveMillingConfig:
    def __init__(self):
        self.config = {
            # Setting up a session
            "session_name": f"Franklin_{datetime.date.today()}",
            "session_dir": f"Z:/structbio/qmm85263/20240509_autolamella_test/AutoLamella-2024-05-09-11-01/{datetime.date.today()}",
            # "session_dir": os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop', 'adapt_mill_session_localtest_temp') ,
            "do_plots": True,
            "display_plots": False,
            "example_SEM": "C:/Users/dmv31621/OneDrive - The Rosalind Franklin Institute/2024/00_Adaptive-milling/test_images/20240429_adaptive_milling_Nadisha2/2024-04-29/Franklin_2024-04-29_lam_000/SEM/",
            "example_FIB": "C:/Users/dmv31621/OneDrive - The Rosalind Franklin Institute/2024/00_Adaptive-milling/test_images/20240429_adaptive_milling_Nadisha2/2024-04-29/Franklin_2024-04-29_lam_000/FIB/",
            "show_DL_segmentation": True,
            "fallback_pixel_size_m": 1.8e-8,
            # Milling cycle control
            "milling_interval_s": 30,
            "gis_stop_m": 2e-7, #default 3e-7
            "max_milling_cycles": 30,
            "reject_GIS_distance_m": 1e-5,
            "lam_height_min_m": 2e-6,
            "window_size_m": 1e-7,
            "max_crack_area_m2": 2e-12,
            "model_path":f"{Path(__file__).parent}/dl_segmentation/2024-02-24_0013_gis_lamela_crack_pytorch_AUnet.ptchkp"
        }

        self.all_gfs = {
            "SEM": {
                "active_view": 1,
                "resolution": "3072x2048",
                "dwell_time": 50e-9,
                "frame_integration": 16,
                "bit_depth": 16,
                "drift_correction": True,
            },
            "FIB": {
                "active_view": 2,
                "resolution": "3072x2048",
                "dwell_time": 50e-9,
                "frame_integration": 32,
                "bit_depth": 16,
                "drift_correction": True,
            },
        }

        self.setup_folders()
        self.save_yaml(
            yaml_path=f"{self.folders['lamella_folder']}/adaptive_milling_config.yaml",
            params=self.config,
        )
        self.save_yaml(
            yaml_path=f"{self.folders['lamella_folder']}/imaging_config.yaml",
            params=self.all_gfs,
        )

    def save_yaml(self, yaml_path: Path, params: dict):
        with open(yaml_path, "w") as f:
            yaml.dump(params, f, indent=4)

    def setup_folders(self):
        session_dir = Path(self.config["session_dir"])
        if not session_dir.exists():
            session_dir.mkdir(exist_ok=True)
        folders = {}
        existing_lamellae_folders = sorted(glob(f"{session_dir}/*lam_*"))
        if len(existing_lamellae_folders) == 0:  # first lamella
            latest_lamella_number = -1
        else:
            latest_lamella_number = int(existing_lamellae_folders[-1].split("_")[-1])
        lamella_number = latest_lamella_number + 1
        folders["lamella_folder"] = Path(
            f"{session_dir}/{self.config['session_name']}_lam_{lamella_number:03}"
        )
        folders["lamella_folder"].mkdir()

        folders["plots_folder"] = Path(f"{folders['lamella_folder']}/Plots")
        folders["plots_folder"].mkdir()
        for gfs_set in list(self.all_gfs.keys()):
            folders[gfs_set] = Path(f"{folders['lamella_folder']}/{gfs_set}")
            folders[gfs_set].mkdir()
        folders["bmp_folder"] = Path(f"{folders['lamella_folder']}/bitmap")
        folders["bmp_folder"].mkdir()

        self.folders = folders
