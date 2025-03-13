from pathlib import Path
import pandas as pd

def setup_lamella_ap_folders(lamella_folder: Path):
    """Sets up the folders for adaptive polish in the lamella folder (path)

    Args:
        lamella_folder (Path): Path to the lamella folder
    """
    ap_folder = f"{lamella_folder}/adaptive_polish"
    Path(ap_folder).mkdir(exist_ok=True)
    Path(f"{ap_folder}/plots").mkdir(exist_ok=True)
    Path(f"{ap_folder}/sem").mkdir(exist_ok=True)
    Path(f"{ap_folder}/fib").mkdir(exist_ok=True)

def setup_results_df():
    results = pd.DataFrame(
        {
            "image": [],
            "milling_time_s": [],
            "min_GIS_um": [],
            "crack_area_um2": [],
        }
    )

    gis_results_detailed = pd.DataFrame(
        {
            "image": [],
            "milling_time_s": [],
            "window": [],
            "gis_windowed_um": [],
        }
    )
    return results, gis_results_detailed
