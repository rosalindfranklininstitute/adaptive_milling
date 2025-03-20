from pathlib import Path
import pandas as pd

def ensure_subdirectories(directory: Path, *subdirectory_names: str) -> list[Path]:
    """Sets up the folders for adaptive polish in the lamella folder (path)

    Args:
        lamella_folder (Path): Path to the lamella folder
    """

    directories = [directory / _ for _ in subdirectory_names]

    for directory in directories:
        # Ensure folders exist
        directory.mkdir(exist_ok=True)
    return directories

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
