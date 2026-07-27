from __future__ import annotations

import json
import statistics
import logging
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    from os import PathLike

_logger = logging.getLogger(__name__)


def get_total_milling_time_from_gis_thickness_json(
    file_path: str | PathLike[str],
) -> float:
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"{file_path} was not found")
    with file_path.open("r") as f:
        times = json.load(f).get("milling_time_s")
    if not isinstance(times, dict):
        raise ValueError(f"Failed to get 'milling_time_s' from {file_path}")

    last_key = max(times)

    # Only get the last time, as it was recorded cumulatively
    last_time = times[last_key]
    try:
        # Recorded time is per-pattern shape, so total is doubled for trench patterns
        return float(last_time) * 2
    except Exception:
        raise ValueError(f"Failed to get total milling time from {file_path}")


def get_statistics_from_time_list(time_list: list[float]) -> dict[str, float]:
    return {
        "min": min(time_list),
        "max": max(time_list),
        "mean": statistics.fmean(time_list),
        "median": statistics.median(time_list),
        "stdev": statistics.stdev(time_list),
    }


def print_statistics(time_list: list[float]) -> None:
    stats_dict = get_statistics_from_time_list(time_list)

    print("\tStatistics:")
    for k, v in stats_dict.items():
        print(f"\t\t{k} milling time (s):\t{v}")


def process_experiment(
    experiment_directory: str | PathLike[str], ignore_lamellae: list[str] | None = None
) -> list[float]:
    experiment_directory = Path(experiment_directory)
    if ignore_lamellae is None:
        ignore_lamellae = []

    id_list: list[str] = []
    time_list: list[float] = []
    for child in experiment_directory.iterdir():
        if child.is_dir():
            if child.name in ignore_lamellae:
                _logger.warning("Skipping lamella '%s' due to ignore list", child.name)
                continue
            ap_subdirs = tuple(child.glob("*adaptive_polish_*"))
            if ap_subdirs:
                for ap_dir in ap_subdirs:
                    try:
                        json_path = ap_dir / "GIS_thickness.json"
                        time_list.append(
                            get_total_milling_time_from_gis_thickness_json(json_path)
                        )
                        id_list.append(f"{ap_dir.parent.name}/{ap_dir.name}")
                    except (FileNotFoundError, ValueError) as e:
                        _logger.error(
                            "Error processing %s: %s", str(experiment_directory), str(e)
                        )

    print(f"Checked {len(time_list)} lamella in {experiment_directory}:")

    if not time_list:
        print("No lamellae found")
        return []

    print("\tMilling times (s):")
    for id_, time in zip(id_list, time_list):
        print(f"\t\t{id_}:\t{time}")

    if len(time_list) > 1:
        print_statistics(time_list=time_list)
    else:
        print("No statistics for single lamella experiment")
    print()

    return time_list


def run(
    *experiment_directories: str | PathLike[str],
    ignore_lamellae: list[str] | None = None,
) -> None:
    print(
        "\n!!!Please note these are the total milling times, not the per-pattern milling times!!!\n"
        "To set the time for a trench pattern based on these times, these values should first be halved.\n"
    )

    time_list: list[float] = []
    for experiment_directory in experiment_directories:
        time_list.extend(
            process_experiment(
                experiment_directory=experiment_directory,
                ignore_lamellae=ignore_lamellae,
            )
        )

    if len(experiment_directories) > 1 and len(time_list) > 1:
        print("\nAll:")
        print_statistics(time_list=time_list)

    return


if __name__ == "__main__":
    # List experiment directories here:
    experiment_directories: list[str | Path] = []
    # If there are any lamellae you do not want to include, their names (e.g.
    # "03-fast-swan") can be added to this list (they should be unique between
    # experiments):
    ignore_lamellae: list[str] = []
    run(*experiment_directories, ignore_lamellae=ignore_lamellae)
