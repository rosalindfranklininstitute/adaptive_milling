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

    last_key = max(times.keys())

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


def process_experiment(
    experiment_path: str | PathLike[str], ignore_lamellae: list[str] | None = None
) -> None:
    experiment_path = Path(experiment_path)
    if ignore_lamellae is None:
        ignore_lamellae = []

    id_list: list[str] = []
    time_list: list[float] = []
    for child in experiment_path.iterdir():
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
                        _logger.error(f"Error processing {experiment_path}: {e}")
    stats_dict = get_statistics_from_time_list(time_list)
    print(f"Checked {len(time_list)} lamella in {experiment_path}:\n")

    print(
        "!!!Please note these are the total milling times, not the per-pattern milling times!!!\n"
        "To set the time for a trench pattern based on these times, these values should first be halved."
    )

    print("\nMilling times (s):")
    for id_, time in zip(id_list, time_list):
        print(f"\t{id_}:\t{time}")

    print("\nStatistics:")
    for k, v in stats_dict.items():
        print(f"\t{k} milling time (s):\t{v}")


if __name__ == "__main__":
    # Set experiment path here:
    experiment_path = Path(
        r""
    )
    # If there are any lamellae you do not want to include, their names
    # (e.g. "03-fast-swan") can be added to this list:
    ignore_lamellae = []
    process_experiment(experiment_path=experiment_path, ignore_lamellae=ignore_lamellae)
