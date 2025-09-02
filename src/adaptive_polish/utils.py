from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Protocol, TypeVar, overload

_TMulDiv = TypeVar("_TMulDiv", bound="_SupportsMulDiv")


class _SupportsMulDiv(Protocol):
    def __mul__(self: "_TMulDiv", other) -> "_TMulDiv": ...

    def __truediv__(self: "_TMulDiv", other) -> "_TMulDiv": ...


_TConvertable = TypeVar(
    "_TConvertable", _SupportsMulDiv, list[_SupportsMulDiv], tuple[_SupportsMulDiv, ...]
)


@overload
def pixels_to_size(
    value: tuple[_SupportsMulDiv], pixel_size: float
) -> tuple[_SupportsMulDiv]: ...


@overload
def pixels_to_size(
    value: list[_SupportsMulDiv], pixel_size: float
) -> list[_SupportsMulDiv]: ...


@overload
def pixels_to_size(value: _TMulDiv, pixel_size: float) -> _TMulDiv: ...


def pixels_to_size(value: _TConvertable, pixel_size: float) -> _TConvertable:
    if isinstance(value, (list, tuple)):
        return type(value)([_ * pixel_size for _ in value])
    return value * pixel_size


@overload
def size_to_pixels(
    value: tuple[_SupportsMulDiv], pixel_size: float
) -> tuple[_SupportsMulDiv]: ...


@overload
def size_to_pixels(
    value: list[_SupportsMulDiv], pixel_size: float
) -> list[_SupportsMulDiv]: ...


@overload
def size_to_pixels(value: _TMulDiv, pixel_size: float) -> _TMulDiv: ...


def size_to_pixels(value: _TConvertable, pixel_size: float) -> _TConvertable:
    if isinstance(value, (list, tuple)):
        return type(value)([_ / pixel_size for _ in value])
    return value / pixel_size


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


def setup_results_df() -> tuple[pd.DataFrame, pd.DataFrame]:
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
            "gis_thickness_um": pd.Series([], dtype=object),
            "gis_thickness_filtered_um": pd.Series([], dtype=object),
            "xlims_px": pd.Series([], dtype=object),
        }
    )
    return results, gis_results_detailed
