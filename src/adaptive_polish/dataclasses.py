from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Any
    import numpy as np
    from numpy.typing import NDArray
    from fibsem.structures import FibsemImage


@dataclass
class LamellaStatistics:
    milling_cycle: int
    milling_time_s: float
    lamella_thickness_um: list[float | int]
    lamella_area_um2: float
    crack_thickness_um: list[float | int]
    crack_area_um2: float
    min_GIS_um: float | None = None
    lamella_bounding_box_px: tuple[float, float, float, float] | None = (
        None  # ymin, xmin, ymax, xmax (image)
    )
    xlims_px: tuple[int, int] | None = None  # min, max (image xlims)
    # The below values are the length of image width, rather than prediction width
    gis_thickness_um: list[float | int] | None = None
    gis_thickness_filtered_um: list[float | int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LamellaInformation:
    identifier: str
    sem_image: FibsemImage
    fib_image: FibsemImage
    prediction: NDArray[np.uint8]
    clean_prediction: NDArray[np.uint8]
    statistics: LamellaStatistics

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
