from __future__ import annotations
import enum


class StopReasons(enum.Enum):
    MIN_GIS_THICKNESS = "minimum GIS thickness below the threshold"
    LAMELLA_DRIFT = "lamella has drifted beyond the threshold"
    CRACK_AREA = "crack area is above the threshold"
    LAMELLA_AREA = "lamella area is below the threshold"
    GIS_REDUCTION_RATE = "reduction in GIS thickness is below the threshold"
    MAX_CYCLES = "run maximum milling cycles"
    USER = "milling stopped by user"
