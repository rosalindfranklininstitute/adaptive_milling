from __future__ import annotations
import enum


class StopReasons(enum.Enum):
    MIN_GIS_THICKNESS = "minimum GIS thickness below the threshold"
    MEDIAN_GIS_THICKNESS = "median GIS thickness below the threshold"
    LAMELLA_DRIFT = "lamella has drifted beyond the threshold"
    CRACK_AREA = "crack area is above the threshold"
    LAMELLA_AREA = "lamella area is below the threshold"
    MAX_CYCLES = "run maximum milling cycles"
    USER = "milling stopped by user"
    MAX_DWELL_MULTIPLIER = "maximum dwell time multiplier is below the threshold"
    MEAN_DWELL_MULTIPLIER = "mean dwell time multiplier is below the threshold"
