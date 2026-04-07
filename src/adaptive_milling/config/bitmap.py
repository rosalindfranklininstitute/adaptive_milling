from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

from adaptive_milling.config import AdaptivePolishMillingConfig


@dataclass
class BitmapAdaptivePolishMillingConfig(AdaptivePolishMillingConfig):
    gis_max: float = field(
        default=2e-6,
        metadata={
            "label": "Max. milling GIS thickness",
            "tooltip": "The milling dwell time will not be reduced where the GIS thickness above this threshold.",
            "type": float,
            "minimum": 0.01,
            "step": 0.01,
            "decimals": 2,
            "scale": 1e6,
            "unit": "m",
        },
    )
    gis_min: float = field(
        default=0.15e-6,
        metadata={
            "label": "Min. milling GIS thickness",
            "tooltip": "The milling dwell time will be 0 where the GIS thickness below this threshold.",
            "type": float,
            "minimum": 0,
            "step": 0.01,
            "decimals": 2,
            "scale": 1e6,
            "unit": "m",
        },
    )
    dwell_multiplier_mean_stop: float = field(
        default=0.03,
        metadata={
            "label": "Mean dwell multiplier threshold",
            "tooltip": "Milling will be stopped if the mean dwell time multiplier drops below this threshold.",
            "type": float,
            "minimum": 0,
            "maximum": 1,
            "step": 0.01,
            "decimal": 0.01,
        },
    )
    dwell_multiplier_max_stop: float = field(
        default=0.05,
        metadata={
            "label": "Max. dwell multiplier threshold",
            "tooltip": "Milling will be stopped if the maximum dwell time multiplier drops below this threshold.",
            "type": float,
            "minimum": 0,
            "maximum": 1,
            "step": 0.01,
            "decimal": 0.01,
        },
    )
    bitmap_erosion_px: int = field(
        default=20,
        metadata={
            "label": "Bitmap erosion",
            "tooltip": "Number of pixels of grayscale erosion to apply when creating the bitmap patterns.",
            "type": int,
            "minimum": 0,
            "step": 1,
            "unit": "px",
        },
    )
    bitmap_gaussian_sigma: float = field(
        default=20,
        metadata={
            "label": "Bitmap sigma",
            "tooltip": "Gaussian blur sigma to apply when creating the bitmap patterns (no-mill areas are preserved).",
            "type": float,
            "minimum": 0,
            "step": 0.1,
            "decimals": 1,
        },
    )

    apply_boundary_smoothing: bool = field(
        default=False,
        metadata={
            "label": "Apply boundary smoothing",
            "tooltip": "Option to apply smoothing to the lamella thickness based on shape of the lamella-GIS boundary when creating the bitmap patterns.",
            "type": bool,
            "advanced": True,
        },
    )
    boundary_smoothing_sigma: float = field(
        default=5.0,
        metadata={
            "label": "Boundary smoothing sigma",
            "tooltip": "Gaussian blur sigma used on the signal for boundary smoothing.",
            "type": float,
            "minimum": 0,
            "advanced": True,
        },
    )
    incident_angle_sputter_ratio: float = field(
        default=4.3,
        metadata={
            "label": "Incident angle sputter ratio",
            "tooltip": "The ratio between the maximum sputter rate and the sputter rate at an incident angle of 0 degrees. Disabled if a ratio of 1 is given.",
            "minimum": 1,
        },
    )
    incident_angle_scale_with_gis_min: bool = field(
        default=False,
        metadata={
            "label": "Incident angle above GIS min",
            "tooltip": "Option to apply incident angle scaling (if used) to the entire signal or above the configured min. milling GIS thickness threshold.",
            "type": bool,
            "advanced": True,
        },
    )
    mask_cracks: bool = field(
        default=True,
        metadata={
            "label": "Mask cracks",
            "tooltip": "Option to mask out cracks in the bitmap pattern.",
        },
    )
    pattern_alignment: Literal["centre", "convolve"] = field(
        default="centre",
        metadata={
            "label": "Pattern alignment method",
            "tooltip": "Pattern alignment method to be used. Convolve is currently experimental and not recommended.",
            "type": str,
            "items": ["centre", "convolve"],
            "advanced": True,
            "hidden": True,
        },
    )
