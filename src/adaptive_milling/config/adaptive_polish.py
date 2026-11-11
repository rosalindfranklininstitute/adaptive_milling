from __future__ import annotations

import math
import os
from dataclasses import dataclass

from fibsem.milling.base import MillingStrategyConfig
from fibsem.structures.params import Display, RangeLimit, param

from adaptive_milling.models import (
    MODEL_GENERATIONS_DICT,
    get_latest_generation_key,
)

DEFAULT_MODEL_GENERATION = get_latest_generation_key()

# limits are in SI (metres, square metres); scale and step are what the form shows
_AT_LEAST_ZERO = RangeLimit(0.0, math.inf)


@dataclass
class AdaptivePolishMillingConfig(MillingStrategyConfig):
    model_path: str = param(
        "",
        doc="Path to the SEM segmentation model weights (*.pth).",
        filepath=True,
        display=Display("Model path"),
    )
    model_generation: str = param(
        DEFAULT_MODEL_GENERATION,
        doc="The generation of model to use (defaults to latest).",
        choices=list(MODEL_GENERATIONS_DICT.keys()),
        display=Display("Model generation"),
    )
    gis_stop_min: float = param(
        0.25e-6,
        unit="m",
        limits=_AT_LEAST_ZERO,
        doc="Stop polishing if the local GIS thickness drops below this threshold.",
        display=Display(
            "Minimum GIS thickness threshold", scale=1e6, step=0.01, decimals=2
        ),
    )
    gis_stop_median: float = param(
        0.3e-6,
        unit="m",
        limits=_AT_LEAST_ZERO,
        doc="Stop polishing if the median GIS thickness across the width of the lamella drops below this threshold.",
        display=Display(
            "Median GIS thickness threshold", scale=1e6, step=0.01, decimals=2
        ),
    )
    max_crack_area: float = param(
        0.2e-12,
        unit="m²",
        limits=_AT_LEAST_ZERO,
        doc="Stop polishing if the total crack area is above this threshold.",
        display=Display(
            "Max. crack area", scale=1e12, unit="µm²", step=0.01, decimals=2
        ),
    )
    max_milling_cycles: int = param(
        60,
        limits=RangeLimit(1, math.inf),
        doc="Maximum adaptive polish cycles per lamella.",
        display=Display("Max. cycles", step=1),
    )
    lamella_spot_area: int = param(
        20,
        unit="pixels",
        limits=RangeLimit(0, 100),
        doc="Connected areas of lamella containing this many pixels or fewer will be filtered out.",
        display=Display("Lamella spot area", step=1, decimals=0, advanced=True),
    )
    minimum_lamella_area: float = param(
        30.0e-12,  # 30 µm²
        unit="m²",
        limits=_AT_LEAST_ZERO,
        doc="Stop polishing if the segmented lamella area is smaller than this size, which may indicate a major segmentation or alignment issue.",
        display=Display(
            "Min. lamella area",
            scale=1e12,
            unit="µm²",
            step=0.1,
            decimals=1,
            advanced=True,
        ),
    )
    maximum_drift: float = param(
        70e-6,
        unit="m",
        limits=_AT_LEAST_ZERO,
        doc="Stop polishing if the segmented lamella area has moved more than this between cycles, which may indicate a major segmentation or alignment issue.",
        display=Display("Max. lamella drift", scale=1e6, decimals=1, advanced=True),
    )
    align_sem: bool = param(
        True,
        doc="Centre the lamella in the SEM image before polishing.",
        display=Display("Align SEM", advanced=True),
    )
    save_predictions: bool = param(
        True,
        doc="Saves the raw and clean SEM segmentations (useful for improving segmentation model).",
        display=Display("Save segmentations", advanced=True),
    )
    gis_filter_sigma: float = param(
        10 / 6,
        limits=_AT_LEAST_ZERO,
        doc="Sigma of the Gaussian used to smooth the GIS thickness measurements.",
        display=Display("GIS filter sigma", step=0.1, decimals=2, advanced=True),
    )
    lamella_pad_x: float = param(
        0.0,
        limits=RangeLimit(0.0, 1.0),
        doc="The fraction of the lamella width that the X-limits will be padded by. These limits are used to determine the region that GIS thickness is measured.",
        display=Display("X-padding fraction", step=0.01, decimals=2, advanced=True),
    )
    imaging_hfw: float = param(
        40e-6,
        unit="m",
        limits=RangeLimit(20e-6, 950e-6),
        doc="The horizontal field width used for SEM imaging.",
        display=Display(
            "Imaging Field of View", scale=1e6, step=10.0, decimals=2, advanced=True
        ),
    )

    @classmethod
    def ensure_model_path(cls, v: str) -> str:
        if v and not os.path.isfile(v):
            raise FileNotFoundError(f"'{v}' is not a valid file path")
        return v

    def get_model_generation(self) -> str | None:
        model_generation = str(self.model_generation).strip()
        if not model_generation:
            # Interpret empty strings as None
            return None
        return model_generation
