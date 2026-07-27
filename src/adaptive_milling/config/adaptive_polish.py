from __future__ import annotations

import os
from dataclasses import dataclass, field

from fibsem.milling.base import MillingStrategyConfig

from adaptive_milling.models import (
    MODEL_GENERATIONS_DICT,
    get_latest_generation_key,
)

DEFAULT_MODEL_GENERATION = get_latest_generation_key()


@dataclass
class AdaptivePolishMillingConfig(MillingStrategyConfig):
    model_path: str = field(
        default="",
        metadata={
            "label": "Model path",
            "tooltip": "Path to the SEM segmentation model weights (*.pth).",
            "type": str,
            "filepath": True,
        },
    )
    model_generation: str = field(
        default=DEFAULT_MODEL_GENERATION,
        metadata={
            "label": "Model generation",
            "tooltip": "The generation of model to use (defaults to latest).",
            "type": str,
            "items": list(MODEL_GENERATIONS_DICT.keys()),
        },
    )
    gis_stop_min: float = field(
        default=0.2e-6,
        metadata={
            "label": "Minimum GIS thickness threshold",
            "tooltip": "Stop polishing if the minimum GIS thickness drops below this threshold.",
            "type": float,
            "minimum": 0,
            "step": 0.01,
            "decimals": 2,
            "scale": 1e6,
            "unit": "m",
        },
    )
    gis_stop_median: float = field(
        default=0.25e-6,
        metadata={
            "label": "Median GIS thickness threshold",
            "tooltip": "Stop polishing if the median GIS thickness drops below this threshold.",
            "type": float,
            "minimum": 0,
            "step": 0.01,
            "decimals": 2,
            "scale": 1e6,
            "unit": "m",
        },
    )
    max_crack_area: float = field(
        default=2e-12,
        metadata={
            "label": "Max. crack area",
            "tooltip": "Stop polishing if the maximum total crack area is above this threshold.",
            "type": float,
            "minimum": 0,
            "step": 0.01,
            "decimals": 2,
            "scale": 1e6,
            "dimensions": 2,
            "unit": "m²",
        },
    )
    max_milling_cycles: int = field(
        default=30,
        metadata={
            "label": "Max. cycles",
            "tooltip": "Maximum adaptive polish cycles per lamella.",
            "type": int,
            "minimum": 1,
            "step": 1,
        },
    )
    minimum_lamella_area: float = field(
        default=30.0e-12,  # 30μm²
        metadata={
            "label": "Min. lamella area",
            "tooltip": "Stop polishing if the segmented lamella area is smaller than this size, which may indicate a segmentation issue.",
            "type": float,
            "minimum": 0,
            "step": 0.1,
            "decimals": 1,
            "scale": 1e6,
            "dimensions": 2,
            "unit": "m²",
            "advanced": True,
        },
    )
    maximum_drift: float = field(
        default=70e-6,
        metadata={
            "label": "Max. lamella drift",
            "tooltip": "Stop polishing if the segmented lamella area has moved more than this between cycles, which would suggest that something is wrong.",
            "type": float,
            "minimum": 0,
            "scale": 1e6,
            "decimals": 1,
            "unit": "m",
            "advanced": True,
        },
    )
    align_sem: bool = field(
        default=True,
        metadata={
            "label": "Align SEM",
            "tooltip": "Centre the lamella in the SEM image before beginning.",
            "type": bool,
            "advanced": True,
        },
    )
    save_predictions: bool = field(
        default=True,
        metadata={
            "label": "Save segmentations",
            "tooltip": "Save the raw and clean SEM segmentations (useful for improving segmentation model).",
            "type": bool,
            "advanced": True,
        },
    )
    gis_filter_sigma: float = field(
        default=10 / 6,
        metadata={
            "label": "GIS filter sigma",
            "tooltip": "Gaussian blur sigma used to smooth the GIS thickness measurements.",
            "type": float,
            "minimum": 0,
            "step": 0.1,
            "decimals": 2,
            "advanced": True,
        },
    )
    lamella_pad_x: float = field(
        default=0.0,
        metadata={
            "label": "X-padding fraction",
            "tooltip": "The fraction of the lamella width that the X-limits will "
            "be padded by. These limits are used to determine the region that GIS "
            "thickness is measured.",
            "type": float,
            "minimum": 0,
            "maximum": 1,
            "step": 0.01,
            "decimals": 2,
            "advanced": True,
        },
    )

    @classmethod
    def ensure_model_path(cls, v: str) -> str:
        if v and not os.path.isfile(v):
            raise FileNotFoundError(f"'{v}' is not a valid file path")
        return v

    def get_model_generation(self) -> str | None:
        model_generation = self.model_generation.strip()
        if not model_generation:
            # Interpret empty strings as None
            return None
        return model_generation
