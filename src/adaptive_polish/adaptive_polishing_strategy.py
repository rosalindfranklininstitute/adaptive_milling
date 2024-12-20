import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.base import (
    FibsemMillingStage,
    MillingStrategy,
    MillingStrategyConfig,
)
from fibsem.milling.patterning.patterns2 import TrenchPattern

from adaptive_polish.franklin_adaptive_milling_dl_fibsem import AdaptiveMilling

@dataclass
class AdaptivePolishingConfig(MillingStrategyConfig):
    milling_interval: int = 10  # seconds
    gis_stop_m: float = 2e-7
    fallback_pixel_size_m: float = 1.8e-8
    max_milling_cycles: int = 3
    reject_GIS_distance_m: float = 1e-5
    window_size_m: float = 1e-7
    max_crack_area_m2: float = 2e-12
    lam_height_min_m: float = 2e-6
    model_path: str = None
    align_sem: bool = True
    sem_resolution: Tuple[int, int] = None
    sem_hfw: float = 40e-6
    sem_dwell_time: float = 200e-9
    sem_frame_integration: int = 8
    fib_resolution: Tuple[int, int] = None
    fib_hfw: float = 40e-6
    fib_dwell_time: float = 200e-9
    fib_frame_integration: int = 8
    do_plots: bool = True
    _advanced_attributes = ["fallback_pixel_size_m", "reject_GIS_distance_m", 
                                       "window_size_m", "max_crack_area_m2", "lam_height_min_m",
                                      "sem_resolution", "sem_hfw", "sem_dwell_time", "sem_frame_integration", 
                                      "fib_resolution", "fib_hfw", "fib_dwell_time", "fib_frame_integration", "do_plots"]

    def __post_init__(self):
        if self.sem_resolution is None:
            self.sem_resolution = [3072, 2048]
        if self.fib_resolution is None:
            self.fib_resolution = [3072, 2048]
        if self.model_path is None:
            self.model_path = f"{Path(__file__).parent}/dl_segmentation/2024-02-24_0013_gis_lamela_crack_pytorch_AUnet.ptchkp"

    @staticmethod
    def from_dict(d: dict) -> "AdaptivePolishingConfig":
        return AdaptivePolishingConfig()  # TODO: proper load

    def to_dict(self) -> dict:
        return self._to_expected_dict()

    def _to_expected_dict(self) -> dict:
        if self.model_path is None:
            self.model_path = f"{Path(__file__).parent}/dl_segmentation/2024-02-24_0013_gis_lamela_crack_pytorch_AUnet.ptchkp"

        return {
            "do_plots": self.do_plots,
            "fallback_pixel_size_m": self.fallback_pixel_size_m,
            "milling_interval_s": self.milling_interval,
            "gis_stop_m": self.gis_stop_m,
            "max_milling_cycles": self.max_milling_cycles,
            "window_size_m": self.window_size_m,
            "max_crack_area_m2": self.max_crack_area_m2,
            "lam_height_min_m": self.lam_height_min_m,
            "model_path": self.model_path,
            "use_sem_beam_shift_alignment_adaptive_polish": self.align_sem,
            "imaging_settings": {
                "electron": {
                    "resolution": self.sem_resolution,
                    "hfw": self.sem_hfw,
                    "dwell_time": self.sem_dwell_time,
                    "frame_integration": self.sem_frame_integration,
                },
                "ion": {
                    "resolution": self.fib_resolution,
                    "hfw": self.fib_hfw,
                    "dwell_time": self.fib_dwell_time,
                    "frame_integration": self.fib_frame_integration,
                },
            },
        }


@dataclass
class AdaptivePolishingStrategy(MillingStrategy):
    """Adaptive polishing strategy"""

    name: str = "AdaptivePolishing"
    fullname: str = "Adaptive Polishing"

    def __init__(self, config: AdaptivePolishingConfig = None):
        self.config = config or AdaptivePolishingConfig()

    def to_dict(self) -> dict:
        return {"name": self.name, "config": self.config.to_dict()}

    @staticmethod
    def from_dict(d: dict) -> "AdaptivePolishingStrategy":
        config = AdaptivePolishingConfig.from_dict(d["config"])
        return AdaptivePolishingStrategy(config=config)

    def run(
        self,
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
        asynch: bool = False,
        parent_ui=None,
    ) -> None:
        """Run adaptive polishing strategy"""
        logging.info(f"Running {self.fullname} for {stage.name}")

        # assert pattern is TrenchPattern
        if not isinstance(stage.pattern, TrenchPattern):
            raise ValueError("Pattern must be TrenchPattern for adaptive polishing")

        # setup milling
        ap = AdaptiveMilling(self.config.to_dict())

        # run adaptive polishing
        ap.adaptive_polish_run(
            microscope,
            None,
            patterns_in=stage.pattern,
            milling_stage=stage,
        )


# TODO: run setup milling before hand for some initialisation (at least once)
# TODO: only have a single layout of the configuration
# TODO: proper load from dict
# TODO: correctly scale values in UI
# QUERY: what is the purpose of the fallback_pixel_size_m?
# QUERY: LOG_AP path is not correct?
