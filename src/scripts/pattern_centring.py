from __future__ import annotations

import csv
import statistics
import logging
import typing
from pathlib import Path

import numpy as np

from fibsem.structures import FibsemImage
from fibsem.milling import FibsemMillingStage, get_milling_stages
from fibsem.utils import load_protocol

from adaptive_polish.config import BitmapAdaptivePolishMillingConfig
from adaptive_polish.strategy import BitmapAdaptivePolishMillingStrategy
from adaptive_polish.gis_measurement import crop_xlims_centre, crop_xlims_minimum

if typing.TYPE_CHECKING:
    from collections.abc import Generator
    from os import PathLike

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())


def read_csv(fp: str | PathLike[str]) -> Generator[tuple[Path, float], None, None]:
    fp = Path(fp)
    with fp.open("r") as f:
        reader = csv.reader(f)
        for row in reader:
            yield Path(row[0]), float(row[1])


def setup_strategy(
    model_path: str | PathLike[str],
    model_generation: str | None = None,
    lamella_pad_x: float = 0.0,
) -> BitmapAdaptivePolishMillingStrategy:
    _logger.info("Setting up strategy")
    kwargs = {"model_path": model_path, "lamella_pad_x": lamella_pad_x}
    if model_generation is not None:
        kwargs["model_generation"] = model_generation
    config = BitmapAdaptivePolishMillingConfig(**kwargs)
    strategy = BitmapAdaptivePolishMillingStrategy(config)
    strategy._load_model()
    return strategy


def get_milling_stage(image_path: str | PathLike[str]) -> FibsemMillingStage | None:
    _logger.info("Loading protocol for %s", image_path)
    protocol_path = Path(image_path).parent.parent.parent.parent / "protocol.yaml"
    protocol = load_protocol(protocol_path)
    _logger.info("Getting milling stage")
    polishing_stages = get_milling_stages("mill_polishing", protocol["milling"])
    for polishing_stage in polishing_stages[::-1]:
        if polishing_stage.strategy.name == "AdaptivePolishing":
            return polishing_stage
    return None


def get_xlims(
    milling_cycle: int,
    identifier: str,
    sem_image: FibsemImage,
    milling_stage: FibsemMillingStage,
    strategy: BitmapAdaptivePolishMillingStrategy,
) -> dict[str, tuple[int, int]]:
    _logger.info("Getting xlims")
    lamella_info = strategy._get_lamella_info(
        milling_cycle=milling_cycle,
        identifier=identifier,
        sem_image=sem_image,
        fib_image=None,  # type: ignore
        milling_stage=milling_stage,
        lamella_pad_x=strategy.config.lamella_pad_x,
    )

    if lamella_info.statistics.gis_thickness_filtered_um is None:
        raise ValueError('"gis_thickness_filtered_um" is not available')
    elif lamella_info.statistics.lamella_thickness_um is None:
        raise ValueError('"lamella_thickness_um" is not available')
    elif lamella_info.statistics.xlims_px is None:
        raise ValueError('"xlims_px" is not defined')

    lamella_width_px = int(
        round(milling_stage.pattern.width / sem_image.metadata.pixel_size.x)  # type: ignore
    )

    gis_thickness_um = np.asarray(
        lamella_info.statistics.gis_thickness_filtered_um, dtype=np.float32
    )
    lamella_thickness_um = np.asarray(
        lamella_info.statistics.lamella_thickness_um, dtype=np.float32
    )
    print(lamella_thickness_um[len(lamella_thickness_um) // 2])
    lamella_thickness_um = np.interp(
        np.linspace(0, len(lamella_thickness_um), len(gis_thickness_um)),
        range(len(lamella_thickness_um)),
        lamella_thickness_um,
    )
    print(lamella_thickness_um[len(gis_thickness_um) // 2])

    results: dict[str, tuple[int, int]] = {}
    results[crop_xlims_centre.__name__] = crop_xlims_centre(
        lamella_width_px, xlims=lamella_info.statistics.xlims_px
    )
    results[crop_xlims_minimum.__name__] = crop_xlims_minimum(
        lamella_thickness=lamella_thickness_um,
        gis_thickness=gis_thickness_um,
        lamella_width=lamella_width_px,
    )
    return results


def run(
    csv_path: str | PathLike[str],
    model_path: str | PathLike[str],
    model_generation: str | None = None,
) -> None:
    csv_path = Path(csv_path)
    strategy = setup_strategy(model_path=model_path, model_generation=model_generation)
    function_diffs: dict[str, list[float]] = {}
    for i, (image_path, centre_x) in enumerate(read_csv(csv_path)):
        full_image_path = csv_path.parent / image_path
        stage = get_milling_stage(full_image_path)
        if stage is None:
            _logger.error("Failed to get stage for %s", image_path)
            continue
        sem_image = FibsemImage.load(str(full_image_path))
        xlim_dict = get_xlims(
            milling_cycle=i,
            identifier=full_image_path.stem,
            sem_image=sem_image,
            milling_stage=stage,
            strategy=strategy,
        )
        for name, xlims in xlim_dict.items():
            diff = centre_x - (sum(xlims) / 2)
            print(f"Difference for {image_path} using {name} is {diff}")
            if name not in function_diffs:
                function_diffs[name] = []
            function_diffs[name].append(diff)
    for name, diffs in function_diffs.items():
        abs_diffs = [abs(_) for _ in diffs]
        median = statistics.median(abs_diffs)
        mean = statistics.fmean(abs_diffs)
        stdev = statistics.stdev(abs_diffs)
        print(f"{name} diff statistics (pixels): {median=}, {mean=}, {stdev=}")


if __name__ == "__main__":
    model_path = ""
    model_generation = "1.4fpn"
    csv_path = ""
    run(csv_path=csv_path, model_path=model_path, model_generation=model_generation)
