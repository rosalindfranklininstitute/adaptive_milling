from __future__ import annotations

import csv
import logging
import typing
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from fibsem.structures import FibsemImage
from fibsem.milling import FibsemMillingStage, get_milling_stages
from fibsem.utils import load_protocol

from adaptive_polish.config import BitmapAdaptivePolishMillingConfig
from adaptive_polish.strategy import BitmapAdaptivePolishMillingStrategy
from adaptive_polish._dataclasses import CycleInformation, CycleTimestamps
from adaptive_polish.gis_measurement import (
    crop_xlims_centre,
    crop_xlims_minimum,
    crop_xlims_minimum2,
    crop_xlims_minimum_log,
    crop_xlims_minimum_log2,
    crop_xlims_minimum_log10,
    crop_xlims_minimum_log1p,
    crop_xlims_minimum_sigmoid,
    crop_xlims_milled_log,
    crop_xlims_milled_log_convolve,
    crop_xlims_milled_log_convolve2,
    crop_xlims_milled_log_convolve3,
)

if typing.TYPE_CHECKING:
    from collections.abc import Generator
    from os import PathLike

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())

mpl.use("Agg")


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
    kwargs = {"model_path": str(model_path), "lamella_pad_x": lamella_pad_x}
    if model_generation is not None:
        kwargs["model_generation"] = model_generation
    config = BitmapAdaptivePolishMillingConfig(**kwargs)  # type: ignore
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
    cycle_info = CycleInformation(
        milling_cycle=milling_cycle, identifier=identifier, timestamps=CycleTimestamps()
    )
    lamella_info = strategy._get_lamella_info(
        cycle_info=cycle_info,
        sem_image=sem_image,
        fib_image=None,  # type: ignore
        lamella_pad_x=strategy.config.lamella_pad_x,
    )

    if lamella_info.statistics.gis_thickness_filtered_um is None:
        raise ValueError('"gis_thickness_filtered_um" is not available')
    elif lamella_info.statistics.lamella_thickness_um is None:
        raise ValueError('"lamella_thickness_um" is not available')
    elif lamella_info.statistics.xlims_image_px is None:
        raise ValueError('"xlims_image_px" is not defined')

    lamella_width_px_float = (
        milling_stage.pattern.width / sem_image.metadata.pixel_size.x
    )
    lamella_width_px = int(round(lamella_width_px_float))
    if lamella_width_px % 2 != 0:
        lamella_width_px_odd = lamella_width_px
    else:
        diff = lamella_width_px_float - lamella_width_px
        if diff == 0:
            lamella_width_px_odd = lamella_width_px - 1
        else:
            # Gets nearest odd pixel count to lamella_width_px_float (for convolution kernel)
            lamella_width_px_odd = lamella_width_px + int(diff > 0)

    gis_thickness_um = np.asarray(
        lamella_info.statistics.gis_thickness_filtered_um, dtype=np.float64
    )
    lamella_thickness_um = np.asarray(
        lamella_info.statistics.lamella_thickness_um, dtype=np.float64
    )
    print(lamella_thickness_um[len(lamella_thickness_um) // 2])
    lamella_thickness_um = np.interp(
        np.linspace(
            0, len(lamella_thickness_um), len(gis_thickness_um), dtype=np.float64
        ),
        range(len(lamella_thickness_um)),
        lamella_thickness_um,
    )
    print(lamella_thickness_um[len(gis_thickness_um) // 2])

    def run_and_store_method(
        results: dict[str, tuple[int, int]], function, *args, **kwargs
    ):
        results[function.__name__] = function(*args, **kwargs)

    results: dict[str, tuple[int, int]] = {}
    run_and_store_method(
        results,
        crop_xlims_centre,
        lamella_width=lamella_width_px,
        xlims=lamella_info.statistics.xlims_image_px,
    )
    # run_and_store_method(
    #     results,
    #     crop_xlims_minimum,
    #     lamella_thickness=lamella_thickness_um,
    #     gis_thickness=gis_thickness_um,
    #     lamella_width=lamella_width_px,
    # )
    run_and_store_method(
        results,
        crop_xlims_minimum2,
        gis_thickness=gis_thickness_um,
        xlims=lamella_info.statistics.xlims_image_px,
        lamella_width=lamella_width_px,
    )
    run_and_store_method(
        results,
        crop_xlims_minimum_log,
        gis_thickness=gis_thickness_um,
        xlims=lamella_info.statistics.xlims_image_px,
        lamella_width=lamella_width_px,
    )

    # run_and_store_method(
    #     results,
    #     crop_xlims_minimum_log2,
    #     gis_thickness=gis_thickness_um,
    #     xlims=lamella_info.statistics.xlims_image_px,
    #     lamella_width=lamella_width_px,
    # )

    # run_and_store_method(
    #     results,
    #     crop_xlims_minimum_log10,
    #     gis_thickness=gis_thickness_um,
    #     xlims=lamella_info.statistics.xlims_image_px,
    #     lamella_width=lamella_width_px,
    # )

    # for fv in np.arange(0.0, 0.3, 0.1):
    #     for es in range(14, 15, 1):
    #         fn = partial(
    #             crop_xlims_milled_log,
    #             fill_value=fv,
    #             edge_size=es,
    #         )
    #         fn.__name__ = f"crop_xlims_milled_log_{fv=:.1f}_{es=}"

    #         run_and_store_method(
    #             results,
    #             fn,
    #             gis_thickness=gis_thickness_um,
    #             xlims=lamella_info.statistics.xlims_image_px,
    #             lamella_width=lamella_width_px,
    #         )

    for fv in np.arange(0.0, 0.1, 0.1):
        for es in range(3, 4, 2):
            # This is basically convolving with 0s so what's the point?
            fn = partial(
                crop_xlims_milled_log_convolve,
                fill_value=fv,
                edge_size=es,
            )
            fn.__name__ = f"crop_xlims_milled_log_convolve_{fv=:.1f}_{es=}"

            run_and_store_method(
                results,
                fn,
                gis_thickness=gis_thickness_um,
                xlims=lamella_info.statistics.xlims_image_px,
                lamella_width=lamella_width_px_odd,
            )

    for es in range(16, 18, 1):
        for sigma in np.arange(0.7, 0.8, 0.1):
            fn = partial(
                crop_xlims_milled_log_convolve2,
                edge_size=es,
                sigma=sigma,
            )
            fn.__name__ = f"crop_xlims_milled_log_convolve2_{sigma=:.1f}_{es=}"

            run_and_store_method(
                results,
                fn,
                gis_thickness=gis_thickness_um,
                xlims=lamella_info.statistics.xlims_image_px,
                lamella_width=lamella_width_px_odd,
            )

            # fn = partial(
            #     crop_xlims_milled_log_convolve2,
            #     edge_size=12,
            #     sigma=0.2,
            # )
            # fn.__name__ = (
            #     "crop_xlims_milled_log_convolve2_sigma=0.2_es=12"
            # )

            # run_and_store_method(
            #     results,
            #     fn,
            #     gis_thickness=gis_thickness_um,
            #     xlims=lamella_info.statistics.xlims_image_px,
            #     lamella_width=lamella_width_px_odd,
            # )

            # fn = partial(
            #     crop_xlims_milled_log_convolve2,
            #     edge_size=17,
            #     sigma=0.7,
            # )
            # fn.__name__ = (
            #     "crop_xlims_milled_log_convolve2_sigma=0.7_es=17"
            # )

            # run_and_store_method(
            #     results,
            #     fn,
            #     gis_thickness=gis_thickness_um,
            #     xlims=lamella_info.statistics.xlims_image_px,
            #     lamella_width=lamella_width_px_odd,
            # )

    # for es in range(12, 18, 5):
    #     for sigma in np.arange(0.2, 0.8, 0.5):
    #         for sigma_trim in np.arange(5.3, 6.5, 0.1):
    #             fn = partial(
    #                 crop_xlims_milled_log_convolve3,
    #                 fill_value=fv,
    #                 edge_size=es,
    #                 sigma=sigma,
    #             )
    #             fn.__name__ = (
    #                 f"crop_xlims_milled_log_convolve3_{sigma=:.1f}_{es=}_{sigma_trim=:.1f}"
    #             )

    #             run_and_store_method(
    #                 results,
    #                 fn,
    #                 gis_thickness=gis_thickness_um,
    #                 xlims=lamella_info.statistics.xlims_image_px,
    #                 lamella_width=lamella_width_px_odd,
    #             )

    # for fv in np.arange(0.0, 0.1, 0.2):
    #     for es in range(12, 13, 1):
    #         for sigma in np.arange(0.2, 0.3, 0.1):
    #             # This is basically convolving with 0s so what's the point?
    #             fn = partial(
    #                 crop_xlims_milled_log_correlate,
    #                 fill_value=fv,
    #                 edge_size=es,
    #                 sigma=sigma,
    #             )
    #             fn.__name__ = (
    #                 f"crop_xlims_milled_log_correlate_{fv=:.1f}_{sigma=:.1f}_{es=}"
    #             )

    #             run_and_store_method(
    #                 results,
    #                 fn,
    #                 gis_thickness=gis_thickness_um,
    #                 xlims=lamella_info.statistics.xlims_image_px,
    #                 lamella_width=lamella_width_px_odd,
    #             )

    # TODO: Combine methods. Crop centrally with a bit of buffer room then apply one of the the other methods?

    # run_and_store_method(
    #     results,
    #     crop_xlims_minimum_log1p,
    #     gis_thickness=gis_thickness_um,
    #     xlims=lamella_info.statistics.xlims_image_px,
    #     lamella_width=lamella_width_px,
    # )

    # run_and_store_method(
    #     results,
    #     crop_xlims_minimum_sigmoid,
    #     gis_thickness=gis_thickness_um,
    #     xlims=lamella_info.statistics.xlims_image_px,
    #     lamella_width=lamella_width_px,
    # )

    return results


def display_diff_stats(
    function_diffs: dict[str, list[float]], plot: bool = True
) -> None:
    names = list(function_diffs.keys())
    diffs = np.asarray(list(function_diffs.values()))

    means = np.mean(diffs, axis=1)

    quartiles1, medians, quartiles3 = np.percentile(diffs, [25, 50, 75], axis=1)
    stdevs = np.std(diffs, axis=1)

    abs_diffs = np.abs(diffs)
    # abs_maxs = np.max(abs_diffs, axis=1)
    abs_means = np.mean(abs_diffs, axis=1)
    abs_quartiles1, abs_medians, abs_quartiles3 = np.percentile(
        abs_diffs, [25, 50, 75], axis=1
    )
    abs_stdevs = np.std(abs_diffs, axis=1)

    for i in np.argsort(abs_means + abs_stdevs * 2)[::-1]:
        print(
            f"""{names[i]} diff statistics (pixels):
    relative:
        median:\t{medians[i]}
        mean:\t{means[i]}
        std:\t{stdevs[i]}
    absolute:
        median:\t{abs_medians[i]}
        mean:\t{abs_means[i]}
        std:\t{abs_stdevs[i]}

"""
        )

    if plot:
        width = 2 + 4 * len(names)
        fig, axs = plt.subplots(2, 1, sharex=True, squeeze=True, figsize=(width, 12))
        axs[0].violinplot(
            [_ for _ in abs_diffs],
            showmeans=False,
            showmedians=False,
        )
        idxs = np.arange(1, len(names) + 1)
        axs[0].errorbar(
            idxs,
            abs_medians,
            yerr=[abs_medians - abs_quartiles1, abs_quartiles3 - abs_medians],
            label="quartiles",
            capsize=0.2,
            fmt="none",
            ecolor="tab:orange",
        )
        axs[0].scatter(
            idxs,
            abs_medians,
            label="median",
            c="tab:green",
            marker="_",
            s=30,
            zorder=4,
        )
        axs[0].scatter(
            idxs,
            abs_means,
            label="mean",
            c="k",
            marker="x",
            s=30,
            zorder=3,
        )

        axs[0].yaxis.grid(True)
        axs[0].set_ylabel("Absolute difference (px)")
        axs[0].legend()

        axs[1].violinplot([_ for _ in diffs], showmeans=False, showmedians=False)
        idxs = np.arange(1, len(names) + 1)
        axs[1].errorbar(
            idxs,
            medians,
            yerr=[medians - quartiles1, quartiles3 - medians],
            label="quartiles",
            capsize=0.2,
            fmt="none",
            ecolor="tab:orange",
        )
        axs[1].scatter(
            idxs,
            medians,
            label="median",
            c="tab:green",
            marker="_",
            s=30,
            zorder=3,
        )
        axs[1].scatter(
            idxs,
            means,
            label="mean",
            c="k",
            marker="x",
            s=30,
            zorder=2,
        )
        axs[1].yaxis.grid(True)
        axs[1].set_xticks(np.arange(1, len(names) + 1), labels=names, minor=False)
        axs[1].set_xticklabels(labels=names, rotation=45)
        axs[1].set_xlim(0.25, len(names) + 0.75)
        axs[1].set_xlabel("Centring method")
        axs[1].set_ylabel("Relative difference (px)")
        fig.tight_layout()

        plot_path = Path(__file__).parent / "centring_plot.png"
        fig.savefig(plot_path)


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
    display_diff_stats(function_diffs=function_diffs)


if __name__ == "__main__":
    model_path = ""
    model_generation = "1.4fpn"
    csv_path = ""
    run(csv_path=csv_path, model_path=model_path, model_generation=model_generation)
