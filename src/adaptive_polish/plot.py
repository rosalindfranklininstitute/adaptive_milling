from __future__ import annotations
import logging
import typing
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from fibsem import constants
from fibsem.structures import Point, FibsemImage
from fibsem.milling import FibsemMillingStage
from fibsem.milling.patterning.plotting import draw_milling_patterns

from adaptive_polish.processing.sem_segmentation import (
    SegmentationLabels as SemSegmentationLabels,
)

if typing.TYPE_CHECKING:
    from os import PathLike
    from collections.abc import Sequence
    from numpy.typing import NDArray, ArrayLike
    from fibsem.structures import FibsemImage
    from adaptive_polish._dataclasses import StrategyRunInformation

_logger = logging.getLogger(__name__)

plt.rc("axes", titlesize="small")
plt.rc("figure", titlesize="large")


def __create_cmap() -> ListedColormap:
    _tab10 = plt.get_cmap("tab10")
    return ListedColormap([_tab10(_.value) for _ in SemSegmentationLabels])


LABEL_CMAP = __create_cmap()


def create_centring_plot(
    sem_image: FibsemImage,
    mask_lamella_clean: NDArray[np.bool_],
    centre_px: Point | None,
    centre_m: Point | None,
    plot_path: str | PathLike[str],
    prediction: NDArray[np.integer] | None = None,
    bounding_box: tuple[float, float, float, float] | None = None,
) -> None:
    _logger.debug("Creating centring plot")
    # Plot centring stuff
    if prediction is not None:
        fig, axs = plt.subplots(1, 2)
        axs = axs.ravel()[::-1]
    else:
        fig, ax = plt.subplots(1, 1)
        axs = [ax]

    _ = axs[0].imshow(sem_image.data, cmap="gray")
    extent = _.get_extent()
    axs[0].set_title("SEM lamella")
    axs[0].imshow(
        # Overlay the cleaned lamella
        mask_lamella_clean,
        cmap=ListedColormap(
            [(0, 0, 0, 0), LABEL_CMAP(SemSegmentationLabels.LAMELLA.value)]
        ),
        extent=extent,
        alpha=0.5,
        interpolation="none",
    )
    if bounding_box is not None:
        axs[0].add_patch(
            Rectangle(
                (bounding_box[1], bounding_box[0]),
                width=bounding_box[3] - bounding_box[1],
                height=bounding_box[2] - bounding_box[0],
                edgecolor="red",
                facecolor="none",
                alpha=0.5,
            )
        )
    if prediction is not None:
        axs[1].set_title("SEM segmentation")
        axs[1].imshow(
            prediction,
            cmap=LABEL_CMAP,
            extent=extent,
            vmin=0,
            vmax=LABEL_CMAP.N,
            interpolation="none",
        )

    for ax in axs:
        # Add centre markers to both
        if centre_px is not None:
            ax.scatter(
                centre_px.x,
                centre_px.y,
                c="r",
                marker="+",
                label="Lamella Centre",
            )
        ax.scatter(
            sem_image.data.shape[1] // 2,
            sem_image.data.shape[0] // 2,
            c="g",
            marker="+",
            label="Image Centre",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    axs[-1].legend()  # No need to have a duplicate legend

    if centre_m is None:
        fig.suptitle("Lamella Centre\nNot found")
    else:
        fig.suptitle(
            "Lamella Centre\n"
            rf"x, y: {centre_m.x * constants.SI_TO_MICRO:.4f}, {centre_m.y * constants.SI_TO_MICRO:.4f} $\mu m$"
        )
    fig.tight_layout()

    fig.savefig(plot_path)
    plt.close(fig)


def plot_segmentation_overlay(
    ax: plt.axes.Axes,
    sem_image: NDArray[typing.Any],
    prediction: NDArray[np.integer[typing.Any]],
    alpha: float = 0.4,
) -> None:
    _ = ax.imshow(sem_image.data, cmap="Greys_r")
    extent = _.get_extent()
    ax.imshow(
        prediction,
        alpha=alpha,
        cmap=LABEL_CMAP,
        vmin=0,
        vmax=LABEL_CMAP.N,
        extent=extent,
        interpolation="none",
    )


def create_milling_cycle_plot(
    save_path: str | PathLike[str],
    sem_image: FibsemImage,
    fib_image: FibsemImage,
    first_prediction: NDArray[np.integer[typing.Any]],
    clean_prediction: NDArray[np.integer[typing.Any]],
    gis_thickness_um: ArrayLike | None = None,
    gis_thickness_min_um: float | None = None,
    crack_area_um2: float | None = None,
    gis_min_stop_threshold_um: float | None = None,
    gis_thickness_median_um: float | None = None,
    gis_median_stop_threshold_um: float | None = None,
    gis_min_threshold_um: float | None = None,
    gis_max_threshold_um: float | None = None,
    milling_stage: FibsemMillingStage | None = None,
    total_milling_time: float | None = None,
    max_crack_area_um2: float | None = None,
    image_xlims: tuple[int, int] | None = None,
    img_name: str | None = None,
    gis_ymax_um: float = 3,
    pattern_dwell_multiplier: Sequence[float] | None = None,
    pattern_xlims: tuple[int, int] | None = None,
) -> None:
    _logger.debug("Creating milling cycle plot")
    fig, axs = plt.subplots(
        nrows=2, ncols=3, figsize=(15, 10), dpi=300, tight_layout=True
    )
    plot_title = img_name
    if total_milling_time is not None:
        plot_title = f"{plot_title}\nMilling time: {total_milling_time:.2f} s"

    if plot_title is not None:
        fig.suptitle(plot_title)

    # SEM
    axs[0, 0].imshow(sem_image.data, cmap="Greys_r")
    axs[0, 0].axis("off")
    axs[0, 0].set_title("SEM")

    # SEM + 1st pass prediction
    plot_segmentation_overlay(
        axs[0, 1], sem_image=sem_image.data, prediction=first_prediction
    )
    axs[0, 1].axis("off")
    axs[0, 1].set_title("SEM segmentation")

    # SEM + clean prediction
    plot_segmentation_overlay(
        axs[0, 2], sem_image=sem_image.data, prediction=clean_prediction
    )
    if image_xlims is not None:
        axs[0, 2].axvline(x=image_xlims[0], color="C4")
        axs[0, 2].axvline(x=image_xlims[1], color="C4")
    axs[0, 2].axis("off")

    _crack_title = "SEM cleaned segmentation"
    _crack_subtitle: list[str] = []
    if crack_area_um2 is not None:
        _crack_subtitle.append(rf"Crack area {crack_area_um2:.2f} $\mu m^2$")
    if max_crack_area_um2 is not None:
        _crack_subtitle.append(f"(threshold {max_crack_area_um2:.2f})")
    if _crack_subtitle:
        _crack_title += "\n" + " ".join(_crack_subtitle)
    axs[0, 2].set_title(_crack_title)

    # FIB image
    axs[1, 0].imshow(fib_image.data, cmap="Greys_r")
    axs[1, 0].axis("off")
    axs[1, 0].set_title("FIB")

    # FIB + milling box
    if milling_stage is not None:
        try:
            draw_milling_patterns(
                fib_image,
                milling_stages=[milling_stage],
                crosshair=True,
                scalebar=False,
                show_preset=False,
                title="Milling pattern",
                ax=axs[1, 1],
            )
        except Exception:
            logging.error("Failed to draw milling patterns", exc_info=True)
        axs[1, 1].axis("off")

    if gis_thickness_um is None:
        axs[1, 2].axis("off")
    else:
        handles = []
        labels = []
        gis_thickness_um = np.asarray(gis_thickness_um)

        _gis_title = "GIS thickness"
        _gis_subtitle: list[str] = []

        # GIS thickness
        thickness_colour = "tab:blue"
        axs[1, 2].plot(gis_thickness_um, ".-", c=thickness_colour)
        axs[1, 2].tick_params(axis="y", labelcolor=thickness_colour)
        axs[1, 2].set_xlabel("Distance along x $px$")
        axs[1, 2].set_ylabel(r"GIS thickness ($\mu m$)", color=thickness_colour)
        axs[1, 2].set_xlim(0, len(gis_thickness_um))
        axs[1, 2].set_ylim(0, gis_ymax_um)

        if gis_min_threshold_um is not None:
            axs[1, 2].hlines(
                y=gis_min_threshold_um,
                xmin=0,
                xmax=len(gis_thickness_um),
                label="Minimum dwell threshold",
                linestyles="dashed",
                colors="orange",
            )
        if gis_max_threshold_um is not None:
            axs[1, 2].hlines(
                y=gis_max_threshold_um,
                xmin=0,
                xmax=len(gis_thickness_um),
                label="Full dwell threshold",
                linestyles="dashed",
                colors="green",
            )

        _min_gis_subtitle: list[str] = ["Minimum"]
        if gis_thickness_min_um is not None:
            _min_gis_subtitle.append(rf"{gis_thickness_min_um:.3f} $\mu m$")
        if gis_min_stop_threshold_um is not None:
            _min_gis_subtitle.append(f"(threshold {gis_min_stop_threshold_um:.3f})")
            axs[1, 2].hlines(
                y=gis_min_stop_threshold_um,
                xmin=0,
                xmax=len(gis_thickness_um),
                label="Min thickness threshold",
                linestyles="dashed",
                colors="red",
            )
        if _min_gis_subtitle:
            _gis_subtitle.append(" ".join(_min_gis_subtitle))

        _median_gis_subtitle: list[str] = ["Median"]
        if gis_thickness_median_um is not None:
            _median_gis_subtitle.append(rf"{gis_thickness_median_um:.3f} $\mu m$")
        if gis_median_stop_threshold_um is not None:
            _median_gis_subtitle.append(
                f"(threshold {gis_median_stop_threshold_um:.3f})"
            )
            axs[1, 2].hlines(
                y=gis_median_stop_threshold_um,
                xmin=0,
                xmax=len(gis_thickness_um),
                label="Median thickness threshold",
                linestyles="-.",
                colors="red",
            )
        if _median_gis_subtitle:
            _gis_subtitle.append(" ".join(_median_gis_subtitle))

        if image_xlims is not None:
            axs[1, 2].axvline(x=image_xlims[0], color="C4")
            axs[1, 2].axvline(x=image_xlims[1], color="C4")

        handles_and_labels = axs[1, 2].get_legend_handles_labels()
        handles.extend(handles_and_labels[0])
        labels.extend(handles_and_labels[1])

        if pattern_dwell_multiplier is not None and pattern_xlims is not None:
            dwell_time_colour = "tab:orange"
            dwell_time_axis = axs[1, 2].twinx()
            dwell_time_axis.plot(
                np.arange(pattern_xlims[0], pattern_xlims[1] + 1),
                pattern_dwell_multiplier,
                "-",
                c=dwell_time_colour,
            )
            dwell_time_axis.tick_params(axis="y", labelcolor=dwell_time_colour)
            dwell_time_axis.set_ylabel(
                r"Dwell time multiplier", color=dwell_time_colour
            )
            dwell_time_axis.set_ylim(0, 1)

            dwell_time_axis.hlines(
                y=np.mean(pattern_dwell_multiplier),
                xmin=0,
                xmax=len(gis_thickness_um),
                label="Mean dwell time multiplier",
                linestyles="dotted",
                colors=dwell_time_colour,
            )
            handles_and_labels = dwell_time_axis.get_legend_handles_labels()
            handles.extend(handles_and_labels[0])
            labels.extend(handles_and_labels[1])

            # Plot the legend on the top axis (dwell_time_axis, if created)
            dwell_time_axis.legend(handles, labels)
        else:
            axs[1, 2].legend(handles, labels)

        if _gis_subtitle:
            _gis_title += "\n" + "\n".join(_gis_subtitle)

        axs[1, 2].set_title(_gis_title)

    fig.savefig(save_path)
    plt.close(fig)


def create_summary_gis_plot(
    run_info: StrategyRunInformation, save_path: str | PathLike[str]
) -> None:
    milling_times: list[float | None] = []
    min_gis_thicknesses: list[float | None] = []
    mean_gis_thicknesses: list[float | None] = []
    median_gis_thicknesses: list[float | None] = []
    for cycle_info in run_info.cycle_information:
        if cycle_info.lamella_statistics is None:
            milling_times.append(None)
            min_gis_thicknesses.append(None)
            mean_gis_thicknesses.append(None)
            median_gis_thicknesses.append(None)
        else:
            milling_times.append(cycle_info.lamella_statistics.estimated_milling_time_s)
            min_gis_thicknesses.append(
                cycle_info.lamella_statistics.gis_thickness_min_um
            )
            mean_gis_thicknesses.append(
                cycle_info.lamella_statistics.gis_thickness_mean_um
            )
            median_gis_thicknesses.append(
                cycle_info.lamella_statistics.gis_thickness_median_um
            )
    _logger.debug("Creating GIS summary plot")
    save_path = Path(save_path)
    fig, ax = plt.subplots(1, 1)
    cumulative_milling_time = np.nancumsum(np.asarray(milling_times, dtype=np.float_))
    ax.plot(
        cumulative_milling_time,
        np.asarray(min_gis_thicknesses, dtype=np.float_),
        label="Minimum",
    )
    ax.plot(
        cumulative_milling_time,
        np.asarray(mean_gis_thicknesses, dtype=np.float_),
        label="Mean",
        linestyle=":",
    )
    ax.plot(
        cumulative_milling_time,
        np.asarray(median_gis_thicknesses, dtype=np.float_),
        label="Median",
        linestyle="--",
    )
    ax.set_xlabel(r"Estimated Milling Time $s$")
    ax.set_ylabel(r"GIS Thickness $\mu m$")
    ax.legend()
    fig.suptitle(save_path.stem)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
