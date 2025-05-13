from __future__ import annotations
import logging
import typing
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from fibsem import constants
from fibsem.structures import Point

from adaptive_polish.dl_segmentation.sem_lamella_segmentor import SegmentationLabels

if typing.TYPE_CHECKING:
    from os import PathLike
    from pandas import DataFrame
    from numpy.typing import NDArray, ArrayLike
    from fibsem.structures import FibsemImage

_logger = logging.getLogger(__name__)

plt.rc("axes", titlesize="small")
plt.rc("figure", titlesize="large")
LABEL_CMAP = plt.get_cmap("tab10")



def create_centring_plot(
    sem_image: FibsemImage,
    mask_lamella_clean: NDArray[np.bool_],
    centre_px: typing.Optional[Point],
    centre_m: typing.Optional[Point],
    plot_path: typing.Union[str, PathLike],
    prediction: typing.Optional[NDArray[np.integer]] = None,
    bounding_box: typing.Optional[tuple[float, float, float, float]] = None,
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
            [(0, 0, 0, 0), LABEL_CMAP(SegmentationLabels.LAMELLA.value)]
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
            vmax=len(LABEL_CMAP.colors),
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


def create_milling_cycle_plot(
    sem_image: NDArray[typing.Any],
    first_prediction: NDArray[np.integer],
    clean_prediction: NDArray[typing.Any],
    fib_image: NDArray[typing.Any],
    gis_thickness_um: ArrayLike,
    gis_stop_um: float,
    crack_area_um2: float,
    min_gis_um: float,
    total_milling_time: typing.Optional[float] = None,
    max_crack_area_um2: typing.Optional[float] = None,
    xlims: typing.Optional[typing.Tuple[int, int]] = None,
    fib_screenshot: typing.Optional[NDArray[typing.Any]] = None,
    img_name: typing.Optional[str] = None,
    save_path: typing.Optional[typing.Union[str, PathLike]] = None,
):
    _logger.debug("Creating milling cycle plot")
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), tight_layout=True)
    plot_title = img_name
    if total_milling_time is not None:
        plot_title = f"{plot_title}\nMilling time: {total_milling_time:.2g} s"

    fig.suptitle(plot_title)

    # SEM
    _ = axs[0, 0].imshow(sem_image, cmap="Greys_r")
    sem_image_extent = _.get_extent()
    axs[0, 0].axis("off")
    axs[0, 0].set_title("SEM")

    # SEM + 1st pass prediction
    axs[0, 1].imshow(sem_image, cmap="Greys_r")
    axs[0, 1].imshow(
        first_prediction,
        alpha=0.4,
        cmap=LABEL_CMAP,
        vmin=0,
        vmax=len(LABEL_CMAP.colors),
        extent=sem_image_extent,
        interpolation="none",
    )
    axs[0, 1].axis("off")
    axs[0, 1].set_title("SEM segmentation")

    # SEM + clean prediction
    axs[0, 2].imshow(sem_image, cmap="Greys_r")
    axs[0, 2].imshow(
        clean_prediction,
        alpha=0.4,
        cmap=LABEL_CMAP,
        vmin=0,
        vmax=len(LABEL_CMAP.colors),
        extent=sem_image_extent,
        interpolation="none",
    )
    if xlims is not None:
        axs[0, 2].axvline(x=xlims[0], color="C4")
        axs[0, 2].axvline(x=xlims[1], color="C4")
    axs[0, 2].axis("off")
    axs[0, 2].set_title(
        "SEM cleaned segmentation\n"
        rf"Crack area {crack_area_um2:.2f} $\mu m^2$ (threshold {max_crack_area_um2:.2f})",
    )

    # FIB image
    axs[1, 0].imshow(fib_image, cmap="Greys_r")
    axs[1, 0].axis("off")
    axs[1, 0].set_title("FIB")

    # FIB + milling box
    if fib_screenshot is not None:
        # Doesn't work -> for some reason I can't open a new napari.Viewer()
        # pattern_viewer = napari.Viewer()
        # pattern_viewer.add_image(fib_image, name="fib_image")
        # _draw_patterns_in_napari(
        #     viewer=pattern_viewer,
        #     ib_image=FibsemImage(data=fib_image),
        #     eb_image=None,
        #     milling_stages=list(adaptive_polish_stage)
        # )
        # screenshot = pattern_viewer.screenshot()
        # axs[1, 1].imshow(screenshot)
        # pattern_viewer.close()
        axs[1, 1].imshow(fib_screenshot[:, int(fib_screenshot.shape[1] / 2) :, :])
    axs[1, 1].axis("off")

    # GIS thickness
    axs[1, 2].plot(gis_thickness_um, ".-")
    axs[1, 2].set_xlabel("Distance along x $px$")
    axs[1, 2].set_ylabel(r"GIS thickness ($\mu m$)")
    axs[1, 2].set_xlim(0, len(gis_thickness_um))
    axs[1, 2].set_ylim(0, 2)
    axs[1, 2].hlines(
        y=gis_stop_um,
        xmin=0,
        xmax=len(gis_thickness_um),
        label="Target GIS",
        linestyles="dashed",
        colors="C1",
    )

    if xlims is not None:
        axs[1, 2].axvline(x=xlims[0], color="C4")
        axs[1, 2].axvline(x=xlims[1], color="C4")

    axs[1, 2].set_title(
        "GIS thickness\n"
        rf"Minimum {min_gis_um:.3f} $\mu m$ (threshold {gis_stop_um:.3f})"
    )
    axs[1, 2].legend()

    fig.savefig(save_path)
    plt.close(fig)


def create_summary_gis_plot(results: DataFrame, save_path: typing.Union[str, PathLike]):
    _logger.debug("Creating GIS summary plot")
    save_path = Path(save_path)
    fig, ax = plt.subplots(1, 1)
    ax.plot(
        results.milling_time_s,
        results.min_GIS_um,
        label=r"Minimum GIS thickness $\mu m$",
    )
    ax.set_xlabel(r"Milling Time $s$")
    ax.set_ylabel(r"GIS Thickness $\mu m$")
    fig.suptitle(save_path.stem)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
