from __future__ import annotations

import typing
from dataclasses import InitVar, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from jinja2 import Environment, FileSystemLoader

from adaptive_milling.processing.segmentation import SEMSegmentationLabels as SemLabels

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


def setup_protocol_path(
    protocol_template_path: Path,
    temporary_directory: Path,
    adaptive_polish_config_dict: dict[str, typing.Any],
    ap_only: bool = False,
    ap_type: typing.Literal["normal", "bitmap"] = "normal",
) -> Path:
    with protocol_template_path.open() as f:
        protocol_dict = yaml.safe_load(f)

    ap_index = 1 if ap_type == "bitmap" else 0

    # Update protocol from config
    stage_dict = protocol_dict["tasks"]["Polishing"]["milling"]["mill_polishing"][
        "stages"
    ][ap_index]
    stage_dict["strategy"]["config"] = adaptive_polish_config_dict

    # Remove the unwanted AP strategy
    protocol_dict["tasks"]["Polishing"]["milling"]["mill_polishing"]["stages"] = [
        stage_dict
    ]

    if ap_only:
        # Optionally remove non-AP milling steps
        protocol_dict["tasks"] = {"Polishing": protocol_dict["tasks"]["Polishing"]}

    protocol_path = temporary_directory / "protocol.yaml"
    with protocol_path.open("w") as f:
        yaml.safe_dump(protocol_dict, f)

    return protocol_path


def setup_test_experiment(
    experiment_template_path: Path, temporary_directory: Path, save_images: bool = False
) -> Path:
    """Sets up a demo autolamella experiment"""
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d-%H-%M")
    environment = Environment(loader=FileSystemLoader(experiment_template_path.parent))
    template = environment.get_template(str(experiment_template_path.name))
    experiment = template.render(
        {"tmp_dir": str(temporary_directory), "now": now, "save_images": save_images}
    )
    experiment_path = temporary_directory / "experiment.yaml"
    with experiment_path.open("w") as f:
        f.write(experiment)

    return experiment_path


def create_mock_prediction_gis_thickness(
    image_shape: tuple[int, int],
    lamella_bbox: tuple[int, int, int, int],
    background_lamella_overlap: int,
    vacuum_bottom_pixels: int,
    add_crack: bool = True,
    top_background_size: int = 3,
    top_vacuum_size: int = 3,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
    segmented_gis_thickness: NDArray[np.uint8] = np.asarray(
        (
            ([0] * lamella_bbox[1])
            + [_ // 2 for _ in range(lamella_bbox[3] - lamella_bbox[1])]
            + ([0] * (image_shape[1] - lamella_bbox[3]))
        ),
        dtype=np.uint8,
    )

    # Add on overlapping background to segmented GIS area
    expected_gis_thickness = segmented_gis_thickness.copy()
    gis_background_thickness = image_shape[0] - (lamella_bbox[2] + 1)
    expected_gis_thickness[
        lamella_bbox[1] : lamella_bbox[1] + background_lamella_overlap
    ] = gis_background_thickness
    expected_gis_thickness[
        lamella_bbox[3] + 1 - background_lamella_overlap : lamella_bbox[3] + 1
    ] = gis_background_thickness

    prediction = np.full(image_shape, SemLabels.VACUUM.value, dtype=np.uint8)

    # Draw background (edges)
    prediction[:, : lamella_bbox[1] + background_lamella_overlap] = (
        SemLabels.BACKGROUND.value
    )
    prediction[:, lamella_bbox[3] + 1 - background_lamella_overlap :] = (
        SemLabels.BACKGROUND.value
    )

    # Add background to bottom of vacuum (should be ignored)
    prediction[image_shape[0] - vacuum_bottom_pixels :, :] = SemLabels.BACKGROUND.value

    # Draw lamella
    prediction[
        lamella_bbox[0] : lamella_bbox[2] + 1, lamella_bbox[1] : lamella_bbox[3] + 1
    ] = SemLabels.LAMELLA.value

    # Draw GIS
    gis_top = lamella_bbox[2] + 1
    for i, thickness in enumerate(segmented_gis_thickness):
        prediction[gis_top : gis_top + thickness, i] = SemLabels.GIS.value

    # Mislable some GIS as background
    # This should not affect the GIS thickness
    hole_coords = (lamella_bbox[2] + 1, lamella_bbox[3])
    prediction[hole_coords[0], hole_coords[1]] = SemLabels.BACKGROUND.value

    # Add hole of vacuum to GIS layer
    hole_coords = (lamella_bbox[2] + 1, lamella_bbox[3] - lamella_bbox[1])
    prediction[hole_coords[0], hole_coords[1]] = SemLabels.VACUUM.value
    expected_gis_thickness[hole_coords[1]] -= 1

    # Add some background to top of lamella
    # This should not affect the GIS thickness
    slicer = (
        slice(None, lamella_bbox[0] + top_background_size),
        slice(lamella_bbox[1], lamella_bbox[1] + top_background_size),
    )
    prediction[slicer] = SemLabels.BACKGROUND.value

    # Add some vacuum to top of lamella
    # This should not affect the GIS thickness
    slicer = (
        slice(None, lamella_bbox[0] + top_vacuum_size),
        slice(lamella_bbox[3], lamella_bbox[3] - top_vacuum_size - 1, -1),
    )
    prediction[slicer] = SemLabels.VACUUM.value

    if add_crack:
        # Add a crack to GIS layer
        remaining_gis_thickness = 1
        crack_coords = (
            lamella_bbox[2] + 1 + remaining_gis_thickness,
            lamella_bbox[3] - lamella_bbox[1] + 1,
        )
        prediction[
            crack_coords[0] : crack_coords[0]
            + expected_gis_thickness[crack_coords[1]]
            - remaining_gis_thickness,
            crack_coords[1],
        ] = SemLabels.CRACK.value
        expected_gis_thickness[crack_coords[1]] = remaining_gis_thickness
        # Must be touching lamella to be considered a crack by post-processing (diagonal is fine)
        prediction[
            crack_coords[0] - remaining_gis_thickness : crack_coords[0],
            crack_coords[1] + 1,
        ] = SemLabels.CRACK.value
        expected_gis_thickness[crack_coords[1] + 1] -= remaining_gis_thickness

    return prediction, segmented_gis_thickness, expected_gis_thickness


@dataclass(repr=False)
class SimpleRectangleLamellaMask:
    shape: InitVar[[NDArray[np.integer] | tuple[int, int]]]
    box_proportion: InitVar[int] = 20
    centre_px: InitVar[tuple[int, int] | None] = None
    array: NDArray[np.bool_] = field(init=False)
    centre: NDArray[np.int_] = field(init=False)
    bounding_box: NDArray[np.int_] = field(init=False)

    def __post_init__(self, shape, box_proportion, centre_px) -> None:
        array_shape = np.asarray(shape)
        array = np.zeros(array_shape, dtype=np.bool_)
        box_centre_to_edge = array_shape // (2 * box_proportion)
        box_size = box_centre_to_edge * 2 + 1
        if centre_px is None:
            centre = np.random.randint(box_size, array_shape - box_size, size=2)
        else:
            centre = np.asarray(centre_px[:2])
        bbox = np.concatenate(
            (centre - box_centre_to_edge, centre + box_centre_to_edge)
        )
        array[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] = True

        self.array = array
        self.centre = centre
        self.bounding_box = bbox
