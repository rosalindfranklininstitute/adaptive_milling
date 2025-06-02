import typing
import yaml
from dataclasses import dataclass, InitVar, field
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from datetime import datetime

import numpy as np



def setup_test_microscope_config(
    microscope_template_path: Path,
    temporary_directory: Path,
    fib_image_dir: typing.Optional[str] = None,
    sem_image_dir: typing.Optional[str] = None,
    cycle_images: bool = True,
) -> Path:
    """Sets up a demo microscope"""
    with microscope_template_path.open() as f:
        microscope_template_dict = yaml.safe_load(f)

    # Update protocol from config
    microscope_template_dict["demo2"] = {
        "SEM_folder_path_str": sem_image_dir,
        "FIB_folder_path_str": fib_image_dir,
        "cycle": cycle_images,
    }

    microscope_config_path = temporary_directory / "microscope.yaml"
    with microscope_config_path.open("w") as f:
        yaml.safe_dump(microscope_template_dict, f)

    return microscope_config_path


def setup_protocol_path(
    protocol_template_path: Path,
    temporary_directory: Path,
    adaptive_polish_config_dict: dict[str, typing.Any],
    ap_only: bool,
) -> Path:
    with protocol_template_path.open() as f:
        protocol_dict = yaml.safe_load(f)

    # Update protocol from config
    protocol_dict["milling"]["mill_polishing"][0]["strategy"]["config"] = (
        adaptive_polish_config_dict
    )

    if ap_only:
        # Optionally remove non-AP milling steps
        protocol_dict["milling"] = {
            "mill_polishing": protocol_dict["milling"]["mill_polishing"]
        }

    protocol_path = temporary_directory / "protocol.yaml"
    with protocol_path.open("w") as f:
        yaml.safe_dump(protocol_dict, f)

    return protocol_path


def setup_test_experiment(
    experiment_template_path: Path, temporary_directory: Path, save_images: bool = False
) -> Path:
    """Sets up a demo autolamella experiment"""
    now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    environment = Environment(loader=FileSystemLoader(experiment_template_path.parent))
    template = environment.get_template(str(experiment_template_path.name))
    experiment = template.render(
        {"tmp_dir": str(temporary_directory), "now": now, "save_images": save_images}
    )
    experiment_path = temporary_directory / "experiment.yaml"
    with experiment_path.open("w") as f:
        f.write(experiment)

    return experiment_path

@dataclass(repr=False)
class SimpleRectangleLamellaMask:
    shape: InitVar[typing.Union[np.typing.NDArray[np.integer], typing.Tuple[int, int]]]
    box_proportion: InitVar[int] = 20
    centre_px: InitVar[typing.Optional[typing.Tuple[int, int]]] = None
    array: np.typing.NDArray[np.integer] = field(init=False)
    centre: np.typing.NDArray[np.integer] = field(init=False)
    bounding_box: np.typing.NDArray[np.integer] = field(init=False)

    def __post_init__(self, shape, box_proportion, centre_px) -> None:
        array_shape = np.asarray(shape)
        array = np.zeros(array_shape, dtype=np.bool_)
        box_centre_to_edge = array_shape // (2 * box_proportion)
        box_size = box_centre_to_edge * 2 + 1
        if centre_px is None:
            centre = np.random.randint(box_size, array_shape - box_size, size=2)
        else:
            centre = np.asarray(centre_px[:2], dtype=np.uint32)
        bbox = np.concatenate(
            (centre - box_centre_to_edge, centre + box_centre_to_edge)
        )
        array[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] = True

        self.array = array
        self.centre = centre
        self.bounding_box = bbox
