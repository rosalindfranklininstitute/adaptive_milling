from __future__ import annotations

import typing
import yaml
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from datetime import datetime


if typing.TYPE_CHECKING:
    from adaptive_polish.strategy import AdaptivePolishMillingConfig


def setup_test_microscope_config(
    microscope_template_path: Path,
    temporary_directory: Path,
    fib_image_dir: str | None = None,
    sem_image_dir: str | None = None,
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
    adaptive_polish_config: AdaptivePolishMillingConfig,
    ap_only: bool,
) -> Path:
    with protocol_template_path.open() as f:
        protocol_dict = yaml.safe_load(f)

    # Update protocol from config
    protocol_dict["milling"]["mill_polishing"][0]["strategy"]["config"] = (
        adaptive_polish_config.to_dict()
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
