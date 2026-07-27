from __future__ import annotations
import argparse
import logging
import typing
from pathlib import Path

import tifffile
import numpy as np

from adaptive_milling.exceptions import SegmentationException

if typing.TYPE_CHECKING:
    from os import PathLike

    from adaptive_milling.models.abstract import AbstractAdaptivePolishingModel


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        dest="image_or_directory",
        type=Path,
        action="append",
        required=True,
        help="Path to an image or directory containing images to be segmented. Multiple instances of this argument are accepted.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_directory",
        type=Path,
        required=True,
        help="Directory to save the segmented images in.",
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="weights_path",
        type=Path,
        required=True,
        help="Path to the model weights file (.pth).",
    )
    parser.add_argument(
        "--gen",
        dest="model_generation",
        type=str,
        required=False,
        help="Model generation (for advanced users) (defaults to the lastest generation).",
    )
    parser.add_argument(
        "--raw",
        dest="raw",
        default=True,
        required=False,
        action="store_false",
        help="Do not apply post-processing to clean up the segmentation.",
    )

    return parser


def _parse_arguments(parser: argparse.ArgumentParser) -> None:
    namespace = parser.parse_args()

    from adaptive_milling.models import load_model
    from adaptive_milling.processing.segmentation import clean_prediction

    def _load_model(
        weights_path: str | PathLike[str], model_generation: str | None = None
    ) -> AbstractAdaptivePolishingModel:
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Failed to find SEM segmentation model '{weights_path}'"
            )
        try:
            return load_model(
                model_path=weights_path,
                generation=model_generation,
            )
        except Exception as e:
            raise SegmentationException("Failed to load SEM segmentation model") from e

    def _infer_image(
        path: Path,
        output_directory: Path,
        model: AbstractAdaptivePolishingModel,
        clean: bool = True,
    ) -> None:
        im = tifffile.imread(path)
        logging.info("Predicting segmentation of image '%s'", path)
        prediction = model.predict(image=im).astype(np.uint8)
        if clean:
            logging.info("Cleaning prediction of image '%s'", path)
            prediction = clean_prediction(prediction)
        with tifffile.TiffWriter(output_directory / path.name) as tiff:
            tiff.write(
                prediction,
                photometric=tifffile.PHOTOMETRIC.MINISBLACK,
                dtype="uint8",
                compression=tifffile.COMPRESSION.LZW,
            )

    def _run_inference(
        *input_paths: str | PathLike[str],
        output_directory: str | PathLike[str],
        model: AbstractAdaptivePolishingModel,
        clean: bool = True,
    ) -> None:
        output_directory = Path(output_directory)

        for path in input_paths:
            path = Path(path)
            if path.is_dir():
                for p in path.rglob("*.tif*"):
                    try:
                        _infer_image(
                            path=p,
                            output_directory=output_directory,
                            model=model,
                            clean=clean,
                        )
                    except Exception:
                        logging.exception("Failed to segment image '%s'", p)
            elif path.is_file():
                try:
                    _infer_image(
                        path=path,
                        output_directory=output_directory,
                        model=model,
                        clean=clean,
                    )
                except Exception:
                    logging.exception("Failed to segment image '%s'", p)
            else:
                logging.warning("No file or directory found at '%s'", p)

    model = _load_model(
        weights_path=namespace.weights_path,
        model_generation=getattr(namespace, "model_generation", None),
    )

    _run_inference(
        *namespace.image_or_directory,
        output_directory=namespace.output_directory,
        model=model,
        clean=not namespace.raw,
    )


def main() -> None:
    _parse_arguments(parser=_create_parser())


if __name__ == "__main__":
    main()
