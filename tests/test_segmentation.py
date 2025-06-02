from __future__ import annotations
import pytest

import itertools
import typing
from pathlib import Path

import cv2
import torch
import albumentations as alb
import albumentations.pytorch
import tifffile
import numpy as np


from adaptive_polish import gis_measurement as gm
from adaptive_polish.dl_segmentation import sem_lamella_segmentor

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


class MockGen1Model(sem_lamella_segmentor.Gen1Model):
    """Useful to avoid using an actual model"""

    def __init__(
        self,
        device: torch.DeviceLikeType,
        max_image_size: int,
        pad: bool = True,
        rgb: bool = True,
        normalise_first: bool = True,
        normalise_version: int = 1,
        resize_version: str = "cv2",
    ) -> None:
        self._rgb = rgb
        self._pad = pad
        self._normalise_first = normalise_first
        self._image_size = max_image_size
        self.device = device
        self._normalise_function = self._get_normalisation_function(normalise_version)
        self._resize_function = self._get_resize_function(resize_version)


def old_model1_preprocessing_function(
    image: NDArray[typing.Any],
    image_size: tuple[int, int],
    pad_size: int,
    rgb: bool,
    normalise_first: bool,
    device: torch.DeviceLikeType,
) -> torch.Tensor:
    # Convert grayscale to 3-channel
    if rgb:
        image = np.stack([image] * 3, axis=-1).astype(np.float32)
    else:
        image = image[..., np.newaxis]

    if normalise_first:
        # Normalization
        mean, std = image.mean(), image.std()
        image = (image - mean) / (3 * std)
        image = np.clip(image, 0, 1)

    # Crop to (3072x3072)
    image = cv2.copyMakeBorder(
        image, pad_size, pad_size, 0, 0, cv2.BORDER_CONSTANT, value=0
    )

    if not normalise_first:
        # Normalization
        mean, std = image.mean(), image.std()
        image = (image - mean) / (3 * std)
        image = np.clip(image, 0, 1)

    # Apply resize transformation
    transform = alb.Compose(
        [
            alb.Resize(*image_size),
            albumentations.pytorch.ToTensorV2(),
        ]
    )
    transformed = transform(image=image)

    return transformed["image"].unsqueeze(0).to(device)


@pytest.mark.usefixtures("skip_if_no_models")
def test_segmentation_model_loads(sem_segmentation_model: tuple[str, Path]):
    generation, model_path = sem_segmentation_model
    """Test that the segmentation model produces a prediction"""
    gm.load_sem_model(model_path, generation=generation)


@pytest.mark.usefixtures("skip_if_no_models")
@pytest.mark.parametrize(["as_array"], [[True], [False]], ids=["array", "path"])
def test_segmentation_model_runs(
    as_array: bool, sem_segmentation_model: tuple[str, Path], sem_image_dir: Path
):
    """Test that the segmentation model produces a prediction"""
    generation, model_path = sem_segmentation_model
    model = gm.load_sem_model(model_path, generation=generation)
    image_path = next(sem_image_dir.glob("*.tif"))
    if as_array:
        image = tifffile.imread(image_path)
    else:
        image = image_path

    prediction = model.predict(image)
    assert prediction.min() == 0
    assert prediction.max() == model.num_classes - 1


@pytest.mark.parametrize(
    "rgb,pad,normalise_first",
    itertools.product([True, False], [True, False], [True, False]),
)
def test_model1_preprocessing_results_match(
    rgb: bool, pad: bool, normalise_first: bool
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_image_size = 100
    scale_multiplier = 3
    output_y_edge_diff = 5 if pad else 0
    pad_size = output_y_edge_diff * scale_multiplier if pad else 0
    unpadded_output_image_size = np.asarray(
        (max_image_size - output_y_edge_diff * 2, max_image_size)
    )
    input_image_size = unpadded_output_image_size * scale_multiplier

    if pad:
        target_image_size = (
            unpadded_output_image_size[1],
            unpadded_output_image_size[1],
        )
    else:
        target_image_size = unpadded_output_image_size

    input_image = np.arange(
        500, 500 + np.multiply(*input_image_size), dtype=np.float32
    ).reshape(input_image_size)
    mock_gen1_model = MockGen1Model(
        device=device,
        max_image_size=max_image_size,
        pad=pad * scale_multiplier,
        rgb=rgb,
        normalise_first=normalise_first,
    )
    output = mock_gen1_model._preprocess(input_image)
    expected_output = old_model1_preprocessing_function(
        input_image,
        image_size=target_image_size,
        pad_size=pad_size,
        rgb=rgb,
        normalise_first=normalise_first,
        device=device,
    )
    assert np.all(expected_output.shape[-2:] == target_image_size), (
        "Expected output isn't the correct shape"
    )
    assert np.all(output.shape[-2:] == target_image_size), (
        "Output isn't the correct shape"
    )

    torch.testing.assert_close(
        output,
        expected_output,
        msg=lambda _: f"Model 1 preprocessing gave different results: {_}",
    )
