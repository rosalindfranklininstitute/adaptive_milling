from __future__ import annotations
import pytest

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
        max_image_size: int,
        device: torch.DeviceLikeType,
    ) -> None:
        self._image_size = max_image_size
        self.device = device


def old_model1_preprocessing_function(
    image: NDArray[typing.Any],
    image_size: int,
    pad_size: int,
    device: torch.DeviceLikeType,
) -> torch.Tensor:
    # Convert grayscale to 3-channel
    image = np.stack([image] * 3, axis=-1).astype(np.float32)

    # Normalization
    mean, std = image.mean(), image.std()
    image = (image - mean) / (3 * std)
    image = np.clip(image, 0, 1)

    # Crop to (3072x3072)
    image = cv2.copyMakeBorder(
        image, pad_size, pad_size, 0, 0, cv2.BORDER_CONSTANT, value=0
    )

    # Apply resize transformation
    transform = alb.Compose(
        [
            alb.Resize(image_size, image_size),
            albumentations.pytorch.ToTensorV2(),
        ]
    )
    transformed = transform(image=image)

    return transformed["image"].unsqueeze(0).to(device)


def test_segmentation_model_loads(sem_segmentation_model: tuple[str, Path]):
    generation, model_path = sem_segmentation_model
    """Test that the segmentation model produces a prediction"""
    gm.load_sem_model(model_path, generation=generation)


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


def test_model1_preprocessing_results_match() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_image_size = 100
    pad_size = 5
    scale_multiplier = 3
    image_size = (
        max_image_size * scale_multiplier,
        max_image_size * scale_multiplier - pad_size * 2,
    )
    image = np.arange(500, 500 + np.multiply(*image_size), dtype=np.float32).reshape(
        image_size
    )
    mock_gen1_model = MockGen1Model(max_image_size=max_image_size, device=device)
    output = mock_gen1_model._preprocess(image)
    expected_output = old_model1_preprocessing_function(
        image, image_size=max_image_size, pad_size=pad_size, device=device
    )

    torch.testing.assert_close(
        output,
        expected_output,
        msg=lambda _: f"Model 1 preprocessing gave different results: {_}",
    )
