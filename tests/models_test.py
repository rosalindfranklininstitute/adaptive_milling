from __future__ import annotations

from pathlib import Path

import pytest
import tifffile

from adaptive_milling.models import load_model


def test_segmentation_model_loads(sem_segmentation_model: tuple[str, Path]):
    generation, model_path = sem_segmentation_model
    """Test that the segmentation model produces a prediction"""
    load_model(model_path, generation=generation)


@pytest.mark.parametrize(["as_array"], [[True], [False]], ids=["array", "path"])
def test_segmentation_model_runs(
    as_array: bool, sem_segmentation_model: tuple[str, Path], sem_image_dir: Path
):
    """Test that the segmentation model produces a prediction"""
    generation, model_path = sem_segmentation_model
    model = load_model(model_path, generation=generation)
    image_path = next(sem_image_dir.glob("*.tif"))
    if as_array:
        image = tifffile.imread(image_path)
    else:
        image = image_path

    prediction = model.predict(image)
    assert prediction.min() == 0
    assert prediction.max() == model.num_classes - 1
