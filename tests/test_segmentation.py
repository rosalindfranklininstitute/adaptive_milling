import pytest

from pathlib import Path
import tifffile

from adaptive_polish import gis_measurement as gm


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
