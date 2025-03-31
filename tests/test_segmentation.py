from pathlib import Path

from adaptive_polish import gis_measurement as gm

def test_segmentation_model_loads(sem_segmentation_model: tuple[str, Path]):
    generation, model_path = sem_segmentation_model
    """Test that the segmentation model produces a prediction"""
    gm.load_sem_model(model_path, generation=generation)


def test_segmentation_model_runs(
    sem_segmentation_model: tuple[str, Path], sem_image_path: Path
):
    """Test that the segmentation model produces a prediction"""
    generation, model_path = sem_segmentation_model
    model = gm.load_sem_model(model_path, generation=generation)
    prediction = model.predict(sem_image_path)
    assert prediction.min() == 0
    assert prediction.max() == model.num_classes - 1
