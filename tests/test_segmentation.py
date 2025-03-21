import pytest

from pathlib import Path

from adaptive_polish import gis_measurement as gm

_MODELS_PATH = (
    Path.home()
    / "OneDrive - The Rosalind Franklin Institute"
    / "Documents"
    / "test data"
    / "adaptive milling"
    / "sem_models"
)
_IMAGES_PATH = _MODELS_PATH.parent

MODEL_PATHS = {
    "0": _MODELS_PATH
    / "Gen0"
    / "2024-02-24_0013_gis_lamela_crack_pytorch_AUnet.ptchkp",
    "1": _MODELS_PATH
    / "Gen1"
    / "gen01_quality_1536_v2"
    / "20250228_gen01_quality_1536_v2.pth",
}

TEST_SEM_IMAGE = (
    _IMAGES_PATH
    / "2024segmentation_testdata"
    / "SEM"
    / "14B_SEM_stop.tif"  # must be an image with cracks
)


@pytest.mark.parametrize("generation,model_path", list(MODEL_PATHS.items()))
def test_segmentation_model_loads(generation, model_path):
    """Test that the segmentation model produces a prediction"""
    gm.load_sem_model(model_path, generation=generation)


@pytest.mark.parametrize("generation,model_path", list(MODEL_PATHS.items()))
def test_segmentation_model_runs(generation, model_path):
    """Test that the segmentation model produces a prediction"""
    model = gm.load_sem_model(model_path, generation=generation)
    prediction = model.predict(TEST_SEM_IMAGE)
    assert prediction.min() == 0
    assert prediction.max() == model.num_classes - 1
