"""

Segmentation of lamella, GIS layer and cracks from SEM images

"""

import numpy as np
from pathlib import Path
import pandas as pd
from pathlib import Path
import torch
import albumentations as alb
import albumentations.pytorch
import segmentation_models_pytorch as smp
import logging

### Utility functions ###


def normalize_by_mean_std_with_clip35(image, **kwargs):
    # Clips data to mean-sigma*std and mean+sigma*std
    # Data is not really clipped. Data outside is just made much less varying
    smooth = 1e-10

    mean = np.mean(image)
    std = np.std(image)
    clipv = 8.0 * std  # 8.0 = 3.0+5.0

    x = (image - mean) / (clipv + smooth)
    data0 = (np.clip(x, -3.0, 5.0) + 3.0) / 8.0
    return data0  # values between 0 and +1


def normalize_by_mean_std_with_clip(image, **kwargs):
    # Clips data to mean-sigma*std and mean+sigma*std
    # Data is not really clipped. Data outside is just made much less varying
    sigma = 3.0
    smooth = 1e-9

    mean = np.mean(image)
    std = np.std(image)
    clipv = sigma * std

    x = (image - mean) / (clipv + smooth)
    data0 = (np.clip(x, -1.0, 1.0) + 1.0) / 2.0
    return data0  # values between 0 and +1


def get_prepreds_tfms(img_size):
    """
    Transforms to prepare data for running predictions
    image shape is preserved but resizes smallest dimension to 512
    """
    assert len(img_size) == 2

    sm_dim = min(*img_size)

    new_size = [int(sz0 / sm_dim * 512) for sz0 in img_size]
    logging.info(f"imgsize:{img_size}, new_size:{new_size}")
    tfm = alb.Compose(
        [
            alb.ToFloat(max_value=65535.0),
            # alb.Lambda(name="normalize_by_mean_std_with_clip35", image=normalize_by_mean_std_with_clip35, always_apply=True),
            alb.Lambda(
                name="normalize_by_mean_std_with_clip",
                image=normalize_by_mean_std_with_clip,
                always_apply=True,
            ),
            # alb.CenterCrop(sqsize, sqsize,always_apply=True),
            alb.Resize(*new_size),
            albumentations.pytorch.ToTensorV2(),
        ]
    )
    return tfm


def get_post_pred_tfms(orig_size):
    """
    Params:
        orig_size: tuple (height, width) of the original image. Image and mask will
        be resized to this size.
    """
    tfm = alb.Compose([alb.Resize(*orig_size)])
    return tfm


def get_data_from_fn(fn0):
    suf0 = Path(fn0).suffix.lower()

    try:
        if "tif" in suf0:
            # Opens as tifffile
            import tifffile

            return tifffile.imread(fn0)
        elif "png" in suf0:
            # opens as png file
            import cv2

            return cv2.imread(fn0, cv2.IMREAD_UNCHANGED)
    except:
        logging.info("Error. Couldn't open image")
        return None

    ValueError(f"Not a valid filename {fn0}")


def load_model(model_path):
    model_fn = str(model_path)
    m=model_fn.lower()

    # CPU only torch.load, needs map_location otherwise throws an error
    _device = "cpu"
    if torch.cuda.is_available():
        _device = "cuda"

    model_st_dict = None
    if "ptstdict" in m:
        model_st_dict=torch.load(model_fn, map_location=_device) 
    elif "ptchkp" in m:
        model_st_dict= torch.load(model_fn, map_location=_device)["model_state_dict"]
    
    
    #Decide what smp model to load to use
    model_arch=None
    model_enc=None

    if "pytorch_u-" in m:
        model_arch=smp.Unet
    elif "pytorch_fpn-" in m:
        model_arch=smp.FPN
    elif "pytorch_link-" in m:
        model_arch=smp.Linknet
    elif "pytorch_manet-" in m or "aunet" in m:
        model_arch=smp.MAnet
    elif "pytorch_pan-" in m:
        model_arch=smp.PAN
    elif "pytorch_psp-" in m:
        model_arch=smp.PSPNet
    else:
        raise ValueError("No architecture could be guessed")

    if "se_resnext50_32x4d" in m:
        model_enc="se_resnext50_32x4d"
    elif "resnext50_32x4d" in m:
        model_enc="resnext50_32x4d"
    elif "resnet34" in m or \
        ("ptstdict" in m or "aunet." in m or "fpn." in m): #ptstdict were all resnet34 encoder
        model_enc="resnet34"
    elif "resnet50" in m:
        model_enc="resnet50"
    else:
        raise ValueError("No encoder architecture could be guessed")

    model1 = model_arch(encoder_name=model_enc, encoder_weights=None,in_channels=1, classes=4, activation=None)

    model1.load_state_dict(model_st_dict)

    return model1


class cSEMLamellaSegmentor:
    def __init__(self, model_path):
        # setup model from model_path provided
        if isinstance(model_path, str):
            self.model_path = Path(model_path)
        elif isinstance(model_path, Path):
            self.model_path = model_path
        else:
            raise ValueError(
                f"model_path:{model_path} is not valid type. Please provide string of Path"
            )

        logging.info(f"model path:{model_path}")

        if torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        logging.info(f"DL device: {self._device}")

        self.model = None
        self._load_pt_model()

    def _load_pt_model(self):
        """
        Loads a SMP model from path in self.model_path and loads it to cuda.

        Only *.ptstdict and *.ptchkp file supported

        """

        self.model=load_model(self.model_path)

        if self._device == "cuda":
            self.model.cuda()

        self.model.eval()

    def get_prediction(self, data2d):
        # Run the transform needed before feeding to model

        # label0_mock = np.zeros_like(data2d, dtype=np.uint8)
        # preprocess0 = get_prepreds_tfms(data2d.shape)(image=data2d, mask=label0_mock)
        # label0_mock = np.zeros_like(data2d, dtype=np.uint8)
        # data1, label1 = preprocess0['image'], preprocess0['mask']

        data1 = get_prepreds_tfms(data2d.shape)(image=data2d)["image"]  # try shortcut

        logging.info(f"data1.shape:{data1.shape}")

        assert torch.is_tensor(data1)
        data_inp = data1[None, :, :, :].to(self._device).float()

        # Run the prediction, one by one
        pred0 = self.model(data_inp)  # gets result in logits form

        logging.info(f"pred0.shape:{pred0.shape}")
        # converts logist to label
        pred0_softmax_labels = torch.squeeze(torch.argmax(pred0, dim=1))
        if self._device == "cuda":
            pred_labels = pred0_softmax_labels.detach().cpu().numpy()
        else:
            pred_labels = pred0_softmax_labels.detach().numpy()

        data_mock = np.zeros_like(pred_labels, dtype=np.uint8)

        postprocess1 = get_post_pred_tfms(data2d.shape)(
            image=data_mock, mask=pred_labels
        )
        lbl0 = postprocess1[
            "mask"
        ]  # Show have the predicted labels in the original size

        return lbl0
