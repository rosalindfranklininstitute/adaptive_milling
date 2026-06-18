from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import albumentations as alb
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from adaptive_milling.models.abstract import AbstractAdaptivePolishingModel

if TYPE_CHECKING:
    from typing import Union, Any
    from os import PathLike
    from numpy.typing import NDArray

    DeviceLikeType = Union[str, torch.device, int]


_logger = logging.getLogger(__name__)


def _normalize_by_mean_std_with_clip(image, **kwargs) -> NDArray[np.float64]:
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


class Gen0Model(AbstractAdaptivePolishingModel):
    def __init__(
        self,
        model_path: str | PathLike[str],
        device: DeviceLikeType,
        max_image_size: int = 512,
    ) -> None:
        self._image_size = max_image_size
        self.num_classes = 4
        super().__init__(model_path=model_path, device=device)

    def _preprocess(self, image: NDArray[Any]) -> torch.Tensor:
        smallest_dim = min(image.shape[0], image.shape[1])
        new_size = tuple(
            int(sz0 / smallest_dim * self._image_size) for sz0 in image.shape[:2]
        )
        _logger.debug(f"imgsize:{image.shape}, new_size:{new_size}")

        transform = alb.Compose(
            [
                alb.ToFloat(max_value=65535.0),
                alb.Lambda(
                    name="normalize_by_mean_std_with_clip",
                    image=_normalize_by_mean_std_with_clip,
                ),
                alb.Resize(new_size[0], new_size[1]),
                ToTensorV2(),
            ]  # type: ignore
        )

        return transform(image=image)["image"].unsqueeze_(0)

    def load(self, model_path: str | PathLike[str]) -> torch.nn.Module:
        model_path = Path(model_path)
        m = Path(model_path).name.lower()

        # CPU only torch.load, needs map_location otherwise throws an error
        model_st_dict = None
        if "ptstdict" in m:
            model_st_dict = torch.load(model_path, map_location=self.device)
        elif m.endswith(".ptchkp"):
            model_st_dict = torch.load(model_path, map_location=self.device)[
                "model_state_dict"
            ]

        # Decide what smp model to load to use
        model_arch = None

        model_kwargs: dict[str, Any] = {
            "classes": self.num_classes,
            "encoder_weights": None,
            "in_channels": 1,
            "activation": None,
        }

        if "pytorch_u-" in m:
            model_arch = smp.Unet
        elif "pytorch_fpn-" in m:
            model_arch = smp.FPN
        elif "pytorch_link-" in m:
            model_arch = smp.Linknet
        elif "pytorch_manet-" in m or "aunet" in m:
            model_arch = smp.MAnet
        elif "pytorch_pan-" in m:
            model_arch = smp.PAN
        elif "pytorch_psp-" in m:
            model_arch = smp.PSPNet
        else:
            raise ValueError("No architecture could be guessed")

        if "se_resnext50_32x4d" in m:
            model_kwargs["encoder_name"] = "se_resnext50_32x4d"
        elif "resnext50_32x4d" in m:
            model_kwargs["encoder_name"] = "resnext50_32x4d"
        elif "resnet34" in m or (
            "ptstdict" in m or "aunet." in m or "fpn." in m
        ):  # ptstdict were all resnet34 encoder
            model_kwargs["encoder_name"] = "resnet34"
        elif "resnet50" in m:
            model_kwargs["encoder_name"] = "resnet50"
        else:
            raise ValueError("No encoder architecture could be guessed")

        model = model_arch(**model_kwargs)

        model.load_state_dict(model_st_dict)

        return model
