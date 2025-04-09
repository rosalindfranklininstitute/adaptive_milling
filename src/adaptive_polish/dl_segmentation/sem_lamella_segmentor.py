"""

Segmentation of lamella, GIS layer and cracks from SEM images

"""

from __future__ import annotations

import enum
import logging
import typing
from pathlib import Path
from abc import ABC, abstractmethod

import torch
from torchvision.transforms import v2
import albumentations as alb
import albumentations.pytorch
import segmentation_models_pytorch as smp
import numpy as np


if typing.TYPE_CHECKING:
    from os import PathLike
    from numpy.typing import NDArray

_logger = logging.getLogger(__name__)


class SegmentationLabels(enum.IntEnum):
    BACKGROUND = 0
    GIS = 1
    LAMELLA = 2
    CRACK = 3
    VACUUM = 4


### Utility functions ###
def open_image(
    path: typing.Union[str, PathLike],
) -> typing.Union[NDArray[typing.Any], None]:
    path = Path(path)
    suffix = path.suffix.lower()

    try:
        if "tif" in suffix:
            # Opens as tifffile
            import tifffile

            return tifffile.imread(path)
        elif "png" in suffix:
            # opens as png file
            import cv2

            return cv2.imread(path, cv2.IMREAD_UNCHANGED)

        raise ValueError(f"Not a supported image '{path}'")
    except Exception:
        logging.error("Failed to open image", exc_info=True)
        return None


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


class AbstractAdaptivePolishingModel(ABC):
    def __init__(
        self,
        model_path: typing.Union[str, PathLike],
        device: torch.DeviceLikeType,
        num_classes: int,
    ) -> None:
        self.num_classes = num_classes
        self.model_path: Path = Path(model_path)
        self.device: torch.DeviceLikeType = torch.device(device)
        self.model: torch.nn.Module = self.load(model_path).to(self.device)
        self.model.eval()

    @abstractmethod
    def _preprocess(
        self, image: NDArray[typing.Any]
    ) -> typing.Union[NDArray[typing.Any], torch.Tensor]:
        raise NotImplementedError(
            "AbstractAdaptivePolishingModel.preprocess must be overridden"
        )

    @staticmethod
    @abstractmethod
    def load(model_path: typing.Union[str, PathLike]) -> torch.nn.Module:
        raise NotImplementedError(
            "AbstractAdaptivePolishingModel.load must be overridden"
        )

    def _postprocess(self, prediction: torch.Tensor) -> NDArray[np.long]:
        # converts logist to label
        return torch.argmax(prediction, dim=1).squeeze(0).numpy(force=True)

    def predict(
        self,
        image: typing.Union[NDArray[typing.Any], typing.Union[str, PathLike]],
        full_size: bool = True,
    ) -> NDArray[typing.Any]:
        if not isinstance(image, np.ndarray):
            try:
                image = Path(image)
            except Exception:
                raise ValueError(f"Failed to parse image '{image}'")
            if not image.is_file():
                raise FileNotFoundError(image)

            image = open_image(image)

        image = image.squeeze()  # Ensure 2D images are 2D

        if len(image.shape) != 2:
            raise ValueError("Adaptive polishing SEM models only supports 2D images")

        with torch.no_grad():
            preprocessed = self._preprocess(image)  # try shortcut

            if not torch.is_tensor(preprocessed):
                preprocessed = torch.from_numpy(preprocessed)

            preprocessed = preprocessed.to(self.device)

            prediction = self.model(preprocessed)  # gets result in logits form

            labels = self._postprocess(prediction)

            if full_size:
                # Resize model to full original dims
                return alb.Resize(image.shape[-2], image.shape[-1]).apply_to_mask(
                    labels
                )
            return labels


class Gen0Model(AbstractAdaptivePolishingModel):
    def __init__(
        self,
        model_path: typing.Union[str, PathLike],
        device: torch.DeviceLikeType,
        max_image_size: int = 512,
    ) -> None:
        self._image_size = max_image_size
        super().__init__(model_path=model_path, device=device, num_classes=4)

    def _preprocess(self, image: NDArray[typing.Any]) -> torch.Tensor:
        smallest_dim = min(*image.shape)
        new_size = tuple(
            int(sz0 / smallest_dim * self._image_size) for sz0 in image.shape
        )
        _logger.debug(f"imgsize:{image.shape}, new_size:{new_size}")

        transform = alb.Compose(
            [
                alb.ToFloat(max_value=65535.0),
                # alb.Lambda(name="normalize_by_mean_std_with_clip35", image=normalize_by_mean_std_with_clip35, always_apply=True),
                alb.Lambda(
                    name="normalize_by_mean_std_with_clip",
                    image=normalize_by_mean_std_with_clip,
                    # always_apply=True,
                ),
                # alb.CenterCrop(sqsize, sqsize,always_apply=True),
                alb.Resize(*new_size),
                albumentations.pytorch.ToTensorV2(),
            ]
        )

        return transform(image=image)["image"].unsqueeze_(0)

    def load(self, model_path: typing.Union[str, PathLike]) -> torch.nn.Module:
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

        model_kwargs = {
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


class Gen1Model(AbstractAdaptivePolishingModel):
    def __init__(
        self,
        model_path: typing.Union[str, PathLike],
        device: torch.DeviceLikeType,
        max_image_size: int = 1536,
    ) -> None:
        self._image_size = max_image_size
        super().__init__(model_path=model_path, device=device, num_classes=5)

    def _preprocess(self, image: NDArray[typing.Any]) -> torch.Tensor:
        # Hopefully this is a more efficient implementation of Casper's preprocessing
        image_shape_array = np.asarray(image.shape)
        axis_multiplier = np.min(self._image_size / image_shape_array)
        target_shape = tuple(
            np.round(image_shape_array * axis_multiplier).astype(int).tolist()
        )

        with torch.no_grad():
            image = torch.from_numpy(image.astype(np.float32)).to(self.device)
            mean, std = image.mean(), image.std()

            # Calculate in place:
            image -= mean
            image /= 3 * std
            image.clamp_(0, 1)

            image = v2.functional.resize(
                image.unsqueeze_(0).unsqueeze_(0),  # expect channel and batch axes
                size=target_shape,
            )
            # No need to pad image as the we're not processing batches

            return v2.functional.grayscale_to_rgb(image).to(
                self.device
            )  # Ensure it's still on the correct device

    def load(self, model_path: typing.Union[str, PathLike]) -> torch.nn.Module:
        model = smp.Unet(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            classes=self.num_classes,
            in_channels=3,
            activation=None,
            decoder_attention_type="scse",
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model


# Using str keys allows for semantic versioning
MODEL_GENERATIONS_DICT: dict[str, AbstractAdaptivePolishingModel] = {
    "0": Gen0Model,
    "1": Gen1Model,
}


def _get_newest_generation_key():
    # If no generation is specified, get the latest generation one
    return sorted(MODEL_GENERATIONS_DICT.keys())[-1]


def load_model(
    model_path: typing.Union[str, PathLike],
    generation: typing.Optional[typing.Union[int, str]] = None,
    device: typing.Optional[torch.DeviceLikeType] = None,
    max_image_size: typing.Optional[int] = None,
) -> AbstractAdaptivePolishingModel:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if generation is None:
        # If no generation is specified, get the latest generation one
        generation = _get_newest_generation_key()

    elif generation not in MODEL_GENERATIONS_DICT:
        raise ValueError(
            f"Invalid model generation '{generation}' specified, available generations are: {', '.join(MODEL_GENERATIONS_DICT.keys())}"
        )

    _logger.info("Loading %s generation model", generation)
    model_class = MODEL_GENERATIONS_DICT.get(str(generation), None)

    model_kwargs = {}
    if max_image_size is not None:
        # If none, use model's default
        model_kwargs["max_image_size"] = max_image_size

    try:
        return model_class(model_path, device=device, **model_kwargs)
    except Exception:
        _logger.error(
            "Failed to load model with '%s'", model_class.__name__, exc_info=True
        )
        raise
