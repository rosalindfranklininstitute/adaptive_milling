"""

Segmentation of lamella, GIS layer and cracks from SEM images

"""

from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING, Protocol
from pathlib import Path
from abc import ABC, abstractmethod

import cv2
import torch
from torchvision.transforms import v2
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import numpy as np


if TYPE_CHECKING:
    from typing import Union, Literal, Any
    from os import PathLike
    from numpy.typing import NDArray

    DeviceLikeType = Union[str, torch.device, int]

    class ResizeProtocol(Protocol):
        def __call__(
            self, image: torch.Tensor, target_shape: tuple[int, int]
        ) -> torch.Tensor: ...

    class NormaliseProtocol(Protocol):
        def __call__(self, image: torch.Tensor) -> torch.Tensor: ...


_logger = logging.getLogger(__name__)


class SegmentationLabels(enum.IntEnum):
    BACKGROUND = 0
    GIS = 1
    LAMELLA = 2
    CRACK = 3
    VACUUM = 4


### Utility functions ###
def open_image(path: str | PathLike[str]) -> NDArray[Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    if "tif" in suffix:
        # Opens as tifffile
        import tifffile

        return tifffile.imread(path)
    elif "png" in suffix:
        # opens as png file
        import cv2

        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    raise ValueError(f"Not a supported image '{path}'")


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
        model_path: str | PathLike[str],
        device: DeviceLikeType,
    ) -> None:
        self.num_classes: int
        self.model_path: Path = Path(model_path)
        self.device: torch.device = torch.device(device)
        self.model: torch.nn.Module = self.load(model_path).to(self.device)
        self.model.eval()

    @abstractmethod
    def _preprocess(self, image: NDArray[Any]) -> NDArray[Any] | torch.Tensor:
        raise NotImplementedError(
            "AbstractAdaptivePolishingModel.preprocess must be overridden"
        )

    @staticmethod
    @abstractmethod
    def load(model_path: str | PathLike[str]) -> torch.nn.Module:
        raise NotImplementedError(
            "AbstractAdaptivePolishingModel.load must be overridden"
        )

    def _postprocess(
        self, prediction: torch.Tensor, image: NDArray[Any]
    ) -> NDArray[np.uint8]:
        # converts logist to label
        return (
            torch.argmax(prediction, dim=1).squeeze(0).to(torch.uint8).numpy(force=True)
        )

    def predict(
        self,
        image: NDArray[Any] | str | PathLike[str],
        full_size: bool = True,
    ) -> NDArray[Any]:
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
                preprocessed_tensor = torch.from_numpy(preprocessed)
            else:
                preprocessed_tensor = preprocessed

            preprocessed_tensor = preprocessed_tensor.to(self.device)

            prediction = self.model(preprocessed_tensor)  # gets result in logits form

            labels = self._postprocess(prediction, image)

            if full_size:
                # Resize model to full original dims
                return alb.Resize(image.shape[-2], image.shape[-1]).apply_to_mask(
                    labels
                )
            return labels


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
        smallest_dim = min(*image.shape)
        new_size = tuple(
            int(sz0 / smallest_dim * self._image_size) for sz0 in image.shape
        )
        _logger.debug(f"imgsize:{image.shape}, new_size:{new_size}")

        transform = alb.Compose(
            [
                alb.ToFloat(max_value=65535.0),
                alb.Lambda(
                    name="normalize_by_mean_std_with_clip",
                    image=normalize_by_mean_std_with_clip,
                ),
                alb.Resize(*new_size),
                ToTensorV2(),
            ]
        )

        return transform(image=image)["image"].unsqueeze_(0)

    def load(self, model_path: str | PathLike[Any]) -> torch.nn.Module:
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
        model_path: str | PathLike[str],
        device: DeviceLikeType,
        max_image_size: int,
        encoder_name: str,
        pad: bool = True,
        rgb: bool = True,
        normalise_first: bool = True,
        normalise_version: Literal[1, 2] = 2,
        resize_version: Literal["cv2", "pytorch"] = "pytorch",
        model_type: Literal["unet", "fpn"] = "unet",
    ) -> None:
        self._rgb = rgb
        self._pad = pad
        self._normalise_first = normalise_first
        self._model_type = model_type
        self._encoder_name = encoder_name
        self._image_size = max_image_size
        self._normalise_function = self._get_normalisation_function(normalise_version)
        self._resize_function = self._get_resize_function(resize_version)

        self.num_classes = 5
        super().__init__(model_path=model_path, device=device)

    def _get_normalisation_function(self, normalise_version: int) -> NormaliseProtocol:
        normalise_functions: dict[int, NormaliseProtocol] = {
            1: self._normalise_1,
            2: self._normalise_2,
        }

        normalise_function = normalise_functions.get(normalise_version)
        if normalise_function is None:
            raise ValueError(
                f"Invalid normalise_version '{normalise_version}', available: {normalise_functions.keys()}"
            )
        _logger.debug(f"Using version {normalise_version} normalisation")
        return normalise_function

    def _get_resize_function(self, resize_version: str) -> ResizeProtocol:
        resize_functions: dict[str, ResizeProtocol] = {
            "cv2": self._resize_cv2,
            "pytorch": self._resize_pytorch,
        }

        resize_function = resize_functions.get(resize_version)
        if resize_function is None:
            raise ValueError(
                f"Invalid resize_version '{resize_version.lower()}', available: {resize_functions.keys()}"
            )
        _logger.debug(f"Using {resize_version} resizing")
        return resize_function

    def _normalise_1(self, image: torch.Tensor) -> torch.Tensor:
        mean = image.mean()
        # correction=0 matches numpy's behaviour (without Bessel's correction)
        std = image.std(correction=0)

        # Calculate in place:
        image -= mean
        image /= 3 * std

        image.clamp_(0, 1)
        return image

    def _normalise_2(self, image: torch.Tensor) -> torch.Tensor:
        mean = image.mean()
        # correction=0 matches numpy's behaviour (without Bessel's correction)
        std = image.std(correction=1)

        # Calculate in place:
        image -= mean
        image /= 3 * std

        image.clamp_(-1, 1)
        return image

    def _resize_cv2(self, image: torch.Tensor, target_shape: tuple[int, int]):
        return torch.from_numpy(
            cv2.resize(
                image.numpy().squeeze(),
                target_shape,
                interpolation=cv2.INTER_LINEAR_EXACT,
            )[np.newaxis, np.newaxis, ...]
        ).to(self.device)

    def _resize_pytorch(
        self, image: torch.Tensor, target_shape: tuple[int, int]
    ) -> torch.Tensor:
        return v2.functional.resize(
            image,
            size=list(target_shape),
            interpolation=v2.InterpolationMode.BICUBIC,
            antialias=True,
        )

    def _preprocess(self, image: NDArray[Any]) -> torch.Tensor:
        with torch.no_grad():
            # the following functions expect channel and batch axes
            image_tensor = torch.from_numpy(
                image[np.newaxis, np.newaxis, ...].astype(np.float32)
            )

            # Calculate padding
            image_shape = (image_tensor.shape[-2], image_tensor.shape[-1])
            large_axis = np.argmax(image_shape)
            small_axis = 1 - large_axis
            axes_diff = image_shape[large_axis] - image_shape[small_axis]
            pad_size, remainder = divmod(axes_diff, 2)
            # Padding is [left, top, right, bottom]
            # Additional padding due to remainder will be added to the top or right
            padding = [0, pad_size + remainder, 0, pad_size]
            if large_axis == 0:
                padding = padding[::-1]

            if self._normalise_first:
                image_tensor = self._normalise_function(image_tensor)

            if self._pad:
                # Pad to square
                image_tensor = v2.functional.pad(image_tensor, padding, fill=0)
                target_shape = (self._image_size, self._image_size)
            else:
                target_shape = self._get_resize_shape(image_tensor)

            if not self._normalise_first:
                image_tensor = self._normalise_function(image_tensor)

            # Resize to input dimensions. Unfortunately albumentations uses cv2
            # which doesn't match the behaviour of pytorch, so numpy has to be
            # used.
            image_tensor = self._resize_function(
                image=image_tensor, target_shape=target_shape
            )

            if self._rgb:
                # Convert grayscale to 3-channel
                image_tensor = v2.functional.grayscale_to_rgb(image_tensor)
            return image_tensor.to(self.device)

    def _get_resize_shape(self, image: NDArray[Any] | torch.Tensor) -> tuple[int, int]:
        image_shape_array = np.asarray(image.shape[-2:])
        axis_multiplier = np.min(self._image_size / image_shape_array)
        new_shape_array = np.round(image_shape_array * axis_multiplier).astype(
            np.uint32
        )
        return (new_shape_array[0], new_shape_array[1])

    def _postprocess(
        self, prediction: torch.Tensor, image: NDArray[Any]
    ) -> NDArray[np.uint8]:
        # Trim off padding
        labels = super()._postprocess(prediction, image)

        if not self._pad:
            return labels

        resized_shape_array = np.asarray(self._get_resize_shape(image))
        labels_shape_array = np.asarray(labels.shape[-2:])

        if np.all(resized_shape_array == labels_shape_array):
            # If they are already the same shape, no need to slice.
            return labels

        padding_array = (labels_shape_array - resized_shape_array) / 2
        return labels[
            int(np.floor(padding_array[0])) : int(
                labels_shape_array[0] - np.ceil(padding_array[0])
            ),
            int(np.floor(padding_array[1])) : int(
                labels_shape_array[1] - np.ceil(padding_array[1])
            ),
        ]

    def load(self, model_path: str | PathLike[str]) -> torch.nn.Module:
        models_dict: dict[str, type[torch.nn.Module]] = {
            "unet": smp.Unet,
            "fpn": smp.FPN,
        }
        model_class = models_dict.get(self._model_type.lower())
        if model_class is None:
            raise ValueError(
                f"Invalid model type '{self._model_type.lower()}', available: {models_dict.keys()}"
            )
        model = model_class(
            encoder_name=self._encoder_name,
            encoder_weights=None,
            classes=self.num_classes,
            in_channels=3 if self._rgb else 1,
            activation=None,
            decoder_attention_type="scse",
        ).to(self.device)

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model


class Gen1QualityModel(Gen1Model):
    def __init__(
        self,
        model_path: str | PathLike[str],
        device: DeviceLikeType,
        max_image_size: int = 1536,
    ) -> None:
        super().__init__(
            model_path=model_path,
            device=device,
            max_image_size=max_image_size,
            encoder_name="efficientnet-b4",
            normalise_first=False,
            normalise_version=1,
            resize_version="cv2",
            model_type="unet",
        )


class Gen1PerformanceModel(Gen1Model):
    def __init__(
        self,
        model_path: str | PathLike[str],
        device: DeviceLikeType,
        max_image_size: int = 768,
    ) -> None:
        super().__init__(
            model_path=model_path,
            device=device,
            max_image_size=max_image_size,
            encoder_name="efficientnet-b3",
            normalise_first=False,
            normalise_version=1,
            resize_version="cv2",
            model_type="unet",
        )


class Gen1ImprovedPerformanceModel(Gen1Model):
    def __init__(
        self,
        model_path: str | PathLike[str],
        device: DeviceLikeType,
        max_image_size: int = 768,
    ) -> None:
        super().__init__(
            model_path=model_path,
            device=device,
            max_image_size=max_image_size,
            encoder_name="efficientnet-b3",
            normalise_first=True,
            normalise_version=1,
            resize_version="cv2",
            model_type="unet",
        )


class Gen1ImprovedQualityModel(Gen1Model):
    def __init__(
        self,
        model_path: str | PathLike[str],
        device: DeviceLikeType,
        max_image_size: int = 1536,
    ) -> None:
        super().__init__(
            model_path=model_path,
            device=device,
            max_image_size=max_image_size,
            encoder_name="efficientnet-b4",
            normalise_first=True,
            normalise_version=1,
            resize_version="cv2",
            model_type="unet",
        )


class Gen1GreyscalePerformanceModel(Gen1Model):
    def __init__(
        self,
        model_path: str | PathLike[str],
        device: DeviceLikeType,
        max_image_size: int = 768,
    ) -> None:
        super().__init__(
            model_path=model_path,
            device=device,
            max_image_size=max_image_size,
            encoder_name="efficientnet-b3",
            pad=False,
            rgb=False,
            normalise_first=True,
            normalise_version=1,
            resize_version="cv2",
            model_type="unet",
        )


class Gen1GreyscaleQualityModel(Gen1Model):
    def __init__(
        self,
        model_path: str | PathLike[str],
        device: DeviceLikeType,
        max_image_size: int = 1536,
    ) -> None:
        super().__init__(
            model_path=model_path,
            device=device,
            max_image_size=max_image_size,
            encoder_name="efficientnet-b4",
            pad=False,
            rgb=False,
            normalise_first=True,
            normalise_version=1,
            resize_version="cv2",
            model_type="unet",
        )


class Gen1ImprovedPreprocessingPerformanceModel(Gen1Model):
    def __init__(
        self,
        model_path: str | PathLike[str],
        device: DeviceLikeType,
        max_image_size: int = 768,
    ) -> None:
        super().__init__(
            model_path=model_path,
            device=device,
            max_image_size=max_image_size,
            encoder_name="efficientnet-b4",
            pad=False,
            rgb=False,
            normalise_first=True,
            normalise_version=2,
            resize_version="pytorch",
            model_type="unet",
        )


class Gen1ImprovedPreprocessingQualityModel(Gen1Model):
    def __init__(
        self,
        model_path: str | PathLike[str],
        device: DeviceLikeType,
        max_image_size: int = 1536,
    ) -> None:
        super().__init__(
            model_path=model_path,
            device=device,
            max_image_size=max_image_size,
            encoder_name="efficientnet-b4",
            pad=False,
            rgb=False,
            normalise_first=True,
            normalise_version=2,
            resize_version="pytorch",
            model_type="unet",
        )


class Gen1ImprovedPreprocessingFPNModel(Gen1Model):
    def __init__(
        self,
        model_path: str | PathLike[str],
        device: DeviceLikeType,
        max_image_size: int = 1536,
    ) -> None:
        super().__init__(
            model_path=model_path,
            device=device,
            max_image_size=max_image_size,
            encoder_name="efficientnet-b6",
            pad=False,
            rgb=False,
            normalise_first=True,
            normalise_version=2,
            resize_version="pytorch",
            model_type="fpn",
        )


class Gen1RGBImprovedPreprocessingFPNModel(Gen1Model):
    def __init__(
        self,
        model_path: str | PathLike[str],
        device: DeviceLikeType,
        max_image_size: int = 1536,
    ) -> None:
        super().__init__(
            model_path=model_path,
            device=device,
            max_image_size=max_image_size,
            encoder_name="efficientnet-b6",
            pad=False,
            rgb=True,
            normalise_first=True,
            normalise_version=2,
            resize_version="pytorch",
            model_type="fpn",
        )


# Using str keys allows for semantic versioning
MODEL_GENERATIONS_DICT: dict[str, type[AbstractAdaptivePolishingModel]] = {
    "0": Gen0Model,
    "1.0p": Gen1PerformanceModel,  # v<=3
    "1.0q": Gen1QualityModel,  # v<=3
    "1.1p": Gen1ImprovedPerformanceModel,  # v4
    "1.1q": Gen1ImprovedQualityModel,  # v4
    "1.2p": Gen1GreyscalePerformanceModel,  # v5
    "1.2q": Gen1GreyscaleQualityModel,  # v5
    # Updated preprocessing:
    "1.3p": Gen1ImprovedPreprocessingPerformanceModel,
    "1.3q": Gen1ImprovedPreprocessingQualityModel,
    "1.3fpn": Gen1ImprovedPreprocessingFPNModel,
    "1.4fpn": Gen1RGBImprovedPreprocessingFPNModel,
}


def _get_newest_generation_key() -> str:
    # If no generation is specified, get the last one specified
    return str(tuple(MODEL_GENERATIONS_DICT.keys())[-1])


def load_model(
    model_path: str | PathLike[str],
    generation: int | str | None = None,
    device: DeviceLikeType | None = None,
) -> AbstractAdaptivePolishingModel:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if generation is None:
        # If no generation is specified, get the latest generation one
        generation = _get_newest_generation_key()

    generation = str(generation)
    if generation not in MODEL_GENERATIONS_DICT:
        raise ValueError(
            f"Invalid model generation '{generation}' specified, available generations are: {', '.join(MODEL_GENERATIONS_DICT.keys())}"
        )

    _logger.info("Loading %s generation model", generation)
    model_class = MODEL_GENERATIONS_DICT[generation]
    if model_class is None:
        raise ValueError("%s")
    try:
        return model_class(model_path, device=device)
    except Exception:
        _logger.error(
            "Failed to load model with '%s'", model_class.__name__, exc_info=True
        )
        raise
