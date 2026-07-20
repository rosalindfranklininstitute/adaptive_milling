from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING, Protocol

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torchvision.transforms import v2

from adaptive_milling.models.abstract import AbstractAdaptivePolishingModel

if TYPE_CHECKING:
    from os import PathLike
    from typing import Any, Literal, Union

    from numpy.typing import NDArray

    DeviceLikeType = Union[str, torch.device, int]

    class ResizeProtocol(Protocol):
        def __call__(
            self, image: torch.Tensor, target_shape: tuple[int, int]
        ) -> torch.Tensor: ...

    class NormaliseProtocol(Protocol):
        def __call__(self, image: torch.Tensor) -> torch.Tensor: ...


_logger = logging.getLogger(__name__)


class AbstractGen1Model(AbstractAdaptivePolishingModel, ABC):
    def __init__(
        self,
        model_path: str | PathLike[str],
        device: DeviceLikeType,
        max_image_size: int,
        encoder_name: str,
        pad: bool = True,
        rgb: bool = True,
        normalise_first: bool = True,
        model_type: Literal["unet", "fpn"] = "unet",
    ) -> None:
        self._rgb = rgb
        self._pad = pad
        self._normalise_first = normalise_first
        self._model_type = model_type
        self._encoder_name = encoder_name
        self._image_size = max_image_size
        self.num_classes = 5
        AbstractAdaptivePolishingModel.__init__(
            self, model_path=model_path, device=device
        )

    def _normalise(self, image: torch.Tensor) -> torch.Tensor:
        mean = image.mean()
        # correction=0 matches numpy's behaviour (without Bessel's correction)
        std = image.std(correction=1)

        # Calculate in place:
        image -= mean
        image /= 3 * std

        image.clamp_(-1, 1)
        return image

    def _resize(
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
                image_tensor = self._normalise(image_tensor)

            if self._pad:
                # Pad to square
                image_tensor = v2.functional.pad(image_tensor, padding, fill=0)
                target_shape = (self._image_size, self._image_size)
            else:
                target_shape = self._get_resize_shape(image_tensor)

            if not self._normalise_first:
                image_tensor = self._normalise(image_tensor)

            # Resize to input dimensions. Unfortunately albumentations uses cv2
            # which doesn't match the behaviour of pytorch, so numpy has to be
            # used.
            image_tensor = self._resize(image=image_tensor, target_shape=target_shape)

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

        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        except Exception:
            _logger.error("Failed to load model state from '%s'", model_path)
            raise
        return model


class Gen1Model(AbstractGen1Model):
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
            model_type="fpn",
        )
