from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from os import PathLike
    from typing import Any, Union

    from numpy.typing import NDArray

    DeviceLikeType = Union[str, torch.device, int]

    class ResizeProtocol(Protocol):
        def __call__(
            self, image: torch.Tensor, target_shape: tuple[int, int]
        ) -> torch.Tensor: ...

    class NormaliseProtocol(Protocol):
        def __call__(self, image: torch.Tensor) -> torch.Tensor: ...


def _open_image(path: str | PathLike[str]) -> NDArray[Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    if "tif" in suffix:
        # Opens as tifffile
        import tifffile

        return tifffile.imread(path)
    elif "png" in suffix:
        # opens as png file
        import cv2

        return np.asarray(cv2.imread(str(path), cv2.IMREAD_UNCHANGED))

    raise ValueError(f"Not a supported image '{path}'")


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
            except TypeError:
                raise ValueError(f"Failed to parse image '{image}'")
            if not image.is_file():
                raise FileNotFoundError(image)

            image = _open_image(image)

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
                return np.asarray(
                    Image.fromarray(labels).resize(
                        (image.shape[-1], image.shape[-2]),
                        resample=Image.Resampling.NEAREST,
                    ),
                    dtype=labels.dtype,
                )
            return labels
