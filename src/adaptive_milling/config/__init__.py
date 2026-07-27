import typing

from adaptive_milling.config.adaptive_polish import AdaptivePolishMillingConfig
from adaptive_milling.config.bitmap import BitmapAdaptivePolishMillingConfig

TAdaptivePolishMillingConfig = typing.TypeVar(
    "TAdaptivePolishMillingConfig", bound=AdaptivePolishMillingConfig
)


__all__ = [
    "TAdaptivePolishMillingConfig",
    "AdaptivePolishMillingConfig",
    "BitmapAdaptivePolishMillingConfig",
]
