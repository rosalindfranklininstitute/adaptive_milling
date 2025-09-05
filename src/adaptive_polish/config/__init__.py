import typing
from adaptive_polish.config.adaptive_polish import AdaptivePolishMillingConfig
from adaptive_polish.config.bitmap import BitmapAdaptivePolishMillingConfig

TAdaptivePolishMillingConfig = typing.TypeVar(
    "TAdaptivePolishMillingConfig", bound=AdaptivePolishMillingConfig
)


__all__ = [
    "TAdaptivePolishMillingConfig",
    "AdaptivePolishMillingConfig",
    "BitmapAdaptivePolishMillingConfig",
]
