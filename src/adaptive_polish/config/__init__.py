import typing
from adaptive_polish.config.adaptive_polish import AdaptivePolishMillingConfig

TAdaptivePolishMillingConfig = typing.TypeVar(
    "TAdaptivePolishMillingConfig", bound=AdaptivePolishMillingConfig
)


__all__ = [
    "TAdaptivePolishMillingConfig",
    "AdaptivePolishMillingConfig",
]
