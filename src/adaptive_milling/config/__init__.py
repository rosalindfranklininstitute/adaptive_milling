import typing

from adaptive_milling.config.adaptive_polish import AdaptivePolishMillingConfig

TAdaptivePolishMillingConfig = typing.TypeVar(
    "TAdaptivePolishMillingConfig", bound=AdaptivePolishMillingConfig
)


__all__ = [
    "AdaptivePolishMillingConfig",
    "TAdaptivePolishMillingConfig",
]
