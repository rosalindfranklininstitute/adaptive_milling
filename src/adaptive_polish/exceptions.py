from __future__ import annotations
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from adaptive_polish.enums import StopReasons


class _AdaptivePolishException(Exception):
    """Base class for all adaptive polish specific errors"""

    pass


class CentringException(_AdaptivePolishException):
    """Exception for centring issues"""

    pass


class SegmentationException(_AdaptivePolishException):
    """Exception for segmentation issues"""

    pass


class _AdaptivePolishMillingException(_AdaptivePolishException):
    """Base class for milling specific exceptions"""

    def __init__(self, *args, reason: StopReasons | str | None = None) -> None:
        if reason is None:
            reason = ", ".join(str(_) for _ in args)
        self.reason = reason
        super().__init__(*args)


class StopMillingException(_AdaptivePolishMillingException):
    """Exception for exiting adaptive milling when conditions are met"""

    pass


class StopEarlyError(_AdaptivePolishMillingException):
    """Error for stopping adaptive milling if something is wrong"""

    pass
