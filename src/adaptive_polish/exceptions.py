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

    pass


class StopMillingException(_AdaptivePolishMillingException):
    """Exception for exiting adaptive milling when conditions are met"""

    pass


class StopEarlyError(_AdaptivePolishMillingException):
    """Error for stopping adaptive milling if something is wrong"""

    pass
