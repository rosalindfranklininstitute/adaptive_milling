import pytest
import typing
from contextlib import contextmanager


class ExceptionForMocking(Exception):
    pass


def assert_raises(
    exception: typing.Union[type[BaseException], None],
):  # -> type[DummyClass] | RaisesContext:# -> type[DummyClass] | RaisesContext:# -> type[DummyClass] | RaisesContext:
    @contextmanager
    def dummy_context() -> typing.Generator[None, None, None]:
        yield None

    if exception is None:
        return dummy_context()
    else:
        return pytest.raises(exception)
