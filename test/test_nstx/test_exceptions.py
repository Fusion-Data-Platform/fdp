
from warnings import warn
import pytest
from fdp.lib.globals import FdpError, FdpWarning


def test_exceptions():
    with pytest.raises(FdpError):
        raise FdpError("error message")
    with pytest.warns(FdpWarning):
        warn("warning message", FdpWarning)
