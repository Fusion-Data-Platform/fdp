
from warnings import warn
import pytest
from fdp.classes.fdp_globals import FdpError, FdpWarning

def test_exceptions():
    with pytest.raises(FdpError):
        raise FdpError("error message")
    with pytest.warns(FdpWarning):
        warn("warning message", FdpWarning)
