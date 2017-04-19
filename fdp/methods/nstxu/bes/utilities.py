# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:33:23 2016

@author: drsmith
"""

from ....classes.globals import FdpError
from ....classes.utilities import isSignal, isContainer


def get_signals_in_container(container):
    """Return list of attribute names corresponding to valid signals"""
    if not isContainer(container):
        raise FdpError("Expecting a container object")
    attrnames = dir(container)
    valid_signals = []
    for attrname in attrnames:
        attr = getattr(container, attrname)
        if isSignal(attr):
            try:
                attr[:]
                valid_signals.append(attr)
            except:
                print('{} is empty, ignoring'.format(attrname))
    return valid_signals
