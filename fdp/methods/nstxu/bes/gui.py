# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:30:04 2016

@author: drsmith
"""

from warnings import warn

import matplotlib.pyplot as plt

from fdp.classes.gui import BaseGui
from fdp.classes.utilities import isSignal
from fdp.classes.fdp_globals import FdpWarning


def gui(signal):
    if not isSignal(signal):
        warn("Method valid only at signal-level", FdpWarning)
        return
    gui = BaseGui(signal)
    #gui.axes = gui.figure.add_subplot(1, 1, 1)
    #gui.axes.plot(signal.time[:], signal[:])
    #gui.canvas.show()
    return gui