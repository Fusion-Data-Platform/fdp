# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 14:54:34 2016

@author: drsmith
"""

from warnings import warn

import numpy as np
import scipy as sp

from fdp.classes.fft import Fft
from fdp.classes.utilities import isSignal, isContainer
from fdp.classes.fdp_globals import FdpWarning

def coh(container, signal1, signal2):
    if not isContainer(container):
        warn("Method valid only at container-level", FdpWarning)
        return
    print('end coh()')
    result = 1
    freq=0
    return result, freq
