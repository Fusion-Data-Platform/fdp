# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 19:10:56 2016

@author: drsmith
"""

import numpy as np


def getTimeIndex(obj, time=None):
    """
    Return time index <= input time
    """
    if not obj.isSignal():
        print('getTimeIndex() is only valid for signals, returning')
        return
    if time is None:
        print('getTimeIndex() requires a numeric scalar value')
        return
    output = None
    indlist = np.nonzero(obj.time > time)
    if indlist[0].size > 0:
        output = indlist[0][0] - 1
    return output
