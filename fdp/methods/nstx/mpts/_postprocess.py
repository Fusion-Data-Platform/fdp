# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:59:54 2015

@author: ktritz
"""


def _postprocess(signal, data):
    if signal._name in 'radius' and signal.units in 'cm':
        data /= 100.
        signal.units = 'm'
    return data
