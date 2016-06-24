# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:59:54 2015

@author: ktritz
"""


def change_units(signal, data):
    if signal.units == 'cm':
        data /= 100.
        signal.units = 'm'
    if signal.units == 'cm^-3':
        data *= 1.e6
        signal.units = 'm^-3'
    return data
    