# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:59:54 2015

@author: ktritz
"""
import numpy as np


def sub_offset(signal, data):
    avg_data = np.mean(data[0:1000])
    return data - avg_data
