#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 09:47:02 2017

@author: drsmith
"""

import fdp

nstxu = fdp.nstxu()

# BES 1D plot and fft plot
bes = nstxu.s204620.bes
bes.ch01.plot()
