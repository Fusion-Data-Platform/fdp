# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:05:55 2016

@author: drsmith
"""

import fdp

nstx=fdp.nstxu()
bes = nstx.s204620.bes
fft = bes.ch01.plotfft(tmin=0.2, tmax=0.45, fmax=200)

psd = bes.ch01.powerspectrum(tmin=0.213, tmax=0.218, fmax=150)

