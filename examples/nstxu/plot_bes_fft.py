# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:05:55 2016

@author: drsmith
"""

import fdp

if __name__=='__main__':
    nstx=fdp.nstxu()
    bes = nstx.s204620.bes
    fft = bes.ch01.plotfft(tmin=0.2, tmax=0.45, fmax=200)
