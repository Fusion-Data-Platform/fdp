# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:09:56 2016

@author: dkriete
"""

import fdp
from fdp.classes.bisignal import bisignal
import matplotlib.pyplot as plt
import numpy as np

nstx = fdp.nstxu()
bes = nstx.s204990.bes

# It would be nice if usage syntax could be like
#   nstx = fdp.nstxu()
#   bes = nstx.s204990.bes
#   bes.bisignal(ch41, ch42).plot_crosspower
# instead of right now where I have to explicitly make a bisignal object

sig1 = bes.ch44 #SOL reference channel
sig2 = bes.ch44 
test = bisignal(sig1, sig1, tmin=0.47, tmax=0.52, nperseg=2048,
                offsetminimum=True, normalizetodc=True)

#loga2power = 10*np.log10(test.asd2)
#
#plt.figure(2)
#plt.plot(test.freqs, loga2power)
#plt.axis([0,250,-80,0])