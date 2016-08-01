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

sig1 = bes.ch47 #SOL reference channel
sig2 = bes.ch46 
test = bisignal(sig1, sig2, tmin=0.47, tmax=0.52, nperseg=2048,
                offsetminimum=True, normalizetodc=True)

loga1power = 10*np.log10(test.asd1)
loga2power = 10*np.log10(test.asd2)
logcpower = 10*np.log10(test.crosspower)

plt.figure(1)
plt.plot(test.freqs, loga1power)
plt.axis([0,250,-80,0])
plt.xlabel('Frequency [kHz]')
plt.ylabel('Autopower ch47 [dB]')

plt.figure(2)
plt.plot(test.freqs, loga2power)
plt.axis([0,250,-80,0])
plt.xlabel('Frequency [kHz]')
plt.ylabel('Autopower ch46 [dB]')

plt.figure(3)
plt.plot(test.freqs, logcpower)
plt.axis([0,250,-80,0])
plt.xlabel('Frequency [kHz]')
plt.ylabel('Crosspower [dB]')

plt.figure(4)
plt.plot(test.freqs, test.crossphase)
plt.axis([0,250,-180,180])
plt.xlabel('Frequency [kHz]')
plt.ylabel('Cross phase [degrees]')

plt.figure(5)
plt.plot(test.freqs, test.cohere)
plt.axis([0,250,0,1])
plt.xlabel('Frequency [kHz]')
plt.ylabel('Coherence')