# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:15:00 2016

@author: drsmith
"""

import fdp
from scipy import signal
nstx = fdp.nstx()
bes = nstx.s204990.bes

#sig1 = bes.ch41
#sig2 = bes.ch41
#cs = CrossSignal(sig1, sig2, tmin=0.3, tmax=0.4, nperseg=4096, 
#                 offsetminimum=True, normalizetodc=True)
#                 
#powerspectrum = 10*np.log10(cs.crosspower_binavg)
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)
#ax.plot(cs.freqs, powerspectrum)
#ax.set_ylabel(r'$10\,\log_{10}(|FFT|^2)$ $(V^2/Hz)$')
#ax.set_xlim([0, 200])
#ax.set_xlabel('Frequency (kHz)')
#Compare autopower using Fft class to autopower from CrossSignal class
#bes.ch41.powerspectrum(tmin=0.3, tmax=0.4, fmax=200, power2=4096)

#bes.plotcrosspower('ch47','ch47',tmin=0.47,tmax=0.52,nperseg=2000,spectrum=False)
#bes.plotcrossphase('ch42','ch46',tmin=0.3,tmax=0.4,nperseg=2000,spectrum=False)
#bes.plotcoherence('ch42','ch46',fmax=1000,tmin=0.3,tmax=0.4,nperseg=2000)


signal.correlate()