 # -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:05:55 2016

@author: drsmith
"""

import fdp

nstx=fdp.nstxu()
bes = nstx.s204670.bes

#bes.ch41.plot()
#bes.ch42.plot()
#bes.ch43.plot()
#bes.ch44.plot()
#bes.ch45.plot()
#bes.ch46.plot()
#bes.ch47.plot()
#bes.ch48.plot()

#bes.ch41.plotfft(tmin=0.260, tmax=0.300, fmax=200)
#bes.ch42.plotfft(tmin=0.260, tmax=0.300, fmax=200)
#bes.ch43.plotfft(tmin=0.260, tmax=0.300, fmax=200)
#bes.ch44.plotfft(tmin=0.260, tmax=0.300, fmax=200)
#bes.ch45.plotfft(tmin=0.260, tmax=0.300, fmax=200)
#bes.ch46.plotfft(tmin=0.260, tmax=0.300, fmax=200)
#bes.ch47.plotfft(tmin=0.260, tmax=0.300, fmax=200)
bes.ch48.plotfft(tmin=0.250, tmax=0.330, fmax=200)

Lmode_spectrum = bes.ch48.powerspectrum(tmin=0.270, tmax=0.280, fmax=200, power2=4e3)
Hmode_spectrum = bes.ch48.powerspectrum(tmin=0.285, tmax=0.295, fmax=200, power2=4e3)