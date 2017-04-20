# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 12:11:09 2016

@author: drsmith
"""

#from fdp.classes.globals import VERBOSE
#VERBOSE = True

import fdp

nstxu = fdp.nstxu()
nstxu.addshot([140000, 141000, 141001, 204620])
print(dir(nstxu))

# diagnostic containers
print(dir(nstxu.s204620))

# logbook
nstxu.s141000.logbook()

# BES 1D plot and fft plot
bes = nstxu.s204620.bes
bes.ch01[:]
bes.ch01.plot()
bes.ch10.plotfft()

# MPTS 2d plot, point-axes for 2d, list signals, list containers, info
mpts = nstxu.s140000.mpts
spline = mpts.spline
mpts.ne.plot()
b = mpts.ne[0,0:12]
print(dir(b))
print(b.axes)
print(b.point_axes)
print(mpts.listSignals())
print(mpts.listContainers())
mpts.info()

# EFIT
eq = nstxu.s204620.equilibria
print(dir(eq))
print(eq.listAttributes())
print(eq.listContainers())
eq.efit02.ipmeas.plot()

# old and new magnetics, 1d slicing
nstxu.s141000.magnetics.highf.plot()
c = nstxu.s204620.magnetics.highn.highn_1[0:200]