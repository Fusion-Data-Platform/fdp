#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:12:34 2017

@author: drsmith
"""

import fdp

nstx = fdp.nstx()

psirz = nstx.s204551.equilibria.efit01.psirz
psirz[:]
print(psirz.shape, psirz.R.axes, dir(psirz.R))
print(psirz)
print(psirz.shape, psirz.R.axes, dir(psirz.R))

b = psirz[0,:,:]
#print(b.shape, b.R.axes, dir(b.R))
#print(b)
#print(b.shape, b.R.axes, dir(b.R))
