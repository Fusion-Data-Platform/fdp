# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:59:49 2016

@author: drsmith
"""

import fdp

nstx = fdp.nstx()

shotnumbers = [141000, 204620]
nstx.addshot(shotnumbers)
for shotnumber in shotnumbers:
    shot = nstx[shotnumber]
    shot.magnetics.highf.plot()
    shot.magnetics.derived.midf_oddn.plot()
    shot.magnetics.highn.highn_1.plot()
