# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:15:00 2016

@author: drsmith
"""

import fdp

nstx = fdp.nstx()

bes = nstx.s204990.bes

bes.plotcrosspower('ch46', 'ch47', tmin=0.47, tmax=0.52, nperseg=2048)

bes.plotcoherence('ch46', 'ch47', tmin=0.47, tmax=0.52, nperseg=2048)
