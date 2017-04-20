# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 12:29:13 2016

@author: drsmith
"""

import fdp

nstx = fdp.nstx()

eq=nstx.s204551.equilibria

efit02 = eq.efit02
efit02.ipmeas.plot()

efit02.bfield(radius=145, time=0.6)