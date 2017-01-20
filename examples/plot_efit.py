# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 12:29:13 2016

@author: drsmith
"""

import fdp

nstx = fdp.nstx()

eq=nstx.s204551.equilibria.efit01
eq.ipmeas.plot()

eq2=nstx.s204551.equilibria.efit02
eq2.kappa.plot()
