# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:21:43 2016

@author: drsmith
"""

import fdp


nstx=fdp.nstx()

shot=nstx.s204620
shot.rwm.irwm1.plot()
shot.rwm.irwm5.plot()