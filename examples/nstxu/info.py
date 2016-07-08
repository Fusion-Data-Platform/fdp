# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 21:42:27 2016

@author: drsmith
"""

import fdp

nstx=fdp.nstx()
#shot=nstx.s204620
#shot.bes.ch01.info()
#shot.bes.info()
#
#shot.mpts.ne.info()
#shot.mpts.info()

shot=nstx.s141000
shot.chers.ti.info()
shot.chers.info()

shot.info(short=True)

