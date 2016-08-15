# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 11:10:19 2016

@author: drsmith
"""

import time

import fdp


nstx=fdp.nstx()

nstx.s204620.bes.gui()
time.sleep(5)

nstx.s204621.bes.ch05.plot()
time.sleep(0.2)

nstx.s204622.bes.ch06.gui()
time.sleep(0.2)

nstx.s204623.bes.ch08.plot()
time.sleep(0.2)

nstx.s204624.bes.gui()
time.sleep(0.2)

nstx.s204625.bes.ch05.plot()
time.sleep(0.2)

nstx.s204626.bes.ch03.gui()
time.sleep(0.2)

nstx.s204627.bes.ch05.plot()
time.sleep(0.2)

nstx.s204628.bes.gui()
time.sleep(0.2)
