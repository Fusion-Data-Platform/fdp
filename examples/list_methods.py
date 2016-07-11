# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:05:55 2016

@author: drsmith
"""

import fdp

nstx=fdp.nstxu()

bes = nstx.s204620.bes
print(bes.listMethods())
print(bes.listSignals())
print(bes.listAttributes())

sig=bes.ch01
print(sig.listMethods())
print(sig.listAttributes())

chers=nstx.s204620.chers
print(chers.listContainers())