# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 12:11:09 2016

@author: drsmith
"""

import fdp

print('*********** Initiate NSTXU object ****************')
nstxu = fdp.nstxu()

print('*********** load shot 204620 ****************')
shot = nstxu.s204620

print('*********** load diagnostic ****************')
mpts = shot.mpts

print('*********** load signal ****************')
ne = mpts.ne

print('*********** load signal.time ****************')
ne.time[:]

print('*********** load signal.radius ****************')
ne.radius[:]

print('*********** load data ****************')
ne[:,:]

print('*********** slice data ****************')
a=ne[0:4,0:12]
