#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:12:34 2017

@author: drsmith
"""

import fdp

nstx = fdp.nstx()

eq = nstx.s204551.equilibria.efit01
print(eq.psirz)