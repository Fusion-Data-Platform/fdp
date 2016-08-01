# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:15:00 2016

@author: drsmith
"""

import fdp
from fdp.classes.bisignal import Bisignal

nstx = fdp.nstx()

shot = nstx.s204620

sig1 = shot.bes.ch41
sig2 = shot.bes.ch42

bs12 = Bisignal(sig1, sig2)

bs11 = Bisignal(sig1, sig1)
