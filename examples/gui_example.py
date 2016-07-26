# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 11:10:19 2016

@author: drsmith
"""

import fdp


nstx=fdp.nstx()
nstx.addxp(1013)
bes=nstx.s204620.bes
gui = bes.gui()


#from fdp.classes.gui import BaseGui
#gui = BaseGui(title='my window')
