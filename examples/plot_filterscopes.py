# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 12:29:13 2016

@author: drsmith
"""

import fdp

nstx = fdp.nstx()

fs = nstx.s141000.filterscopes

fs.bayg_dalpha_eies.plot()
fs.bayi_opipe_dalpha.plot()
fs.baye_dalf_haifa.plot()
fs.bayc_opipe_dalpha.plot()
