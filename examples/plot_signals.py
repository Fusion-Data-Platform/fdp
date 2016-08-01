# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 12:11:09 2016

@author: drsmith
"""

import fdp
    
nstxu = fdp.nstxu()

#shotlist = [nstxu.s140000, nstxu.s141000, nstxu.s141001, nstxu.s204620]
shotlist = [nstxu.s204620]

for shot in shotlist:
    print('SHOT {}'.format(shot.shot))
#    shot.bes.ch01.plot(tmin=0.1,tmax=0.5)
#    shot.mpts.ne.plot(tmin=0.1, tmax=0.4)
#    #shot.mpts.te.plot()
#    #shot.mpts.plot()
#    #shot.chers.ti.plot()
#    #shot.chers.spline.tis.plot()
#    shot.usxr.hdown.hdown01.plot(tmax=1)
#    #shot.usxr.hdown.plot()
#    shot.magnetics.highf.plot(tmin=0.5)
#    shot.magnetics.highn.highn_1.plot()
    shot.rwm.irwm1.plot()