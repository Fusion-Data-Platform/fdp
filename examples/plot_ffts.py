# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 16:39:53 2016

@author: drsmith
"""

import fdp

nstx=fdp.nstx()

shot=nstx.s141000

kwargs = {'tmin':0.2, 'tmax':0.3}

shot.bes.ch01.plotfft(**kwargs)

shot.magnetics.highf.plotfft(**kwargs)
shot.magnetics.highn.highn_1.plotfft(**kwargs)
shot.magnetics.highn.highn_7.plotfft(**kwargs)
shot.magnetics.highn.highn_16.plotfft(**kwargs)

shot.usxr.hdown.hdown00.plotfft(**kwargs)
shot.usxr.hdown.hdown15.plotfft(**kwargs)
shot.usxr.hup.hup00.plotfft(**kwargs)
shot.usxr.hup.hup15.plotfft(**kwargs)
shot.usxr.vtop.vtop00.plotfft(**kwargs)
shot.usxr.vtop.vtop15.plotfft(**kwargs)
