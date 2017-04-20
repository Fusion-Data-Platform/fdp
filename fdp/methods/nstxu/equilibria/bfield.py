# -*- coding: utf-8 -*-

import warnings

def bfield(obj, radius=None, z=None, time=None):
    # must call from equilibrium container
    if not obj.isContainer():
        warnings.warn(
            'Must call bfield() from equilibrium container, returning')
        return

    # if nstx.shot.equilibria.bfield(), default to efit02
    if obj._name == 'equilibria':
        print('Using EFIT02')
        eqdata = obj.efit02
    else:
        eqdata = obj

    # defaults
    if not radius:
        radius = 1.4   # major radius in m
    if not z:
        z = 0.0   # height in m
    if not time:
        time = 0.5   # time in s

    # get time index
    tindex = eqdata.rmaxis.getTimeIndex(time=time)

    if eqdata.mw[tindex] <= 1 or eqdata.mh[tindex] <= 1:
        warnings.warn(
            'Invalid reconstruction, returning')
        return