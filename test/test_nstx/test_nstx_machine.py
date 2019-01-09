#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:26:14 2017

@author: drsmith
"""

import pytest
import fdp

from . import shotlist


def test_no_connection(setup_nstx):
    nstx = setup_nstx
    with pytest.raises(AttributeError):
        nstx.BadAttribute
    repr(nstx)
    nstx[0]
    nstx.find('radius')
    nstx = setup_nstx
    nstx.addshot(shotlist=shotlist[0:2])
    for shot in nstx:
        pass
    shotlist[0] in nstx
    del nstx[shotlist[0]]


def test_shot_management():
    ishot = 0
    # load shot with machine instantiation
    nstx = fdp.Nstxu(shotlist=shotlist[ishot])
    assert shotlist[ishot] in nstx
    assert isinstance(nstx[shotlist[ishot]], fdp.lib.shot.Shot)
    # load shot with adshot() method
    ishot += 1
    nstx.addshot(shotlist=shotlist[ishot])
    assert shotlist[ishot] in nstx
    assert isinstance(nstx[shotlist[ishot]], fdp.lib.shot.Shot)
    # add shot by attribute reference
    ishot += 1
    shot = getattr(nstx, 's{}'.format(shotlist[ishot]))
    assert shotlist[ishot] in nstx
    assert isinstance(shot, fdp.lib.shot.Shot)
    # add shot by item index
    ishot += 1
    shot = nstx[shotlist[ishot]]
    assert shotlist[ishot] in nstx
    assert isinstance(shot, fdp.lib.shot.Shot)
    # check shotlist
    assert len(nstx) == 4
    nstx.shotlist()


def test_load_xp():
    nstx = fdp.Nstxu()
    nstx.addshot(xp=1038)
    assert len(nstx) == 24


def test_load_date():
    nstx = fdp.Nstxu()
    nstx.addshot(date='20160506')
    assert len(nstx) == 30


def test_filter_xp(setup_nstx):
    nstx = setup_nstx
    xp1037 = nstx.filter(xp=1038)
    repr(xp1037)
    with pytest.raises(AttributeError):
        xp1037.BadAttribute
    assert len(xp1037) == 24
    for shot in xp1037:
        pass
    #xp1037.logbook()
    xp1037.shotlist()
    dir(xp1037)
    keys = list(xp1037._shots.keys())
    assert keys[0] in xp1037


def test_filter_date(setup_nstx):
    nstx = setup_nstx
    runday = nstx.filter(date='20160506')
    assert len(runday) == 30
