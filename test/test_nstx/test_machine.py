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
    nstx.__delitem__(shotlist[0])

def test_shot_management():
    ishot = 0
    # load shot with machine instantiation
    nstx = fdp.nstx(shotlist=shotlist[ishot])
    assert shotlist[ishot] in nstx._shots
    assert isinstance(nstx[shotlist[ishot]], fdp.classes.shot.Shot)
    # load shot with adshot() method
    ishot += 1
    nstx.addshot(shotlist=shotlist[ishot])
    assert shotlist[ishot] in nstx._shots
    assert isinstance(nstx[shotlist[ishot]], fdp.classes.shot.Shot)
    # add shot by attribute reference
    ishot += 1
    shot = getattr(nstx, 's{}'.format(shotlist[ishot]))
    assert shotlist[ishot] in nstx._shots
    assert isinstance(shot, fdp.classes.shot.Shot)
    # check shotlist
    assert len(dir(nstx)) == 4
    assert len(nstx) == 3
    nstx.listshot()

def test_load_xp():
    nstx = fdp.nstx()
    nstx.addxp(xp=1038)
    assert len(nstx) == 24

def test_load_date():
    nstx = fdp.nstx()
    nstx.adddate(date='20160506')
    assert len(nstx) == 30

def test_filter_xp(setup_nstx):
    nstx = setup_nstx
    xp1037 = nstx.filter_shots(xp=1038)
    assert len(xp1037) == 24

def test_filter_date(setup_nstx):
    nstx = setup_nstx
    runday = nstx.filter_shots(date='20160506')
    assert len(runday) == 30