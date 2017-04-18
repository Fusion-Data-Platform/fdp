#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:26:14 2017

@author: drsmith
"""

import pytest
import fdp

from . import shotlist

def test_nstx_machine(setup_nstx):
    nstx = setup_nstx
    nstx.addshot(shotlist=shotlist[0:2])
    for shot in nstx:
        pass
    shotlist[0] in nstx
    repr(nstx)
    nstx.__delitem__(shotlist[0])
    with pytest.raises(AttributeError):
        nstx.BadAttribute
    nstx[0]
    nstx.find('radius')

def test_shots():
    """Test shot addition and management"""
    nstx = fdp.nstx(shotlist=shotlist[2])
    assert shotlist[2] in nstx._shots
    assert isinstance(nstx[shotlist[2]], fdp.classes.shot.Shot)
    # add shot by method
    nstx.addshot(shotlist=shotlist[0])
    assert shotlist[0] in nstx._shots
    assert isinstance(nstx[shotlist[0]], fdp.classes.shot.Shot)
    # add shot by attribute reference
    shot = getattr(nstx, 's{}'.format(shotlist[1]))
    assert shotlist[1] in nstx._shots
    assert isinstance(shot, fdp.classes.shot.Shot)
    # check shotlist
    assert len(dir(nstx)) == 4
    assert len(nstx) == 3
    nstx.listshot()

def test_xp():
    nstx = fdp.nstx()
    nstx.addxp(xp=1037)
    assert len(nstx) == 95

def test_date():
    nstx = fdp.nstx()
    nstx.adddate(date='20160506')
    assert len(nstx) == 30

def test_filter_xp():
    nstx = fdp.nstx()
    xp1037 = nstx.filter_shots(xp=1037)
    assert len(xp1037) == 95

def test_filter_date():
    nstx = fdp.nstx()
    d = nstx.filter_shots(date='20160506')
    assert len(d) == 30