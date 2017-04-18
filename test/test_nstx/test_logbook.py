#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 21:28:09 2017

@author: drsmith
"""

from . import shotlist

def test_logbook(setup_nstx):
    nstx = setup_nstx
    # load a valid shot and check logbook
    nstx.addshot(shotlist=shotlist[0])
    shot = nstx[shotlist[0]]
    assert isinstance(shot.get_logbook(), list)
    assert len(shot.get_logbook()) > 0
    shot.logbook()
    # load an invalid shot and check logbook
    assert isinstance(nstx.s888888.get_logbook(), list)
    assert len(nstx.s888888.get_logbook()) == 0
