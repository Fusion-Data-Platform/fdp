#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 21:28:09 2017

@author: drsmith
"""


def test_logbook(setup_nstx):
    nstx = setup_nstx
    shots = dir(nstx)
    # load a valid shot and check logbook
    shot = getattr(nstx, shots[1])
    assert isinstance(shot.get_logbook(), list)
    assert len(shot.get_logbook()) > 0
    shot.logbook()
    # load an invalid shot and check logbook
    shot = nstx.s888888
    assert isinstance(shot.get_logbook(), list)
    assert len(shot.get_logbook()) == 0
    shot.logbook()
