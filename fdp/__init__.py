# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 19:34:56 2015

@author: ktritz
"""

from .classes import fdp, fdp_globals

VERBOSE = fdp_globals.VERBOSE
TKROOT = fdp_globals.TKROOT

def nstxu():
    return fdp.Fdp().nstxu
    
def nstx():
    return nstxu()
    
def diiid():
    return fdp.Fdp().diiid
    
def d3d():
    return diiid()
    
def cmod():
    return fdp.Fdp().cmod
