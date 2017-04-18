# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 19:34:56 2015

@author: ktritz
"""

from .classes import fdp

__version__ = '0.1.4'


def nstxu(*args, **kwargs):
    return fdp.Fdp(*args, **kwargs).nstxu


def nstx(*args, **kwargs):
    return nstxu(*args, **kwargs)


def diiid(*args, **kwargs):
    return fdp.Fdp(*args, **kwargs).diiid


def d3d(*args, **kwargs):
    return diiid(*args, **kwargs)


def cmod(*args, **kwargs):
    return fdp.Fdp(*args, **kwargs).cmod
