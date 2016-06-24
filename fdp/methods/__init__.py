# -*- coding: utf-8 -*-
"""
``fdp.methods`` contains methods for FDP objects (e.g. a plot method, ``>>> mpts.te.plot()``).  Methods can be specified at different levels: global, machine-specific, or diagnostic-specific.
"""

"""
Created on Thu Oct 29 10:20:43 2015

@author: ktritz
"""
from .plot import plot
from .utilities import listSignals, listContainers, \
    listMethods, listAttributes

__all__ = ['plot', 'listSignals', 'listContainers', 
           'listMethods', 'listAttributes']
