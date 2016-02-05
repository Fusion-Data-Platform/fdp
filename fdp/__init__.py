# -*- coding: utf-8 -*-
"""
FDP is a data framework package for magnetic fusion experiments.

The 'fdp' package name is remapped to classes.fdp.Fdp().

**Usage**::

    >>> import fdp
    >>> dir(fdp)
    ['cmod', 'diiid', 'nstx']
    >>> nstx = fdp.nstx

"""

"""
Created on Wed Nov 25 19:34:56 2015

@author: ktritz
"""

import sys
from .classes.fdp import Fdp

# The 'fdp' package is remapping from fdp.__init__.py module to fdp.Fdp() object.
# The remapping provides this functionality:
# >>> import fdp
# >>> nstx = fdp.nstx
sys.modules[__name__] = Fdp()
