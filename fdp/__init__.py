# -*- coding: utf-8 -*-
"""
FDP is a data framework for magnetic fusion experiments

The 'fdp' module name is remapped to classes.fdp.Fdp()

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

# Change fdp module mapping from fdp.__init__.py module to fdp.Fdp() object
# This change enables:
# >>> import fdp
# >>> nstx = fdp.nstx
sys.modules[__name__] = Fdp()
