# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 19:34:56 2015

@author: ktritz
"""

import sys

#from . import classes
from .classes.fdp import Fdp
from . import methods

# make subpackages importable
#__all__ = ['classes', 'methods']

# The 'fdp' package is remapping from fdp.__init__.py to classes.fdp.Fdp() object.
# The remapping provides this functionality:
# >>> import fdp
# >>> nstx = fdp.nstx
sys.modules[__name__] = Fdp()
