# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 19:34:56 2015

@author: ktritz
"""

import sys
import os

from .classes.fdp import Fdp
from . import methods

# The 'fdp' package is remapping from fdp.__init__.py to classes.fdp.Fdp() object.
# The remapping provides this functionality:
# >>> import fdp
# >>> nstx = fdp.nstx
sys.modules[__name__] = Fdp()
