# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 19:34:56 2015

@author: ktritz
"""

import sys
import os

#from . import classes
from .classes.fdp import Fdp
from . import methods

MDSplusEgg = os.path.join(os.path.dirname(os.path.abspath(__name__)), 'libs', 'MDSplus-7.0.62-py2.7.egg')
sys.path.append(MDSplusEgg)

# The 'fdp' package is remapping from fdp.__init__.py to classes.fdp.Fdp() object.
# The remapping provides this functionality:
# >>> import fdp
# >>> nstx = fdp.nstx
sys.modules[__name__] = Fdp()
