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

LIBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__name__)), 'libs')
mdsplus_egg = os.path.join(LIBS_DIR, 'mdsplus_alpha-7.0.218-py2.7.egg')
sys.path.insert(0, mdsplus_egg)
pymssql_whl = os.path.join(LIBS_DIR, 'pymssql-2.1.1-cp27-none-win_amd64.whl')
sys.path.append(pymssql_whl)

# The 'fdp' package is remapping from fdp.__init__.py to classes.fdp.Fdp() object.
# The remapping provides this functionality:
# >>> import fdp
# >>> nstx = fdp.nstx
sys.modules[__name__] = Fdp()
