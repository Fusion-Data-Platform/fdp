# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 19:34:56 2015

@author: ktritz
"""

import sys
from classes import fdp

# Change fdp module mapping from fdp.__init__.py module to fdp.Fdp() object
# This change enables:
# >>> import fdp
# >>> nstx = fdp.nstx
sys.modules[__name__] = fdp.Fdp()
