# -*- coding: utf-8 -*-
"""
Run the FDP test suite:

    % python run.py

"""

import os.path
import unittest

loader = unittest.defaultTestLoader.discover(os.path.dirname(__file__))
unittest.TextTestRunner(verbosity=2).run(loader)
