# -*- coding: utf-8 -*-
"""
Run the FDP test suite:

    % python testfdp.py
    
"""

import os.path
import unittest

if __name__=='__main__':
    TEST_DIR = os.path.join(os.path.dirname(__file__), 'testscripts')
    loader = unittest.defaultTestLoader.discover(TEST_DIR)
    unittest.TextTestRunner(verbosity=2).run(loader)