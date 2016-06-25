# -*- coding: utf-8 -*-
"""
Run the FDP test suite:

    % python testfdp.py
    
"""

import os.path
import unittest

if __name__=='__main__':
    TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            'test')
    print('running tests in {}'.format(TEST_DIR))
    loader = unittest.defaultTestLoader.discover(TEST_DIR)
    unittest.TextTestRunner().run(loader)